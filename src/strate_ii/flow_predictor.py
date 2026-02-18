"""OT-CFM (Optimal Transport Continuous Flow Matching) predictor for Fin-JEPA.

Replaces the heuristic z ~ N(0, I) injection in the stochastic MLP predictor
with a learned vector field v_θ (Lipman et al. 2022).

v6-FINAL upgrade: Mini-batch OT coupling (Tong et al. 2023) matches noise
samples to targets via linear_sum_assignment, producing globally straighter
inter-sample trajectories. This enables 1-2 Euler steps at inference instead
of 10-50 (critical for real-time MCTS planning in Strate IV).

Training objective (per batch):
    t ~ U[0, 1],   x_0 ~ N(0, I)
    π = OT_coupling(x_0, h_y_tgt)          (mini-batch optimal transport)
    x_0 ← x_0[π]                           (permute noise by OT assignment)
    x_t = (1 - t) * x_0 + t * h_y_tgt      (OT straight-path interpolant)
    L_CFM = || v_θ(x_t, t, c) - (h_y_tgt - x_0) ||²

Inference: Euler integration from x_0 ~ N(0, I) to x_1 in 1-2 steps.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    from scipy.optimize import linear_sum_assignment

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous time t ∈ [0, 1].

    n_freq log-spaced frequencies → [sin, cos] → linear projection to d_model.
    """

    def __init__(self, d_model: int):
        super().__init__()
        n_freq = d_model // 2
        # Log-spaced from 1 to 10 000 (covers low and high frequency)
        exponent = torch.arange(n_freq, dtype=torch.float32) / max(n_freq - 1, 1)
        self.register_buffer("freqs", 10_000.0 ** exponent)  # (n_freq,)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

    def forward(self, t: Tensor) -> Tensor:
        """(B,) -> (B, d_model)."""
        # t may be any dtype; cast to float32 for stability then restore
        args = t.float().unsqueeze(-1) * self.freqs  # (B, n_freq)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, d_model)
        return self.proj(emb.to(t.dtype))


class FlowPredictor(nn.Module):
    """OT-CFM predictor: learns v_θ(x_t, t, context) → velocity field.

    Args:
        d_model: Encoder representation dimension.
        hidden_dim: Hidden dimension of the velocity-field MLP.
        n_layers: Number of MLP layers.
        seq_len: Maximum sequence length (for positional embeddings).
        dropout: Dropout probability.
        ot: Enable mini-batch Optimal Transport coupling (Tong et al. 2023).
        ot_batch_size: Sub-batch size for Hungarian assignment (O(n³)).
    """

    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 1024,
        n_layers: int = 4,
        seq_len: int = 128,
        dropout: float = 0.05,
        ot: bool = True,
        ot_batch_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.ot = ot and HAS_SCIPY
        self.ot_batch_size = ot_batch_size

        if ot and not HAS_SCIPY:
            import warnings
            warnings.warn(
                "OT-CFM requested but scipy not installed. "
                "Falling back to random coupling.",
                stacklevel=2,
            )

        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)

        # Velocity-field network: [x_t || t_emb || ctx || pos] → d_model
        # Named `vf` (not `mlp`) so checkpoint detection can tell CFM from old MLP.
        in_dim = d_model * 4
        layers: list[nn.Module] = []
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else d_model
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.vf = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # OT coupling
    # ------------------------------------------------------------------

    @staticmethod
    def _ot_permutation(
        x_0: Tensor,
        x_1: Tensor,
        sub_batch: int = 256,
    ) -> Tensor:
        """Mini-batch OT coupling via Hungarian algorithm (Tong et al. 2023).

        For each sub-batch of size min(B, sub_batch):
          1. Flatten spatial dims to (sub_B, N_tgt * d_model)
          2. Compute pairwise L2² cost matrix
          3. Solve assignment via scipy linear_sum_assignment
          4. Return permutation indices for x_0

        Args:
            x_0: (B, N_tgt, d_model) source samples (noise).
            x_1: (B, N_tgt, d_model) target samples (h_y_tgt).
            sub_batch: Size of sub-batches for assignment.

        Returns:
            (B,) int64 permutation. x_0[perm] is the OT-coupled noise.
        """
        B = x_0.shape[0]
        # Flatten to (B, D_flat) for cost computation
        x0_flat = x_0.reshape(B, -1).detach().cpu().float().numpy()
        x1_flat = x_1.reshape(B, -1).detach().cpu().float().numpy()

        perm = np.arange(B)

        for start in range(0, B, sub_batch):
            end = min(start + sub_batch, B)
            x0_sub = x0_flat[start:end]  # (n, D)
            x1_sub = x1_flat[start:end]  # (n, D)

            # C[i,j] = ||x_0[i] - x_1[j]||²
            x0_sq = (x0_sub**2).sum(axis=1, keepdims=True)  # (n, 1)
            x1_sq = (x1_sub**2).sum(axis=1, keepdims=True)  # (n, 1)
            cost = x0_sq + x1_sq.T - 2.0 * (x0_sub @ x1_sub.T)  # (n, n)

            _, col_idx = linear_sum_assignment(cost)
            perm[start:end] = col_idx + start

        return torch.from_numpy(perm).long()

    # ------------------------------------------------------------------
    # Core velocity prediction
    # ------------------------------------------------------------------

    def forward_velocity(
        self,
        x_t: Tensor,             # (B, N_tgt, d_model)
        t: Tensor,               # (B,) ∈ [0, 1]
        h_x: Tensor,             # (B, S, d_model)
        target_positions: Tensor,  # (B, N_tgt) int64
    ) -> Tensor:
        """Compute predicted velocity v_θ(x_t, t, context). Returns (B, N_tgt, d_model)."""
        N_tgt = x_t.shape[1]

        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, N_tgt, -1)  # (B, N_tgt, d_model)
        ctx = h_x[:, -1:, :].expand(-1, N_tgt, -1)                     # (B, N_tgt, d_model)
        pos = self.pos_embed(target_positions)                           # (B, N_tgt, d_model)

        inp = torch.cat([x_t, t_emb, ctx, pos], dim=-1)  # (B, N_tgt, 4*d_model)
        return self.vf(inp)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def forward(
        self,
        h_x: Tensor,             # (B, S, d_model)
        target_positions: Tensor,  # (B, N_tgt) int64
        h_y_tgt: Tensor,         # (B, N_tgt, d_model) = x_1 (target)
    ) -> tuple[Tensor, Tensor]:
        """OT-CFM training step (vectorized over t).

        Samples t ~ U[0,1] and x_0 ~ N(0,I), applies OT coupling to match
        noise samples to targets within the mini-batch, then returns
        (v_pred, v_tgt) so the caller can compute the CFM loss.

        Returns:
            v_pred: (B, N_tgt, d_model) predicted velocity.
            v_tgt:  (B, N_tgt, d_model) target velocity = h_y_tgt - x_0.
        """
        B = h_y_tgt.shape[0]
        t = torch.rand(B, device=h_y_tgt.device, dtype=h_y_tgt.dtype)  # (B,)
        x_0 = torch.randn_like(h_y_tgt)

        # OT coupling: permute x_0 so each noise sample is matched to its
        # nearest target under L2 cost → globally straighter trajectories.
        if self.ot and B > 1:
            perm = self._ot_permutation(x_0, h_y_tgt, self.ot_batch_size)
            x_0 = x_0[perm.to(x_0.device)]

        t_e = t.view(B, 1, 1)
        x_t = (1.0 - t_e) * x_0 + t_e * h_y_tgt
        v_tgt = h_y_tgt - x_0
        v_pred = self.forward_velocity(x_t, t, h_x, target_positions)
        return v_pred, v_tgt

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        h_x: Tensor,             # (B, S, d_model)
        target_positions: Tensor,  # (B, N_tgt) int64
        n_steps: int = 2,
    ) -> Tensor:
        """Euler integration from x_0 ~ N(0, I) to x_1. Returns (B, N_tgt, d_model)."""
        B, N_tgt = target_positions.shape
        x = torch.randn(B, N_tgt, self.d_model, device=h_x.device, dtype=h_x.dtype)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=h_x.device, dtype=h_x.dtype)
            x = x + dt * self.forward_velocity(x, t, h_x, target_positions)
        return x
