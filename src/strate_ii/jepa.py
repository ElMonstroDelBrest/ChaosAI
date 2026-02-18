"""Fin-JEPA: Financial Joint-Embedding Predictive Architecture.

Combines context encoder (E_x, Mamba-2), target encoder (E_y, EMA of E_x),
predictor, block masking, and VICReg loss.

Strate III additions:
- Stochastic predictor with latent noise z ~ N(0, I)
- output_proj: d_model → codebook_dim for decoding back to Strate I space
- generate_futures(): sample N divergent latent trajectories

Key invariants:
- E_y has NO gradients. All parameters have requires_grad=False.
- E_y weights are updated via EMA: param_y ← τ*param_y + (1-τ)*param_x
- VICReg covariance is computed in float32.
"""

import copy
import math

import torch
from torch import Tensor, nn

from .encoder import Mamba2Encoder
from .flow_predictor import FlowPredictor
from .predictor import Predictor
from .masking import generate_batch_masks
from .vicreg import VICRegLoss

_ENCODER_REGISTRY: dict[str, type] = {
    "mamba": Mamba2Encoder,
}


class FinJEPA(nn.Module):
    """Fin-JEPA model.

    Args:
        num_codes: Codebook size from Strate I.
        codebook_dim: Codebook vector dimension.
        d_model: Model hidden dimension.
        d_state: SSM state dimension.
        n_layers: Number of Mamba-2 blocks.
        n_heads: Number of SSM heads.
        expand_factor: Inner dim expansion factor.
        conv_kernel: Causal conv kernel size.
        seq_len: Maximum sequence length.
        pred_hidden_dim: Predictor hidden dimension.
        pred_n_layers: Predictor MLP depth.
        pred_dropout: Predictor dropout.
        pred_z_dim: Predictor latent noise dimension (0 = deterministic).
        cfm_weight: Weight of CFM loss relative to VICReg (0.0 = disable CFM).
        cfm_n_steps: Euler integration steps at inference.
        encoder_type: Which encoder backbone to use — "mamba" (default) or
            "transformer" (causal GPT-style, ablation counterpart).
        mask_ratio: JEPA mask ratio.
        block_size_min: Minimum mask block size.
        block_size_max: Maximum mask block size.
        inv_weight: VICReg invariance weight.
        var_weight: VICReg variance weight.
        cov_weight: VICReg covariance weight.
        var_gamma: VICReg variance target std.
        tau: Initial EMA momentum.
    """

    def __init__(
        self,
        num_codes: int = 1024,
        codebook_dim: int = 64,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 6,
        n_heads: int = 2,
        expand_factor: int = 2,
        conv_kernel: int = 4,
        seq_len: int = 64,
        pred_hidden_dim: int = 256,
        pred_n_layers: int = 2,
        pred_dropout: float = 0.1,
        pred_z_dim: int = 32,
        cfm_weight: float = 1.0,
        cfm_n_steps: int = 2,
        cfm_ot: bool = True,
        cfm_ot_batch_size: int = 256,
        encoder_type: str = "mamba",
        mask_ratio: float = 0.5,
        block_size_min: int = 4,
        block_size_max: int = 8,
        inv_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        var_gamma: float = 1.0,
        tau: float = 0.996,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.block_size_min = block_size_min
        self.block_size_max = block_size_max
        self.tau = tau
        self.d_model = d_model
        self.codebook_dim = codebook_dim
        self.cfm_weight = cfm_weight
        self.cfm_n_steps = cfm_n_steps

        # Resolve encoder class (lazy-import TransformerEncoder to avoid
        # importing torch transformer ops when using Mamba).
        if encoder_type == "transformer":
            from .transformer_encoder import TransformerEncoder
            _ENCODER_REGISTRY["transformer"] = TransformerEncoder
        if encoder_type not in _ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder_type={encoder_type!r}. "
                f"Choose from {list(_ENCODER_REGISTRY)}"
            )
        enc_cls = _ENCODER_REGISTRY[encoder_type]

        # Context encoder E_x (trained via gradient)
        self.context_encoder = enc_cls(
            num_codes=num_codes,
            codebook_dim=codebook_dim,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            n_heads=n_heads,
            expand_factor=expand_factor,
            conv_kernel=conv_kernel,
            seq_len=seq_len,
        )

        # Target encoder E_y (EMA of E_x, NO gradient)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor (stochastic with z noise)
        self.predictor = Predictor(
            d_model=d_model,
            hidden_dim=pred_hidden_dim,
            n_layers=pred_n_layers,
            seq_len=seq_len,
            dropout=pred_dropout,
            z_dim=pred_z_dim,
        )

        # Flow predictor (OT-CFM, Phase D) — None when cfm_weight == 0 to save memory
        if cfm_weight > 0.0:
            self.flow_predictor: FlowPredictor | None = FlowPredictor(
                d_model=d_model,
                hidden_dim=pred_hidden_dim,
                n_layers=pred_n_layers,
                seq_len=seq_len,
                dropout=pred_dropout,
                ot=cfm_ot,
                ot_batch_size=cfm_ot_batch_size,
            )
        else:
            self.flow_predictor = None

        # VICReg loss
        self.vicreg = VICRegLoss(
            inv_weight=inv_weight,
            var_weight=var_weight,
            cov_weight=cov_weight,
            var_gamma=var_gamma,
        )

        # Output projection: d_model → codebook_dim (for Strate I decoder)
        self.output_proj = nn.Linear(d_model, codebook_dim)
        self._init_output_proj()

    def _init_output_proj(self):
        """Initialize output_proj via pseudo-inverse of context_encoder.input_proj."""
        with torch.no_grad():
            W = self.context_encoder.input_proj.weight  # (d_model, codebook_dim)
            # Pseudo-inverse: (codebook_dim, d_model)
            W_pinv = torch.linalg.pinv(W)
            self.output_proj.weight.copy_(W_pinv)
            if self.context_encoder.input_proj.bias is not None:
                # Approximate bias correction
                self.output_proj.bias.zero_()
            else:
                self.output_proj.bias.zero_()

    def project_to_codebook_space(self, h: Tensor) -> Tensor:
        """Project from d_model space back to codebook_dim space.

        Args:
            h: (*, d_model) latent representations.

        Returns:
            (*, codebook_dim) projected representations.
        """
        return self.output_proj(h)

    def load_codebook(self, codebook_weights: Tensor):
        """Load frozen codebook weights into both encoders."""
        self.context_encoder.load_codebook(codebook_weights)
        self.target_encoder.load_codebook(codebook_weights)

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update: param_y ← τ*param_y + (1-τ)*param_x.

        Called AFTER each backward pass.
        """
        for p_y, p_x in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            p_y.data.mul_(self.tau).add_(p_x.data, alpha=1.0 - self.tau)

    def set_tau(self, tau: float):
        """Update EMA momentum (for cosine annealing schedule)."""
        self.tau = tau

    @torch.no_grad()
    def generate_futures(
        self,
        token_indices: Tensor,
        weekend_mask: Tensor | None,
        target_positions: Tensor,
        n_samples: int = 16,
        exo_clock: Tensor | None = None,
    ) -> Tensor:
        """Generate N stochastic future trajectories in latent space.

        Args:
            token_indices: (B, S) context token indices.
            weekend_mask: (B, S) apathy mask (or None).
            target_positions: (B, N_tgt) positions to predict.
            n_samples: Number of divergent futures to generate.
            exo_clock: (B, S, 2) exogenous [RV, Volume] clock (or None).

        Returns:
            (N, B, N_tgt, d_model) — N latent trajectories.
        """
        B = token_indices.shape[0]
        N_tgt = target_positions.shape[1]
        device = token_indices.device

        h_x = self.context_encoder(
            token_indices, weekend_mask=weekend_mask, exo_clock=exo_clock
        )

        futures = []
        if self.flow_predictor is not None:
            # CFM: each sample() call starts from a different x_0 ~ N(0, I)
            for _ in range(n_samples):
                h_pred = self.flow_predictor.sample(
                    h_x, target_positions, n_steps=self.cfm_n_steps
                )
                futures.append(h_pred)
        else:
            # Fallback: original Gaussian noise injection
            for _ in range(n_samples):
                z = torch.randn(B, N_tgt, self.predictor.z_dim, device=device)
                h_pred = self.predictor(h_x, target_positions, z=z)
                futures.append(h_pred)

        return torch.stack(futures)  # (N, B, N_tgt, d_model)

    def forward(
        self,
        token_indices: Tensor,
        weekend_mask: Tensor | None = None,
        exo_clock: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass: mask → encode → predict (with z noise) → VICReg loss.

        Args:
            token_indices: (B, S) int64 token indices.
            weekend_mask: (B, S) float {0.0, 1.0} weekend indicator.
            exo_clock: (B, S, 2) float exogenous [RV, Volume] clock (or None).

        Returns:
            dict with loss components and diagnostics.
        """
        B, S = token_indices.shape
        device = token_indices.device

        # 1. Generate block masks
        block_mask = generate_batch_masks(
            B, S,
            mask_ratio=self.mask_ratio,
            block_size_min=self.block_size_min,
            block_size_max=self.block_size_max,
            device=device,
        )

        # 2. Context encoder: sees full sequence with [MASK] at target positions
        h_x = self.context_encoder(
            token_indices,
            weekend_mask=weekend_mask,
            block_mask=block_mask,
            exo_clock=exo_clock,
        )  # (B, S, d_model)

        # 3. Target encoder: sees full sequence WITHOUT masking (no grad)
        with torch.no_grad():
            h_y = self.target_encoder(
                token_indices,
                weekend_mask=weekend_mask,
                block_mask=None,  # Target sees everything
                exo_clock=exo_clock,
            )  # (B, S, d_model)

        # 4. Extract target positions for each sample
        # Collect target positions (variable per sample, pad to max)
        target_positions_list = []
        max_targets = 0
        for b in range(B):
            tgt_pos = block_mask[b].nonzero(as_tuple=True)[0]
            target_positions_list.append(tgt_pos)
            max_targets = max(max_targets, len(tgt_pos))

        # Pad target positions and gather representations
        target_positions = torch.zeros(B, max_targets, dtype=torch.long, device=device)
        target_mask = torch.zeros(B, max_targets, dtype=torch.bool, device=device)

        for b in range(B):
            n = len(target_positions_list[b])
            target_positions[b, :n] = target_positions_list[b]
            target_mask[b, :n] = True

        # 5. Predict target representations with stochastic noise z
        z = torch.randn(B, max_targets, self.predictor.z_dim, device=device)
        h_y_pred = self.predictor(h_x, target_positions, z=z)  # (B, N_tgt, d_model)

        # 6. Gather true target representations from target encoder
        h_y_tgt = torch.gather(
            h_y, 1,
            target_positions.unsqueeze(-1).expand(-1, -1, h_y.shape[-1]),
        )  # (B, N_tgt, d_model)

        # 7. Flatten valid targets for VICReg (exclude padding)
        h_pred_flat = h_y_pred[target_mask]  # (N_valid, d_model)
        h_tgt_flat = h_y_tgt[target_mask]    # (N_valid, d_model)

        # 8. VICReg loss
        loss_dict = self.vicreg(h_pred_flat, h_tgt_flat.detach())
        total_loss = loss_dict["total"]

        # 9. CFM loss (Phase D) — train v_θ to transport N(0,I) → h_y_tgt
        cfm_loss = h_y_pred.new_tensor(0.0)
        if self.flow_predictor is not None and self.cfm_weight > 0.0:
            v_pred, v_tgt = self.flow_predictor(h_x, target_positions, h_y_tgt.detach())
            # Mask padding positions before MSE
            v_pred_flat = v_pred[target_mask]
            v_tgt_flat = v_tgt[target_mask]
            cfm_loss = (v_pred_flat - v_tgt_flat).pow(2).mean()
            total_loss = total_loss + self.cfm_weight * cfm_loss

        return {
            "loss": total_loss,
            "invariance": loss_dict["invariance"],
            "variance": loss_dict["variance"],
            "covariance": loss_dict["covariance"],
            "cfm_loss": cfm_loss,
            "mask_ratio": block_mask.float().mean(),
            "n_targets": target_mask.float().sum() / B,
        }
