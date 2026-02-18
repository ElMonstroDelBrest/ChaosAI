"""Finite Scalar Quantization codebook (Mentzer et al., 2023 — DeepMind).

Replaces SphericalVQ-VAE's SphericalCodebook with a codebook-free quantizer:
no learned dictionary, no EMA updates, no dead-code collapse.

Theory:
  FSQ projects the encoder output to a low-dimensional FSQ space (fsq_dim),
  then quantizes each dimension independently to a fixed integer grid.
  The total number of codes is product(levels), e.g. [8,8,8,2] → 1024.

  No codebook collapse by construction:
    - Every code is reachable (the grid is fixed, not learned).
    - Fat tails are covered: tanh bounds the input to the grid range,
      so extreme events don't fall "outside" the dictionary.
    - No commitment loss needed (STE through the projection layers).

API:
  Drop-in replacement for SphericalCodebook. Same forward() dict keys:
    z_q, indices, commitment_loss (=0), codebook_loss (=0),
    perplexity, utilization.
  Same encode() and reset_usage() methods.
  Same `embeddings` buffer shape (num_codes, latent_dim).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class FSQCodebook(nn.Module):
    """Finite Scalar Quantization codebook.

    Args:
        num_codes: Total number of discrete codes = product(levels).
        latent_dim: Encoder/decoder interface dimension (e.g. 64).
        levels: Number of quantization levels per FSQ dimension, e.g. [8,8,8,2].
            The product must equal num_codes.
    """

    def __init__(
        self,
        num_codes: int,
        latent_dim: int,
        levels: list[int],
        # Unused VQ-VAE kwargs — accepted for drop-in config compatibility
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        dead_threshold: int = 2,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        assert math.prod(levels) == num_codes, (
            f"product(levels)={math.prod(levels)} != num_codes={num_codes}. "
            f"Adjust levels so their product equals num_codes."
        )
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.levels = levels
        self.fsq_dim = len(levels)

        # Learned projections: latent_dim ↔ fsq_dim
        # proj_in compresses to the FSQ grid space (high compression, e.g. 64→4).
        # proj_out expands back to latent_dim for the decoder.
        self.proj_in = nn.Linear(latent_dim, self.fsq_dim, bias=False)
        self.proj_out = nn.Linear(self.fsq_dim, latent_dim, bias=False)
        nn.init.orthogonal_(self.proj_in.weight)
        nn.init.orthogonal_(self.proj_out.weight)

        # Level metadata (non-learnable buffers)
        self.register_buffer(
            "levels_buf", torch.tensor(levels, dtype=torch.float32)
        )
        # Mixed-radix strides: stride[i] = product(levels[0..i-1])
        strides = torch.ones(self.fsq_dim, dtype=torch.long)
        for i in range(1, self.fsq_dim):
            strides[i] = strides[i - 1] * levels[i - 1]
        self.register_buffer("strides", strides)

        # `embeddings`: (num_codes, latent_dim) — proj_out(all grid points).
        # Used by FinJEPA.load_codebook() and decode_from_indices().
        # Refreshed via sync_embeddings() after each training epoch.
        self.register_buffer("embeddings", torch.zeros(num_codes, latent_dim))
        self.sync_embeddings()  # initial sync with orthogonal proj_out

    # ─── Quantization helpers ────────────────────────────────────────────────

    def _quantize(self, z_fsq: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize FSQ-space vectors with straight-through estimator.

        Maps each dimension i to the non-negative integer grid {0, ..., L_i-1},
        then centers to {-(L_i-1)/2, ..., (L_i-1)/2} for symmetric proj_out.

        Args:
            z_fsq: (B, fsq_dim) unconstrained encoder projections.

        Returns:
            z_q_nonneg: (B, fsq_dim) integer-valued non-negative grid coords.
            z_q_centered: (B, fsq_dim) centered grid coords (for proj_out).
        """
        L = self.levels_buf  # (fsq_dim,)

        # Map to (0, 1) via tanh, then scale to (0, L_i - 1)
        z_unit = (torch.tanh(z_fsq) + 1.0) / 2.0           # ∈ (0, 1)
        z_scaled = z_unit * (L - 1.0)                       # ∈ (0, L_i - 1)

        # Round STE: forward = rounded, backward = identity (gradient through)
        z_q_nonneg = z_scaled + (z_scaled.round() - z_scaled).detach()
        z_q_nonneg = z_q_nonneg.clamp(torch.zeros_like(L), L - 1.0)

        # Center: shift to symmetric representation for proj_out
        z_q_centered = z_q_nonneg - (L - 1.0) / 2.0        # ∈ [-(L-1)/2, (L-1)/2]

        return z_q_nonneg, z_q_centered

    def _to_indices(self, z_q_nonneg: Tensor) -> Tensor:
        """Convert non-negative grid coords to scalar integer indices.

        Uses mixed-radix encoding consistent with _enumerate_grid().

        Args:
            z_q_nonneg: (B, fsq_dim) non-negative integer-valued grid coords.

        Returns:
            (B,) int64 indices in [0, num_codes).
        """
        z_int = z_q_nonneg.round().long()                   # ensure integer
        L_int = self.levels_buf.long()
        z_int = z_int.clamp(
            torch.zeros_like(L_int), L_int - 1
        )
        return (z_int * self.strides).sum(dim=-1)           # (B,)

    def _enumerate_grid(self) -> tuple[Tensor, Tensor]:
        """Enumerate all num_codes grid points.

        Returns:
            nonneg: (num_codes, fsq_dim) — all non-negative integer grid coords.
            centered: (num_codes, fsq_dim) — same, centered around 0.
        """
        device = self.levels_buf.device
        # Stride-order enumeration: dim 0 varies fastest (stride=1)
        k = torch.arange(self.num_codes, dtype=torch.long, device=device)
        nonneg = torch.zeros(self.num_codes, self.fsq_dim, dtype=torch.float32, device=device)
        tmp = k.clone()
        for i, L in enumerate(self.levels):
            nonneg[:, i] = (tmp % L).float()
            tmp = tmp // L
        centered = nonneg - (self.levels_buf - 1.0) / 2.0
        return nonneg, centered

    # ─── Public API ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def sync_embeddings(self):
        """Refresh `embeddings` buffer from current proj_out weights.

        Call once after each training epoch (not every step).
        Cost: num_codes × fsq_dim projection — negligible.
        """
        _, centered = self._enumerate_grid()
        dtype = self.proj_out.weight.dtype
        self.embeddings.copy_(self.proj_out(centered.to(dtype)))

    def forward(self, z_e: Tensor) -> dict[str, Tensor]:
        """Quantize encoder output z_e via FSQ.

        Args:
            z_e: (B, latent_dim) encoder outputs.

        Returns:
            dict with z_q (STE quantized, same shape as z_e), indices,
            commitment_loss (=0), codebook_loss (=0), perplexity, utilization.
        """
        # Project to FSQ space and quantize
        z_fsq = self.proj_in(z_e)                           # (B, fsq_dim)
        z_q_nonneg, z_q_centered = self._quantize(z_fsq)   # both (B, fsq_dim)

        # Project quantized grid coords back to latent_dim
        z_q_latent = self.proj_out(z_q_centered)            # (B, latent_dim)

        # Straight-through estimator through the full projection chain.
        # Forward: use the quantized+projected representation.
        # Backward: gradient flows directly to z_e (as if quantization=identity).
        z_q_ste = z_e + (z_q_latent - z_e).detach()

        # Integer indices (detached — no gradient through index selection)
        indices = self._to_indices(z_q_nonneg.detach())     # (B,)

        # Batch perplexity and utilization metrics
        with torch.no_grad():
            onehot = torch.zeros(
                z_e.shape[0], self.num_codes, device=z_e.device, dtype=z_e.dtype
            )
            onehot.scatter_(1, indices.unsqueeze(1), 1.0)
            avg_probs = onehot.mean(0)
            perplexity = torch.exp(
                -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
            )
            utilization = (avg_probs > 0).float().mean()

        return {
            "z_q": z_q_ste,
            "indices": indices,
            "commitment_loss": z_e.new_tensor(0.0),   # no commitment loss in FSQ
            "codebook_loss": z_e.new_tensor(0.0),
            "perplexity": perplexity,
            "utilization": utilization,
        }

    @torch.no_grad()
    def encode(self, z_e: Tensor) -> Tensor:
        """Return discrete token indices. (B, latent_dim) -> (B,).

        Used by TopologicalTokenizer.tokenize() — inference only.
        """
        z_fsq = self.proj_in(z_e)
        z_q_nonneg, _ = self._quantize(z_fsq)
        return self._to_indices(z_q_nonneg)

    def reset_usage(self):
        """No-op. Required for API compatibility with SphericalCodebook."""
        pass
