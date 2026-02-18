"""Mamba-2 Encoder: token embedding + positional embedding + Mamba-2 stack.

Takes discrete token indices from Strate I codebook, embeds them via the
frozen codebook vectors, projects to d_model, adds positional embeddings,
and passes through N Mamba-2 blocks.

Volume-Clock (Phase B — v6):
  The encoder derives a per-token volatility proxy from the L2 distance
  between consecutive codebook embeddings and passes it to each Mamba-2
  block as `vol_clock`. This is zero-overhead (the codebook lookup is
  already done) and requires no external data or pipeline changes.
"""

import torch
from torch import Tensor, nn

from .mamba2_block import Mamba2Block


class Mamba2Encoder(nn.Module):
    """Encoder stack for Fin-JEPA.

    Embeds token indices using frozen codebook, projects to d_model,
    adds sinusoidal positional embeddings, then applies N Mamba-2 blocks.

    Args:
        num_codes: Size of codebook vocabulary (K=1024).
        codebook_dim: Dimension of codebook vectors (D=64).
        d_model: Model hidden dimension (128).
        d_state: SSM state dimension (16).
        n_layers: Number of Mamba-2 blocks (6).
        n_heads: Number of SSM heads (2).
        expand_factor: Expansion factor for inner dim (2).
        conv_kernel: Causal conv kernel size (4).
        seq_len: Maximum sequence length (64).
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
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Token embedding: frozen codebook lookup + learned projection
        self.codebook_embed = nn.Embedding(num_codes, codebook_dim)
        self.codebook_embed.weight.requires_grad = False  # Frozen
        self.input_proj = nn.Linear(codebook_dim, d_model)

        # Learned [MASK] embedding for masked positions
        self.mask_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Sinusoidal positional embedding
        self.register_buffer("pos_embed", self._sinusoidal_embed(seq_len, d_model))

        # Mamba-2 stack
        self.layers = nn.ModuleList([
            Mamba2Block(d_model, d_state, n_heads, expand_factor, conv_kernel)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _sinusoidal_embed(seq_len: int, d_model: int) -> Tensor:
        """Generate sinusoidal positional embeddings."""
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        dim = torch.arange(0, d_model, 2, dtype=torch.float32)
        angle = pos / (10000.0 ** (dim / d_model))
        embed = torch.zeros(seq_len, d_model)
        embed[:, 0::2] = torch.sin(angle)
        embed[:, 1::2] = torch.cos(angle)
        return embed.unsqueeze(0)  # (1, S, d_model)

    def load_codebook(self, codebook_weights: Tensor):
        """Load frozen codebook weights from Strate I.

        Args:
            codebook_weights: (K, D) codebook embedding matrix from Strate I.
        """
        assert codebook_weights.shape == self.codebook_embed.weight.shape
        self.codebook_embed.weight.data.copy_(codebook_weights)

    @staticmethod
    def _compute_vol_clock(x_embed: Tensor) -> Tensor:
        """Compute realized-volatility proxy from codebook embedding transitions.

        vol_clock_t = ||embed_t - embed_{t-1}||_2 (L2 distance between
        consecutive codebook vectors), z-scored per sequence.

        High values signal regime changes (proxy for local market volatility).
        Position 0 is assigned the sequence mean (no previous token exists).

        Args:
            x_embed: (B, S, codebook_dim) raw codebook embeddings (pre-proj).

        Returns:
            (B, S) float32 z-scored volatility proxy, detached from graph.
        """
        with torch.no_grad():
            # (B, S-1, D) → (B, S-1) L2 distances
            diffs = x_embed[:, 1:, :] - x_embed[:, :-1, :]
            dist = diffs.norm(dim=-1)  # (B, S-1)

            # Prepend sequence mean for position 0 (no predecessor)
            dist0 = torch.cat([dist.mean(dim=1, keepdim=True), dist], dim=1)  # (B, S)

            # Z-score per sequence: zero-mean, unit-variance
            mu = dist0.mean(dim=1, keepdim=True)
            sigma = dist0.std(dim=1, keepdim=True).clamp(min=1e-6)
            return ((dist0 - mu) / sigma).detach()

    def forward(
        self,
        token_indices: Tensor,
        weekend_mask: Tensor | None = None,
        block_mask: Tensor | None = None,
        exo_clock: Tensor | None = None,
    ) -> Tensor:
        """Encode a sequence of token indices.

        Args:
            token_indices: (B, S) int64 token indices [0, K-1].
            weekend_mask: (B, S) float {0.0, 1.0} weekend indicator.
            block_mask: (B, S) bool where True = masked (target) positions.
                If provided, masked positions get [MASK] embedding.
            exo_clock: (B, S, 2) float exogenous [RV, Volume] clock signals.
                When present, the endogenous L2 vol_clock is NOT computed
                (kills the codebook feedback loop). Falls back to L2 when None.

        Returns:
            (B, S, d_model) encoded representations.
        """
        B, S = token_indices.shape

        # Embed tokens: codebook lookup → projection
        x_embed = self.codebook_embed(token_indices)  # (B, S, codebook_dim)
        x = self.input_proj(x_embed)  # (B, S, d_model)

        # Clock signals: exogenous takes priority, L2 is fallback
        vol_clock = None
        if exo_clock is None:
            # Endogenous fallback: L2 distance between consecutive codebook
            # embeddings, z-scored per sequence. WARNING: creates a feedback
            # loop if the codebook collapses. Use exo_clock when available.
            vol_clock = self._compute_vol_clock(x_embed)  # (B, S)

        # Replace masked positions with [MASK] embedding
        if block_mask is not None:
            mask_expanded = block_mask.unsqueeze(-1).float()  # (B, S, 1)
            x = x * (1.0 - mask_expanded) + self.mask_embed * mask_expanded

        # Add positional embedding
        x = x + self.pos_embed[:, :S, :]

        # Pass through Mamba-2 stack with clock conditioning
        for layer in self.layers:
            x = layer(
                x,
                weekend_mask=weekend_mask,
                vol_clock=vol_clock,
                exo_clock=exo_clock,
            )

        return self.norm(x)
