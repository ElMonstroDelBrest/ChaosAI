"""Transformer-based Encoder for Fin-JEPA ablation study.

Drop-in replacement for Mamba2Encoder. Accepts the same __init__ signature
(d_state and conv_kernel are silently ignored — no SSM).

Differences vs Mamba2Encoder:
  - Causal GPT-style multi-head self-attention instead of Mamba-2 SSM.
  - Weekend masking via key_padding_mask (don't attend TO weekend positions)
    instead of Δ=0 gating.
  - No vol_clock computation (vol_clock is the Phase-B Mamba-specific
    contribution being ablated here).

This lets us compare:
  Proof A — temporal extrapolation (Transformer vs Mamba on held-out futures)
  Proof B — Black Swan regime detection (vol_clock signal ablation)
"""

import torch
import torch.nn as nn
from torch import Tensor


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block (causal self-attention + FFN).

    Uses Pre-LN for training stability (same pattern as GPT-2).
    GELU activation in FFN matches modern practice.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(
        self,
        x: Tensor,
        causal_mask: Tensor,
        weekend_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (B, S, d_model)
            causal_mask: (S, S) float additive mask, -inf at future positions.
            weekend_mask: (B, S) float {0.0, 1.0} weekend indicator.
                If provided, weekend positions are masked as keys (key_padding_mask).

        Returns:
            (B, S, d_model)
        """
        # Weekend positions → float key_padding_mask (B, S), 0.0 / -inf.
        # Using float (same type as causal_mask) avoids the deprecation warning
        # from PyTorch about mixed bool/float attention masks.
        if weekend_mask is not None:
            # weekend_mask: 1.0 = weekend → -inf added to key logits → no attention
            float_kpm = weekend_mask.masked_fill(weekend_mask.bool(), float("-inf"))
            float_kpm = float_kpm.masked_fill(~weekend_mask.bool(), 0.0)
        else:
            float_kpm = None

        # Pre-norm self-attention (causal + optional weekend key mask)
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=causal_mask,   # (S, S) upper-triangular float -inf
            key_padding_mask=float_kpm,  # (B, S) float: 0.0 or -inf
            need_weights=False,
        )
        x = residual + attn_out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Causal Transformer encoder for Fin-JEPA (ablation counterpart to Mamba2Encoder).

    Identical embedding pipeline: frozen codebook → input_proj → [MASK] → pos_embed.
    Identical interface: forward(token_indices, weekend_mask, block_mask) → (B, S, d_model).
    Identical auxiliary method: load_codebook(weights).

    Args:
        num_codes: Size of codebook vocabulary (K=1024).
        codebook_dim: Dimension of codebook vectors (D=64).
        d_model: Model hidden dimension (128).
        d_state: SSM state dimension — ACCEPTED BUT IGNORED (no SSM here).
        n_layers: Number of Transformer blocks (6).
        n_heads: Number of attention heads (2).
        expand_factor: FFN hidden dim = expand_factor × d_model (2 → ffn_dim=256).
        conv_kernel: Causal conv kernel — ACCEPTED BUT IGNORED (no conv here).
        seq_len: Maximum sequence length (64).
    """

    def __init__(
        self,
        num_codes: int = 1024,
        codebook_dim: int = 64,
        d_model: int = 128,
        d_state: int = 16,        # ignored
        n_layers: int = 6,
        n_heads: int = 2,
        expand_factor: int = 2,
        conv_kernel: int = 4,     # ignored
        seq_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # ── Token embedding (identical to Mamba2Encoder) ──────────────────
        self.codebook_embed = nn.Embedding(num_codes, codebook_dim)
        self.codebook_embed.weight.requires_grad = False   # Frozen codebook

        self.input_proj = nn.Linear(codebook_dim, d_model)

        # Learned [MASK] token for masked positions
        self.mask_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Sinusoidal positional embedding (identical to Mamba2Encoder)
        self.register_buffer("pos_embed", self._sinusoidal_embed(seq_len, d_model))

        # ── Causal attention mask (static, registered as buffer) ──────────
        # Upper triangular: future positions get -inf so softmax → 0
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal)

        # ── Transformer blocks ────────────────────────────────────────────
        ffn_dim = expand_factor * d_model
        self.layers = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

    # ── Static helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _sinusoidal_embed(seq_len: int, d_model: int) -> Tensor:
        """Sinusoidal positional embedding — identical to Mamba2Encoder."""
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        dim = torch.arange(0, d_model, 2, dtype=torch.float32)
        angle = pos / (10000.0 ** (dim / d_model))
        embed = torch.zeros(seq_len, d_model)
        embed[:, 0::2] = torch.sin(angle)
        embed[:, 1::2] = torch.cos(angle)
        return embed.unsqueeze(0)  # (1, S, d_model)

    # ── Public API (mirrors Mamba2Encoder) ────────────────────────────────

    def load_codebook(self, codebook_weights: Tensor):
        """Load frozen codebook weights from Strate I.

        Args:
            codebook_weights: (K, D) codebook embedding matrix.
        """
        assert codebook_weights.shape == self.codebook_embed.weight.shape
        self.codebook_embed.weight.data.copy_(codebook_weights)

    def forward(
        self,
        token_indices: Tensor,
        weekend_mask: Tensor | None = None,
        block_mask: Tensor | None = None,
        exo_clock: Tensor | None = None,
    ) -> Tensor:
        """Encode a sequence of token indices (causal Transformer).

        Args:
            token_indices: (B, S) int64 token indices [0, K-1].
            weekend_mask: (B, S) float {0.0, 1.0} weekend indicator.
                Weekend positions are masked as attention keys.
            block_mask: (B, S) bool where True = masked (target) positions.
                Masked positions receive the learned [MASK] embedding.
            exo_clock: (B, S, 2) float exogenous [RV, Volume] clock signals.
                ACCEPTED BUT IGNORED — interface uniformity with Mamba2Encoder.

        Returns:
            (B, S, d_model) encoded representations.
        """
        B, S = token_indices.shape

        # 1. Codebook lookup → project to d_model
        x_embed = self.codebook_embed(token_indices)   # (B, S, codebook_dim)
        x = self.input_proj(x_embed)                   # (B, S, d_model)

        # 2. Replace masked positions with [MASK] embedding
        if block_mask is not None:
            mask_expanded = block_mask.unsqueeze(-1).float()   # (B, S, 1)
            x = x * (1.0 - mask_expanded) + self.mask_embed * mask_expanded

        # 3. Add positional embedding
        x = x + self.pos_embed[:, :S, :]

        # 4. Slice causal mask to actual sequence length
        causal = self.causal_mask[:S, :S]  # (S, S)

        # 5. Transformer stack
        for layer in self.layers:
            x = layer(x, causal_mask=causal, weekend_mask=weekend_mask)

        return self.norm(x)
