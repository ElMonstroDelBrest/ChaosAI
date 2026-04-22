"""Asymmetric Transformer predictor for JEPA (port from ../JEPA).

Deep-and-narrow Transformer predictor:
  - d_pred < d_model (intentionally weaker than encoder — anti-collapse)
  - Cross-attends to the FULL context sequence (not just the last token)
  - AttnRes: each block's query input is a softmax-mix of all prior queries

Anti-collapse argument (inherited from the ../JEPA reference): the asymmetric
predictor + EMA target encoder is sufficient to prevent representation
collapse on legitimate-correlated data (markets). Paired with a weak/no
VICReg var-cov term, this should work as well as the strong VICReg currently
used in Fin-JEPA, while giving up less bias from the regulariser.

Shape contract:
  Input:  h_x (B, S, d_model) — full context sequence
          target_positions (B, N_tgt) int64 — indices into a (B, tgt_len, d)
                                              output grid. Assumes sorted
                                              positions; masks handle padding.
  Output: (B, tgt_len, d_model) — predictions at all trained target slots.
          FinJEPA gathers the requested positions from this grid.

Attention is computed with a manual softmax(QK/sqrt(d)) @ V rather than
jax.nn.dot_product_attention, because on v6e XLA flags the reduce inside
SDPA's backward as "Reduction Function not modeled" (same bug fix as in
../JEPA/src/model.py:643-656).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array


class PredictorBlock(nn.Module):
    """One transformer predictor block: pre-norm SA -> pre-norm CA -> pre-norm FFN.

    Zero-init on the output projections of each sub-block so the model starts
    as a no-op residual (identity on pos_queries).
    """

    d_pred: int = 64
    n_heads: int = 4

    def setup(self):
        assert self.d_pred % self.n_heads == 0, (
            f"d_pred ({self.d_pred}) must be divisible by n_heads ({self.n_heads})"
        )
        self.head_dim = self.d_pred // self.n_heads
        self.scale = self.head_dim ** -0.5

    def _attn(self, q: Array, k: Array, v: Array) -> Array:
        """Manual softmax(QK/sqrt(d)) @ V — XLA-friendly on v6e."""
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        q = q.reshape(B, Tq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, Tk, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, Tk, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        attn = nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        return out.transpose(0, 2, 1, 3).reshape(B, Tq, -1)

    @nn.compact
    def __call__(self, q: Array, kv_ctx: Array) -> Array:
        # Self-attention on queries
        q_n = nn.RMSNorm(name="norm_sa")(q)
        sa_qkv = nn.Dense(3 * self.d_pred, use_bias=False, name="sa_qkv")(q_n)
        q_s, k_s, v_s = jnp.split(sa_qkv, 3, axis=-1)
        sa_out = nn.Dense(
            self.d_pred, use_bias=False, name="sa_out",
            kernel_init=nn.initializers.zeros,
        )(self._attn(q_s, k_s, v_s))
        q = q + sa_out

        # Cross-attention: queries read from the full context sequence
        q_c_n = nn.RMSNorm(name="norm_ca")(q)
        q_c = nn.Dense(self.d_pred, use_bias=False, name="ca_q")(q_c_n)
        ca_kv = nn.Dense(2 * self.d_pred, use_bias=False, name="ca_kv")(kv_ctx)
        k_c, v_c = jnp.split(ca_kv, 2, axis=-1)
        ca_out = nn.Dense(
            self.d_pred, use_bias=False, name="ca_out",
            kernel_init=nn.initializers.zeros,
        )(self._attn(q_c, k_c, v_c))
        q = q + ca_out

        # FFN: SiLU gated 2x expansion
        q_f_n = nn.RMSNorm(name="norm_ff")(q)
        ff = nn.Dense(2 * self.d_pred, use_bias=False, name="ff_in")(q_f_n)
        ff = nn.silu(ff)
        ff = nn.Dense(
            self.d_pred, use_bias=False, name="ff_out",
            kernel_init=nn.initializers.zeros,
        )(ff)
        return q + ff


class TransformerPredictor(nn.Module):
    """Asymmetric Transformer JEPA predictor.

    Args:
        d_model: encoder width (input to ctx_proj).
        d_pred: predictor width (narrower — anti-collapse).
        tgt_len: number of trained target slots (pos_queries size).
        n_layers: number of predictor blocks.
        n_heads: attention heads (d_pred % n_heads == 0).
        use_attn_res: if True, each block's query input is a softmax-mix
            of all prior block outputs (same idea as encoder AttnRes).
    """

    d_model: int = 128
    d_pred: int = 64
    tgt_len: int = 8
    n_layers: int = 6
    n_heads: int = 4
    use_attn_res: bool = True

    @nn.compact
    def __call__(self, ctx_seq: Array) -> Array:
        """ctx_seq: (B, S, d_model) — full context sequence.

        Returns: (B, tgt_len, d_model) — predictions at the tgt_len trained slots.
        """
        B = ctx_seq.shape[0]

        kv = nn.Dense(self.d_pred, use_bias=False, name="ctx_proj")(ctx_seq)
        pos_queries = self.param(
            "pos_queries",
            nn.initializers.normal(stddev=0.02),
            (self.tgt_len, self.d_pred),
        )
        q = jnp.broadcast_to(pos_queries[None, :, :], (B, self.tgt_len, self.d_pred))

        if self.use_attn_res:
            attnres_queries = self.param(
                "attnres_queries",
                nn.initializers.zeros,
                (self.n_layers, self.d_pred),
            )
            key_norm = nn.RMSNorm(name="attnres_keynorm")
            hidden = [q]
            normed = [key_norm(q)]
        else:
            attnres_queries = None
            key_norm = None
            hidden = None
            normed = None

        for i in range(self.n_layers):
            if self.use_attn_res and i > 0:
                V = jnp.stack(hidden, axis=0)              # (L_i, B, T, d_pred)
                K = jnp.stack(normed, axis=0)
                logits = jnp.einsum("d,lbtd->lbt", attnres_queries[i], K)
                alpha = nn.softmax(logits, axis=0)
                q_in = jnp.einsum("lbt,lbtd->btd", alpha, V)
            else:
                q_in = q

            q = PredictorBlock(
                d_pred=self.d_pred, n_heads=self.n_heads, name=f"blocks_{i}",
            )(q_in, kv)

            if self.use_attn_res:
                hidden.append(q)
                normed.append(key_norm(q))

        q = nn.RMSNorm(name="out_norm")(q)
        out = nn.Dense(self.d_model, use_bias=False, name="out_proj")(q)
        return out
