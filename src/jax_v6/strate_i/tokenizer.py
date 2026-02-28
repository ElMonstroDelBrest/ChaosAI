"""Strate I tokenizer ported to JAX/Flax — inference only (encode path).

Reproduces the exact tokenization of the PyTorch TopologicalTokenizer:
  patches (B, L, 5) → RevIN → Encoder → FSQ → (B,) int64 indices

Designed to run on TPU v6e-8 for massively parallel pretokenization.
All 440M patches processed in a single vectorized pass via jax.pmap.

Weight conversion from PyTorch checkpoint:
  load_from_pytorch_checkpoint() handles all shape transpositions:
    - Conv1d: PyTorch (out, in, k) → JAX (k, in, out)
    - Linear: PyTorch (out, in) → JAX (in, out)

Usage:
    from src.jax_v6.strate_i.tokenizer import StrateITokenizer, load_from_pytorch_checkpoint

    params = load_from_pytorch_checkpoint("checkpoints/strate-i-*.ckpt")
    tokenizer = StrateITokenizer(
        hidden_channels=128, latent_dim=64, n_layers=4,
        fsq_levels=[8, 8, 8, 2],
    )
    indices = tokenizer.apply({"params": params}, patches)  # (B, L, 5) → (B,)
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array


# ─── Causal Conv1d ────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """1D causal convolution with left-padding (channels-last).

    JAX conv operates channels-last: (B, L, C) → (B, L, C_out).
    Causal: pad (kernel_size - 1) * dilation on the left, zero on the right.
    """
    features: int
    kernel_size: int = 3
    dilation: int = 1

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """(B, L, C_in) → (B, L, C_out)."""
        pad = (self.kernel_size - 1) * self.dilation
        # Left-pad only (causal): (B, L, C) → (B, L+pad, C)
        x = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))
        return nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            kernel_dilation=(self.dilation,),
            padding="VALID",
            name="conv",
        )(x)


class CausalResidualBlock(nn.Module):
    """Two causal dilated convolutions with GELU + residual."""
    channels: int
    kernel_size: int = 3
    dilation: int = 1

    @nn.compact
    def __call__(self, x: Array) -> Array:
        residual = x
        x = nn.gelu(CausalConv1d(self.channels, self.kernel_size, self.dilation, name="conv1")(x))
        x = nn.gelu(CausalConv1d(self.channels, self.kernel_size, self.dilation, name="conv2")(x))
        return x + residual


# ─── Encoder ──────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Causal dilated conv encoder → latent on the unit sphere.

    (B, L, 5) → (B, latent_dim), L2-normalized.
    """
    hidden_channels: int = 128
    latent_dim: int = 64
    n_layers: int = 4
    dilation_base: int = 2
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """(B, L, C=5) → (B, latent_dim) on unit sphere."""
        # input_proj: pointwise conv (kernel_size=1)
        x = nn.Conv(
            features=self.hidden_channels,
            kernel_size=(1,),
            padding="VALID",
            name="input_proj",
        )(x)  # (B, L, hidden)

        # Dilated causal residual blocks
        for i in range(self.n_layers):
            dilation = self.dilation_base ** i
            x = CausalResidualBlock(
                self.hidden_channels,
                self.kernel_size,
                dilation,
                name=f"layers_{i}",
            )(x)

        # Take last timestep (causal: contains full receptive field)
        z = x[:, -1, :]  # (B, hidden)

        # Project to latent space
        z = nn.Dense(self.latent_dim, name="output_proj")(z)  # (B, latent_dim)

        # L2 normalize → unit sphere
        norm = jnp.maximum(jnp.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
        return z / norm


# ─── FSQ (Finite Scalar Quantization) ────────────────────────────────────────

class FSQQuantizer(nn.Module):
    """FSQ codebook — encode path only (no STE, no gradients).

    Projects latent → 4D FSQ space → quantize → mixed-radix scalar index.
    Also holds the `embeddings` buffer (1024, 64) for Strate II codebook init.
    """
    latent_dim: int = 64
    levels: tuple = (8, 8, 8, 2)

    def setup(self):
        self.fsq_dim = len(self.levels)
        self.num_codes = math.prod(self.levels)
        self.levels_arr = jnp.array(self.levels, dtype=jnp.float32)
        # Mixed-radix strides: stride[i] = product(levels[0..i-1])
        strides = [1]
        for i in range(1, self.fsq_dim):
            strides.append(strides[-1] * self.levels[i - 1])
        self.strides = jnp.array(strides, dtype=jnp.int32)

    @nn.compact
    def __call__(self, z_e: Array) -> Array:
        """(B, latent_dim) → (B,) int64 token indices."""
        # Project to FSQ space
        z_fsq = nn.Dense(self.fsq_dim, use_bias=False, name="proj_in")(z_e)  # (B, 4)

        # Quantize: tanh → scale → round → clamp
        L = self.levels_arr
        z_unit = (jnp.tanh(z_fsq) + 1.0) / 2.0            # ∈ (0, 1)
        z_scaled = z_unit * (L - 1.0)                       # ∈ (0, L_i-1)
        z_q = jnp.round(z_scaled)                           # integer-valued
        z_q = jnp.clip(z_q, 0.0, L - 1.0)                  # clamp to valid range

        # Mixed-radix → scalar index
        z_int = z_q.astype(jnp.int32)                       # (B, 4)
        indices = jnp.sum(z_int * self.strides, axis=-1)    # (B,)
        return indices.astype(jnp.int64)

    def get_embeddings(self, params: dict) -> Array:
        """Compute all 1024 embeddings = proj_out(centered_grid).

        Used by Strate II to initialize the codebook.
        Requires proj_out weights (not used in encode path).
        """
        L = self.levels_arr
        k = jnp.arange(self.num_codes, dtype=jnp.int32)
        nonneg = jnp.zeros((self.num_codes, self.fsq_dim), dtype=jnp.float32)
        tmp = k
        for i, lev in enumerate(self.levels):
            nonneg = nonneg.at[:, i].set((tmp % lev).astype(jnp.float32))
            tmp = tmp // lev
        centered = nonneg - (L - 1.0) / 2.0
        # proj_out: (fsq_dim, latent_dim)
        proj_out_kernel = params["proj_out"]["kernel"]  # (4, 64)
        return centered @ proj_out_kernel  # (1024, 64)


# ─── Full Tokenizer ──────────────────────────────────────────────────────────

class StrateITokenizer(nn.Module):
    """Complete Strate I tokenizer: Encoder + FSQ.

    Reproduces TopologicalTokenizer.tokenize() from PyTorch.
    RevIN is skipped for patch_length=1 (L=1 degeneracy).
    For L>1, RevIN normalizes per-instance before encoding.
    """
    hidden_channels: int = 128
    latent_dim: int = 64
    n_layers: int = 4
    dilation_base: int = 2
    kernel_size: int = 3
    fsq_levels: tuple = (8, 8, 8, 2)
    use_revin: bool = False  # True only for patch_length > 1

    @nn.compact
    def __call__(self, patches: Array) -> Array:
        """(B, L, 5) → (B,) int64 token indices.

        For L=1: patches are (B, 1, 5), RevIN skipped.
        For L>1: per-instance normalize before encoding.
        """
        B, L, C = patches.shape

        # RevIN: normalize per-instance (skip for L=1)
        if self.use_revin and L > 1:
            means = jnp.mean(patches, axis=1, keepdims=True)     # (B, 1, C)
            var = jnp.var(patches, axis=1, keepdims=True) + 1e-5
            stds = jnp.sqrt(var)
            patches = (patches - means) / stds

        # Encode → unit sphere
        z_e = Encoder(
            hidden_channels=self.hidden_channels,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
            dilation_base=self.dilation_base,
            kernel_size=self.kernel_size,
            name="encoder",
        )(patches)  # (B, latent_dim)

        # FSQ quantize → indices
        indices = FSQQuantizer(
            latent_dim=self.latent_dim,
            levels=self.fsq_levels,
            name="fsq",
        )(z_e)  # (B,)

        return indices


# ─── Weight Conversion ────────────────────────────────────────────────────────

def load_from_pytorch_checkpoint(ckpt_path: str) -> dict:
    """Convert PyTorch Strate I checkpoint to JAX param dict.

    Handles all shape transpositions:
      - Conv1d: PyTorch (C_out, C_in, K) → JAX Conv (K, C_in, C_out)
      - Linear: PyTorch (out, in) → JAX Dense kernel (in, out)

    Args:
        ckpt_path: Path to PyTorch Lightning .ckpt file.

    Returns:
        JAX params dict ready for StrateITokenizer.apply({"params": params}, ...).
    """
    import torch
    import numpy as np

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    def t(key: str) -> np.ndarray:
        """Extract tensor as numpy."""
        return state[key].detach().cpu().numpy()

    # ── Encoder ──
    encoder_params = {}

    # input_proj: Conv1d(5, 128, 1) — PyTorch (128, 5, 1) → JAX (1, 5, 128)
    encoder_params["input_proj"] = {
        "kernel": t("tokenizer.vqvae.encoder.input_proj.weight").transpose(2, 1, 0),
        "bias": t("tokenizer.vqvae.encoder.input_proj.bias"),
    }

    # 4 × CausalResidualBlock, each with conv1 and conv2
    for i in range(4):
        prefix = f"tokenizer.vqvae.encoder.layers.{i}"
        block_params = {}
        for conv_name in ["conv1", "conv2"]:
            # CausalConv1d wraps nn.Conv1d — PyTorch (128, 128, 3) → JAX (3, 128, 128)
            w = t(f"{prefix}.{conv_name}.conv.weight").transpose(2, 1, 0)
            b = t(f"{prefix}.{conv_name}.conv.bias")
            block_params[conv_name] = {"conv": {"kernel": w, "bias": b}}
        encoder_params[f"layers_{i}"] = block_params

    # output_proj: Linear(128, 64) — PyTorch (64, 128) → JAX (128, 64)
    encoder_params["output_proj"] = {
        "kernel": t("tokenizer.vqvae.encoder.output_proj.weight").T,
        "bias": t("tokenizer.vqvae.encoder.output_proj.bias"),
    }

    # ── FSQ ──
    fsq_params = {
        # proj_in: Linear(64, 4, no bias) — PyTorch (4, 64) → JAX (64, 4)
        "proj_in": {
            "kernel": t("tokenizer.vqvae.codebook.proj_in.weight").T,
        },
    }

    # proj_out is not used in encode path, but needed for embeddings sync
    if "tokenizer.vqvae.codebook.proj_out.weight" in state:
        fsq_params["proj_out"] = {
            "kernel": t("tokenizer.vqvae.codebook.proj_out.weight").T,  # (4, 64)
        }

    params = {
        "encoder": encoder_params,
        "fsq": fsq_params,
    }

    return params
