"""Shared utilities — framework-agnostic where possible.

metrics, data_io, env_setup: numpy-only, no framework deps.
jax_checkpoint, jax_encoder: lazy-import JAX (only on TPU VMs).
math_utils: PyTorch (existing).
"""

from .metrics import compute_sharpe, compute_max_drawdown, compute_cum_return
from .env_setup import setup_tpu_env

# data_io, jax_checkpoint, jax_encoder have heavy deps (tf, jax) — import explicitly
# from .data_io import read_arrayrecord_tokens
# from .jax_checkpoint import load_jepa_checkpoint
# from .jax_encoder import create_encoder_from_config

__all__ = [
    "compute_sharpe",
    "compute_max_drawdown",
    "compute_cum_return",
    "setup_tpu_env",
]
