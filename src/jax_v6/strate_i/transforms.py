"""JAX data transforms for Strate I: log-returns, patch extraction, exo_clock.

All operations are pure JAX — runs on TPU. No PyTorch dependencies.
Vectorized across the full dataset (no per-file Python loop).

Usage:
    import jax.numpy as jnp
    from src.jax_v6.strate_i.transforms import compute_log_returns, extract_patches

    ohlcv = jnp.array(np.load("pair.npy"))  # (T, 5)
    log_ret = compute_log_returns(ohlcv)     # (T-1, 5)
    patches = extract_patches(log_ret, 1, 1) # (T-1, 1, 5)
"""

import jax
import jax.numpy as jnp
from jax import Array


def compute_log_returns(ohlcv: Array) -> Array:
    """Log-returns for OHLC + volume transform.

    Matches PyTorch src/strate_i/data/transforms.py:compute_log_returns exactly.

    Input:  (T, 5) — OHLCV
    Output: (T-1, 5) — log-returns for prices, log(1 + v/mean_v) for volume.
    """
    prices = ohlcv[:, :4]
    price_log_returns = jnp.log(prices[1:] + 1e-9) - jnp.log(prices[:-1] + 1e-9)

    volume = ohlcv[:, 4]
    mean_volume = jnp.mean(volume)

    volume_transform = jnp.where(
        mean_volume > 1e-8,
        jnp.log1p(volume[1:] / jnp.maximum(mean_volume, 1e-8)),
        jnp.zeros_like(volume[1:]),
    )

    return jnp.concatenate([price_log_returns, volume_transform[:, None]], axis=-1)


def extract_patches(series: Array, patch_length: int = 1, stride: int = 1) -> Array:
    """Extract sliding window patches from time series.

    Input:  (T, C)
    Output: (N, L, C) where N = (T - patch_length) // stride + 1

    For patch_length=1, stride=1: equivalent to series[:, None, :] → (T, 1, C).
    """
    T, C = series.shape

    if patch_length == 1 and stride == 1:
        # Fast path: no sliding window needed
        return series[:, None, :]  # (T, 1, C)

    # General sliding window via lax.dynamic_slice
    n_patches = (T - patch_length) // stride + 1
    starts = jnp.arange(n_patches) * stride  # (N,)

    def _extract_one(start):
        return jax.lax.dynamic_slice(series, (start, 0), (patch_length, C))

    return jax.vmap(_extract_one)(starts)  # (N, L, C)


def compute_exo_clock(patches: Array) -> Array:
    """Exogenous clock signals (RV + Volume), z-scored per asset.

    Input:  (N, L, 5) patches
    Output: (N, 2) — [RV_zscore, Volume_zscore]

    For L=1: RV = abs(OHLC mean), Volume = abs(vol).
    """
    N = patches.shape[0]
    ohlc = patches[:, :, :4]
    rv = jnp.std(ohlc.reshape(N, -1), axis=1)
    vol = jnp.mean(jnp.abs(patches[:, :, 4]), axis=1)

    def _zscore(x: Array) -> Array:
        mu = jnp.mean(x)
        sigma = jnp.maximum(jnp.std(x), 1e-6)
        return (x - mu) / sigma

    return jnp.stack([_zscore(rv), _zscore(vol)], axis=-1)  # (N, 2)


def compute_apathy_mask(patches: Array, percentile: float = 10.0) -> Array:
    """Apathy mask: 1.0 for low-volatility patches (below percentile).

    Input:  (N, L, 5) patches
    Output: (N,) float32 — 1.0 = low-vol (masked), 0.0 = normal
    """
    ohlc = patches[:, :, :4]
    N = patches.shape[0]
    volatilities = jnp.std(ohlc.reshape(N, -1), axis=1)

    # jnp.percentile for the threshold
    threshold = jnp.percentile(volatilities, percentile)
    return (volatilities < threshold).astype(jnp.float32)
