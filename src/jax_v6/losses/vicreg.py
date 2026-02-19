"""VICReg loss: Variance-Invariance-Covariance Regularization in JAX.

Port of strate_ii/vicreg.py. Float32 enforced for covariance to ensure
proper orthogonalization of Momentum vs Volatility dimensions.
"""

import jax.numpy as jnp
from jax import Array


def invariance_loss(z_a: Array, z_b: Array, mask: Array | None = None) -> Array:
    """MSE between paired representations. (N, D) -> scalar.

    Args:
        mask: optional (N,) float mask (1=valid, 0=padding).
    """
    sq = jnp.sum((z_a - z_b) ** 2, axis=-1)  # (N,)
    if mask is not None:
        return jnp.sum(sq * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.mean(sq)


def variance_loss(z: Array, gamma: float = 1.0, mask: Array | None = None) -> Array:
    """Hinge loss on per-dimension std to prevent collapse. (N, D) -> scalar."""
    if mask is not None:
        w = mask[:, None].astype(jnp.float32)  # (N, 1)
        n_valid = jnp.maximum(jnp.sum(mask), 1.0)
        mean = jnp.sum(z * w, axis=0) / n_valid
        var = jnp.sum(w * (z - mean) ** 2, axis=0) / n_valid
    else:
        var = jnp.var(z, axis=0)
    std = jnp.sqrt(var + 1e-4)
    return jnp.mean(jnp.maximum(gamma - std, 0.0))


def covariance_loss(z: Array, mask: Array | None = None) -> Array:
    """Off-diagonal covariance penalty. (N, D) -> scalar.

    MUST be called with float32 inputs for numerical stability.
    """
    z_f32 = z.astype(jnp.float32)
    d = z_f32.shape[-1]
    if mask is not None:
        w = mask[:, None].astype(jnp.float32)  # (N, 1)
        n_valid = jnp.maximum(jnp.sum(mask).astype(jnp.float32), 2.0)
        mean = jnp.sum(z_f32 * w, axis=0) / n_valid
        z_centered = (z_f32 - mean) * w
        cov = (z_centered.T @ z_centered) / (n_valid - 1)
    else:
        n = z_f32.shape[0]
        z_centered = z_f32 - jnp.mean(z_f32, axis=0)
        cov = (z_centered.T @ z_centered) / (n - 1)
    off_diag = jnp.sum(cov ** 2) - jnp.sum(jnp.diag(cov) ** 2)
    return off_diag / d


def vicreg_loss(
    z_a: Array,
    z_b: Array,
    inv_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    var_gamma: float = 1.0,
    mask: Array | None = None,
) -> dict[str, Array]:
    """Compute VICReg loss.

    Args:
        z_a: Predicted representations (N, D).
        z_b: Target representations (N, D). Should be stop_gradient'd.
        inv_weight: Weight for invariance (MSE) term.
        var_weight: Weight for variance (hinge) term.
        cov_weight: Weight for covariance (decorrelation) term.
        var_gamma: Target std for variance hinge.
        mask: optional (N,) float mask (1=valid, 0=padding).

    Returns:
        dict with total, invariance, variance, covariance losses.
    """
    inv = invariance_loss(z_a, z_b, mask)

    # Float32 for covariance and variance
    z_a_f = z_a.astype(jnp.float32)
    z_b_f = z_b.astype(jnp.float32)
    var = variance_loss(z_a_f, var_gamma, mask) + variance_loss(z_b_f, var_gamma, mask)
    cov = covariance_loss(z_a_f, mask) + covariance_loss(z_b_f, mask)

    total = inv_weight * inv + var_weight * var + cov_weight * cov

    return {
        "total": total,
        "invariance": inv,
        "variance": var,
        "covariance": cov,
    }
