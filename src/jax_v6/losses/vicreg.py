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
    """Log-variance loss to prevent collapse. (N, D) -> scalar.

    Uses -log(std/γ + ε) instead of hinge max(0, γ-std). The log creates
    infinite gradient pressure as std→0, making collapse structurally impossible.
    Falls to 0 when std >= gamma (via ReLU clamp).
    """
    if mask is not None:
        w = mask[:, None].astype(jnp.float32)  # (N, 1)
        n_valid = jnp.maximum(jnp.sum(mask), 1.0)
        mean = jnp.sum(z * w, axis=0) / n_valid
        var = jnp.sum(w * (z - mean) ** 2, axis=0) / n_valid
    else:
        var = jnp.var(z, axis=0)
    std = jnp.sqrt(var + 1e-4)
    return jnp.mean(jnp.maximum(-jnp.log(std / gamma + 1e-6), 0.0))


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


def barlow_twins_loss(
    z_a: Array,
    z_b: Array,
    lambda_off: float = 0.0051,
    mask: Array | None = None,
) -> dict[str, Array]:
    """Barlow Twins loss — cross-correlation matrix pushed to identity.

    Per-dimension normalization ensures equal gradient on ALL dims, preventing
    the structural dimensional collapse caused by KAN polynomial basis correlations.

    Args:
        z_a: (N, D) predicted representations (float32 recommended).
        z_b: (N, D) target representations.
        lambda_off: Weight for off-diagonal terms (redundancy reduction).
        mask: optional (N,) float mask (1=valid, 0=padding).

    Returns:
        dict with total, on_diag, off_diag losses.
    """
    z_a = z_a.astype(jnp.float32)
    z_b = z_b.astype(jnp.float32)

    if mask is not None:
        w = mask[:, None]
        n_valid = jnp.maximum(jnp.sum(mask), 2.0)
        mean_a = jnp.sum(z_a * w, axis=0) / n_valid
        mean_b = jnp.sum(z_b * w, axis=0) / n_valid
        z_a = (z_a - mean_a) * w
        z_b = (z_b - mean_b) * w
        std_a = jnp.sqrt(jnp.sum(z_a ** 2, axis=0) / n_valid + 1e-4)
        std_b = jnp.sqrt(jnp.sum(z_b ** 2, axis=0) / n_valid + 1e-4)
    else:
        n_valid = float(z_a.shape[0])
        z_a = z_a - z_a.mean(axis=0)
        z_b = z_b - z_b.mean(axis=0)
        std_a = jnp.sqrt(jnp.var(z_a, axis=0) + 1e-4)
        std_b = jnp.sqrt(jnp.var(z_b, axis=0) + 1e-4)

    z_a_n = z_a / std_a
    z_b_n = z_b / std_b
    C = (z_a_n.T @ z_b_n) / n_valid  # (D, D) cross-correlation matrix

    D = C.shape[0]
    on_diag = jnp.sum((1.0 - jnp.diag(C)) ** 2)
    off_diag = (jnp.sum(C ** 2) - jnp.sum(jnp.diag(C) ** 2)) / (D * (D - 1))
    total = on_diag / D + lambda_off * off_diag

    return {"total": total, "on_diag": on_diag / D, "off_diag": off_diag}


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
        var_weight: Weight for variance (log) term.
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


def cross_resolution_loss(
    h_x: Array,
    R: int = 4,
    inv_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    var_gamma: float = 1.0,
) -> dict[str, Array]:
    """Temporal scale-space consistency loss (Option C).

    Forces mean(h_x) ≈ mean(h_x[:, ::R, :]).
    Both views represent the same sequence; one uses all positions,
    the other uses every R-th (coarser temporal view).
    Grounded in Lindeberg scale-space theory: a good representation
    should be stable under temporal downsampling.

    Args:
        h_x: (B, S, D) context encoder output (float32 or bf16).
        R: Temporal subsampling factor (default 4).
        inv_weight: VICReg invariance weight.
        var_weight: VICReg variance weight.
        cov_weight: VICReg covariance weight.

    Returns:
        dict with total, invariance, variance, covariance.
    """
    h_fine = h_x.mean(axis=1)              # (B, D)
    h_coarse = h_x[:, ::R, :].mean(axis=1)  # (B, D) — every R-th position
    return vicreg_loss(
        h_fine, h_coarse,
        inv_weight=inv_weight,
        var_weight=var_weight,
        cov_weight=cov_weight,
        var_gamma=var_gamma,
    )
