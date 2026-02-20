"""Multiverse Crossing — convergence analysis across M perturbed latent universes.

Each universe starts from a slightly perturbed JEPA context h_x, generating
independent future trajectories. By analyzing how these universes converge or
diverge over time, we extract:

  - convergence_score:  inter-universe agreement (high = consensus)
  - divergence_rate:    temporal change in inter-universe spread
  - bifurcation_index:  soft cluster count via eigenvalue analysis of covariance
  - lyapunov_proxy:     log(spread / perturbation) / t — chaos indicator
  - inter_mv_std:       mean std across universe means
  - intra_mv_std:       mean std within each universe

These signals modulate the CVaR alpha for risk-aware planning:
  convergence → 1 → alpha ↑ → aggressive
  convergence → 0 → alpha ↓ → conservative / no trade
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import EPS


def perturb_latent(
    h_x: jnp.ndarray,
    key: jax.random.PRNGKey,
    n_multiverses: int,
    sigma: float,
) -> jnp.ndarray:
    """Generate M perturbed copies of h_x via geodesic perturbation on the
    representation hypersphere — stays on the learned manifold by construction.

    Naive additive Gaussian noise (h_x + ε) moves points off the manifold
    learned by the JEPA encoder, producing inputs the OT-CFM predictor was
    never trained on. Instead, we:

      1. Project noise onto the tangent plane at h_x (remove radial component)
      2. Add scaled tangent noise to h_x
      3. Re-project to the same L2 norm as h_x (back on the sphere)

    This is a first-order geodesic perturbation on S^(d-1), equivalent to
    moving along great circles. VICReg's variance regularization naturally
    pushes representations toward a hypersphere, making this geometrically
    consistent with the training objective.

    Args:
        h_x: (d_model,) JEPA context representation.
        key: PRNG key for reproducibility.
        n_multiverses: Number of parallel universes M.
        sigma: Perturbation std on the tangent plane (small, e.g. 0.01).

    Returns:
        (M, d_model) perturbed latent states, all with ||h|| = ||h_x||.
    """
    d = h_x.shape[-1]
    noise = jax.random.normal(key, shape=(n_multiverses, d))

    # Step 1: Project noise onto tangent plane at h_x
    # Remove the component parallel to h_x (radial direction)
    h_norm = h_x / (jnp.linalg.norm(h_x) + EPS)  # unit direction
    radial = jnp.sum(noise * h_norm[None, :], axis=-1, keepdims=True)  # (M, 1)
    tangent_noise = noise - radial * h_norm[None, :]  # (M, d)

    # Step 2: Perturb along tangent directions
    perturbed = h_x[None, :] + sigma * tangent_noise  # (M, d)

    # Step 3: Re-project to original L2 norm (back on the sphere)
    r = jnp.linalg.norm(h_x)
    perturbed = perturbed * (r / (jnp.linalg.norm(perturbed, axis=-1, keepdims=True) + EPS))

    return perturbed


def compute_convergence(
    mv_futures_t: jnp.ndarray,
    prev_inter_std: float,
    sigma: float,
    t: int,
) -> dict[str, jnp.ndarray]:
    """Compute multiverse convergence metrics at step t.

    Args:
        mv_futures_t: (M, N, d_model) — future latents across M universes
            and N samples at current step t.
        prev_inter_std: Inter-multiverse std from previous step (for rate).
        sigma: Original perturbation sigma (for Lyapunov normalization).
        t: Current time step (1-indexed for Lyapunov calculation).

    Returns:
        Dict with convergence metrics (all scalars or 1D arrays).
    """
    # mv_futures_t: (M, N, d_model)
    # Mean within each universe → (M, d_model)
    mv_means = mv_futures_t.mean(axis=1)

    # Inter-multiverse std: how much universe means disagree → (d_model,) → scalar
    inter_mv_std = mv_means.std(axis=0).mean()

    # Intra-multiverse std: average noise within each universe → scalar
    intra_mv_std = mv_futures_t.std(axis=1).mean()

    # Convergence score ∈ [0, 1]: high when universes agree
    convergence_score = 1.0 / (1.0 + inter_mv_std / (intra_mv_std + EPS))

    # Divergence rate: temporal change in inter-multiverse spread
    divergence_rate = inter_mv_std - prev_inter_std

    # Bifurcation index: effective number of clusters via eigenvalue analysis
    # Covariance of universe means → eigenvalues → entropy-based count
    M = mv_means.shape[0]
    centered = mv_means - mv_means.mean(axis=0, keepdims=True)  # (M, d_model)
    # Use a low-rank approach: (M, M) gram matrix instead of (d_model, d_model) cov
    gram = (centered @ centered.T) / (M - 1 + EPS)  # (M, M)
    eigvals = jnp.linalg.eigvalsh(gram)  # (M,)
    eigvals = jnp.maximum(eigvals, 0.0)  # numerical safety
    eigvals_norm = eigvals / (eigvals.sum() + EPS)
    # Soft cluster count: exp(entropy) — 1 = single mode, M = all modes active
    entropy = -jnp.sum(eigvals_norm * jnp.log(eigvals_norm + EPS))
    bifurcation_index = jnp.exp(entropy)

    # Lyapunov proxy: log(spread / perturbation) / t
    t_safe = jnp.maximum(jnp.float32(t), 1.0)
    lyapunov_proxy = jnp.log(inter_mv_std / (sigma + EPS) + EPS) / t_safe

    return {
        "convergence_score": convergence_score,
        "divergence_rate": divergence_rate,
        "bifurcation_index": bifurcation_index,
        "lyapunov_proxy": lyapunov_proxy,
        "inter_mv_std": inter_mv_std,
        "intra_mv_std": intra_mv_std,
    }


def dynamic_cvar_alpha(
    convergence_score: jnp.ndarray,
    alpha_min: float,
    alpha_max: float,
) -> jnp.ndarray:
    """Compute dynamic CVaR alpha based on multiverse convergence.

    High convergence → higher alpha → more aggressive (larger tail window).
    Low convergence  → lower alpha  → more conservative (focus on worst-case).

    Args:
        convergence_score: Scalar ∈ [0, 1].
        alpha_min: Minimum CVaR alpha (conservative).
        alpha_max: Maximum CVaR alpha (aggressive).

    Returns:
        Scalar CVaR alpha ∈ [alpha_min, alpha_max].
    """
    return alpha_min + (alpha_max - alpha_min) * convergence_score
