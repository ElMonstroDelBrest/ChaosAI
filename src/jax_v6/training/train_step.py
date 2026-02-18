"""JIT-compiled training step for Fin-JEPA with GSPMD.

Single @jax.jit function that:
1. Computes loss + grads
2. Updates params via optimizer
3. Updates EMA target params
4. Advances RNG state

GSPMD handles gradient All-Reduce automatically via sharding annotations.
"""

import jax
import jax.numpy as jnp
from jax import Array

from .train_state import FinJEPATrainState, update_target_ema


@jax.jit
def train_step(
    state: FinJEPATrainState,
    batch: dict[str, Array],
    model,
) -> tuple[FinJEPATrainState, dict[str, Array]]:
    """Single training step.

    Args:
        state: Current training state (params, target_params, optimizer, rng).
        batch: Dict with token_indices, weekend_mask, block_mask, exo_clock,
               target_positions, target_mask.
        model: FinJEPA Flax module (for apply).

    Returns:
        (new_state, metrics) where metrics contains loss components.
    """
    rng, step_rng = jax.random.split(state.rng)
    dropout_rng, noise_rng = jax.random.split(step_rng)

    def loss_fn(params):
        outputs = model.apply(
            {"params": params},
            batch,
            target_params=state.target_params,
            key=noise_rng,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )
        return outputs["loss"], outputs

    (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Update params via optimizer (GSPMD auto-syncs grads)
    state = state.apply_gradients(grads=grads)

    # EMA update of target encoder
    state = update_target_ema(state)

    # Advance RNG
    state = state.replace(rng=rng)

    metrics = {
        "loss": outputs["loss"],
        "invariance": outputs["invariance"],
        "variance": outputs["variance"],
        "covariance": outputs["covariance"],
        "cfm_loss": outputs["cfm_loss"],
        "mask_ratio": outputs["mask_ratio"],
        "n_targets": outputs["n_targets"],
        "lr": state.opt_state.inner_states["0"].inner_state.hyperparams.get(
            "learning_rate", jnp.float32(0.0)
        ) if hasattr(state.opt_state, "inner_states") else jnp.float32(0.0),
    }

    return state, metrics


@jax.jit
def eval_step(
    state: FinJEPATrainState,
    batch: dict[str, Array],
    model,
) -> dict[str, Array]:
    """Single evaluation step (no grad, no dropout).

    Args:
        state: Current training state.
        batch: Validation batch.
        model: FinJEPA Flax module.

    Returns:
        metrics dict with loss components.
    """
    rng, step_rng = jax.random.split(state.rng)

    outputs = model.apply(
        {"params": state.params},
        batch,
        target_params=state.target_params,
        key=step_rng,
        deterministic=True,
    )

    return {
        "loss": outputs["loss"],
        "invariance": outputs["invariance"],
        "variance": outputs["variance"],
        "covariance": outputs["covariance"],
        "cfm_loss": outputs["cfm_loss"],
    }
