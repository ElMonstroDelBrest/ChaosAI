"""Custom TrainState with EMA target params for Fin-JEPA.

Extends Flax train_state.TrainState with:
  - target_params: EMA copy of context encoder params
  - tau: current EMA momentum (annealed during training)
  - rng: PRNGKey for stochastic noise in training step

Optimizer: optax.adamw + linear warmup + cosine decay.
Checkpointing: orbax.checkpoint.CheckpointManager for sharded saves.
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import struct
from jax import Array
from jax.random import PRNGKey
import optax
import orbax.checkpoint as ocp


@struct.dataclass
class FinJEPATrainState(train_state.TrainState):
    """TrainState with EMA target encoder and RNG state."""
    target_params: dict  # EMA of context_encoder params
    tau: float           # EMA momentum (current)
    rng: PRNGKey         # PRNG state for noise sampling


def update_target_ema(state: FinJEPATrainState) -> FinJEPATrainState:
    """EMA update: target = tau * target + (1 - tau) * context.

    Only updates the context_encoder subset of params.
    """
    tau = state.tau

    # Extract context encoder params from both sets
    ctx_params = state.params["context_encoder"]
    tgt_params = state.target_params

    new_target = jax.tree.map(
        lambda t, c: tau * t + (1.0 - tau) * c,
        tgt_params, ctx_params,
    )

    return state.replace(target_params=new_target)


def compute_tau(epoch: int, tau_start: float, tau_end: float, anneal_epochs: int) -> float:
    """Cosine annealing of EMA momentum from tau_start to tau_end."""
    if epoch >= anneal_epochs:
        return tau_end
    progress = epoch / anneal_epochs
    return tau_end - (tau_end - tau_start) * (1.0 + jnp.cos(jnp.pi * progress)) / 2.0


def create_optimizer(
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
) -> optax.GradientTransformation:
    """AdamW with linear warmup + cosine decay schedule."""
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr, warmup_steps),
            optax.cosine_decay_schedule(lr, total_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )
    return optax.adamw(learning_rate=schedule, weight_decay=weight_decay)


def create_train_state(
    model,
    key: PRNGKey,
    dummy_batch: dict,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    tau_start: float = 0.996,
) -> FinJEPATrainState:
    """Initialize FinJEPATrainState with model params and EMA copy.

    Args:
        model: FinJEPA Flax module.
        key: PRNGKey for initialization.
        dummy_batch: Example batch for shape inference.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        total_steps: Total training steps (for cosine schedule).
        tau_start: Initial EMA momentum.

    Returns:
        Initialized FinJEPATrainState.
    """
    init_key, rng_key, target_key = jax.random.split(key, 3)

    # Initialize model params
    dummy_target_params = None  # Will be set after init
    variables = model.init(
        {"params": init_key, "dropout": init_key},
        dummy_batch,
        target_params=None,  # Not used during init
        key=init_key,
        deterministic=True,
    )
    params = variables["params"]

    # EMA target: deep copy of context_encoder params
    target_params = jax.tree.map(lambda x: x.copy(), params["context_encoder"])

    optimizer = create_optimizer(lr, weight_decay, warmup_steps, total_steps)

    return FinJEPATrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        target_params=target_params,
        tau=tau_start,
        rng=rng_key,
    )


def create_checkpoint_manager(
    directory: str,
    max_to_keep: int = 3,
) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager for sharded saves."""
    return ocp.CheckpointManager(
        directory,
        options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep),
    )
