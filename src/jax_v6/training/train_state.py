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
    grad_clip: float = 1.0,
    n_restarts: int = 4,
) -> optax.GradientTransformation:
    """AdamW with SGDR warm restarts + gradient clipping.

    Warm restarts periodically bump the LR back up, helping the optimizer
    escape narrow local minima (the likely cause of NaN at step 2750).
    Each cycle is longer than the previous (T_mult=2): if the first cycle
    is 100 steps, the next is 200, then 400, etc.
    """
    train_steps = total_steps - warmup_steps

    # SGDR: cosine cycles with increasing period (T_mult=2)
    # Cycle lengths: T, 2T, 4T, ... = T * (2^n_restarts - 1) = train_steps
    first_cycle = train_steps // (2 ** n_restarts - 1) if n_restarts > 0 else train_steps

    cycle_schedules = []
    cycle_boundaries = [warmup_steps]
    cycle_len = max(first_cycle, 1)
    for i in range(n_restarts):
        cycle_schedules.append(optax.cosine_decay_schedule(lr, cycle_len))
        if i < n_restarts - 1:
            cycle_boundaries.append(cycle_boundaries[-1] + cycle_len)
        cycle_len *= 2

    schedule = optax.join_schedules(
        schedules=[optax.linear_schedule(0.0, lr, warmup_steps)] + cycle_schedules,
        boundaries=cycle_boundaries,
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay, b2=0.95)
    if grad_clip > 0.0:
        tx = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.zero_nans(),
            tx,
        )
    return tx


def create_train_state(
    model,
    key: PRNGKey,
    dummy_batch: dict,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    tau_start: float = 0.996,
    grad_clip: float = 1.0,
    n_restarts: int = 4,
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

    optimizer = create_optimizer(lr, weight_decay, warmup_steps, total_steps, grad_clip, n_restarts)

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
