"""JEPA checkpoint loading — extracts the ~50-line boilerplate shared across scripts."""

import os


def load_jepa_checkpoint(scale_config_path, tpu_gen=None, scale_tier=None):
    """Load JEPA model + checkpoint, returning (config, state, mesh, n_params, d_model, latest_step).

    Handles batch_size adaptation to local device count, mesh creation,
    train_state init from dummy batch, sharding, and orbax restore.

    Args:
        scale_config_path: Path to scaling YAML config.
        tpu_gen: TPU generation string (default: env TPU_GEN or 'v6e').
        scale_tier: Scale tier string (default: env SCALE_TIER or '54m_gnn_cfm').

    Returns:
        dict with keys: config, state, mesh, n_params, d_model, latest_step, target_params.
    """
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    from src.jax_v6.config import load_config
    from src.jax_v6.jepa import FinJEPA
    from src.jax_v6.training.sharding import create_mesh, shard_train_state
    from src.jax_v6.training.train_state import create_checkpoint_manager, create_train_state

    if tpu_gen is None:
        tpu_gen = os.environ.get("TPU_GEN", "v6e")
    if scale_tier is None:
        scale_tier = os.environ.get("SCALE_TIER", "54m_gnn_cfm")

    config = load_config(scale_config_path)

    # Adapt batch_size to local device count
    n_devices = len(jax.devices())
    config_chips = 64
    if n_devices != config_chips:
        per_chip = config.training.batch_size // config_chips
        object.__setattr__(config.training, "batch_size", per_chip * n_devices)

    mesh = create_mesh()
    model = FinJEPA.from_config(config)
    B = config.training.batch_size // n_devices
    S = config.embedding.seq_len
    max_tgt = int(S * config.masking.mask_ratio) + config.masking.block_size_max
    d_model = config.mamba2.d_model

    dummy_batch = {
        "token_indices": jnp.zeros((B, S), dtype=jnp.int32),
        "weekend_mask": jnp.zeros((B, S), dtype=jnp.float32),
        "exo_clock": jnp.zeros((B, S, 2), dtype=jnp.float32),
        "block_mask": jnp.zeros((B, S), dtype=jnp.bool_),
        "target_positions": jnp.zeros((B, max_tgt), dtype=jnp.int32),
        "target_mask": jnp.ones((B, max_tgt), dtype=jnp.bool_),
    }
    if config.mamba2.gnn_dim > 0:
        dummy_batch["gnn_embeddings"] = jnp.zeros(
            (B, S, config.mamba2.gnn_dim), dtype=jnp.float32,
        )
        dummy_batch["gnn_mask"] = jnp.zeros((B, S), dtype=jnp.float32)

    key = jax.random.PRNGKey(42)
    state = create_train_state(
        model, key, dummy_batch,
        lr=config.training.lr, weight_decay=config.training.weight_decay,
        warmup_steps=0, total_steps=1, tau_start=config.ema.tau_start,
        grad_clip=config.training.grad_clip, n_restarts=1,
    )
    n_params = sum(x.size for x in jax.tree.leaves(state.params))

    state = shard_train_state(state, mesh)
    ckpt_dir = os.path.join(os.getcwd(), f"checkpoints/jax_{tpu_gen}/{scale_tier}")
    ckpt_mgr = create_checkpoint_manager(ckpt_dir, max_to_keep=5)
    latest = ckpt_mgr.latest_step()
    state = ckpt_mgr.restore(latest, args=ocp.args.StandardRestore(state))

    return {
        "config": config,
        "state": state,
        "mesh": mesh,
        "n_params": n_params,
        "d_model": d_model,
        "latest_step": latest,
        "target_params": state.target_params,
    }
