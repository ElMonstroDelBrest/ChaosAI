"""GSPMD sharding setup for TPU v4-32 (32 chips).

Pure Data Parallelism: model 654 MB << 32 GB HBM per chip.
Batch axis is sharded across chips, params are replicated.
XLA auto-inserts All-Reduce for gradient synchronization.

Global batch 1024 / 32 chips = 32 local batch per chip.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def create_mesh() -> Mesh:
    """Create a 1D mesh over all available TPU chips.

    For v4-32: 32 chips arranged in a single 'batch' axis.
    For CPU/GPU dev: falls back to available devices.
    """
    devices = jax.devices()
    return Mesh(devices, axis_names=("batch",))


def data_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding spec for batch data: split along batch axis."""
    return NamedSharding(mesh, P("batch"))


def param_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding spec for model params: replicated across all chips."""
    return NamedSharding(mesh, P())


def shard_batch(batch: dict, mesh: Mesh) -> dict:
    """Shard a batch dict across TPU chips.

    Each leaf array with a leading batch dimension gets sharded.
    Scalar or None values are left untouched.

    Args:
        batch: dict of arrays (from Grain dataloader).
        mesh: TPU mesh.

    Returns:
        dict of sharded arrays.
    """
    d_sharding = data_sharding(mesh)

    def shard_leaf(x):
        if x is None:
            return None
        if isinstance(x, jnp.ndarray) and x.ndim >= 1:
            return jax.device_put(x, d_sharding)
        return x

    return jax.tree.map(shard_leaf, batch)


def shard_params(params, mesh: Mesh):
    """Replicate params across all chips."""
    p_sharding = param_sharding(mesh)
    return jax.device_put(params, p_sharding)
