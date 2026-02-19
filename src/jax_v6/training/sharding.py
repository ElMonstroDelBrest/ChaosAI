"""GSPMD sharding setup — auto-detects TPU topology.

Pure Data Parallelism: model ~654 MB << HBM per chip.
Batch axis is sharded across chips, params are replicated.
XLA auto-inserts All-Reduce for gradient synchronization.

Supported topologies (auto-detected via jax.devices()):
  - TPU v6e-8  (Trillium): 8 chips, ~32 GB HBM each
  - TPU v4-32: 32 chips, ~32 GB HBM each
  - CPU/GPU:   falls back to available devices
"""

import logging
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

log = logging.getLogger(__name__)


def create_mesh() -> Mesh:
    """Create a 1D mesh over all available TPU/GPU/CPU devices.

    Auto-detects the device count and platform. For pure data parallelism
    a flat 1D mesh is optimal (model fits entirely in one chip's HBM).
    """
    devices = jax.devices()
    n_devices = len(devices)
    platform = devices[0].platform if devices else "cpu"
    log.info(
        "GSPMD Mesh: %d %s device(s) — pure data parallelism (batch axis)",
        n_devices, platform.upper(),
    )
    if n_devices > 1:
        log.info(
            "  Local batch per chip: global_batch / %d", n_devices,
        )
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
