"""Auto-Sharder GSPMD topologique — copié/généralisé depuis ChaosAI.

Usage:
    mesh = create_mesh()
    state = shard_train_state(state, mesh)
    batch = shard_batch(batch, mesh)

Topologies supportées automatiquement:
    v5p-8   → (8,1)   DP pur
    v5p-32  → (8,4)   DP+FSDP
    v5p-128 → (32,4)
    v5p-768 → (192,4)

Principe clé: fsdp_dim=4 correspond aux 4 chips d'un tray v5p (liens ICI
les plus rapides). FSDP reste intra-tray → zero sauts inter-serveurs.
"""

from __future__ import annotations
import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

log = logging.getLogger(__name__)
PyTree = Any

# {num_devices: (data_dim, fsdp_dim)}
_MESH_SHAPES: dict[int, tuple[int, int]] = {
    1:   (1, 1),
    4:   (4, 1),
    8:   (8, 1),
    16:  (8, 2),
    32:  (8, 4),
    64:  (16, 4),
    128: (32, 4),
    256: (64, 4),
    512: (128, 4),
    768: (192, 4),
}


def get_mesh_shape(num_devices: int) -> tuple[int, int]:
    if num_devices in _MESH_SHAPES:
        return _MESH_SHAPES[num_devices]
    fsdp_dim = 4 if num_devices >= 16 else (2 if num_devices >= 4 else 1)
    while fsdp_dim > 1 and num_devices % fsdp_dim != 0:
        fsdp_dim //= 2
    return (num_devices // fsdp_dim, fsdp_dim)


def create_mesh() -> Mesh:
    """Crée un Mesh 2D ('data', 'fsdp') topology-aware."""
    num_devices = jax.device_count()
    data_dim, fsdp_dim = get_mesh_shape(num_devices)
    device_mesh = create_device_mesh(
        mesh_shape=(data_dim, fsdp_dim),
        devices=jax.devices(),
    )
    mesh = Mesh(device_mesh, axis_names=("data", "fsdp"))
    log.info(
        "Mesh: %d devices → (data=%d, fsdp=%d) | platform=%s",
        num_devices, data_dim, fsdp_dim,
        jax.devices()[0].platform.upper(),
    )
    return mesh


def data_sharding(mesh: Mesh) -> NamedSharding:
    """Batch → split sur 'data'. Shape: (batch, ...) → P('data', None, ...)"""
    return NamedSharding(mesh, P("data"))

def param_sharding(mesh: Mesh) -> NamedSharding:
    """Params/opt_state → FSDP sur 'fsdp'. Axe 0 partitionné."""
    return NamedSharding(mesh, P("fsdp"))

def replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Scalaires/step → répliqués sur tous les chips."""
    return NamedSharding(mesh, P())


def shard_batch(batch: dict, mesh: Mesh) -> dict:
    """Shard un dict de batch sur l'axe 'data'."""
    d_sharding = data_sharding(mesh)
    def _shard(x):
        if x is None: return None
        if isinstance(x, jnp.ndarray) and x.ndim >= 1:
            return jax.device_put(x, d_sharding)
        return x
    return jax.tree.map(_shard, batch)


def shard_train_state(state: PyTree, mesh: Mesh) -> PyTree:
    """FSDP sur params/opt_state, répliqué pour le reste."""
    fsdp = param_sharding(mesh)
    repl = replicated_sharding(mesh)
    def _shard_leaf(path: str, x):
        if any(k in path for k in ("params", "opt_state", "mu", "nu")):
            return jax.device_put(x, fsdp)
        return jax.device_put(x, repl)
    flat_state, treedef = jax.tree.flatten_with_path(state)
    sharded = [_shard_leaf("/".join(str(p) for p in path), leaf)
               for path, leaf in flat_state]
    return treedef.unflatten(sharded)


def shard_rng(rng: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
    """Split la clé RNG sur data_dim replicas (dropout indépendant par replica)."""
    data_dim = mesh.shape["data"]
    if rng.ndim == 0 or (rng.ndim == 1 and rng.shape[0] == 2):
        rngs = jax.random.split(rng, data_dim)
    else:
        rngs = rng
    return jax.device_put(rngs, NamedSharding(mesh, P("data")))
