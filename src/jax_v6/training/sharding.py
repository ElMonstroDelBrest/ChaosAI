"""Auto-Sharder Topology-Aware — 2D Mesh ('data', 'fsdp') pour TPU Pod slices.

Mappe automatiquement un maillage 2D virtuel (data parallelism × FSDP) sur la
topologie physique du Tore 3D des TPUs via jax.experimental.mesh_utils, minimisant
les sauts ICI (Inter-Chip Interconnect) pour maximiser le débit.

Topologies supportées (auto-détectées via jax.device_count()) :
  TPU v5p/v5e: 1 TensorCore par puce → le chiffre = le nombre de puces.
  - v5p-8    :   8 puces → DP pur (8, 1)
  - v5p-16   :  16 puces → (8, 2)
  - v5p-32   :  32 puces → (8, 4)
  - v5p-64   :  64 puces → (16, 4)
  - v5p-128  : 128 puces → (32, 4)
  - v5p-256  : 256 puces → (64, 4)
  - v5p-768  : 768 puces → (192, 4)
  - CPU/GPU  : fallback gracieux

Sharding strategy:
  - Batch (data) : shardé sur l'axe 'data' → P("data", None, ...)
  - Params/OptState : shardés sur l'axe 'fsdp' → P("fsdp", ...) (FSDP)
  - RNG keys : shardés sur 'data' pour reproductibilité par replica
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

# ──────────────────────────────────────────────────────────────────────
# Topology → Mesh shape mapping
# ──────────────────────────────────────────────────────────────────────

# Predefined optimal shapes: {num_devices: (data_dim, fsdp_dim)}
# Rationale: fsdp_dim=4 saturates the ICI bandwidth on v5p (4 chips per
# tray share the fastest links). data_dim scales linearly with pod size.
_MESH_SHAPES: dict[int, tuple[int, int]] = {
    8:   (8, 1),      # Single host — DP pur, modèle tient dans 1 chip
    16:  (8, 2),      # 2 hosts — léger FSDP
    32:  (8, 4),      # 4 hosts — FSDP complet sur 1 tray
    64:  (16, 4),     # 8 hosts
    128: (32, 4),     # 16 hosts
    256: (64, 4),     # 32 hosts
    768: (192, 4),    # 96 hosts — full v5p-768 supercalculateur
}


def get_optimal_mesh_shape(num_devices: int) -> tuple[int, int]:
    """Détermine le maillage 2D optimal (data, fsdp) pour N devices.

    Utilise les modes prédéfinis si N est une taille connue, sinon applique
    la règle de fallback : fsdp_dim=4 (ou 2 si N<16), data_dim=N//fsdp_dim.

    Args:
        num_devices: Nombre total de devices (jax.device_count()).

    Returns:
        (data_dim, fsdp_dim) — dimensions du maillage 2D.

    Raises:
        ValueError: Si num_devices < 1 ou non divisible par fsdp_dim.
    """
    if num_devices < 1:
        raise ValueError(f"num_devices must be >= 1, got {num_devices}")

    # Lookup direct
    if num_devices in _MESH_SHAPES:
        return _MESH_SHAPES[num_devices]

    # Fallback: fsdp=4 pour >=16 devices, fsdp=2 pour 2-15, fsdp=1 pour 1
    if num_devices == 1:
        return (1, 1)
    elif num_devices < 16:
        fsdp_dim = 2 if num_devices >= 4 else 1
    else:
        fsdp_dim = 4

    # Assurer divisibilité
    while fsdp_dim > 1 and num_devices % fsdp_dim != 0:
        fsdp_dim //= 2

    data_dim = num_devices // fsdp_dim
    return (data_dim, fsdp_dim)


# ──────────────────────────────────────────────────────────────────────
# Mesh creation
# ──────────────────────────────────────────────────────────────────────

def create_mesh() -> Mesh:
    """Crée un Mesh 2D ('data', 'fsdp') topology-aware.

    Utilise create_device_mesh pour mapper les axes virtuels sur le Tore 3D
    physique des TPUs, minimisant les sauts ICI pour les collectives FSDP.

    Returns:
        jax.sharding.Mesh configuré pour le pod slice détecté.
    """
    num_devices = jax.device_count()
    platform = jax.devices()[0].platform if jax.devices() else "cpu"
    data_dim, fsdp_dim = get_optimal_mesh_shape(num_devices)

    # create_device_mesh mappe intelligemment sur la topologie physique ICI
    device_mesh = create_device_mesh(
        mesh_shape=(data_dim, fsdp_dim),
        devices=jax.devices(),
    )

    mesh = Mesh(device_mesh, axis_names=("data", "fsdp"))

    # Logging production — confirmation du mode au lancement
    mode_name = f"x{num_devices}"
    log.info("=" * 70)
    log.info(
        "Auto-Sharder activé: Mode %s détecté -> Mesh 2D (data=%d, fsdp=%d)",
        mode_name, data_dim, fsdp_dim,
    )
    log.info(
        "  Platform: %s | Devices: %d | Topology-aware: %s",
        platform.upper(), num_devices, "OUI" if num_devices > 1 else "NON (single device)",
    )
    if fsdp_dim > 1:
        log.info(
            "  FSDP: params shardés sur %d chips | DP: %d replicas parallèles",
            fsdp_dim, data_dim,
        )
        log.info(
            "  Batch effectif par chip: global_batch / %d",
            data_dim,
        )
    else:
        log.info(
            "  Mode DP pur: params répliqués, batch / %d par chip",
            data_dim,
        )
    log.info("=" * 70)

    return mesh


# ──────────────────────────────────────────────────────────────────────
# PartitionSpec helpers
# ──────────────────────────────────────────────────────────────────────

def data_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding pour les données de batch: split sur l'axe 'data'.

    Shape attendue: (batch, seq_len, ...) → P("data", None, ...)
    """
    return NamedSharding(mesh, P("data"))


def param_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding pour les paramètres: FSDP sur l'axe 'fsdp'.

    L'axe 0 de chaque tenseur de poids est shardé sur les chips FSDP.
    Si fsdp_dim=1, équivalent à réplication complète.
    """
    return NamedSharding(mesh, P("fsdp"))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding répliqué — chaque chip a une copie complète."""
    return NamedSharding(mesh, P())


# ──────────────────────────────────────────────────────────────────────
# Batch sharding
# ──────────────────────────────────────────────────────────────────────

def shard_batch(batch: dict, mesh: Mesh) -> dict:
    """Shard un batch dict sur les chips TPU (axe 'data').

    Chaque feuille avec une dimension batch (ndim >= 1) est shardée.
    Scalaires et None sont laissés intacts.

    Args:
        batch: dict d'arrays (depuis Grain dataloader).
        mesh: TPU mesh 2D.

    Returns:
        dict d'arrays shardés.
    """
    d_sharding = data_sharding(mesh)

    def _shard_leaf(x):
        if x is None:
            return None
        if isinstance(x, jnp.ndarray) and x.ndim >= 1:
            return jax.device_put(x, d_sharding)
        return x

    return jax.tree.map(_shard_leaf, batch)


# ──────────────────────────────────────────────────────────────────────
# Train state sharding (FSDP)
# ──────────────────────────────────────────────────────────────────────

def shard_params(params: PyTree, mesh: Mesh) -> PyTree:
    """Shard les paramètres sur l'axe FSDP.

    L'axe 0 de chaque tenseur est partitionné sur les fsdp_dim chips.
    Compatible avec les checkpoints : le restore reconstruit automatiquement.
    """
    p_sharding = param_sharding(mesh)
    return jax.device_put(params, p_sharding)


def shard_train_state(state: PyTree, mesh: Mesh) -> PyTree:
    """Applique le sharding FSDP sur un TrainState complet.

    Stratégie :
      - state.params     → NamedSharding(mesh, P('fsdp'))  [FSDP]
      - state.opt_state  → NamedSharding(mesh, P('fsdp'))  [FSDP, co-localisé avec params]
      - state.step       → répliqué P()
      - Tout autre champ → répliqué P()

    Args:
        state: Flax TrainState (ou tout pytree avec .params et .opt_state).
        mesh: Mesh 2D ('data', 'fsdp').

    Returns:
        TrainState avec sharding appliqué sur chaque feuille.
    """
    fsdp = param_sharding(mesh)
    replicated = replicated_sharding(mesh)

    def _shard_leaf(path: str, x):
        """Shard une feuille selon son chemin dans le pytree."""
        if any(k in path for k in ("params", "opt_state", "mu", "nu")):
            return jax.device_put(x, fsdp)
        return jax.device_put(x, replicated)

    # Flatten avec paths pour router le sharding
    flat_state, treedef = jax.tree.flatten_with_path(state)
    sharded_leaves = []
    for path_parts, leaf in flat_state:
        path_str = "/".join(str(p) for p in path_parts)
        sharded_leaves.append(_shard_leaf(path_str, leaf))

    return treedef.unflatten(sharded_leaves)


def shard_rng(rng: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
    """Shard une clé RNG sur l'axe 'data' pour reproductibilité par replica.

    Chaque replica data-parallel reçoit une clé différente → dropout et
    augmentation indépendants par replica, mais reproductibles avec le même seed.

    Args:
        rng: Clé PRNG JAX (ou array de clés shape (data_dim,)).
        mesh: Mesh 2D.

    Returns:
        Clé(s) PRNG shardée(s) sur l'axe 'data'.
    """
    data_dim = mesh.shape["data"]
    if rng.ndim == 0 or (rng.ndim == 1 and rng.shape[0] == 2):
        # Single key → split en data_dim clés
        rngs = jax.random.split(rng, data_dim)
    else:
        rngs = rng
    return jax.device_put(rngs, NamedSharding(mesh, P("data")))
