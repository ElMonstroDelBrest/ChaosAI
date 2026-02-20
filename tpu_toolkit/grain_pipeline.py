"""Template Grain — pipeline de données async multi-host pour TPU.

Grain charge les données directement depuis GCS (ArrayRecord format)
vers la HBM des TPU sans passer par le CPU host, via double-buffering.

Architecture:
    GCS (ArrayRecord shards)
      → grain.ArrayRecordDataSource
      → grain.IndexSampler (ShardByJaxProcess: chaque host lit ses shards)
      → transforms (parse + preprocess, dans workers séparés)
      → grain.Batch
      → DataLoader (worker_count=32, prefetch_buffer_size=128)
      → jax.device_put (async H2D)

Paramètres importants:
    worker_count: min(cpu_count, 32). Au-delà de 32, rendements décroissants.
    prefetch_buffer_size: >= worker_count pour ne jamais affamer les TPU.
    batch_size: global / jax.process_count() par host.
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable

import grain.python as grain
import jax
import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Split train/val déterministe
# ─────────────────────────────────────────────────────────────────────────────

def hash_split(key: str, val_ratio: float = 0.2) -> str:
    """Split train/val déterministe par hash MD5 d'une clé (ex: nom de fichier).

    Avantage vs. shuffle+split: reproductible sans seed, et le même fichier
    est toujours dans le même split quel que soit l'ordre de traitement.

    Usage:
        for shard_path in all_shards:
            split = hash_split(shard_path.stem)  # "train" ou "val"
    """
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return "val" if (h % 1000) < int(val_ratio * 1000) else "train"


# ─────────────────────────────────────────────────────────────────────────────
# Auto-détection workers
# ─────────────────────────────────────────────────────────────────────────────

def auto_worker_count(requested: int = 0, cap: int = 32) -> int:
    """Résout le nombre de workers Grain.

    0 → auto depuis os.cpu_count(), plafonné à cap.
    Sur v6e-8: 180 CPUs → 32 workers.
    Sur v5p-8: ~90 CPUs → 32 workers.
    Au-delà de 32: contention sur les locks SharedMemoryArray.
    """
    if requested > 0:
        return requested
    cpus = os.cpu_count() or 4
    count = min(cpus, cap)
    count = max(count, 2)
    log.info("Grain workers: %d (cpu_count=%s)", count, cpus)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Template de transform
# ─────────────────────────────────────────────────────────────────────────────

class BaseTransform(grain.MapTransform):
    """Template de transform Grain.

    Sous-classer et implémenter map().
    Important: map() est appelé dans des processus worker séparés.
    → Pas de state JAX. Tout en numpy.
    → Pas de print() (multiprocess). Utiliser logging.
    """

    def map(self, record: bytes) -> dict:
        """Traiter un enregistrement brut → dict numpy.

        Args:
            record: Enregistrement brut (bytes depuis ArrayRecord).

        Returns:
            dict de numpy arrays. Toutes les clés doivent avoir
            des shapes compatibles pour grain.Batch.
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader générique
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloader(
    shard_paths: list[str],
    transform: grain.MapTransform,
    batch_size: int,
    split: str = "train",
    worker_count: int = 0,
    prefetch_buffer_size: int = 128,
    seed: int = 42,
    drop_remainder: bool = True,
) -> grain.DataLoader:
    """Crée un DataLoader Grain multi-host.

    Args:
        shard_paths: Chemins vers les fichiers .arrayrecord.
        transform: Instance de grain.MapTransform pour le preprocessing.
        batch_size: Batch GLOBAL. Divisé par jax.process_count() en interne.
        split: "train" (shuffle infini) ou "val" (1 epoch, pas de shuffle).
        worker_count: Workers par host. 0 = auto.
        prefetch_buffer_size: Buffer de préchargement (>= worker_count).
        seed: Seed pour le shuffle.
        drop_remainder: Si True, drop le dernier batch incomplet.

    Returns:
        grain.DataLoader — itérable infini (train) ou 1-epoch (val).

    Note multi-host:
        grain.ShardByJaxProcess() distribue automatiquement les shards
        entre les hosts JAX. Chaque host voit une partition disjointe.
        Le batch_size est divisé par jax.process_count() → chaque host
        produit batch_size/n_hosts exemples par step.
    """
    source = grain.ArrayRecordDataSource(shard_paths)
    worker_count = auto_worker_count(worker_count)

    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=(split == "train"),
        seed=seed,
        shard_options=grain.ShardByJaxProcess(),
        num_epochs=None if split == "train" else 1,
    )

    per_host_batch = max(batch_size // jax.process_count(), 1)

    transforms = [
        transform,
        grain.Batch(batch_size=per_host_batch, drop_remainder=drop_remainder),
    ]

    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=transforms,
        worker_count=worker_count,
        read_options=grain.ReadOptions(prefetch_buffer_size=prefetch_buffer_size),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Async device_put helper
# ─────────────────────────────────────────────────────────────────────────────

def to_device_async(batch: dict, sharding=None) -> dict:
    """Transfère un batch numpy vers TPU HBM de façon asynchrone.

    jax.device_put() est non-bloquant: il retourne immédiatement un
    DeviceArray future. Le transfert H2D se fait en arrière-plan.
    Le calcul ne bloque que quand la valeur est effectivement nécessaire.

    vs jnp.array(): bloquant, attend la fin du transfert.

    Usage:
        for batch in dataloader:
            batch = to_device_async(batch, sharding=data_sharding(mesh))
            state, metrics = train_step(state, batch)
            # H2D overlap avec le calcul du step précédent ici
    """
    def _put(x):
        if x is None: return None
        if sharding is not None:
            return jax.device_put(x, sharding)
        return jax.device_put(x)
    return {k: _put(v) for k, v in batch.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Manifest helper (pour organiser les shards)
# ─────────────────────────────────────────────────────────────────────────────

def load_shards_from_manifest(
    arrayrecord_dir: str,
    split: str,
    val_ratio: float = 0.2,
) -> list[str]:
    """Charge les paths de shards depuis un manifest.json.

    Format manifest.json:
        {
          "shards": [
            {"path": "gs://bucket/data/shard_000.arrayrecord", "key": "BTCUSDT"},
            ...
          ]
        }

    Le split est déterministe par hash de "key".
    """
    manifest_path = Path(arrayrecord_dir) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    return [
        s["path"] for s in manifest["shards"]
        if hash_split(s["key"], val_ratio) == split
    ]
