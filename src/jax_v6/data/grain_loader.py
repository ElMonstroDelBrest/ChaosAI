"""Grain async multi-host data pipeline for ArrayRecord shards.

Pipeline:
  GCS bucket (ArrayRecord shards)
    -> grain.ArrayRecordDataSource
    -> grain.IndexSampler(shard_options=grain.ShardByJaxProcess())
    -> grain.DataLoader(worker_count=auto, prefetch_buffer_size=auto)
    -> dict of jnp.array per batch

Each host reads different shards automatically (ShardByJaxProcess).
Block masks are pre-computed in the transform (numpy, avoids JIT issues).
Val split is deterministic by hash of pair_name (reproducible).

Worker count is auto-tuned: min(os.cpu_count(), 32) to saturate v6e IO.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)

from ..masking import generate_batch_masks


def _parse_example(serialized: bytes, seq_len: int = 128) -> dict:
    """Parse a tf.train.Example protobuf into numpy arrays."""
    example = tf.train.Example()
    example.ParseFromString(serialized)
    features = example.features.feature

    token_indices = np.array(features["token_indices"].int64_list.value, dtype=np.int64)
    weekend_mask = np.array(features["weekend_mask"].float_list.value, dtype=np.float32)
    exo_clock_flat = np.array(features["exo_clock"].float_list.value, dtype=np.float32)
    exo_clock = exo_clock_flat.reshape(seq_len, 2)
    pair_name = features["pair_name"].bytes_list.value[0].decode("utf-8")
    original_len = int(features["original_len"].int64_list.value[0])

    return {
        "token_indices": token_indices,
        "weekend_mask": weekend_mask,
        "exo_clock": exo_clock,
        "pair_name": pair_name,
        "original_len": original_len,
    }


def _pair_to_split(pair_name: str, val_ratio: float = 0.2) -> str:
    """Deterministic train/val split by hashing pair name."""
    h = int(hashlib.md5(pair_name.encode()).hexdigest(), 16)
    return "val" if (h % 1000) < int(val_ratio * 1000) else "train"


class ParseAndMask(grain.MapTransform):
    """Grain transform: deserialize protobuf, pre-compute block masks."""

    def __init__(
        self,
        seq_len: int = 128,
        mask_ratio: float = 0.5,
        block_size_min: int = 4,
        block_size_max: int = 8,
    ):
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.block_size_min = block_size_min
        self.block_size_max = block_size_max

    def map(self, serialized: bytes) -> dict:
        parsed = _parse_example(serialized, self.seq_len)
        rng = np.random.default_rng()

        # Pre-compute block mask (numpy, not JAX)
        orig_len = parsed["original_len"]
        mask = generate_batch_masks(
            1, orig_len,
            mask_ratio=self.mask_ratio,
            block_size_min=self.block_size_min,
            block_size_max=self.block_size_max,
            rng=rng,
        )[0]  # (orig_len,)

        # Pad mask to seq_len
        if orig_len < self.seq_len:
            mask = np.pad(mask, (0, self.seq_len - orig_len), constant_values=False)

        # Extract target positions (where mask is True)
        target_positions = np.where(mask)[0].astype(np.int64)
        max_targets = int(self.seq_len * self.mask_ratio) + self.block_size_max
        target_mask = np.zeros(max_targets, dtype=bool)
        n_targets = min(len(target_positions), max_targets)
        padded_positions = np.zeros(max_targets, dtype=np.int64)
        padded_positions[:n_targets] = target_positions[:n_targets]
        target_mask[:n_targets] = True

        return {
            "token_indices": parsed["token_indices"],
            "weekend_mask": parsed["weekend_mask"],
            "exo_clock": parsed["exo_clock"],
            "block_mask": mask.astype(bool),
            "target_positions": padded_positions,
            "target_mask": target_mask,
        }


def _auto_worker_count(requested: int) -> int:
    """Resolve worker_count: 0 means auto-detect from os.cpu_count().

    Capped at 32 to avoid over-subscribing (diminishing returns past that).
    Minimum 2 to keep the pipeline overlapping IO and compute.
    """
    if requested > 0:
        return requested
    cpus = os.cpu_count() or 4
    count = min(cpus, 32)
    count = max(count, 2)
    log.info("Grain worker_count auto-detected: %d (cpu_count=%s)", count, cpus)
    return count


def create_dataloader(
    arrayrecord_dir: str,
    split: str = "train",
    batch_size: int = 1024,
    seq_len: int = 128,
    mask_ratio: float = 0.5,
    block_size_min: int = 4,
    block_size_max: int = 8,
    val_ratio: float = 0.2,
    worker_count: int = 0,
    prefetch_buffer_size: int = 4,
    seed: int = 42,
) -> grain.DataLoader:
    """Create a Grain DataLoader for ArrayRecord shards.

    Args:
        arrayrecord_dir: Directory with .arrayrecord files + manifest.json.
        split: "train" or "val".
        batch_size: Global batch size (will be sharded across hosts).
        seq_len: Sequence length (records are padded to this).
        mask_ratio: JEPA mask ratio for block masks.
        block_size_min: Min block size for masking.
        block_size_max: Max block size for masking.
        val_ratio: Fraction of pairs used for validation.
        worker_count: Number of parallel Grain workers (0 = auto from cpu_count).
        prefetch_buffer_size: Prefetch buffer size.
        seed: Random seed for sampler.

    Returns:
        grain.DataLoader yielding batched dicts of numpy arrays.
    """
    worker_count = _auto_worker_count(worker_count)

    ar_dir = Path(arrayrecord_dir)
    manifest_path = ar_dir / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Filter shards by split (deterministic hash of pair_name)
    shard_paths = []
    for shard_info in manifest["shards"]:
        pair_split = _pair_to_split(shard_info["pair"], val_ratio)
        if pair_split == split:
            shard_paths.append(shard_info["path"])

    if not shard_paths:
        raise ValueError(f"No shards found for split={split}")

    # Create data source from ArrayRecord files
    source = grain.ArrayRecordDataSource(shard_paths)

    # Sampler: automatically shards by JAX process for multi-host
    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=split == "train",
        seed=seed,
        shard_options=grain.ShardByJaxProcess(),
        num_epochs=None if split == "train" else 1,
    )

    # Transforms
    transforms = [
        ParseAndMask(
            seq_len=seq_len,
            mask_ratio=mask_ratio,
            block_size_min=block_size_min,
            block_size_max=block_size_max,
        ),
        grain.Batch(batch_size=batch_size // jax.process_count(), drop_remainder=True),
    ]

    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=transforms,
        worker_count=worker_count,
        read_options=grain.ReadOptions(prefetch_buffer_size=prefetch_buffer_size),
    )

    return loader
