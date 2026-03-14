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
import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)

from ..masking import generate_batch_masks


def _parse_example(serialized: bytes, seq_len: int = 128, gnn_dim: int = 0,
                   exo_clock_dim: int = 2) -> dict:
    """Parse a tf.train.Example protobuf into numpy arrays.

    Args:
        exo_clock_dim: expected exo_clock dimension (2 or 4). If the record
            contains a different dim, it is padded with zeros or truncated.
    """
    example = tf.train.Example()
    example.ParseFromString(serialized)
    features = example.features.feature

    token_indices = np.array(features["token_indices"].int64_list.value, dtype=np.int64)
    weekend_mask = np.array(features["weekend_mask"].float_list.value, dtype=np.float32)
    exo_clock_flat = np.array(features["exo_clock"].float_list.value, dtype=np.float32)

    # Detect stored exo_clock dim from the record (if metadata present)
    if "exo_clock_dim" in features:
        stored_dim = int(features["exo_clock_dim"].int64_list.value[0])
    else:
        stored_dim = 2  # legacy records are always 2D

    exo_clock = exo_clock_flat.reshape(seq_len, stored_dim)

    # Adapt to requested exo_clock_dim (pad or truncate)
    if stored_dim < exo_clock_dim:
        pad_width = exo_clock_dim - stored_dim
        exo_clock = np.pad(exo_clock, ((0, 0), (0, pad_width)), constant_values=0.0)
    elif stored_dim > exo_clock_dim:
        exo_clock = exo_clock[:, :exo_clock_dim]

    pair_name = features["pair_name"].bytes_list.value[0].decode("utf-8")
    original_len = int(features["original_len"].int64_list.value[0])

    # scale_id: 0=1min, 1=hourly, 2=daily (default 0 for legacy shards without this field)
    if "scale_id" in features:
        scale_id = np.int32(features["scale_id"].int64_list.value[0])
    else:
        scale_id = np.int32(0)

    result = {
        "token_indices": token_indices,
        "weekend_mask": weekend_mask,
        "exo_clock": exo_clock,
        "pair_name": pair_name,
        "original_len": original_len,
        "scale_id": scale_id,
    }

    # GNN on-chain embeddings (Strate V) — present only when gnn_dim > 0
    if gnn_dim > 0 and "gnn_embeddings" in features:
        gnn_flat = np.array(features["gnn_embeddings"].float_list.value, dtype=np.float32)
        result["gnn_embeddings"] = gnn_flat.reshape(seq_len, gnn_dim)
        result["gnn_mask"] = np.array(features["gnn_mask"].float_list.value, dtype=np.float32)
    elif gnn_dim > 0:
        # Field not in record — fill zeros (backward compat with old ArrayRecords)
        result["gnn_embeddings"] = np.zeros((seq_len, gnn_dim), dtype=np.float32)
        result["gnn_mask"] = np.zeros(seq_len, dtype=np.float32)

    return result


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
        gnn_dim: int = 0,
        exo_clock_dim: int = 2,
    ):
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.block_size_min = block_size_min
        self.block_size_max = block_size_max
        self.gnn_dim = gnn_dim
        self.exo_clock_dim = exo_clock_dim

    def map(self, serialized: bytes) -> dict:
        parsed = _parse_example(serialized, self.seq_len, gnn_dim=self.gnn_dim,
                                exo_clock_dim=self.exo_clock_dim)
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

        result = {
            "token_indices": parsed["token_indices"],
            "weekend_mask": parsed["weekend_mask"],
            "exo_clock": parsed["exo_clock"],
            "scale_id": parsed["scale_id"],
            "block_mask": mask.astype(bool),
            "target_positions": padded_positions,
            "target_mask": target_mask,
        }

        # GNN on-chain embeddings (Strate V)
        if self.gnn_dim > 0 and "gnn_embeddings" in parsed:
            result["gnn_embeddings"] = parsed["gnn_embeddings"]
            result["gnn_mask"] = parsed["gnn_mask"]

        return result


def _auto_worker_count(requested: int) -> int:
    """Resolve worker_count: 0 means in-process (no multiprocessing).

    Values > 0 are used as-is. 0 means run in the main process,
    avoiding JAX TPU conflicts in Grain worker subprocesses.
    """
    if requested == 0:
        log.info("Grain worker_count=0: running in main process (no multiprocessing)")
        return 0
    return requested


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
    gnn_dim: int = 0,
    exo_clock_dim: int = 2,
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
    import jax  # lazy import — avoid TPU init in Grain worker processes

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
            gnn_dim=gnn_dim,
            exo_clock_dim=exo_clock_dim,
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


class InMemoryLoader:
    """Pre-load ALL records into RAM, then serve batches at TPU speed.

    With 1.4 TB RAM and ~6M records (~8 GB), everything fits in memory.
    Startup: parallel parse with 128 threads → ~30s for 6M records.
    Training: zero I/O, batch assembly from numpy arrays, ~0.5ms/batch.

    Usage:
        loader = InMemoryLoader(arrayrecord_dir, split="train", ...)
        for batch in loader:  # infinite iterator
            ...
    """

    def __init__(
        self,
        arrayrecord_dir: str,
        split: str = "train",
        batch_size: int = 512,
        seq_len: int = 128,
        mask_ratio: float = 0.5,
        block_size_min: int = 4,
        block_size_max: int = 8,
        val_ratio: float = 0.2,
        n_threads: int = 128,
        gnn_dim: int = 0,
        exo_clock_dim: int = 2,
        seed: int = 42,
    ):
        import array_record.python.array_record_module as arrm
        from concurrent.futures import ThreadPoolExecutor
        import jax

        self.batch_size = batch_size // jax.process_count()
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.block_size_min = block_size_min
        self.block_size_max = block_size_max
        self.gnn_dim = gnn_dim
        self.exo_clock_dim = exo_clock_dim
        self.rng = np.random.default_rng(seed)

        ar_dir = Path(arrayrecord_dir)
        with open(ar_dir / "manifest.json") as f:
            manifest = json.load(f)

        # Filter shards by split
        shard_paths = []
        for shard_info in manifest["shards"]:
            if _pair_to_split(shard_info["pair"], val_ratio) == split:
                shard_paths.append(shard_info["path"])

        log.info("InMemoryLoader: loading %d shards with %d threads...", len(shard_paths), n_threads)
        t0 = __import__("time").time()

        # Read all raw bytes from ArrayRecord in parallel threads
        def _read_shard(path):
            reader = arrm.ArrayRecordReader(path)
            records = reader.read_all()
            reader.close()
            return records

        all_raw = []
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for records in pool.map(_read_shard, shard_paths):
                all_raw.extend(records)

        n_records = len(all_raw)
        log.info("InMemoryLoader: read %d records in %.1fs", n_records, __import__("time").time() - t0)

        # Parse all records in parallel threads
        max_tgt = int(seq_len * mask_ratio) + block_size_max
        parse_fn = ParseAndMask(
            seq_len=seq_len, mask_ratio=mask_ratio,
            block_size_min=block_size_min, block_size_max=block_size_max,
            gnn_dim=gnn_dim, exo_clock_dim=exo_clock_dim,
        )

        # Parse + pre-allocate arrays in chunks (avoids 5.9M list-of-dicts → np.stack OOM)
        t1 = __import__("time").time()
        CHUNK = 50_000
        first = parse_fn.map(all_raw[0])
        keys = list(first.keys())

        # Pre-allocate contiguous arrays
        self.data = {}
        for k in keys:
            shape = (n_records,) + first[k].shape
            self.data[k] = np.empty(shape, dtype=first[k].dtype)
        self.data[keys[0]][0] = first[keys[0]]
        for k in keys[1:]:
            self.data[k][0] = first[k]

        # Parse directly into pre-allocated arrays (zero intermediate allocation)
        for start in range(1, n_records, CHUNK):
            end = min(start + CHUNK, n_records)
            for i in range(start, end):
                parsed = parse_fn.map(all_raw[i])
                for k in keys:
                    self.data[k][i] = parsed[k]
            if start % (CHUNK * 4) == 0 or end == n_records:
                elapsed = __import__("time").time() - t1
                rate = end / elapsed
                log.info("InMemoryLoader: parsed %d/%d (%.0f rec/s, ETA %.0fs)",
                         end, n_records, rate, (n_records - end) / max(rate, 1))

        del all_raw
        log.info("InMemoryLoader: parsed %d records in %.1fs", n_records, __import__("time").time() - t1)

        self.n_records = n_records
        self.indices = np.arange(n_records)
        self._pos = 0
        self._shuffle()

        elapsed = __import__("time").time() - t0
        mem_mb = sum(v.nbytes for v in self.data.values()) / 1e6
        log.info("InMemoryLoader: READY — %d records, %.0f MB in RAM, %.1fs total",
                 n_records, mem_mb, elapsed)

    def _shuffle(self):
        self.rng.shuffle(self.indices)
        self._pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos + self.batch_size > self.n_records:
            self._shuffle()
        idx = self.indices[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        # Contiguous slice → near-zero copy, numpy advanced indexing is fast
        batch = {k: v[idx] for k, v in self.data.items()}
        return batch

    def precompute_epoch_batches(self, n_batches: int) -> list[dict]:
        """Pre-assemble N batches as contiguous numpy arrays.

        Call once per epoch to eliminate per-step indexing overhead.
        With 180 CPUs, this takes <1s for 10K batches.
        """
        self._shuffle()
        batches = []
        for i in range(n_batches):
            if self._pos + self.batch_size > self.n_records:
                self._shuffle()
            idx = self.indices[self._pos:self._pos + self.batch_size]
            self._pos += self.batch_size
            batches.append({k: v[idx].copy() for k, v in self.data.items()})
        return batches


def count_train_records(arrayrecord_dir: str, val_ratio: float = 0.2) -> int:
    """Count training records from manifest (lightweight, no Grain source needed)."""
    ar_dir = Path(arrayrecord_dir)
    manifest_path = ar_dir / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    total = 0
    for shard_info in manifest["shards"]:
        if _pair_to_split(shard_info["pair"], val_ratio) == "train":
            total += shard_info.get("count", 0)
    return total
