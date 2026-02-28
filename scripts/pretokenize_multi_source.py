"""Multi-source pretokenizer with multiprocessing.

Parallelizes across all available CPU cores (180 vCPUs on TPU v6e-8 host).
Each worker loads the Strate I tokenizer independently and processes files.

Usage:
    PYTHONPATH=. python scripts/pretokenize_multi_source.py \
        --source_dirs data/ohlcv_crypto_1min/ data/ohlcv_stocks_daily/ \
        --source_ids 0 1 \
        --strate_i_config configs/strate_i/1m_p1.yaml \
        --checkpoint checkpoints/strate-i-best.ckpt \
        --output_dir data/arrayrecord_multi/ \
        --seq_len 128 --workers 64
"""

import argparse
import json
import logging
import os
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tensorflow as tf
from array_record.python.array_record_module import ArrayRecordWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(process)d] %(levelname)s %(message)s")
log = logging.getLogger("pretok_multi")

# ── Globals set per-worker via initializer ──
_tokenizer = None
_patch_length = None
_stride = None
_macro_signals = None
_macro_signal_names = None


def _worker_init(config_path, checkpoint_path, macro_context_path):
    """Initialize tokenizer in each worker process."""
    global _tokenizer, _patch_length, _stride, _macro_signals, _macro_signal_names

    from src.strate_i.config import load_config as load_strate_i_config
    from src.strate_i.lightning_module import StrateILightningModule

    config = load_strate_i_config(config_path)
    model = StrateILightningModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    _tokenizer = model.tokenizer
    _patch_length = config.patch.patch_length
    _stride = config.patch.stride

    if macro_context_path and os.path.exists(macro_context_path):
        data = np.load(macro_context_path, allow_pickle=True)
        _macro_signals = data["signals"]
        _macro_signal_names = list(data["signal_names"])


def compute_patch_volatility(log_ret_patches: torch.Tensor) -> torch.Tensor:
    ohlc = log_ret_patches[:, :, :4]
    return ohlc.reshape(ohlc.shape[0], -1).std(dim=1)


def compute_exo_clock(patches: torch.Tensor) -> torch.Tensor:
    N = patches.shape[0]
    ohlc = patches[:, :, :4]
    rv = ohlc.reshape(N, -1).std(dim=1)
    vol = patches[:, :, 4].abs().mean(dim=1)

    def _zscore(x):
        mu = x.mean()
        sigma = x.std().clamp(min=1e-6)
        return (x - mu) / sigma

    return torch.stack([_zscore(rv), _zscore(vol)], dim=-1)


def compute_apathy_mask(volatilities: torch.Tensor, percentile: float = 10.0) -> torch.Tensor:
    threshold = np.percentile(volatilities.numpy(), percentile)
    return (volatilities < threshold).float()


def load_ohlcv(path: Path) -> Optional[torch.Tensor]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pt":
            data = torch.load(path, weights_only=True)
        elif suffix == ".npy":
            data = torch.from_numpy(np.load(path)).float()
        else:
            return None
        if data.ndim != 2 or data.shape[1] != 5:
            return None
        return data
    except Exception:
        return None


def enrich_exo_clock_4d(exo_2d, n_patches, macro_signals, macro_signal_names):
    vix_idx = macro_signal_names.index("VIX") if "VIX" in macro_signal_names else None
    credit_idx = macro_signal_names.index("HY_SPREAD") if "HY_SPREAD" in macro_signal_names else None
    extra = np.zeros((n_patches, 2), dtype=np.float32)
    if vix_idx is not None and macro_signals.shape[0] >= n_patches:
        extra[:, 0] = macro_signals[:n_patches, vix_idx]
    if credit_idx is not None and macro_signals.shape[0] >= n_patches:
        extra[:, 1] = macro_signals[:n_patches, credit_idx]
    return np.concatenate([exo_2d, extra], axis=-1)


def to_example(token_indices, weekend_mask, exo_clock, pair_name, original_len,
               target_seq_len=128, source_id=0):
    seq_len = len(token_indices)
    if seq_len < target_seq_len:
        pad = target_seq_len - seq_len
        token_indices = np.pad(token_indices, (0, pad), constant_values=0)
        weekend_mask = np.pad(weekend_mask, (0, pad), constant_values=0.0)
        exo_clock = np.pad(exo_clock, ((0, pad), (0, 0)), constant_values=0.0)
    elif seq_len > target_seq_len:
        token_indices = token_indices[:target_seq_len]
        weekend_mask = weekend_mask[:target_seq_len]
        exo_clock = exo_clock[:target_seq_len]
        original_len = target_seq_len

    feature = {
        "token_indices": tf.train.Feature(
            int64_list=tf.train.Int64List(value=token_indices.tolist())),
        "weekend_mask": tf.train.Feature(
            float_list=tf.train.FloatList(value=weekend_mask.tolist())),
        "exo_clock": tf.train.Feature(
            float_list=tf.train.FloatList(value=exo_clock.flatten().tolist())),
        "exo_clock_dim": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[exo_clock.shape[1]])),
        "pair_name": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[pair_name.encode("utf-8")])),
        "original_len": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[original_len])),
        "source_id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[source_id])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def _process_file(args):
    """Process a single OHLCV file → ArrayRecord shard. Runs in worker process."""
    file_path, source_id, output_dir, seq_len, apathy_percentile, batch_size = args

    from src.strate_i.data.transforms import compute_log_returns, extract_patches

    file_path = Path(file_path)
    pair_name = file_path.stem
    ohlcv = load_ohlcv(file_path)
    if ohlcv is None:
        return None

    if ohlcv.shape[0] < seq_len * 2:
        return None

    log_ret = compute_log_returns(ohlcv)
    patches = extract_patches(log_ret, _patch_length, _stride)
    if patches.shape[0] == 0:
        return None

    # Tokenize
    with torch.no_grad():
        all_tokens = []
        for i in range(0, patches.shape[0], batch_size):
            batch = patches[i:i + batch_size]
            all_tokens.append(_tokenizer.tokenize(batch))
        token_indices = torch.cat(all_tokens)

    volatilities = compute_patch_volatility(patches)
    apathy_mask = compute_apathy_mask(volatilities, apathy_percentile)
    exo_2d = compute_exo_clock(patches)
    exo_np = exo_2d.numpy().astype(np.float32)

    if _macro_signals is not None and _macro_signal_names is not None:
        exo_np = enrich_exo_clock_4d(exo_np, patches.shape[0], _macro_signals, _macro_signal_names)

    n_patches = token_indices.shape[0]
    n_seqs = n_patches // seq_len
    if n_seqs == 0:
        return None

    shard_path = Path(output_dir) / f"{pair_name}.arrayrecord"
    writer = ArrayRecordWriter(str(shard_path), "group_size:1")

    for i in range(n_seqs):
        start = i * seq_len
        end = start + seq_len
        record = to_example(
            token_indices[start:end].numpy().astype(np.int64),
            apathy_mask[start:end].numpy().astype(np.float32),
            exo_np[start:end],
            pair_name, seq_len, seq_len, source_id=source_id,
        )
        writer.write(record)
    writer.close()

    return {"pair": pair_name, "path": str(shard_path), "count": n_seqs, "source_id": source_id}


def pretokenize_multi_source(
    strate_i_config_path: str,
    checkpoint_path: str,
    source_dirs: list[str],
    source_ids: list[int],
    output_dir: str,
    seq_len: int = 128,
    apathy_percentile: float = 10.0,
    batch_size: int = 8192,
    macro_context_path: Optional[str] = None,
    max_pairs: int = 0,
    workers: int = 0,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if workers <= 0:
        workers = min(os.cpu_count() or 1, 128)

    # Build flat task list: (file_path, source_id, output_dir, seq_len, ...)
    tasks = []
    for source_dir, source_id in zip(source_dirs, source_ids):
        data_path = Path(source_dir)
        if not data_path.exists():
            log.warning("Source dir not found: %s — skipping", source_dir)
            continue
        files = sorted(list(data_path.glob("*.pt")) + list(data_path.glob("*.npy")))
        if max_pairs > 0:
            files = files[:max_pairs]
        log.info("Source %d (%s): %d files", source_id, source_dir, len(files))
        for f in files:
            tasks.append((str(f), source_id, output_dir, seq_len, apathy_percentile, batch_size))

    log.info("Total: %d files to process with %d workers", len(tasks), workers)

    # Detect exo_clock_dim
    exo_clock_dim = 4 if macro_context_path else 2

    # Process with multiprocessing pool
    manifest_shards = []
    done = 0

    with mp.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(strate_i_config_path, checkpoint_path, macro_context_path),
    ) as pool:
        for result in pool.imap_unordered(_process_file, tasks):
            done += 1
            if result is not None:
                manifest_shards.append(result)
            if done % 100 == 0 or done == len(tasks):
                total_recs = sum(s["count"] for s in manifest_shards)
                log.info("Progress: %d/%d files | %d shards | %d records",
                         done, len(tasks), len(manifest_shards), total_recs)

    total_records = sum(s["count"] for s in manifest_shards)
    manifest = {
        "shards": manifest_shards,
        "total_records": total_records,
        "seq_len": seq_len,
        "exo_clock_dim": exo_clock_dim,
        "sources": {str(sid): sdir for sid, sdir in zip(source_ids, source_dirs)},
    }
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Done. %d records across %d pairs from %d sources. Manifest: %s",
             total_records, len(manifest_shards), len(source_dirs), manifest_path)


if __name__ == "__main__":
    mp.set_start_method("fork")  # fork shares memory (tokenizer weights)

    parser = argparse.ArgumentParser(description="Multi-source OHLCV → ArrayRecord (parallel)")
    parser.add_argument("--source_dirs", nargs="+", required=True)
    parser.add_argument("--source_ids", nargs="+", type=int, default=None)
    parser.add_argument("--strate_i_config", type=str, default="configs/strate_i/1m_p1.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/arrayrecord_multi/")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--apathy_percentile", type=float, default=10.0)
    parser.add_argument("--macro_context", type=str, default=None)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = auto, up to 128)")
    args = parser.parse_args()

    source_ids = args.source_ids or list(range(len(args.source_dirs)))
    if len(source_ids) != len(args.source_dirs):
        parser.error("--source_ids must match --source_dirs in length")

    pretokenize_multi_source(
        strate_i_config_path=args.strate_i_config,
        checkpoint_path=args.checkpoint,
        source_dirs=args.source_dirs,
        source_ids=source_ids,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        apathy_percentile=args.apathy_percentile,
        macro_context_path=args.macro_context,
        max_pairs=args.max_pairs,
        workers=args.workers,
    )
