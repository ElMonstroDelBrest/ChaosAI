"""Combined pretokenize + ArrayRecord: OHLCV → ArrayRecord in one pass.

Bypasses the .pt intermediate files that cause disk explosion (~2MB overhead
per ~2.5KB of actual data). Writes directly to ArrayRecord format.

Pipeline per pair:
  OHLCV .pt → log-returns → patches → tokenize → apathy mask + exo_clock
           → tf.train.Example → ArrayRecordWriter (1 shard per pair)

Optionally includes GNN on-chain embeddings (Strate V):
  --gnn_embeddings_path: parquet from compute_gnn_embeddings.py (timestamp → emb)
  --raw_parquet_dir: raw parquet dir with open_time column (for temporal alignment)
  --gnn_dim: embedding dimension (default 256)

Expected output size: ~1-2 GB for 432 pairs (vs 604 GB with .pt intermediates).

Usage:
    PYTHONPATH=. python scripts/pretokenize_to_arrayrecord.py \
        --strate_i_config configs/strate_i_1m.yaml \
        --checkpoint checkpoints/strate-i-best.ckpt \
        --data_dir data/ohlcv_1m/ \
        --output_dir data/arrayrecord_1m/ \
        --seq_len 128

    # With GNN embeddings:
    PYTHONPATH=. python scripts/pretokenize_to_arrayrecord.py \
        --strate_i_config configs/strate_i_1m.yaml \
        --checkpoint checkpoints/strate-i-best.ckpt \
        --data_dir data/ohlcv_1m/ \
        --output_dir data/arrayrecord_1m_gnn/ \
        --seq_len 128 \
        --gnn_embeddings_path data/onchain/embeddings/eth_embeddings.parquet \
        --raw_parquet_dir data/raw/1m/ \
        --gnn_dim 256
"""

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from array_record.python.array_record_module import ArrayRecordWriter

from src.strate_i.config import load_config as load_strate_i_config
from src.strate_i.data.transforms import compute_log_returns, extract_patches
from src.strate_i.lightning_module import StrateILightningModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pretok_ar")


def compute_patch_volatility(log_ret_patches: torch.Tensor) -> torch.Tensor:
    """Per-patch volatility = std of OHLC log-returns."""
    ohlc = log_ret_patches[:, :, :4]
    return ohlc.reshape(ohlc.shape[0], -1).std(dim=1)


def compute_exo_clock(patches: torch.Tensor) -> torch.Tensor:
    """Exogenous clock signals (RV + Volume), z-scored per asset."""
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
    """1.0 for low-volatility patches (below percentile), 0.0 otherwise."""
    threshold = np.percentile(volatilities.numpy(), percentile)
    return (volatilities < threshold).float()


def load_gnn_embeddings(path: str, gnn_dim: int) -> dict:
    """Load GNN embeddings parquet → {timestamp_str: np.array(gnn_dim,)}.

    The parquet is produced by compute_gnn_embeddings.py with columns:
      timestamp (str, e.g. "2024-01-15_14"), emb_0..emb_{gnn_dim-1} (float32).
    """
    df = pd.read_parquet(path)
    emb_cols = [f"emb_{i}" for i in range(gnn_dim)]
    missing = [c for c in emb_cols if c not in df.columns]
    if missing:
        raise ValueError(f"GNN embeddings parquet missing columns: {missing[:5]}...")
    result = {}
    for _, row in df.iterrows():
        result[row["timestamp"]] = row[emb_cols].values.astype(np.float32)
    log.info("Loaded %d GNN embeddings from %s", len(result), path)
    return result


def load_pair_timestamps(raw_parquet_dir: str, pair_name: str) -> Optional[np.ndarray]:
    """Load open_time timestamps from raw parquet for a pair.

    Returns array of pd.Timestamp (UTC) aligned 1:1 with the .pt tensor rows.
    Returns None if parquet not found.
    """
    pq_path = Path(raw_parquet_dir) / f"{pair_name}.parquet"
    if not pq_path.exists():
        return None
    df = pd.read_parquet(pq_path, columns=["open_time"])
    df.columns = [c.lower() for c in df.columns]
    if "open_time" not in df.columns:
        return None
    timestamps = pd.to_datetime(df["open_time"], utc=True)
    return timestamps.values


def align_gnn_to_patches(
    n_patches: int,
    stride: int,
    timestamps: np.ndarray,
    gnn_emb_dict: dict,
    gnn_dim: int,
) -> tuple:
    """Map patch indices to hourly GNN embeddings via timestamps.

    Each patch j starts at candle index j*stride. The hour key for that candle
    is its open_time truncated to hour ("YYYY-MM-DD_HH").

    Returns:
        gnn_embeddings: (n_patches, gnn_dim) float32
        gnn_mask: (n_patches,) float32 — 1.0 if embedding found, 0.0 otherwise
    """
    embeddings = np.zeros((n_patches, gnn_dim), dtype=np.float32)
    mask = np.zeros(n_patches, dtype=np.float32)

    for j in range(n_patches):
        candle_idx = j * stride
        if candle_idx >= len(timestamps):
            break
        ts = pd.Timestamp(timestamps[candle_idx], tz="UTC")
        hour_key = ts.strftime("%Y-%m-%d_%H")

        if hour_key in gnn_emb_dict:
            embeddings[j] = gnn_emb_dict[hour_key]
            mask[j] = 1.0

    return embeddings, mask


def to_example(
    token_indices: np.ndarray,
    weekend_mask: np.ndarray,
    exo_clock: np.ndarray,
    pair_name: str,
    original_len: int,
    target_seq_len: int = 128,
    gnn_embeddings: Optional[np.ndarray] = None,
    gnn_mask: Optional[np.ndarray] = None,
) -> bytes:
    """Serialize one sequence to tf.train.Example protobuf bytes.

    Args:
        gnn_embeddings: (seq_len, gnn_dim) float32 — GNN on-chain embeddings.
        gnn_mask: (seq_len,) float32 — 1.0 where GNN embedding exists, 0.0 otherwise.
    """
    seq_len = len(token_indices)

    # Pad to target_seq_len
    if seq_len < target_seq_len:
        pad = target_seq_len - seq_len
        token_indices = np.pad(token_indices, (0, pad), constant_values=0)
        weekend_mask = np.pad(weekend_mask, (0, pad), constant_values=0.0)
        exo_clock = np.pad(exo_clock, ((0, pad), (0, 0)), constant_values=0.0)
        if gnn_embeddings is not None:
            gnn_embeddings = np.pad(gnn_embeddings, ((0, pad), (0, 0)), constant_values=0.0)
            gnn_mask = np.pad(gnn_mask, (0, pad), constant_values=0.0)
    elif seq_len > target_seq_len:
        token_indices = token_indices[:target_seq_len]
        weekend_mask = weekend_mask[:target_seq_len]
        exo_clock = exo_clock[:target_seq_len]
        if gnn_embeddings is not None:
            gnn_embeddings = gnn_embeddings[:target_seq_len]
            gnn_mask = gnn_mask[:target_seq_len]
        original_len = target_seq_len

    feature = {
        "token_indices": tf.train.Feature(
            int64_list=tf.train.Int64List(value=token_indices.tolist())
        ),
        "weekend_mask": tf.train.Feature(
            float_list=tf.train.FloatList(value=weekend_mask.tolist())
        ),
        "exo_clock": tf.train.Feature(
            float_list=tf.train.FloatList(value=exo_clock.flatten().tolist())
        ),
        "exo_clock_dim": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[2])
        ),
        "pair_name": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[pair_name.encode("utf-8")])
        ),
        "original_len": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[original_len])
        ),
    }

    # GNN on-chain embeddings (Strate V) — optional
    if gnn_embeddings is not None:
        feature["gnn_embeddings"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=gnn_embeddings.flatten().tolist())
        )
        feature["gnn_mask"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=gnn_mask.tolist())
        )

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def pretokenize_to_arrayrecord(
    strate_i_config_path: str,
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    seq_len: int = 128,
    apathy_percentile: float = 10.0,
    batch_size: int = 8192,
    gnn_embeddings_path: Optional[str] = None,
    gnn_dim: int = 256,
    raw_parquet_dir: Optional[str] = None,
):
    config = load_strate_i_config(strate_i_config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load Strate I model
    log.info("Loading Strate I from %s", checkpoint_path)
    model = StrateILightningModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    tokenizer = model.tokenizer

    patch_length = config.patch.patch_length
    stride = config.patch.stride

    # Load GNN embeddings if provided
    gnn_emb_dict = None
    if gnn_embeddings_path:
        if not raw_parquet_dir:
            raise ValueError("--raw_parquet_dir required for GNN temporal alignment")
        gnn_emb_dict = load_gnn_embeddings(gnn_embeddings_path, gnn_dim)
        log.info("GNN mode: dim=%d, %d hourly embeddings loaded", gnn_dim, len(gnn_emb_dict))

    data_path = Path(data_dir)
    pt_files = sorted(list(data_path.glob("*.pt")) + list(data_path.glob("*.npy")))
    log.info("Found %d OHLCV files in %s", len(pt_files), data_dir)

    manifest = {"shards": [], "total_records": 0, "seq_len": seq_len, "gnn_dim": gnn_dim if gnn_emb_dict else 0}
    total_records = 0
    total_pairs = 0
    gnn_hits = 0
    gnn_misses = 0

    for idx, pt_file in enumerate(pt_files):
        pair_name = pt_file.stem
        if pt_file.suffix == ".npy":
            ohlcv = torch.from_numpy(np.load(pt_file)).float()
        else:
            ohlcv = torch.load(pt_file, weights_only=True)  # (T, 5)

        # Log-returns and patches
        log_ret = compute_log_returns(ohlcv)
        patches = extract_patches(log_ret, patch_length, stride)

        if patches.shape[0] == 0:
            log.warning("Skipping %s: no patches", pair_name)
            continue

        # Tokenize in batches
        with torch.no_grad():
            all_tokens = []
            for i in range(0, patches.shape[0], batch_size):
                batch = patches[i : i + batch_size]
                tokens = tokenizer.tokenize(batch)
                all_tokens.append(tokens)
            token_indices = torch.cat(all_tokens)

        # Volatility + masks
        volatilities = compute_patch_volatility(patches)
        apathy_mask = compute_apathy_mask(volatilities, apathy_percentile)
        exo_clock = compute_exo_clock(patches)

        # GNN alignment: map patches to hourly embeddings via raw parquet timestamps
        pair_gnn_emb = None
        pair_gnn_mask = None
        if gnn_emb_dict is not None:
            timestamps = load_pair_timestamps(raw_parquet_dir, pair_name)
            if timestamps is not None:
                pair_gnn_emb, pair_gnn_mask = align_gnn_to_patches(
                    patches.shape[0], stride, timestamps, gnn_emb_dict, gnn_dim,
                )
                gnn_hits += int(pair_gnn_mask.sum())
                gnn_misses += int((pair_gnn_mask == 0).sum())
            else:
                log.warning("No raw parquet for %s — GNN embeddings will be zeros", pair_name)

        n_patches = token_indices.shape[0]
        n_seqs = n_patches // seq_len
        if n_seqs == 0:
            log.warning("Skipping %s: %d patches < seq_len=%d", pair_name, n_patches, seq_len)
            continue

        # Write directly to ArrayRecord (one shard per pair)
        shard_path = output_path / f"{pair_name}.arrayrecord"
        writer = ArrayRecordWriter(str(shard_path), "group_size:1")
        count = 0

        for i in range(n_seqs):
            start = i * seq_len
            end = start + seq_len

            ti = token_indices[start:end].numpy().astype(np.int64)
            wm = apathy_mask[start:end].numpy().astype(np.float32)
            ec = exo_clock[start:end].numpy().astype(np.float32)

            # Slice GNN embeddings for this sequence
            seq_gnn_emb = None
            seq_gnn_mask = None
            if pair_gnn_emb is not None:
                seq_gnn_emb = pair_gnn_emb[start:end]
                seq_gnn_mask = pair_gnn_mask[start:end]

            record = to_example(
                ti, wm, ec, pair_name, seq_len, seq_len,
                gnn_embeddings=seq_gnn_emb, gnn_mask=seq_gnn_mask,
            )
            writer.write(record)
            count += 1

        writer.close()

        manifest["shards"].append({
            "pair": pair_name,
            "path": str(shard_path),
            "count": count,
        })
        total_records += count
        total_pairs += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(pt_files):
            log.info(
                "Progress: %d/%d pairs | %s: %d patches, %d seqs",
                idx + 1, len(pt_files), pair_name, n_patches, count,
            )

    manifest["total_records"] = total_records
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Done. %d records across %d pairs. Manifest: %s", total_records, total_pairs, manifest_path)
    if gnn_emb_dict is not None:
        coverage = gnn_hits / max(gnn_hits + gnn_misses, 1) * 100
        log.info("GNN coverage: %d/%d patches (%.1f%%)", gnn_hits, gnn_hits + gnn_misses, coverage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OHLCV -> ArrayRecord (no .pt intermediates)")
    parser.add_argument("--strate_i_config", type=str, default="configs/strate_i_1m.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/ohlcv_1m/")
    parser.add_argument("--output_dir", type=str, default="data/arrayrecord_1m/")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--apathy_percentile", type=float, default=10.0)
    # GNN on-chain embeddings (Strate V) — optional
    parser.add_argument("--gnn_embeddings_path", type=str, default=None,
                        help="Parquet from compute_gnn_embeddings.py (timestamp -> embedding)")
    parser.add_argument("--gnn_dim", type=int, default=256,
                        help="GNN embedding dimension (default: 256)")
    parser.add_argument("--raw_parquet_dir", type=str, default=None,
                        help="Raw parquet dir with open_time (for GNN temporal alignment)")
    args = parser.parse_args()

    pretokenize_to_arrayrecord(
        strate_i_config_path=args.strate_i_config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        apathy_percentile=args.apathy_percentile,
        gnn_embeddings_path=args.gnn_embeddings_path,
        gnn_dim=args.gnn_dim,
        raw_parquet_dir=args.raw_parquet_dir,
    )
