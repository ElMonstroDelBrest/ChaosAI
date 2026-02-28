"""TPU-native multi-source pretokenizer — fully batched.

All OHLCV candles are concatenated into one giant array, transferred to TPU,
and processed in a single vectorized pass. No Python for-loop per asset.

Phases:
  1. CPU (64 workers): load all files → numpy, compute per-asset log_returns
  2. CPU: concatenate all candles + build boundary index
  3. CPU → TPU: transfer giant array to HBM (~3 GB, < 1 sec at 32 GB/s PCIe)
  4. TPU: tokenize all candles in batched passes (12.8 TB/s HBM bandwidth)
  5. TPU → CPU: transfer indices + exo_clock back
  6. CPU (64 workers): split by asset + write ArrayRecord shards

Usage:
    PYTHONPATH=. python scripts/pretokenize_tpu.py \
        --source_dirs data/ohlcv_crypto_1min/ data/ohlcv_stocks_daily/ \
        --source_ids 0 1 \
        --checkpoint checkpoints/strate-i-epoch=02-val/loss/total=0.0063.ckpt \
        --output_dir data/arrayrecord_multi/ \
        --seq_len 128
"""

import argparse
import json
import logging
import os
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pretok_tpu")


# ── Phase 1: Parallel CPU I/O + log_returns ──────────────────────────────────

def _load_and_transform(args):
    """Load one file (.pt, .npy, or .parquet), compute log_returns on CPU.

    Returns (name, log_ret, source_id, dir_tag) or None.
    """
    file_path, source_id = args
    path = Path(file_path)
    dir_tag = path.parent.name
    try:
        if path.suffix == ".pt":
            import torch
            data = torch.load(path, weights_only=True).numpy().astype(np.float32)
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(path, columns=["open", "high", "low", "close", "volume"])
            data = table.to_pandas().values.astype(np.float32)
        else:
            data = np.load(path).astype(np.float32)

        if data.ndim != 2 or data.shape[1] != 5 or data.shape[0] < 3:
            return None

        # compute_log_returns on CPU (numpy)
        prices = data[:, :4]
        log_ret_prices = np.log(prices[1:] + 1e-9) - np.log(prices[:-1] + 1e-9)
        volume = data[:, 4]
        mean_vol = volume.mean()
        if mean_vol > 1e-8:
            vol_transform = np.log1p(volume[1:] / max(mean_vol, 1e-8))
        else:
            vol_transform = np.zeros_like(volume[1:])
        log_ret = np.concatenate([log_ret_prices, vol_transform[:, None]], axis=-1)

        return (path.stem, log_ret.astype(np.float32), source_id, dir_tag)
    except Exception as e:
        return None


def load_all_files(source_dirs, source_ids, max_pairs=0, workers=64):
    """Load all files + compute log_returns in parallel on CPU."""
    tasks = []
    for source_dir, source_id in zip(source_dirs, source_ids):
        data_path = Path(source_dir)
        if not data_path.exists():
            log.warning("Source dir not found: %s", source_dir)
            continue
        files = sorted(list(data_path.glob("*.pt")) + list(data_path.glob("*.npy")) + list(data_path.glob("*.parquet")))
        if max_pairs > 0:
            files = files[:max_pairs]
        log.info("Source %d (%s): %d files", source_id, source_dir, len(files))
        for f in files:
            tasks.append((str(f), source_id))

    log.info("Loading %d files with %d workers...", len(tasks), workers)
    t0 = time.time()
    results = []
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(_load_and_transform, tasks, chunksize=32):
            if r is not None:
                results.append(r)
    log.info("Loaded %d files in %.1fs", len(results), time.time() - t0)
    return results


# ── Phase 2: Concatenate + build boundary index ─────────────────────────────

def build_concat_array(all_data, seq_len):
    """Concatenate all log_returns, track per-asset boundaries.

    Returns:
        all_candles: (N_total, 5) float32
        assets: list of (pair_name, start, length, source_id, n_seqs, dir_tag)
    """
    assets = []
    arrays = []
    offset = 0
    skipped = 0

    for pair_name, log_ret, source_id, dir_tag in all_data:
        T = log_ret.shape[0]
        n_seqs = T // seq_len
        if n_seqs == 0:
            skipped += 1
            continue
        usable = n_seqs * seq_len
        arrays.append(log_ret[:usable])
        assets.append((pair_name, offset, usable, source_id, n_seqs, dir_tag))
        offset += usable

    log.info("Concat: %d assets, %d candles total, %d skipped (too short)",
             len(assets), offset, skipped)
    all_candles = np.concatenate(arrays, axis=0)  # (N_total, 5)
    return all_candles, assets


# ── Phase 3: TPU tokenization (fully batched) ────────────────────────────────

def tokenize_batched_tpu(all_candles_np, params, tokenizer, batch_size=2_000_000):
    """Tokenize ALL candles in batched passes on TPU.

    Args:
        all_candles_np: (N_total, 5) numpy — all log_returns concatenated
        params: JAX params on TPU
        tokenizer: StrateITokenizer
        batch_size: candles per TPU forward pass (2M = ~40 MB, fits easily)

    Returns:
        token_indices: (N_total,) int32 numpy
        exo_clocks: (N_total, 2) float32 numpy
    """
    import jax
    import jax.numpy as jnp

    N = all_candles_np.shape[0]
    log.info("TPU tokenization: %d candles in batches of %d", N, batch_size)

    # JIT the tokenizer for fixed batch shape (avoids retracing)
    @jax.jit
    def _tokenize_batch(params, patches):
        return tokenizer.apply({"params": params}, patches)

    all_indices = []
    all_exo = []
    t0 = time.time()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk_np = all_candles_np[start:end]  # (B, 5)

        # Reshape to (B, 1, 5) for patch_length=1
        patches = jnp.array(chunk_np[:, None, :])  # CPU → TPU transfer

        # Tokenize on TPU
        indices = _tokenize_batch(params, patches)  # (B,)

        # Exo-clock on TPU: RV = std(OHLC), Volume = mean(abs(vol))
        ohlc = patches[:, :, :4]
        rv = jnp.std(ohlc.reshape(end - start, -1), axis=1)
        vol = jnp.mean(jnp.abs(patches[:, :, 4]), axis=1)
        exo = jnp.stack([rv, vol], axis=-1)  # (B, 2)

        # Transfer back to CPU (async — JAX pipelines this)
        all_indices.append(np.asarray(indices))
        all_exo.append(np.asarray(exo))

        if (start // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            speed = (start + batch_size) / max(elapsed, 0.01)
            eta = (N - start - batch_size) / max(speed, 1)
            log.info("TPU: %d/%d (%.0f candles/sec, ETA %.0fs)",
                     min(start + batch_size, N), N, speed, max(eta, 0))

    token_indices = np.concatenate(all_indices)  # (N,)
    exo_clocks = np.concatenate(all_exo)          # (N, 2)

    elapsed = time.time() - t0
    log.info("TPU tokenization done: %d candles in %.1fs (%.0f candles/sec)",
             N, elapsed, N / elapsed)
    return token_indices, exo_clocks


# ── Phase 4: Compute apathy + z-score exo per asset (CPU vectorized) ────────

def compute_per_asset_features(token_indices, exo_clocks, assets, seq_len):
    """Split results by asset, z-score exo_clock per asset, compute apathy mask.

    Returns list of (pair_name, ti, ec, am, source_id, n_seqs) ready for writing.
    """
    results = []
    for pair_name, start, length, source_id, n_seqs, dir_tag in assets:
        ti = token_indices[start:start + length].reshape(n_seqs, seq_len)
        exo_raw = exo_clocks[start:start + length]  # (length, 2)

        # Z-score exo_clock per asset
        for col in range(2):
            mu = exo_raw[:, col].mean()
            sigma = max(exo_raw[:, col].std(), 1e-6)
            exo_raw[:, col] = (exo_raw[:, col] - mu) / sigma

        ec = exo_raw.reshape(n_seqs, seq_len, 2)

        # Apathy mask: low-volatility patches (below 10th percentile per asset)
        vols = np.abs(exo_raw[:, 0])  # RV is already computed
        threshold = np.percentile(vols, 10.0)
        apathy = (vols < threshold).astype(np.float32)
        am = apathy.reshape(n_seqs, seq_len)

        # Unique shard name: dir_tag + pair_name (avoids cross-source collisions)
        unique_name = f"{dir_tag}__{pair_name}"
        results.append((unique_name, ti.astype(np.int64), ec.astype(np.float32),
                        am, source_id, n_seqs))
    return results


# ── Phase 5: Write ArrayRecords (parallel CPU) ──────────────────────────────

def _write_shard(args):
    """Write one asset's ArrayRecord shard."""
    import tensorflow as tf
    from array_record.python.array_record_module import ArrayRecordWriter

    pair_name, ti, ec, am, source_id, n_seqs, output_dir, seq_len, exo_clock_dim = args
    shard_path = os.path.join(output_dir, f"{pair_name}.arrayrecord")
    writer = ArrayRecordWriter(shard_path, "group_size:1")

    for i in range(n_seqs):
        feature = {
            "token_indices": tf.train.Feature(
                int64_list=tf.train.Int64List(value=ti[i].tolist())),
            "weekend_mask": tf.train.Feature(
                float_list=tf.train.FloatList(value=am[i].tolist())),
            "exo_clock": tf.train.Feature(
                float_list=tf.train.FloatList(value=ec[i].flatten().tolist())),
            "exo_clock_dim": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[exo_clock_dim])),
            "pair_name": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[pair_name.encode("utf-8")])),
            "original_len": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[seq_len])),
            "source_id": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[source_id])),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    return {"pair": pair_name, "path": shard_path, "count": n_seqs, "source_id": int(source_id)}


def write_arrayrecords(tokenized_data, output_dir, seq_len, exo_clock_dim=2, workers=64):
    """Write all shards in parallel using spawn context (fork-safe after JAX init)."""
    os.makedirs(output_dir, exist_ok=True)
    tasks = [
        (name, ti, ec, am, sid, ns, output_dir, seq_len, exo_clock_dim)
        for name, ti, ec, am, sid, ns in tokenized_data
    ]
    log.info("Writing %d shards with %d workers (spawn)...", len(tasks), workers)
    t0 = time.time()
    shards = []
    # Use spawn context to avoid fork after JAX multithreaded init
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for r in pool.imap_unordered(_write_shard, tasks, chunksize=16):
            shards.append(r)
    total = sum(s["count"] for s in shards)
    log.info("Wrote %d shards (%d records) in %.1fs", len(shards), total, time.time() - t0)
    return shards


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TPU-native multi-source pretokenizer")
    parser.add_argument("--source_dirs", nargs="+", required=True)
    parser.add_argument("--source_ids", nargs="+", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/arrayrecord_multi/")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--io_workers", type=int, default=64)
    parser.add_argument("--tpu_batch", type=int, default=2_000_000,
                        help="Candles per TPU forward pass (default 2M)")
    args = parser.parse_args()
    source_ids = args.source_ids or list(range(len(args.source_dirs)))
    t_total = time.time()

    # ── Phase 1: Load + log_returns (CPU parallel) ──
    all_data = load_all_files(args.source_dirs, source_ids, args.max_pairs, args.io_workers)
    if not all_data:
        log.error("No data loaded"); return

    # ── Phase 2: Concatenate into giant array ──
    all_candles, assets = build_concat_array(all_data, args.seq_len)
    del all_data  # free ~3 GB
    log.info("Giant array: %s (%.1f MB)", all_candles.shape,
             all_candles.nbytes / 1e6)

    # ── Phase 3: Init JAX + TPU tokenize ──
    import jax
    import jax.numpy as jnp
    log.info("JAX: %d devices (%s)", jax.device_count(), jax.devices()[0].device_kind)

    from src.jax_v6.strate_i.tokenizer import StrateITokenizer, load_from_pytorch_checkpoint

    if args.checkpoint.endswith(".npz"):
        # JAX checkpoint saved as numpy (from train_strate_i_jax.py)
        raw = dict(np.load(args.checkpoint))
        # Reconstruct param tree: "['encoder']/['input_proj']/['kernel']" → nested dict
        params = {}
        for flat_key, arr in raw.items():
            # Strip brackets: "['encoder']/['input_proj']/['kernel']" → "encoder/input_proj/kernel"
            clean = flat_key.replace("['", "").replace("']", "")
            parts = clean.split("/")
            d = params
            for p in parts[:-1]:
                if p not in d:
                    d[p] = {}
                d = d[p]
            d[parts[-1]] = jnp.array(arr)
        # Map StrateIVAE keys → StrateITokenizer keys
        tok_params = {"encoder": params.get("encoder", {})}
        tok_params["fsq"] = {"proj_in": {"kernel": params["fsq_proj_in"]["kernel"]}}
        if "fsq_proj_out" in params:
            tok_params["fsq"]["proj_out"] = {"kernel": params["fsq_proj_out"]["kernel"]}
        params = tok_params
        log.info("Loaded JAX npz checkpoint: %s (%d params)", args.checkpoint,
                 sum(v.size for v in jax.tree.leaves(params)))
    else:
        params = load_from_pytorch_checkpoint(args.checkpoint)

    params = jax.device_put(params, jax.devices()[0])

    tokenizer = StrateITokenizer(
        hidden_channels=128, latent_dim=64, n_layers=4,
        dilation_base=2, kernel_size=3,
        fsq_levels=(8, 8, 8, 2), use_revin=False,
    )

    # Warm-up JIT with actual batch size
    log.info("JIT compiling (batch=%d)...", args.tpu_batch)
    dummy = jnp.zeros((min(args.tpu_batch, 1000), 1, 5))
    _ = jax.jit(tokenizer.apply)({"params": params}, dummy)
    log.info("JIT done")

    token_indices, exo_clocks = tokenize_batched_tpu(
        all_candles, params, tokenizer, batch_size=args.tpu_batch,
    )
    del all_candles  # free

    # ── Phase 4: Per-asset features (CPU) ──
    log.info("Computing per-asset features...")
    tokenized = compute_per_asset_features(token_indices, exo_clocks, assets, args.seq_len)
    del token_indices, exo_clocks

    # ── Phase 5: Write ArrayRecords (CPU parallel) ──
    shards = write_arrayrecords(tokenized, args.output_dir, args.seq_len, workers=args.io_workers)

    # ── Manifest ──
    total = sum(s["count"] for s in shards)
    manifest = {
        "shards": shards, "total_records": total, "seq_len": args.seq_len,
        "exo_clock_dim": 2,
        "sources": {str(sid): sdir for sid, sdir in zip(source_ids, args.source_dirs)},
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t_total
    log.info("=== DONE: %d records, %d shards in %.1fs (%.1f min) ===",
             total, len(shards), elapsed, elapsed / 60)


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
