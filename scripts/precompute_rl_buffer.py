"""Precompute RL buffer: JEPA inference on TPU + real close log-returns.

Phase 0 (before JAX): Load raw OHLCV files → compute close log-returns
    per asset, segment into seq_len windows → per-sequence signed returns.
Phase 1 (main thread): Read ArrayRecord shards → token_indices, exo_clock.
Phase 2 (after JAX init): JEPA context_encoder batch inference → h_last.
Phase 3: Merge embeddings + returns + vol → save per-asset npz files.

Usage:
    PYTHONPATH=. python scripts/precompute_rl_buffer.py \
        --raw_dirs data/raw_1m_parquet/ data/ohlcv_stocks_daily/ \
        --arrayrecord_dir data/arrayrecord_multi/ \
        --jepa_ckpt_dir checkpoints/jax_v6e/multi/21825/ \
        --config configs/scaling/v6e_multi.yaml \
        --output_dir data/rl_buffer/

    # Smoke test (5 batches only)
    PYTHONPATH=. python scripts/precompute_rl_buffer.py \
        --arrayrecord_dir data/arrayrecord_multi/ \
        --config configs/scaling/v6e_multi.yaml \
        --max_batches 5
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
log = logging.getLogger("precompute_rl")

# Resolution scale IDs — must match pretokenize_tpu.py
_SCALE_ID_MAP = {
    'futures_1m_parquet': 0, 'spot_1m_parquet': 0,
    '1m_parquet': 0, 'ohlcv_1m': 0, 'arrayrecord_1m': 0,
    'ohlcv_stocks_1h': 1,
    'ohlcv_stocks_daily': 2, 'ohlcv_sp500': 2,
    'ohlcv_forex': 2, 'ohlcv_commodities': 2,
    'yfinance_parquet': 2, 'sp500': 2, 'stocks': 2,
}


def _get_scale_id(dir_name: str) -> int:
    """Get scale_id from directory name."""
    return _SCALE_ID_MAP.get(dir_name, 0)


# ── Phase 0: Load raw OHLCV + compute close returns (BEFORE JAX init) ──────

def _load_raw_close_returns(args):
    """Load one raw file, compute close log-returns segmented by seq_len.

    Returns (unique_name, per_seq_returns) or None.
    unique_name matches ArrayRecord pair_name format: "{dir_tag}__{file_stem}"
    """
    file_path, seq_len = args
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

        # Close log-returns (same as pretokenize_tpu.py: log(close[t+1]) - log(close[t]))
        close = data[:, 3]
        close_lr = np.log(close[1:] + 1e-9) - np.log(close[:-1] + 1e-9)
        close_lr = np.clip(close_lr, -5.0, 5.0)  # clip listing/delisting noise

        # Segment into seq_len windows (must match pretokenizer exactly)
        T = close_lr.shape[0]
        n_seqs = T // seq_len
        if n_seqs == 0:
            return None

        usable = n_seqs * seq_len
        segmented = close_lr[:usable].reshape(n_seqs, seq_len)  # (n_seqs, seq_len)
        per_seq_returns = segmented.sum(axis=1).astype(np.float32)  # (n_seqs,)

        unique_name = f"{dir_tag}__{path.stem}"
        return (unique_name, per_seq_returns)
    except Exception as e:
        log.debug("Failed to load %s: %s", file_path, e)
        return None


def load_raw_returns(raw_dirs, seq_len=128, workers=8):
    """Load all raw files, compute per-sequence close returns.

    Must be called BEFORE JAX init (uses fork-based multiprocessing).

    Args:
        raw_dirs: list of directories with raw OHLCV files (.pt, .parquet, .npy)
        seq_len: sequence length (must match ArrayRecord pretokenization)
        workers: CPU worker count (keep low on TPU VMs, ~8)

    Returns:
        dict: unique_pair_name → np.array of per-sequence signed returns
    """
    tasks = []
    for raw_dir in raw_dirs:
        p = Path(raw_dir)
        if not p.exists():
            log.warning("Raw dir not found: %s", raw_dir)
            continue
        files = sorted(
            list(p.glob("*.pt")) + list(p.glob("*.npy")) + list(p.glob("*.parquet"))
        )
        log.info("Raw dir %s: %d files", raw_dir, len(files))
        for f in files:
            tasks.append((str(f), seq_len))

    if not tasks:
        return {}

    log.info("Loading %d raw files with %d workers...", len(tasks), workers)
    t0 = time.time()
    returns_dict = {}
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(_load_raw_close_returns, tasks, chunksize=16):
            if r is not None:
                returns_dict[r[0]] = r[1]
    log.info("Loaded close returns for %d assets in %.1fs", len(returns_dict), time.time() - t0)
    return returns_dict


# ── Phase 1: Read ArrayRecords (main thread, no multiprocessing) ────────────

def parse_example(serialized: bytes, seq_len: int, exo_clock_dim: int = 2) -> dict:
    """Parse ArrayRecord example → dict with token_indices, exo_clock, pair_name."""
    import tensorflow as tf

    example = tf.train.Example()
    example.ParseFromString(serialized)
    features = example.features.feature

    token_indices = np.array(features["token_indices"].int64_list.value, dtype=np.int64)
    exo_clock_flat = np.array(features["exo_clock"].float_list.value, dtype=np.float32)

    stored_dim = int(features["exo_clock_dim"].int64_list.value[0]) if "exo_clock_dim" in features else 2
    exo_clock = exo_clock_flat.reshape(seq_len, stored_dim)

    if stored_dim < exo_clock_dim:
        exo_clock = np.pad(exo_clock, ((0, 0), (0, exo_clock_dim - stored_dim)))
    elif stored_dim > exo_clock_dim:
        exo_clock = exo_clock[:, :exo_clock_dim]

    weekend_mask = np.array(features["weekend_mask"].float_list.value, dtype=np.float32)
    pair_name = features["pair_name"].bytes_list.value[0].decode("utf-8")

    return {
        "token_indices": token_indices,
        "exo_clock": exo_clock,
        "weekend_mask": weekend_mask,
        "pair_name": pair_name,
    }


def read_arrayrecords(arrayrecord_dir, seq_len, exo_clock_dim):
    """Read all ArrayRecord shards, group by pair_name (main thread only).

    Returns:
        asset_data: dict pair_name → list of parsed examples (ordered chronologically)
        total_records: int
    """
    from array_record.python.array_record_module import ArrayRecordReader

    ar_dir = Path(arrayrecord_dir)
    with open(ar_dir / "manifest.json") as f:
        manifest = json.load(f)

    asset_data = {}
    total_records = 0

    for shard_info in manifest["shards"]:
        shard_path = shard_info["path"]
        if not os.path.exists(shard_path):
            alt_path = str(ar_dir / Path(shard_path).name)
            if os.path.exists(alt_path):
                shard_path = alt_path
            else:
                continue

        reader = ArrayRecordReader(shard_path)
        n = reader.num_records()
        # read() without args returns one record; read([indices]) returns batch
        records = reader.read(list(range(n))) if n > 0 else []
        for raw in records:
            parsed = parse_example(raw, seq_len, exo_clock_dim)
            pair = parsed["pair_name"]
            if pair not in asset_data:
                asset_data[pair] = []
            asset_data[pair].append(parsed)
            total_records += 1
        reader.close()

    log.info("Read %d records from %d assets", total_records, len(asset_data))
    return asset_data, total_records


# ── Phase 2: JEPA inference ────────────────────────────────────────────────

def load_jepa_params(ckpt_dir: str, config_path: str):
    """Load JEPA params from orbax checkpoint.

    Matches run_training.py init exactly: same dummy_batch (with macro_context
    if macro_dim > 0), same optimizer chain (create_train_state).
    """
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    from src.jax_v6.config import load_config
    from src.jax_v6.jepa import FinJEPA
    from src.jax_v6.training.train_state import create_train_state

    config = load_config(config_path)
    model = FinJEPA.from_config(config)

    B, S = 1, config.embedding.seq_len
    dummy_batch = {
        "token_indices": jnp.zeros((B, S), dtype=jnp.int64),
        "weekend_mask": jnp.zeros((B, S), dtype=jnp.float32),
        "block_mask": jnp.zeros((B, S), dtype=jnp.bool_),
        "exo_clock": jnp.zeros((B, S, config.mamba2.exo_clock_dim), dtype=jnp.float32),
        "target_positions": jnp.zeros((B, 1), dtype=jnp.int64),
        "target_mask": jnp.zeros((B, 1), dtype=jnp.bool_),
    }
    # Include macro_context if model expects it (must match training init)
    if config.mamba2.macro_dim > 0:
        dummy_batch["macro_context"] = jnp.zeros((B, S, config.mamba2.macro_dim), dtype=jnp.float32)
    if config.mamba2.gnn_dim > 0:
        dummy_batch["gnn_embeddings"] = jnp.zeros((B, S, config.mamba2.gnn_dim), dtype=jnp.float32)
        dummy_batch["gnn_mask"] = jnp.zeros((B, S), dtype=jnp.float32)

    key = jax.random.PRNGKey(42)

    # Use create_train_state (same optimizer chain as run_training.py)
    dummy_state = create_train_state(
        model, key, dummy_batch,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        warmup_steps=1000,
        total_steps=100000,
        grad_clip=config.training.grad_clip,
        n_restarts=config.training.n_restarts,
    )

    # Restore via CheckpointManager (handles default/ subdir)
    ckpt_base = os.path.abspath(os.path.dirname(ckpt_dir.rstrip("/")))
    ckpt_step = int(os.path.basename(ckpt_dir.rstrip("/")))
    ckpt_mgr = ocp.CheckpointManager(ckpt_base)
    restored = ckpt_mgr.restore(ckpt_step, args=ocp.args.StandardRestore(dummy_state))
    log.info("Restored JEPA params from %s (step %d)", ckpt_base, ckpt_step)
    return restored.params, config, model


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute RL buffer from JEPA + raw returns")
    parser.add_argument("--raw_dirs", nargs="*", default=[],
                        help="Directories with raw OHLCV files for close returns")
    parser.add_argument("--arrayrecord_dir", type=str, default="data/arrayrecord_multi/")
    parser.add_argument("--jepa_ckpt_dir", type=str, default="checkpoints/jax_v6e/multi/21825/")
    parser.add_argument("--config", type=str, default="configs/scaling/v6e_multi.yaml")
    parser.add_argument("--output_dir", type=str, default="data/rl_buffer/")
    parser.add_argument("--max_batches", type=int, default=0, help="0 = all")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--io_workers", type=int, default=8,
                        help="CPU workers for raw data loading (keep <=8 on TPU VMs)")
    args = parser.parse_args()

    t0 = time.time()

    # ── Phase 0: Load raw close returns BEFORE JAX (fork-safe) ──
    returns_lookup = {}
    if args.raw_dirs:
        from src.jax_v6.config import load_config
        config_pre = load_config(args.config)
        seq_len_pre = config_pre.embedding.seq_len
        returns_lookup = load_raw_returns(args.raw_dirs, seq_len=seq_len_pre, workers=args.io_workers)
    else:
        log.warning("No --raw_dirs provided. Returns will use vol proxy (unsigned). "
                     "Pass --raw_dirs for real close log-returns.")

    # ── Phase 1 + 2: JAX init → read ArrayRecords → JEPA inference ──
    import jax
    import jax.numpy as jnp
    log.info("JAX: %d devices (%s)", jax.device_count(), jax.devices()[0].device_kind)

    from src.jax_v6.config import load_config
    config = load_config(args.config)
    seq_len = config.embedding.seq_len
    exo_clock_dim = config.mamba2.exo_clock_dim
    d_model = config.mamba2.d_model

    # Load JEPA params
    params, _, model = load_jepa_params(args.jepa_ckpt_dir, args.config)
    params = jax.device_put(params, jax.devices()[0])
    log.info("JEPA params loaded, d_model=%d", d_model)

    # JIT context encoder (macro_context=None is safe: encoder checks None)
    # Use lambda method — submodules from setup() aren't accessible outside apply()
    def _encode_fn(self, token_indices, weekend_mask, exo_clock, scale_id):
        h_x = self.context_encoder(
            token_indices,
            weekend_mask=weekend_mask,
            block_mask=None,
            exo_clock=exo_clock,
            scale_id=scale_id,
        )
        return h_x[:, -1, :]  # (B, d_model)

    @jax.jit
    def encode_batch(params, token_indices, weekend_mask, exo_clock, scale_id):
        return model.apply(
            {"params": params},
            token_indices, weekend_mask, exo_clock, scale_id,
            method=_encode_fn,
        )

    # Read ArrayRecords (main thread, no multiprocessing — safe after JAX init)
    log.info("Reading ArrayRecord shards...")
    asset_data, total_records = read_arrayrecords(args.arrayrecord_dir, seq_len, exo_clock_dim)

    # Flatten all sequences for batched inference
    all_tokens = []
    all_exo = []
    all_weekend = []
    all_pairs = []
    all_scale_ids = []

    for pair_name, examples in sorted(asset_data.items()):
        # Determine scale_id from dir_tag (first part of pair_name before __)
        dir_tag = pair_name.split("__")[0] if "__" in pair_name else ""
        scale_id_val = _get_scale_id(dir_tag)
        for ex in examples:
            all_tokens.append(ex["token_indices"])
            all_exo.append(ex["exo_clock"])
            all_weekend.append(ex["weekend_mask"])
            all_pairs.append(pair_name)
            all_scale_ids.append(scale_id_val)

    N = len(all_tokens)
    all_tokens_np = np.stack(all_tokens)                              # (N, seq_len)
    all_exo_np = np.stack(all_exo)                                    # (N, seq_len, exo_clock_dim)
    all_weekend_np = np.stack(all_weekend)                            # (N, seq_len)
    all_scale_ids_np = np.array(all_scale_ids, dtype=np.int32)        # (N,)
    del all_tokens, all_exo, all_weekend, asset_data  # free
    log.info("Total sequences: %d, tokens=%s, exo=%s", N, all_tokens_np.shape, all_exo_np.shape)

    # JIT warmup
    log.info("JIT compiling encoder...")
    bs = min(args.batch_size, N)
    _ = encode_batch(
        params,
        jnp.array(all_tokens_np[:bs]),
        jnp.array(all_weekend_np[:bs]),
        jnp.array(all_exo_np[:bs]),
        jnp.array(all_scale_ids_np[:bs]),
    )
    log.info("JIT done")

    # Batched inference
    all_h_last = []
    batch_count = 0
    t_inf = time.time()

    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        h_last = encode_batch(
            params,
            jnp.array(all_tokens_np[start:end]),
            jnp.array(all_weekend_np[start:end]),
            jnp.array(all_exo_np[start:end]),
            jnp.array(all_scale_ids_np[start:end]),
        )
        all_h_last.append(np.asarray(h_last))
        batch_count += 1

        if batch_count % 50 == 0:
            elapsed = time.time() - t_inf
            speed = end / max(elapsed, 0.01)
            log.info("Batch %d: %d/%d (%.0f seq/sec)", batch_count, end, N, speed)

        if args.max_batches > 0 and batch_count >= args.max_batches:
            log.info("Stopping at max_batches=%d", args.max_batches)
            N = end
            all_pairs = all_pairs[:N]
            all_exo_np = all_exo_np[:N]
            all_scale_ids_np = all_scale_ids_np[:N]
            break

    all_h_last = np.concatenate(all_h_last)  # (N, d_model)
    log.info("JEPA inference done: %d sequences in %.1fs", N, time.time() - t_inf)
    del all_tokens_np, all_weekend_np  # free

    # ── Bifurcation index per sequence (M=3 geodesic perturbations) ──────────
    # Measures market chaos: high = bifurcating regimes, low = universe consensus.
    # Used by: CQL alpha modulation (tdmpc2.py) and PER (train_cross_sectional.py).
    # Cost: M=3 geodesic perturbations + 3×3 eigenvalue decomposition per sequence.
    log.info("Computing bifurcation indices (M=3 perturbations, N=%d)...", N)
    t_bif = time.time()
    M_BIF = 3
    SIGMA_BIF = 0.01
    rng_bif = np.random.default_rng(42)
    all_bifurcation = np.zeros(N, dtype=np.float32)

    for i in range(N):
        h = all_h_last[i]  # (d_model,)
        h_norm = h / (np.linalg.norm(h) + 1e-8)
        perturbed = []
        for _ in range(M_BIF):
            noise = rng_bif.standard_normal(h.shape).astype(np.float32) * SIGMA_BIF
            noise = noise - np.dot(noise, h_norm) * h_norm  # tangent plane projection
            h_p = h_norm + noise
            h_p = h_p / (np.linalg.norm(h_p) + 1e-8)
            perturbed.append(h_p)
        H_mat = np.stack(perturbed)                     # (M, d)
        cov = H_mat @ H_mat.T                           # (M, M)
        eigvals = np.abs(np.linalg.eigvalsh(cov))
        eigvals = eigvals / (eigvals.sum() + 1e-10)
        all_bifurcation[i] = -np.sum(eigvals * np.log(eigvals + 1e-10))  # entropy

    log.info("Bifurcation done: %.1fs, mean=%.4f, max=%.4f",
             time.time() - t_bif, all_bifurcation.mean(), all_bifurcation.max())

    # ── Phase 3: Merge returns + vol + embeddings → save per-asset ──
    vol_proxy = all_exo_np[:, :, 0].mean(axis=1)  # (N,) mean RV per sequence
    del all_exo_np

    # Build per-asset arrays
    asset_buffers = {}
    seq_counters = {}  # pair_name → running sequence index (for returns alignment)

    for i in range(N):
        pair = all_pairs[i]
        if pair not in asset_buffers:
            asset_buffers[pair] = {"h_last": [], "vol": [], "returns": [], "bifurcation": []}
            seq_counters[pair] = 0

        asset_buffers[pair]["h_last"].append(all_h_last[i])
        asset_buffers[pair]["vol"].append(vol_proxy[i])

        # Real returns from raw data, or fallback to vol proxy
        seq_idx = seq_counters[pair]
        if pair in returns_lookup and seq_idx < len(returns_lookup[pair]):
            asset_buffers[pair]["returns"].append(returns_lookup[pair][seq_idx])
        else:
            # Fallback: vol proxy (unsigned) — will be logged as warning
            asset_buffers[pair]["returns"].append(vol_proxy[i])

        asset_buffers[pair]["bifurcation"].append(all_bifurcation[i])
        seq_counters[pair] = seq_idx + 1

    # Count matches
    matched = sum(1 for p in asset_buffers if p in returns_lookup)
    total_assets = len(asset_buffers)
    log.info("Returns alignment: %d/%d assets matched with raw data (%.1f%%)",
             matched, total_assets, 100.0 * matched / max(total_assets, 1))

    if matched < total_assets:
        unmatched = [p for p in sorted(asset_buffers) if p not in returns_lookup][:5]
        log.warning("Unmatched examples: %s%s", unmatched,
                     " ..." if len(unmatched) < total_assets - matched else "")

    # Save per-asset npz files
    os.makedirs(args.output_dir, exist_ok=True)
    asset_meta = []

    for pair_name, buf in sorted(asset_buffers.items()):
        h = np.stack(buf["h_last"])      # (T, d_model)
        v = np.array(buf["vol"])         # (T,)
        r = np.array(buf["returns"])     # (T,)
        bif = np.array(buf["bifurcation"], dtype=np.float32)  # (T,)

        if h.shape[0] < 64:
            continue

        # Z-score returns per-asset (normalizes across asset classes)
        r_mean = r.mean()
        r_std = max(r.std(), 1e-8)
        r_normalized = (r - r_mean) / r_std

        out_path = os.path.join(args.output_dir, f"{pair_name}.npz")
        np.savez_compressed(out_path, h_last=h, vol=v, returns=r_normalized,
                            bifurcation_index=bif,
                            returns_mean=np.float32(r_mean), returns_std=np.float32(r_std))
        asset_meta.append({
            "pair": pair_name,
            "n_steps": int(h.shape[0]),
            "d_model": int(d_model),
            "path": out_path,
            "has_real_returns": pair_name in returns_lookup,
        })

    manifest_out = {
        "assets": asset_meta,
        "n_assets": len(asset_meta),
        "total_steps": sum(a["n_steps"] for a in asset_meta),
        "d_model": int(d_model),
        "seq_len": int(seq_len),
        "exo_clock_dim": int(exo_clock_dim),
        "assets_with_real_returns": matched,
        "returns_normalized": True,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest_out, f, indent=2)

    elapsed = time.time() - t0
    log.info("=== DONE: %d assets (%d with real returns), %d total steps, "
             "saved to %s in %.1fs ===",
             len(asset_meta), matched, manifest_out["total_steps"],
             args.output_dir, elapsed)


if __name__ == "__main__":
    main()
