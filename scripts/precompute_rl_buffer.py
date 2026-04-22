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
    # scale_id: must be present if scale_emb_dim > 0, so that Flax init traces
    # scale_embed and includes it in the params pytree (matching the checkpoint)
    if getattr(config.mamba2, 'scale_emb_dim', 0) > 0:
        dummy_batch["scale_id"] = jnp.zeros((B,), dtype=jnp.int32)
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
    restored = ckpt_mgr.restore(ckpt_step, args=ocp.args.PyTreeRestore(dummy_state, partial_restore=True))
    log.info("Restored JEPA params from %s (step %d)", ckpt_base, ckpt_step)
    return restored.params, config, model


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute RL buffer from JEPA + raw returns")
    parser.add_argument("--raw_dirs", nargs="*", default=[],
                        help="Directories with raw OHLCV files for close returns")
    parser.add_argument("--arrayrecord_dir", type=str, default="data/arrayrecord_multi/")
    parser.add_argument("--jepa_ckpt_dir", "--jepa_ckpt", type=str, default="checkpoints/jax_v6e/multi/21825/",
                        dest="jepa_ckpt_dir")
    parser.add_argument("--config", type=str, default="configs/scaling/v6e_multi.yaml")
    parser.add_argument("--output_dir", type=str, default="data/rl_buffer/")
    parser.add_argument("--max_batches", type=int, default=0, help="0 = all")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--io_workers", type=int, default=8,
                        help="CPU workers for raw data loading (keep <=8 on TPU VMs)")
    parser.add_argument("--seq_cutoff_ratio", type=float, default=0.0,
                        help="If >0, save only the first fraction of each asset's sequences "
                             "to output_dir (train split). E.g. 0.8 = first 80%% = pre-2024 approx.")
    parser.add_argument("--oos_dir", type=str, default="",
                        help="If set (and seq_cutoff_ratio>0), also save the last "
                             "(1-ratio) sequences here for OOS evaluation.")
    parser.add_argument("--mvx_cfm", type=int, default=0,
                        help="If >0 AND model has CFM, use M trajectories from "
                             "flow_predictor as bifurcation source (proper MVX). "
                             "0 = legacy tangent-plane noise (default).")
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

    # JIT context encoder. When --mvx_cfm > 0 AND the model has flow_predictor,
    # also returns CFM-based bifurcation per sequence (proper Multiverse Crossing
    # surrogate, Rice-theorem grounded — sample M alternative latent futures via
    # the CFM flow predictor, measure eigenvalue entropy of their Gram matrix).
    use_cfm_mvx = args.mvx_cfm > 0 and getattr(config.predictor, 'cfm_weight', 0.0) > 0.0
    M_MVX = args.mvx_cfm if use_cfm_mvx else 0
    log.info("MVX mode: %s (M=%d, cfm_weight=%g)",
             "CFM-based" if use_cfm_mvx else "noise (legacy)",
             M_MVX, getattr(config.predictor, 'cfm_weight', 0.0))

    def _encode_fn(self, token_indices, weekend_mask, exo_clock, scale_id):
        out = self.context_encoder(
            token_indices,
            weekend_mask=weekend_mask,
            block_mask=None,
            exo_clock=exo_clock,
            scale_id=scale_id,
        )
        # Encoder may return (h_x, hidden_states) when return_hidden_states=True
        h_x = out[0] if isinstance(out, tuple) else out
        return h_x  # (B, S, d_model) — return full context for CFM sampling

    @jax.jit
    def encode_full(params, token_indices, weekend_mask, exo_clock, scale_id):
        return model.apply(
            {"params": params},
            token_indices, weekend_mask, exo_clock, scale_id,
            method=_encode_fn,
        )

    if use_cfm_mvx:
        # Build a standalone FlowPredictor instance, sample method takes its own params
        from src.jax_v6.predictors.flow_predictor import FlowPredictor
        flow_pred = FlowPredictor(
            d_model=d_model,
            hidden_dim=config.predictor.hidden_dim,
            n_layers=config.predictor.n_layers,
            seq_len=seq_len,
            dropout=config.predictor.dropout,
            ot=config.predictor.cfm_ot,
        )

        # Manual Euler integration — diffrax 0.7 is incompatible with our JIT
        # boundaries (ODETerm wrapping fails the AbstractTerm check inside jit).
        # We reimplement a 2-step Euler explicitly using the velocity field.
        @jax.jit
        def sample_one(flow_params, h_x, target_positions, key):
            """One CFM trajectory at target_positions via 2-step Euler.

            Returns: (B, d_model) — sampled latent at the (single) target position.
            """
            B, N_tgt = target_positions.shape
            n_steps = 2
            x0 = jax.random.normal(key, (B, N_tgt, d_model), dtype=h_x.dtype)
            dt = 1.0 / n_steps
            y = x0
            for step in range(n_steps):
                t_batch = jnp.full((B,), step * dt, dtype=h_x.dtype)
                v = flow_pred.apply(
                    {"params": flow_params},
                    y, t_batch, h_x, target_positions, True,
                    method=flow_pred._forward_velocity,
                )
                y = y + dt * v
            return y[:, 0, :]  # (B, d_model)

        @jax.jit
        def bifurcation_from_samples(samples_stack):
            """Eigenvalue entropy per batch elem from (M, B, d) samples."""
            samples_norm = samples_stack / (
                jnp.linalg.norm(samples_stack, axis=-1, keepdims=True) + 1e-8)
            gram = jnp.einsum("kbd,lbd->bkl", samples_norm, samples_norm)
            eigvals = jnp.abs(jnp.linalg.eigvalsh(gram))
            eigvals = eigvals / (eigvals.sum(axis=-1, keepdims=True) + 1e-10)
            return -jnp.sum(eigvals * jnp.log(eigvals + 1e-10), axis=-1)

        def sample_and_bif(flow_params, h_x, key):
            """Sample M CFM trajectories at last position, return bifurcation per seq.

            Python-loop over M (diffrax doesn't compose with vmap).
            """
            B, S = h_x.shape[0], h_x.shape[1]
            target_positions = jnp.full((B, 1), S - 1, dtype=jnp.int32)
            keys = jax.random.split(key, M_MVX)
            samples = [sample_one(flow_params, h_x, target_positions, k) for k in keys]
            samples = jnp.stack(samples, axis=0)  # (M, B, d_model)
            return bifurcation_from_samples(samples)

    def encode_batch(params, token_indices, weekend_mask, exo_clock, scale_id, key=None):
        """Returns h_last (B, d_model) and optionally cfm_bifurcation (B,)."""
        h_x = encode_full(params, token_indices, weekend_mask, exo_clock, scale_id)
        h_last = h_x[:, -1, :]
        cfm_bif = None
        if use_cfm_mvx and key is not None:
            cfm_bif = sample_and_bif(params["flow_predictor"], h_x, key)
        # Block on h_last to release h_x from device memory between batches
        h_last.block_until_ready()
        return h_last, cfm_bif

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
    warmup_key = jax.random.PRNGKey(0)
    _ = encode_batch(
        params,
        jnp.array(all_tokens_np[:bs]),
        jnp.array(all_weekend_np[:bs]),
        jnp.array(all_exo_np[:bs]),
        jnp.array(all_scale_ids_np[:bs]),
        warmup_key,
    )
    log.info("JIT done")

    # Batched inference
    all_h_last = []
    all_cfm_bif = [] if use_cfm_mvx else None
    batch_count = 0
    t_inf = time.time()
    rng = np.random.default_rng(42)

    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        bk = jax.random.PRNGKey(int(rng.integers(2**31)))
        h_last, cfm_bif = encode_batch(
            params,
            jnp.array(all_tokens_np[start:end]),
            jnp.array(all_weekend_np[start:end]),
            jnp.array(all_exo_np[start:end]),
            jnp.array(all_scale_ids_np[start:end]),
            bk,
        )
        all_h_last.append(np.asarray(h_last))
        if use_cfm_mvx:
            all_cfm_bif.append(np.asarray(cfm_bif))
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

    # ── Bifurcation index per sequence ──────────────────────────────────────
    # CFM-based MVX (proper, when --mvx_cfm > 0 and v6.5+ checkpoint with CFM):
    #   M alternative latent futures sampled via flow_predictor → eigenvalue
    #   entropy of their Gram matrix. This IS the model's response to perturbations,
    #   not pure noise around the latent.
    #
    # Legacy noise-based (fallback, M=3 tangent-plane noise on JEPA latent):
    #   measures eigenvalue entropy of M random points near h_norm. Not actually
    #   model-dependent (saturated at sigma+M values). Kept for backward compat.
    if use_cfm_mvx:
        all_bifurcation = np.concatenate(all_cfm_bif)  # already computed during encode
        log.info("Bifurcation (CFM, M=%d): mean=%.4f, std=%.4f, max=%.4f",
                 M_MVX, all_bifurcation.mean(), all_bifurcation.std(), all_bifurcation.max())
    else:
        log.info("Computing legacy bifurcation indices (M=3 noise, N=%d)...", N)
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
    oos_dir = args.oos_dir if args.oos_dir else None
    if oos_dir:
        os.makedirs(oos_dir, exist_ok=True)
    asset_meta = []
    oos_meta = []

    for pair_name, buf in sorted(asset_buffers.items()):
        h = np.stack(buf["h_last"])      # (T, d_model)
        v = np.array(buf["vol"])         # (T,)
        r = np.array(buf["returns"])     # (T,)
        bif = np.array(buf["bifurcation"], dtype=np.float32)  # (T,)

        if h.shape[0] < 30:  # lowered from 64: daily stocks have ~40 seqs at seq_len=128
            continue

        # Temporal split: first cutoff sequences = train, rest = OOS
        if args.seq_cutoff_ratio > 0.0:
            cutoff = max(30, int(h.shape[0] * args.seq_cutoff_ratio))
            h_train, h_oos = h[:cutoff], h[cutoff:]
            v_train, v_oos = v[:cutoff], v[cutoff:]
            r_train, r_oos = r[:cutoff], r[cutoff:]
            bif_train, bif_oos = bif[:cutoff], bif[cutoff:]
        else:
            h_train, v_train, r_train, bif_train = h, v, r, bif
            h_oos, v_oos, r_oos, bif_oos = None, None, None, None

        # Z-score returns using TRAIN stats only (no leakage into OOS)
        r_mean = r_train.mean()
        r_std = max(r_train.std(), 1e-8)
        r_train_norm = (r_train - r_mean) / r_std

        out_path = os.path.join(args.output_dir, f"{pair_name}.npz")
        np.savez_compressed(out_path, h_last=h_train, vol=v_train, returns=r_train_norm,
                            bifurcation_index=bif_train,
                            returns_mean=np.float32(r_mean), returns_std=np.float32(r_std))
        asset_meta.append({
            "pair": pair_name,
            "n_steps": int(h_train.shape[0]),
            "d_model": int(d_model),
            "path": out_path,
            "has_real_returns": pair_name in returns_lookup,
        })

        # Save OOS portion if requested
        if oos_dir and h_oos is not None and h_oos.shape[0] >= 10:
            r_oos_norm = (r_oos - r_mean) / r_std  # same stats as train
            oos_path = os.path.join(oos_dir, f"{pair_name}.npz")
            np.savez_compressed(oos_path, h_last=h_oos, vol=v_oos, returns=r_oos_norm,
                                bifurcation_index=bif_oos,
                                returns_mean=np.float32(r_mean), returns_std=np.float32(r_std))
            oos_meta.append({
                "pair": pair_name,
                "n_steps": int(h_oos.shape[0]),
                "d_model": int(d_model),
                "path": oos_path,
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
        "seq_cutoff_ratio": args.seq_cutoff_ratio,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest_out, f, indent=2)

    if oos_dir and oos_meta:
        oos_manifest = {
            "assets": oos_meta,
            "n_assets": len(oos_meta),
            "total_steps": sum(a["n_steps"] for a in oos_meta),
            "d_model": int(d_model),
            "seq_len": int(seq_len),
            "exo_clock_dim": int(exo_clock_dim),
            "assets_with_real_returns": sum(1 for a in oos_meta if a["has_real_returns"]),
            "returns_normalized": True,
            "seq_cutoff_ratio": args.seq_cutoff_ratio,
            "split": "oos",
        }
        with open(os.path.join(oos_dir, "manifest.json"), "w") as f:
            json.dump(oos_manifest, f, indent=2)
        log.info("OOS buffer: %d assets, %d steps → %s",
                 len(oos_meta), oos_manifest["total_steps"], oos_dir)

    elapsed = time.time() - t0
    log.info("=== DONE: %d assets (%d with real returns), %d total steps, "
             "saved to %s in %.1fs ===",
             len(asset_meta), matched, manifest_out["total_steps"],
             args.output_dir, elapsed)


if __name__ == "__main__":
    main()
