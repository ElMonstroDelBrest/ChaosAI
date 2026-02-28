"""Pre-compute dual-scale (micro + macro) replay buffer for Strate IV.

Uses the SAME JEPA encoder on two scales:
  - Micro: per-window embedding (128 1-min candles = ~2h context)
  - Macro: rolling mean of past N_MACRO windows (~4 days at 128 min/window)

Computes multiverse convergence on both scales + cross-scale coherence.
Saves transitions as NPZ shards for offline TD-MPC2 training.

Usage on TPU VM:
    PYTHONPATH=. SCALE_CONFIG=configs/scaling/v6e_54m_gnn_cfm.yaml \
    SCALE_TIER=54m_gnn_cfm TPU_GEN=v6e \
    python3 -u scripts/precompute_dual_buffer.py
"""

import os
import sys

if __name__ != "__main__":
    sys.exit(0)

from src.common.env_setup import setup_tpu_env
setup_tpu_env()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
from pathlib import Path

import numpy as np
import torch

import jax
import jax.numpy as jnp

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
EVAL_DATA_DIR = os.environ.get("EVAL_DATA_DIR", "data/ohlcv_1m/")
MACRO_DATA_DIR = os.environ.get("MACRO_DATA_DIR", "data/ohlcv_v5/")
ARRAYRECORD_DIR = os.environ.get("ARRAYRECORD_DIR", "data/arrayrecord_1m_p1/")
OUTPUT_DIR = os.environ.get("BUFFER_DIR", "data/dual_buffer/")
SEQ_LEN = 128
TRAIN_RATIO = 0.7
BATCH_ENC = 256
MAX_WINDOWS = 150
FUTURE_LEN = 32
N_MACRO = 48          # rolling window for macro embedding (~48×128min ≈ 4 days)
N_MULTIVERSES = 5
PERTURB_SIGMA = 0.01
TX_COST = 0.0008


def p(msg):
    print(msg, flush=True)


from src.common.data_io import read_arrayrecord_tokens


def compute_convergence_np(embeddings, n_multi, sigma):
    """Simplified convergence from M perturbed embeddings."""
    # embeddings: (M, d) — perturbed versions of a single embedding
    means = embeddings  # each "universe" is a single point
    inter_std = float(np.std(np.mean(embeddings, axis=-1)))
    intra_std = max(float(np.mean(np.std(embeddings, axis=-1))), 1e-8)
    convergence = 1.0 / (1.0 + inter_std / intra_std)

    # Gram matrix eigenvalues for bifurcation
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    gram = centered @ centered.T / max(embeddings.shape[1], 1)
    eigvals = np.abs(np.linalg.eigvalsh(gram))
    eigvals = eigvals / (eigvals.sum() + 1e-10)
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-10))
    bifurcation = float(np.exp(entropy))

    lyapunov = float(np.log(max(inter_std, 1e-10) / max(sigma, 1e-10)))

    return np.array([
        convergence,
        0.0,  # divergence_rate (need prev step, skip for offline)
        bifurcation,
        lyapunov,
        inter_std,
    ], dtype=np.float32)


def perturb_embedding_np(h, n_multi, sigma, rng):
    """Geodesic perturbation on hypersphere (numpy version)."""
    d = h.shape[0]
    h_norm = np.linalg.norm(h)
    h_hat = h / (h_norm + 1e-10)

    noise = rng.standard_normal((n_multi, d)).astype(np.float32)
    # Project onto tangent plane
    dot = (noise * h_hat[None, :]).sum(axis=-1, keepdims=True)
    tangent = noise - dot * h_hat[None, :]
    # Perturb + re-project to sphere
    perturbed = h[None, :] + sigma * tangent
    norms = np.linalg.norm(perturbed, axis=-1, keepdims=True)
    perturbed = perturbed * (h_norm / (norms + 1e-10))
    return perturbed


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: Load JEPA
# ═════════════════════════════════════════════════════════════════════════════
p("=" * 65)
p("  PRE-COMPUTE DUAL BUFFER — Strate IV")
p("=" * 65)
t_global = time.time()

p("\n[1/5] Loading JEPA model + checkpoint...")
from src.common.jax_checkpoint import load_jepa_checkpoint
ckpt = load_jepa_checkpoint(os.environ["SCALE_CONFIG"])
config, state, mesh = ckpt["config"], ckpt["state"], ckpt["mesh"]
n_params, d_model, latest = ckpt["n_params"], ckpt["d_model"], ckpt["latest_step"]
p("  JEPA: %.1fM params (d=%d), checkpoint step %d" % (n_params / 1e6, d_model, latest))

# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Read data + collect per-pair info
# ═════════════════════════════════════════════════════════════════════════════
p("\n[2/5] Reading pre-tokenized data + prices...")

ar_dir = Path(ARRAYRECORD_DIR)
ohlcv_dir = Path(EVAL_DATA_DIR)
macro_dir = Path(MACRO_DATA_DIR)
ar_shards = sorted(ar_dir.glob("*.arrayrecord"))
p("  Found %d ArrayRecord shards" % len(ar_shards))

# Collect per-pair data
pairs_data = []
t0 = time.time()

for si, shard_path in enumerate(ar_shards):
    pair_name = shard_path.stem
    ohlcv_path = ohlcv_dir / f"{pair_name}.pt"
    macro_path = macro_dir / f"{pair_name}.pt"
    if not ohlcv_path.exists():
        continue

    ohlcv = torch.load(str(ohlcv_path), map_location="cpu", weights_only=True)
    closes = ohlcv[:, 3].numpy()
    T = len(closes)

    # Load macro (1h) OHLCV if available
    has_macro = macro_path.exists()
    if has_macro:
        macro_ohlcv = torch.load(str(macro_path), map_location="cpu", weights_only=True)
        macro_closes = macro_ohlcv[:, 3].numpy()
    else:
        macro_closes = None

    pair_tokens = read_arrayrecord_tokens(shard_path)
    n_records = pair_tokens.shape[0]

    tokens_seq = []
    returns_seq = []
    vol_labels = []
    revin_stds_seq = []
    log_rets = np.diff(np.log(closes + 1e-10))

    for w in range(n_records):
        entry_idx = w * SEQ_LEN + SEQ_LEN + 1
        exit_idx = entry_idx + FUTURE_LEN
        if exit_idx >= T:
            continue

        tokens_seq.append(pair_tokens[w])
        ret = np.log(closes[exit_idx] / (closes[entry_idx] + 1e-10))
        returns_seq.append(ret)

        # Realized vol in context window
        ctx_start = max(0, w * SEQ_LEN)
        ctx_end = w * SEQ_LEN + SEQ_LEN
        ctx_rets = log_rets[ctx_start:min(ctx_end, len(log_rets))]
        vol = np.std(ctx_rets) if len(ctx_rets) > 1 else 0.0
        vol_labels.append(vol)

        # RevIN stds from context OHLCV
        ctx_ohlcv = ohlcv[ctx_start:ctx_end].numpy()
        revin_std = np.std(ctx_ohlcv, axis=0) if len(ctx_ohlcv) > 1 else np.ones(5)
        revin_stds_seq.append(revin_std.astype(np.float32))

    if len(tokens_seq) < 4:
        continue

    if len(tokens_seq) > MAX_WINDOWS:
        tokens_seq = tokens_seq[-MAX_WINDOWS:]
        returns_seq = returns_seq[-MAX_WINDOWS:]
        vol_labels = vol_labels[-MAX_WINDOWS:]
        revin_stds_seq = revin_stds_seq[-MAX_WINDOWS:]

    pairs_data.append({
        "name": pair_name,
        "tokens": np.array(tokens_seq, dtype=np.int64),
        "returns": np.array(returns_seq, dtype=np.float64),
        "revin_stds": np.array(revin_stds_seq, dtype=np.float32),
        "has_macro": has_macro,
        "macro_closes": macro_closes,
    })

    if (si + 1) % 100 == 0:
        p("  Processed %d/%d shards (%d pairs)" % (si + 1, len(ar_shards), len(pairs_data)))

n_pairs = len(pairs_data)
total_windows = sum(len(pd["tokens"]) for pd in pairs_data)
p("  %d pairs, %d total windows in %.1fs" % (n_pairs, total_windows, time.time() - t0))

# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Extract JEPA embeddings for ALL windows
# ═════════════════════════════════════════════════════════════════════════════
p("\n[3/5] Extracting JEPA embeddings...")

from src.common.jax_encoder import create_encoder_from_config
encoder, encode_batch = create_encoder_from_config(config)

target_params = state.target_params

emb_dim = 2 * d_model

# Concatenate all tokens, encode in batches, then split back per pair
all_tokens = np.concatenate([pd["tokens"] for pd in pairs_data])
n_total = len(all_tokens)
embeddings = np.zeros((n_total, emb_dim), dtype=np.float32)
t0 = time.time()

for i in range(0, n_total, BATCH_ENC):
    end = min(i + BATCH_ENC, n_total)
    bt = jnp.array(all_tokens[i:end], dtype=jnp.int32)
    bx = jnp.zeros((end - i, SEQ_LEN, 2), dtype=jnp.float32)
    embeddings[i:end] = np.array(encode_batch(target_params, bt, bx))
    if (i // BATCH_ENC + 1) % 50 == 0:
        p("  Encoded %d/%d" % (end, n_total))

p("  Embeddings (%d, %d) in %.1fs" % (n_total, emb_dim, time.time() - t0))

# Split embeddings back per pair
idx = 0
for pd in pairs_data:
    n = len(pd["tokens"])
    pd["embeddings"] = embeddings[idx:idx + n]
    idx += n

# ═════════════════════════════════════════════════════════════════════════════
# Phase 4: Build dual-scale observations + transitions
# ═════════════════════════════════════════════════════════════════════════════
p("\n[4/5] Building dual-scale observations + transitions...")

rng = np.random.default_rng(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_obs = []
all_next_obs = []
all_rewards = []
all_dones = []
all_actions = []
all_pair_ids = []
all_is_train = []

for pid, pd in enumerate(pairs_data):
    embs = pd["embeddings"]       # (N, emb_dim)
    rets = pd["returns"]          # (N,)
    revins = pd["revin_stds"]     # (N, 5)
    n_win = len(embs)
    split = int(n_win * TRAIN_RATIO)

    for t in range(n_win):
        # --- Micro embedding (current window) ---
        h_micro = embs[t]  # (emb_dim,)

        # --- Macro embedding (rolling mean of past N_MACRO windows) ---
        start_macro = max(0, t - N_MACRO)
        h_macro = embs[start_macro:t + 1].mean(axis=0)  # (emb_dim,)

        # --- Multiverse perturbation + convergence (micro) ---
        perturbed_micro = perturb_embedding_np(h_micro, N_MULTIVERSES, PERTURB_SIGMA, rng)
        conv_micro = compute_convergence_np(perturbed_micro, N_MULTIVERSES, PERTURB_SIGMA)

        # --- Multiverse perturbation + convergence (macro) ---
        perturbed_macro = perturb_embedding_np(h_macro, N_MULTIVERSES, PERTURB_SIGMA, rng)
        conv_macro = compute_convergence_np(perturbed_macro, N_MULTIVERSES, PERTURB_SIGMA)

        # --- Cross-scale coherence ---
        cos_sim = np.dot(h_micro, h_macro) / (
            np.linalg.norm(h_micro) * np.linalg.norm(h_macro) + 1e-10
        )
        cross_tf = np.array([cos_sim], dtype=np.float32)

        # --- Build observation ---
        obs = np.concatenate([
            h_micro.astype(np.float32),     # (emb_dim,)
            h_macro.astype(np.float32),     # (emb_dim,)
            conv_micro,                      # (5,)
            conv_macro,                      # (5,)
            cross_tf,                        # (1,)
            revins[t],                       # (5,)
            np.array([rets[t]], dtype=np.float32),  # (1,)
        ])

        all_obs.append(obs)
        all_rewards.append(rets[t])
        all_actions.append(np.array([1.0], dtype=np.float32))  # behavior: always long
        all_pair_ids.append(pid)
        all_is_train.append(t < split)
        all_dones.append(t == n_win - 1)

    if (pid + 1) % 100 == 0:
        p("  Built obs for %d/%d pairs" % (pid + 1, n_pairs))

# Build next_obs (shifted by 1, last gets zeros)
for i in range(len(all_obs) - 1):
    # next_obs is the obs of the NEXT window in the SAME pair
    if all_pair_ids[i] == all_pair_ids[i + 1]:
        all_next_obs.append(all_obs[i + 1])
    else:
        all_next_obs.append(np.zeros_like(all_obs[i]))
        all_dones[i] = True  # end of pair
all_next_obs.append(np.zeros_like(all_obs[-1]))
all_dones[-1] = True

obs_arr = np.array(all_obs, dtype=np.float32)
next_obs_arr = np.array(all_next_obs, dtype=np.float32)
reward_arr = np.array(all_rewards, dtype=np.float32)
action_arr = np.array(all_actions, dtype=np.float32)
done_arr = np.array(all_dones, dtype=np.float32)
pair_id_arr = np.array(all_pair_ids, dtype=np.int32)
is_train_arr = np.array(all_is_train, dtype=bool)

obs_dim = obs_arr.shape[1]
p("  Total transitions: %d (obs_dim=%d)" % (len(obs_arr), obs_dim))
p("  Train: %d, Test: %d" % (is_train_arr.sum(), (~is_train_arr).sum()))

# ═════════════════════════════════════════════════════════════════════════════
# Phase 5: Save as NPZ shards
# ═════════════════════════════════════════════════════════════════════════════
p("\n[5/5] Saving buffer shards...")

# Save train and test separately
train_mask = is_train_arr
test_mask = ~is_train_arr

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "train.npz"),
    obs=obs_arr[train_mask],
    next_obs=next_obs_arr[train_mask],
    action=action_arr[train_mask],
    reward=reward_arr[train_mask],
    done=done_arr[train_mask],
    pair_ids=pair_id_arr[train_mask],
)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "test.npz"),
    obs=obs_arr[test_mask],
    next_obs=next_obs_arr[test_mask],
    action=action_arr[test_mask],
    reward=reward_arr[test_mask],
    done=done_arr[test_mask],
    pair_ids=pair_id_arr[test_mask],
)

# Also save metadata
import json
meta = {
    "obs_dim": int(obs_dim),
    "emb_dim": int(emb_dim),
    "n_pairs": int(n_pairs),
    "n_train": int(train_mask.sum()),
    "n_test": int(test_mask.sum()),
    "n_macro_window": N_MACRO,
    "n_multiverses": N_MULTIVERSES,
    "perturb_sigma": PERTURB_SIGMA,
    "future_len": FUTURE_LEN,
    "seq_len": SEQ_LEN,
    "jepa_checkpoint_step": int(latest),
    "jepa_params_M": round(n_params / 1e6, 1),
    "d_model": int(d_model),
}
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

elapsed = time.time() - t_global
p("\n  Buffer saved to %s" % OUTPUT_DIR)
p("  Train: %s" % os.path.join(OUTPUT_DIR, "train.npz"))
p("  Test:  %s" % os.path.join(OUTPUT_DIR, "test.npz"))
p("  obs_dim=%d, %d train, %d test transitions" % (obs_dim, train_mask.sum(), test_mask.sum()))
p("  Total time: %.1fs" % elapsed)
p("=" * 65)
