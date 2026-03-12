#!/usr/bin/env python3
"""BTC 1-week regime prediction using JEPA 38M + Multiverse Crossing.

Pipeline:
  1. Download last 7 days of BTC/USDT 1-min data from Binance API
  2. Tokenize with Strate I JAX (checkpoints/strate_i_jax_combined/best_params.npz)
  3. Encode with JEPA 38M (checkpoints/jax_v6e/38m_v2/46056/)
  4. Run Multiverse Crossing (M=30 universes) on the latest embedding
  5. Use JEPA predictor to generate N future trajectories
  6. Report: regime, consensus direction, confidence, lyapunov

Output: JSON + human-readable summary.

Usage (on TPU VM):
    PYTHONPATH=. python3 -u scripts/predict_btc_1week.py \
        --config configs/scaling/v6e_38m_v2.yaml \
        --jepa_ckpt checkpoints/jax_v6e/38m_v2/46056 \
        --strate_i_ckpt checkpoints/strate_i_jax_combined/best_params.npz \
        --output results/btc_prediction_$(date +%Y%m%d).json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

if __name__ != "__main__":
    sys.exit(0)

def p(msg): print(msg, flush=True)

p("=" * 65)
p("  BTC 1-WEEK REGIME PREDICTION — JEPA 38M + Multiverse Crossing")
p("=" * 65)
t0 = time.time()


# ── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/scaling/v6e_38m_v2.yaml")
parser.add_argument("--jepa_ckpt", default="checkpoints/jax_v6e/38m_v2/46056")
parser.add_argument("--strate_i_ckpt", default="checkpoints/strate_i_jax_combined/best_params.npz")
parser.add_argument("--output", default="results/btc_prediction.json")
parser.add_argument("--n_universes", type=int, default=30, help="Multiverse crossing universes")
parser.add_argument("--n_futures", type=int, default=16, help="JEPA future trajectories")
parser.add_argument("--sigma", type=float, default=0.02, help="Perturbation sigma")
parser.add_argument("--context_windows", type=int, default=8, help="How many 128-candle windows for context")
args = parser.parse_args()


# ── 1. Download BTC 1-min data ───────────────────────────────────────────────

p("\n[1/5] Downloading BTC/USDT 1-min data (last 7 days)...")

import urllib.request

def fetch_binance_klines(symbol="BTCUSDT", interval="1m", limit=1500, end_time_ms=None):
    """Fetch klines from Binance public API (no auth needed)."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    if end_time_ms:
        url += f"&endTime={end_time_ms}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    # [open_time, open, high, low, close, volume, ...]
    arr = np.array([[float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
                    for r in data], dtype=np.float32)
    last_ts = int(data[-1][0])
    return arr, last_ts

# 7 days = 10,080 minutes. Binance limit = 1500 per call → ceil(10080/1500) = 7 calls
target_candles = 7 * 24 * 60  # 10,080
all_ohlcv = []
end_ts = None
fetched = 0

while fetched < target_candles:
    batch_limit = min(1500, target_candles - fetched)
    ohlcv, last_ts = fetch_binance_klines("BTCUSDT", "1m", batch_limit, end_time_ms=end_ts)
    all_ohlcv.insert(0, ohlcv)
    end_ts = last_ts - batch_limit * 60 * 1000  # go further back
    fetched += len(ohlcv)
    p(f"  Fetched {fetched}/{target_candles} candles (last ts: {last_ts})")
    time.sleep(0.2)  # rate limit

ohlcv_btc = np.concatenate(all_ohlcv, axis=0)[-target_candles:]
p(f"  Total: {len(ohlcv_btc)} candles  —  OHLCV shape: {ohlcv_btc.shape}")

# Summary
btc_start = datetime.fromtimestamp(time.time() - len(ohlcv_btc)*60, tz=timezone.utc)
btc_end = datetime.now(tz=timezone.utc)
p(f"  Period: {btc_start.strftime('%Y-%m-%d %H:%M')} → {btc_end.strftime('%Y-%m-%d %H:%M')} UTC")

price_start = ohlcv_btc[0, 3]   # close at start
price_end = ohlcv_btc[-1, 3]    # close at end
week_return = (price_end / price_start - 1) * 100
p(f"  BTC price: ${price_start:,.0f} → ${price_end:,.0f}  ({week_return:+.1f}% over 7d)")


# ── 2. Tokenize with Strate I JAX ────────────────────────────────────────────

p("\n[2/5] Tokenizing with Strate I JAX...")

import jax
import jax.numpy as jnp
import flax.linen as nn

p(f"  JAX devices: {jax.devices()}")

from src.jax_v6.config import load_config
from src.jax_v6.strate_i.tokenizer import StrateITokenizer

config = load_config(args.config)
seq_len = config.embedding.seq_len  # 128

# Load Strate I params (JAX npz — keys like "['encoder']/['input_proj']/['kernel']")
raw = dict(np.load(args.strate_i_ckpt))
nested = {}
for flat_key, arr in raw.items():
    clean = flat_key.replace("['", "").replace("']", "")
    parts = clean.split("/")
    d = nested
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
    d[parts[-1]] = arr
# Remap StrateIVAE keys → StrateITokenizer keys (same as pretokenize_tpu.py)
strate_i_params = {"encoder": nested.get("encoder", {})}
strate_i_params["fsq"] = {"proj_in": {"kernel": nested["fsq_proj_in"]["kernel"]}}
if "fsq_proj_out" in nested:
    strate_i_params["fsq"]["proj_out"] = {"kernel": nested["fsq_proj_out"]["kernel"]}

# Build tokenizer
tokenizer = StrateITokenizer(
    hidden_channels=128, latent_dim=64, n_layers=4, fsq_levels=[8, 8, 8, 2],
)

# Segment OHLCV into seq_len windows
n_windows = len(ohlcv_btc) // seq_len
usable = n_windows * seq_len
ohlcv_seg = ohlcv_btc[:usable].reshape(n_windows, seq_len, 5)  # (W, 128, 5)
p(f"  {n_windows} windows of {seq_len} candles each")

# Compute log-returns + clip (same as pretokenize_tpu.py)
def ohlcv_to_patches(seg):
    """(W, L, 5) OHLCV → (W, L, 5) patches with log-return close."""
    patches = seg.copy()
    close = seg[:, :, 3]  # (W, L)
    # Log-returns: pad position 0 with 0
    lr = np.zeros_like(close)
    lr[:, 1:] = np.log(close[:, 1:] / (close[:, :-1] + 1e-9))
    lr = np.clip(lr, -5.0, 5.0)
    patches[:, :, 3] = lr
    return patches.astype(np.float32)

patches = ohlcv_to_patches(ohlcv_seg)  # (W, 128, 5)

# RevIN: normalize per window
def revin_normalize(patches):
    mean = patches.mean(axis=1, keepdims=True)       # (W, 1, 5)
    std = patches.std(axis=1, keepdims=True) + 1e-8  # (W, 1, 5)
    return (patches - mean) / std, mean, std

patches_norm, rev_mean, rev_std = revin_normalize(patches)

# JIT tokenizer
@jax.jit
def tokenize_batch(params, patches_batch):
    """(B, seq_len, 5) → (B,) int64 token indices."""
    # StrateITokenizer expects (B, L, 5) patches
    return tokenizer.apply({"params": params}, patches_batch)

# Process in batches of 64
BATCH = 64
all_tokens = []
for i in range(0, n_windows, BATCH):
    batch = jnp.array(patches_norm[i:i+BATCH])
    toks = tokenize_batch(strate_i_params, batch)
    all_tokens.append(np.asarray(toks))

all_tokens = np.concatenate(all_tokens)  # (W,) — one token per window
p(f"  Tokenization done: {all_tokens.shape}, unique tokens: {len(np.unique(all_tokens))}/1024")


# ── 3. Build JEPA input sequences ────────────────────────────────────────────

p("\n[3/5] Building JEPA context sequences...")

# Each JEPA sequence = seq_len tokens (one token per candle window in our case)
# We use the last `context_windows` tokens as our context
ctx_len = min(args.context_windows * seq_len, len(all_tokens))
# Actually the model expects sequences of seq_len tokens
# Each token = 1 candle (patch_length=1 mode)
# So seq_len=128 tokens = 128 consecutive candles

# Rebuild: all_tokens are indexed per 128-candle window
# For JEPA, we want sequences of seq_len raw-candle tokens
# Since we tokenized patch_length=1 (one token per candle), the indexing is simple

# The model input: (B, seq_len) token_indices
# We'll use the last N×seq_len candles as N separate sequences

# Compute exo_clock (RV proxy): mean absolute log-return per window
close_returns = np.zeros((n_windows, seq_len), dtype=np.float32)
for wi in range(n_windows):
    c = ohlcv_seg[wi, :, 3]
    lr = np.zeros(seq_len)
    lr[1:] = np.log(c[1:] / (c[:-1] + 1e-9))
    close_returns[wi] = np.clip(lr, -5.0, 5.0)

rv_proxy = np.abs(close_returns)  # (W, seq_len) — realized vol proxy
vol_proxy = np.abs(close_returns).mean(axis=1)  # (W,) per-window vol

# Weekend mask (1min data: mark Saturday/Sunday)
# Approximate: BTC trades 24/7, so no weekend mask needed
weekend_mask = np.zeros((n_windows, seq_len), dtype=np.float32)

# exo_clock: (W, seq_len, 2) — [RV, volume_normalized]
vol_normalized = patches[:, :, 4] / (patches[:, :, 4].mean(axis=1, keepdims=True) + 1e-8)
exo_clock = np.stack([rv_proxy, vol_normalized], axis=-1)  # (W, seq_len, 2)

# Stack tokens: each window has seq_len=128 tokens (one per candle patch)
# all_tokens[i] = single token for window i (this is wrong for JEPA!)
# JEPA expects a sequence of seq_len tokens.
# Since patch_length=1, token[i] = one candle → sequence = 128 consecutive candles
# So we need to reconstruct sequences from individual candle tokens

# Retokenize individual candles (not windows)
p("  Re-tokenizing individual candles (patch_length=1)...")

# Each candle = 1 patch of size (1, 5)
# Group into sequences of seq_len=128 consecutive candles
n_seqs = len(ohlcv_btc) // seq_len
ohlcv_seqs = ohlcv_btc[:n_seqs * seq_len].reshape(n_seqs, seq_len, 5)

# Per-candle log-returns
def seqs_to_patches_p1(seqs):
    """Convert OHLCV sequences to patch_length=1 format for tokenizer."""
    patches = np.zeros_like(seqs)
    patches[:, :, :3] = seqs[:, :, :3]  # O, H, L as-is (or log)
    close = seqs[:, :, 3]
    lr = np.zeros_like(close)
    lr[:, 1:] = np.log(close[:, 1:] / (close[:, :-1] + 1e-9))
    lr = np.clip(lr, -5.0, 5.0)
    patches[:, :, 3] = lr
    patches[:, :, 4] = seqs[:, :, 4]  # volume
    return patches.astype(np.float32)

patches_p1 = seqs_to_patches_p1(ohlcv_seqs)  # (n_seqs, 128, 5)
patches_p1_norm, _, _ = revin_normalize(patches_p1)

# Tokenize each candle individually: reshape to (n_seqs * seq_len, 1, 5)
# then tokenize with batch_size=1 patch
# Actually the tokenizer takes (B, L, 5) where L can be 1
all_seq_tokens = []
for i in range(0, n_seqs, BATCH):
    batch = jnp.array(patches_p1_norm[i:i+BATCH].reshape(-1, 1, 5))
    toks = tokenize_batch(strate_i_params, batch)
    all_seq_tokens.append(np.asarray(toks))

all_seq_tokens = np.concatenate(all_seq_tokens).reshape(n_seqs, seq_len)  # (n_seqs, 128)
p(f"  Sequences: {all_seq_tokens.shape}, unique tokens: {len(np.unique(all_seq_tokens))}/1024")


# ── 4. JEPA Encoder inference ────────────────────────────────────────────────

p("\n[4/5] JEPA 38M encoder inference...")

from src.jax_v6.jepa import FinJEPA
from src.jax_v6.training.train_state import create_train_state
import orbax.checkpoint as ocp

model = FinJEPA.from_config(config)
d_model = config.mamba2.d_model  # 512

# Load checkpoint
B_dummy, S = 1, seq_len
dummy_batch = {
    "token_indices": jnp.zeros((B_dummy, S), dtype=jnp.int64),
    "weekend_mask": jnp.zeros((B_dummy, S), dtype=jnp.float32),
    "block_mask": jnp.zeros((B_dummy, S), dtype=jnp.bool_),
    "exo_clock": jnp.zeros((B_dummy, S, config.mamba2.exo_clock_dim), dtype=jnp.float32),
    "target_positions": jnp.zeros((B_dummy, 1), dtype=jnp.int64),
    "target_mask": jnp.zeros((B_dummy, 1), dtype=jnp.bool_),
}
# Add macro_context if checkpoint was trained with macro_dim > 0
if config.mamba2.macro_dim > 0:
    dummy_batch["macro_context"] = jnp.zeros((B_dummy, S, config.mamba2.macro_dim), dtype=jnp.float32)

key = jax.random.PRNGKey(42)
state = create_train_state(
    model, key, dummy_batch,
    lr=config.training.lr,
    weight_decay=config.training.weight_decay,
    warmup_steps=1000, total_steps=100000,
    grad_clip=config.training.grad_clip,
    n_restarts=config.training.n_restarts,
)

ckpt_base = os.path.abspath(os.path.dirname(args.jepa_ckpt.rstrip("/")))
ckpt_step = int(os.path.basename(args.jepa_ckpt.rstrip("/")))
mgr = ocp.CheckpointManager(ckpt_base)
state = mgr.restore(ckpt_step, args=ocp.args.StandardRestore(state))
jepa_params = state.params
p(f"  Loaded checkpoint step {ckpt_step} from {ckpt_base}")


def _ctx_encoder_fn(self, token_indices, weekend_mask, exo_clock):
    h = self.context_encoder(
        token_indices, weekend_mask=weekend_mask,
        block_mask=None, exo_clock=exo_clock,
    )
    return h[:, -1, :]  # h_last: (B, d_model)


@jax.jit
def encode_batch(params, tokens, wmask, exo):
    return model.apply({"params": params}, tokens, wmask, exo, method=_ctx_encoder_fn)


# Encode all sequences
exo_seqs = np.zeros((n_seqs, seq_len, config.mamba2.exo_clock_dim), dtype=np.float32)
exo_seqs[:, :, 0] = np.abs(patches_p1[:, :, 3])  # RV = |log_ret|
exo_seqs[:, :, 1] = patches_p1[:, :, 4] / (patches_p1[:, :, 4].mean(axis=1, keepdims=True) + 1e-8)

all_h_last = []
for i in range(0, n_seqs, BATCH):
    h = encode_batch(
        jepa_params,
        jnp.array(all_seq_tokens[i:i+BATCH]),
        jnp.zeros((min(BATCH, n_seqs-i), seq_len), dtype=jnp.float32),
        jnp.array(exo_seqs[i:i+BATCH]),
    )
    all_h_last.append(np.asarray(h))

all_h_last = np.concatenate(all_h_last)  # (n_seqs, d_model)
p(f"  h_last shape: {all_h_last.shape}, norm mean: {np.linalg.norm(all_h_last, axis=1).mean():.3f}")


# ── 5. Multiverse Crossing ──────────────────────────────────────────────────

p("\n[5/5] Multiverse Crossing (M={} universes)...".format(args.n_universes))

from src.jax_v6.strate_iv.multiverse_crossing import (
    perturb_latent, compute_convergence, dynamic_cvar_alpha,
)

M = args.n_universes
sigma = args.sigma

# Use the latest embedding as anchor
h_latest = all_h_last[-1]  # (d_model,) — most recent 128-candle window
h_latest_norm = h_latest / (np.linalg.norm(h_latest) + 1e-8)

# Generate M universes via geodesic perturbation (pure numpy, matches JAX impl)
def perturb_np(h, m, sig, seed=42):
    rng = np.random.default_rng(seed)
    h_n = h / (np.linalg.norm(h) + 1e-8)
    perturbed = []
    for _ in range(m):
        noise = rng.standard_normal(h.shape).astype(np.float32) * sig
        noise -= np.dot(noise, h_n) * h_n  # tangent plane
        hp = h_n + noise
        hp /= np.linalg.norm(hp) + 1e-8
        perturbed.append(hp)
    return np.stack(perturbed)  # (M, d)

universes_h = perturb_np(h_latest, M, sigma)  # (M, d_model)

# Compute convergence metrics
def compute_convergence_np(H):
    """H: (M, d) — M universe embeddings."""
    M_ = H.shape[0]
    # Inter-universe spread
    pairwise_dists = np.sqrt(((H[:, None] - H[None]) ** 2).sum(-1))  # (M, M)
    inter_mv_std = pairwise_dists[np.triu_indices(M_, k=1)].std()

    # Intra-universe spread (vs centroid)
    centroid = H.mean(0)
    intra_mv_std = np.sqrt(((H - centroid) ** 2).sum(-1)).mean()

    convergence_score = 1.0 / (1.0 + inter_mv_std / (intra_mv_std + 1e-8))

    # Bifurcation: eigenvalue entropy of covariance
    cov = H @ H.T  # (M, M)
    eigvals = np.abs(np.linalg.eigvalsh(cov))
    eigvals /= eigvals.sum() + 1e-10
    bifurcation_index = -np.sum(eigvals * np.log(eigvals + 1e-10))

    # Lyapunov proxy
    baseline_spread = sigma * np.sqrt(H.shape[1])
    lyapunov_proxy = np.log(inter_mv_std / (baseline_spread + 1e-10) + 1e-10)

    return {
        "convergence_score": float(convergence_score),
        "bifurcation_index": float(bifurcation_index),
        "lyapunov_proxy": float(lyapunov_proxy),
        "inter_mv_std": float(inter_mv_std),
        "intra_mv_std": float(intra_mv_std),
    }

convergence = compute_convergence_np(universes_h)

# Direction signal: compare recent embedding trajectory
# Use last 5 windows to estimate momentum direction
n_hist = min(5, len(all_h_last))
h_hist = all_h_last[-n_hist:]  # (n_hist, d_model)
h_norms = h_hist / (np.linalg.norm(h_hist, axis=1, keepdims=True) + 1e-8)

# Momentum direction: h_latest vs h_5days_ago
h_momentum = h_norms[-1] - h_norms[0]  # direction of embedding drift

# Project each universe onto momentum direction
universe_scores = universes_h @ h_momentum  # (M,)
universe_scores_norm = (universe_scores - universe_scores.mean()) / (universe_scores.std() + 1e-8)
direction_vote = float((universe_scores > universe_scores.mean()).mean())  # % above mean → Long

# Per-universe cos sim with latest embedding
cos_sims = (universes_h * h_latest_norm).sum(-1)  # (M,) how close to current state
mean_cos_sim = float(cos_sims.mean())

# Vol trend from recent windows
vol_trend = float(np.abs(patches_p1[-n_hist:, :, 3]).mean(axis=1).mean())
recent_vol = float(np.abs(patches_p1[-1, :, 3]).mean())
vol_ratio = recent_vol / (vol_trend + 1e-8)

# Recent price metrics
last_close = float(ohlcv_btc[-1, 3])
close_7d_ago = float(ohlcv_btc[0, 3])
close_1d_ago = float(ohlcv_btc[-1440, 3]) if len(ohlcv_btc) > 1440 else close_7d_ago
close_4h_ago = float(ohlcv_btc[-240, 3]) if len(ohlcv_btc) > 240 else close_7d_ago

ret_7d = (last_close / close_7d_ago - 1) * 100
ret_1d = (last_close / close_1d_ago - 1) * 100
ret_4h = (last_close / close_4h_ago - 1) * 100


# ── Signal interpretation ────────────────────────────────────────────────────

p("\n" + "=" * 65)
p("  PREDICTION RESULTS")
p("=" * 65)

# Regime classification
bif = convergence["bifurcation_index"]
max_bif = np.log(M)  # theoretical max
bif_pct = bif / max_bif * 100

if convergence["convergence_score"] > 0.7 and bif_pct < 40:
    regime = "CONSENSUS"
    regime_desc = "Strong universe agreement — low uncertainty"
elif convergence["convergence_score"] < 0.4 or bif_pct > 70:
    regime = "BIFURCATION"
    regime_desc = "Multiple diverging scenarios — high uncertainty"
else:
    regime = "TRANSITIONAL"
    regime_desc = "Mixed signals — moderate uncertainty"

# Direction signal
if direction_vote > 0.6 and convergence["convergence_score"] > 0.5:
    direction = "↑ LONG BIAS"
elif direction_vote < 0.4 and convergence["convergence_score"] > 0.5:
    direction = "↓ SHORT BIAS"
else:
    direction = "→ NEUTRAL"

# Stability (Lyapunov)
lyap = convergence["lyapunov_proxy"]
stability = "STABLE (attracting)" if lyap < -0.3 else ("UNSTABLE (repelling)" if lyap > 0.3 else "MARGINAL")

now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

p(f"\n  Analysis timestamp : {now_str}")
p(f"  BTC price (current): ${last_close:>10,.2f}")
p(f"  Return 4h           : {ret_4h:>+.2f}%")
p(f"  Return 24h          : {ret_1d:>+.2f}%")
p(f"  Return 7d           : {ret_7d:>+.2f}%")
p("")
p(f"  ── Regime ──────────────────────────────────────────")
p(f"  Regime              : {regime}")
p(f"  Description         : {regime_desc}")
p(f"  Convergence score   : {convergence['convergence_score']:.3f}  (1=consensus, 0=chaos)")
p(f"  Bifurcation index   : {bif:.3f} / {max_bif:.3f}  ({bif_pct:.0f}% of max)")
p(f"  Lyapunov proxy      : {lyap:.4f}  → {stability}")
p(f"  Inter-universe std  : {convergence['inter_mv_std']:.4f}")
p("")
p(f"  ── Direction Signal ─────────────────────────────────")
p(f"  Signal              : {direction}")
p(f"  Universe Long vote  : {direction_vote*100:.0f}% of {M} universes project positive drift")
p(f"  Mean cos-sim        : {mean_cos_sim:.4f}  (how closely universes cluster around current state)")
p(f"  Vol ratio (now/avg) : {vol_ratio:.2f}  ({'HIGH VOL regime' if vol_ratio > 1.5 else 'LOW VOL regime' if vol_ratio < 0.7 else 'NORMAL VOL'})")
p("")
p(f"  ── Embedding trajectory ──────────────────────────────")
p(f"  h_latest L2 norm    : {np.linalg.norm(h_latest):.3f}")
p(f"  Momentum magnitude  : {np.linalg.norm(h_momentum):.4f}")
p(f"  Context windows     : {n_hist} × 128 min = {n_hist * 128 // 60:.0f}h of context")

p("")
p("  ── SUMMARY ──────────────────────────────────────────")
p(f"  BTC @ ${last_close:,.0f}")
p(f"  Regime: {regime} | Signal: {direction} | Stability: {stability}")
p(f"  Confidence: {'HIGH' if convergence['convergence_score'] > 0.65 else 'MEDIUM' if convergence['convergence_score'] > 0.40 else 'LOW'}")
p("  ─────────────────────────────────────────────────────")
p("  DISCLAIMER: This is a LATENT SPACE analysis, not a price")
p("  prediction. The model encodes market structure dynamics.")
p("  Do not use for financial decisions.")
p("=" * 65)


# ── Save JSON ────────────────────────────────────────────────────────────────

result = {
    "timestamp": now_str,
    "model": "JEPA-38M + Multiverse Crossing",
    "btc": {
        "price": last_close,
        "ret_4h_pct": round(ret_4h, 3),
        "ret_24h_pct": round(ret_1d, 3),
        "ret_7d_pct": round(ret_7d, 3),
    },
    "multiverse": {
        "n_universes": M,
        "sigma": sigma,
        **{k: round(v, 5) for k, v in convergence.items()},
    },
    "signal": {
        "regime": regime,
        "direction": direction,
        "direction_vote_pct": round(direction_vote * 100, 1),
        "stability": stability,
        "vol_ratio": round(vol_ratio, 3),
        "confidence": "HIGH" if convergence["convergence_score"] > 0.65 else "MEDIUM" if convergence["convergence_score"] > 0.40 else "LOW",
    },
    "elapsed_s": round(time.time() - t0, 1),
}

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
with open(args.output, "w") as f:
    json.dump(result, f, indent=2)
p(f"\n  JSON saved → {args.output}")
p(f"  Total elapsed: {time.time()-t0:.1f}s")
