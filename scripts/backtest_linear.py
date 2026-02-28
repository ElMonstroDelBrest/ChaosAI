"""Trading Backtest v2 — Latent-State Strategies

Backtest JEPA representations via latent-state strategies on real trading returns.
Reads pre-tokenized data from ArrayRecord (same tokens the JEPA was trained on)
+ close prices from raw OHLCV for return computation.

3 latent-state strategies:
  S1: Vol-Target — always long, size by predicted vol regime (61.2% accuracy)
  S2: Regime K-Means — K=4 clusters on embeddings, size by historical Sharpe per regime
  S3: Anomaly Filter — flat when embedding distance > p80 threshold (unknown regime)

Per-pair computation with median + IQR aggregation (fixes v1 sequential compounding bug).

Usage:
    SCALE_CONFIG=configs/scaling/v6e_54m_gnn_cfm.yaml SCALE_TIER=54m_gnn_cfm \
    TPU_GEN=v6e EVAL_DATA_DIR=data/ohlcv_1m/ \
    ARRAYRECORD_DIR=data/arrayrecord_1m_p1/ \
    PYTHONPATH=. python3 -u scripts/backtest_linear.py
"""

import os
import sys

# Anti-hang: Grain worker subprocess protection
if __name__ != "__main__":
    sys.exit(0)

# Fix OpenBLAS thread explosion on TPU VMs (128+ cores)
from src.common.env_setup import setup_tpu_env
setup_tpu_env()

import json
import time
from pathlib import Path

import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax.numpy as jnp
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
EVAL_DATA_DIR = os.environ.get("EVAL_DATA_DIR", "data/ohlcv_1m/")
ARRAYRECORD_DIR = os.environ.get("ARRAYRECORD_DIR", "data/arrayrecord_1m_p1/")
SEQ_LEN = 128
TRAIN_RATIO = 0.7
BATCH_ENC = 256
MAX_WINDOWS = 150  # per pair (~19200 candles worth)
FUTURE_LEN = 32    # prediction horizon (minutes)
TX_COST = 0.0008   # round-trip Binance Futures taker fee (0.04% x 2)
ANNUALIZE = np.sqrt(252 * 24 * 60 / FUTURE_LEN)  # windows/year -> annualization
K_REGIMES = 4      # number of K-Means regimes
ANOMALY_PERCENTILE = 80  # distance threshold for anomaly filter


def p(msg):
    print(msg, flush=True)


from src.common.metrics import compute_sharpe as _compute_sharpe, compute_max_drawdown, compute_cum_return
from src.common.data_io import read_arrayrecord_tokens


def compute_sharpe(returns):
    return _compute_sharpe(returns, annualize=ANNUALIZE)


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: Load JEPA model + checkpoint
# ═════════════════════════════════════════════════════════════════════════════
p("=" * 65)
p("  BACKTEST v2 — Latent-State Strategies")
p("=" * 65)
t_global = time.time()

p("\n[1/6] Loading JEPA model + checkpoint...")
from src.common.jax_checkpoint import load_jepa_checkpoint
ckpt = load_jepa_checkpoint(os.environ["SCALE_CONFIG"])
config, state, mesh = ckpt["config"], ckpt["state"], ckpt["mesh"]
n_params, d_model, latest = ckpt["n_params"], ckpt["d_model"], ckpt["latest_step"]
S = config.embedding.seq_len
p("  JEPA: %.1fM params (d=%d), checkpoint step %d" % (n_params / 1e6, d_model, latest))

# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Read pre-tokenized data + collect prices and labels
# ═════════════════════════════════════════════════════════════════════════════
p("\n[2/6] Reading pre-tokenized data + prices...")

ar_dir = Path(ARRAYRECORD_DIR)
ohlcv_dir = Path(EVAL_DATA_DIR)
ar_shards = sorted(ar_dir.glob("*.arrayrecord"))
p("  Found %d ArrayRecord shards" % len(ar_shards))

all_tokens = []
all_vol_regime = []
all_actual_returns = []
all_pair_ids = []
all_is_train = []
n_pairs_ok = 0

t0 = time.time()
for si, shard_path in enumerate(ar_shards):
    pair_name = shard_path.stem
    ohlcv_path = ohlcv_dir / f"{pair_name}.pt"
    if not ohlcv_path.exists():
        continue

    # Read OHLCV for close prices
    ohlcv = torch.load(str(ohlcv_path), map_location="cpu", weights_only=True)
    closes = ohlcv[:, 3].numpy()
    T = len(closes)

    # Read pre-tokenized windows from arrayrecord
    pair_tokens = read_arrayrecord_tokens(shard_path)  # (N, 128)
    n_records = pair_tokens.shape[0]

    # Each record w covers candles w*128..w*128+128 (patch_length=1, stride=1)
    # compute_log_returns shifts by 1: token 0 = log_ret(candle 0, candle 1)
    # Entry price: closes[w*128 + 128], Exit price: closes[w*128 + 128 + FUTURE_LEN]
    tokens_seq = []
    vol_labels = []
    returns_seq = []
    log_rets = np.diff(np.log(closes + 1e-10))

    for w in range(n_records):
        # +1 offset from compute_log_returns (T-1 log-returns from T candles)
        entry_idx = w * SEQ_LEN + SEQ_LEN + 1
        exit_idx = entry_idx + FUTURE_LEN
        if exit_idx >= T:
            continue

        tokens_seq.append(pair_tokens[w])
        ret = (closes[exit_idx] - closes[entry_idx]) / (closes[entry_idx] + 1e-10)
        returns_seq.append(ret)

        # Realized volatility in future window
        fut_rets = log_rets[entry_idx:min(exit_idx, len(log_rets))]
        vol = np.std(fut_rets) if len(fut_rets) > 1 else 0.0
        vol_labels.append(vol)

    if len(tokens_seq) < 4:
        continue

    # Limit to last MAX_WINDOWS (most recent data)
    if len(tokens_seq) > MAX_WINDOWS:
        tokens_seq = tokens_seq[-MAX_WINDOWS:]
        vol_labels = vol_labels[-MAX_WINDOWS:]
        returns_seq = returns_seq[-MAX_WINDOWS:]

    # Volatility regime: median split per pair
    vol_arr = np.array(vol_labels)
    vol_med = np.median(vol_arr[vol_arr > 0]) if np.any(vol_arr > 0) else 0.0
    vol_binary = (vol_arr > vol_med).astype(np.int64)

    arr = np.array(tokens_seq, dtype=np.int64)
    split = int(len(arr) * TRAIN_RATIO)
    itr = np.zeros(len(arr), dtype=bool)
    itr[:split] = True

    all_tokens.append(arr)
    all_vol_regime.append(vol_binary)
    all_actual_returns.append(np.array(returns_seq, dtype=np.float64))
    all_pair_ids.append(np.full(len(arr), n_pairs_ok, dtype=np.int32))
    all_is_train.append(itr)
    n_pairs_ok += 1

    if (si + 1) % 100 == 0:
        p("  Processed %d/%d shards (%d pairs ok)" % (si + 1, len(ar_shards), n_pairs_ok))

tokens = np.concatenate(all_tokens)
labels_vol = np.concatenate(all_vol_regime)
actual_returns = np.concatenate(all_actual_returns)
pair_ids = np.concatenate(all_pair_ids)
is_train = np.concatenate(all_is_train)

n_total = len(tokens)
n_train = int(is_train.sum())
n_test = n_total - n_train

p("  %d samples (%d train, %d test) from %d pairs in %.1fs" % (
    n_total, n_train, n_test, n_pairs_ok, time.time() - t0))
p("  Vol: %.1f%% high" % (labels_vol.mean() * 100))

# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Extract JEPA embeddings
# ═════════════════════════════════════════════════════════════════════════════
p("\n[3/6] Extracting JEPA embeddings...")
target_params = state.target_params

from src.common.jax_encoder import create_encoder_from_config
encoder, encode_batch = create_encoder_from_config(config)

emb_dim = 2 * d_model
embeddings = np.zeros((n_total, emb_dim), dtype=np.float32)
t0 = time.time()

for i in range(0, n_total, BATCH_ENC):
    end = min(i + BATCH_ENC, n_total)
    bt = jnp.array(tokens[i:end], dtype=jnp.int32)
    bx = jnp.zeros((end - i, SEQ_LEN, 2), dtype=jnp.float32)
    embeddings[i:end] = np.array(encode_batch(target_params, bt, bx))
    if (i // BATCH_ENC + 1) % 50 == 0:
        p("  Encoded %d/%d" % (end, n_total))

p("  Embeddings (%d, %d) in %.1fs" % (n_total, emb_dim, time.time() - t0))

# ═════════════════════════════════════════════════════════════════════════════
# Phase 4: Train latent-state strategies on train set
# ═════════════════════════════════════════════════════════════════════════════
p("\n[4/6] Training latent-state strategies...")

X_train = embeddings[is_train]
X_test = embeddings[~is_train]

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)

# --- S1: Vol-Target (logistic regression on vol regime) ---
best_vol = {"bal_acc": 0.0, "clf": None, "C": 0.0}
for C in [0.001, 0.01, 0.1, 1.0]:
    clf = LogisticRegression(max_iter=3000, C=C, solver="lbfgs", class_weight="balanced")
    clf.fit(Xtr, labels_vol[is_train])
    ba = balanced_accuracy_score(labels_vol[~is_train], clf.predict(Xte))
    if ba > best_vol["bal_acc"]:
        best_vol = {"bal_acc": ba, "clf": clf, "C": C}

clf_vol = best_vol["clf"]
p("  S1 Vol classifier: bal_acc=%.1f%% (C=%.3f)" % (best_vol["bal_acc"] * 100, best_vol["C"]))

# --- S2: Regime K-Means (K=4 on train embeddings) ---
kmeans = KMeans(n_clusters=K_REGIMES, random_state=42, n_init=10)
kmeans.fit(Xtr)
train_cluster_ids = kmeans.labels_
train_returns = actual_returns[is_train]

# Compute Sharpe per cluster on train set
regime_sharpe = {}
for c in range(K_REGIMES):
    mask_c = train_cluster_ids == c
    if mask_c.sum() > 1:
        regime_sharpe[c] = compute_sharpe(train_returns[mask_c])
    else:
        regime_sharpe[c] = 0.0

max_pos_sharpe = max(max(regime_sharpe.values()), 1e-6)
p("  S2 K-Means regime Sharpes: %s" % {c: "%.2f" % s for c, s in regime_sharpe.items()})
p("  S2 Favorable regimes (Sharpe>0): %s" % [c for c, s in regime_sharpe.items() if s > 0])

# --- S3: Anomaly Filter (centroid + distance threshold) ---
centroid = Xtr.mean(axis=0)
train_dists = np.linalg.norm(Xtr - centroid, axis=1)
threshold = np.percentile(train_dists, ANOMALY_PERCENTILE)
p("  S3 Anomaly threshold (p%d): %.2f" % (ANOMALY_PERCENTILE, threshold))

# ═════════════════════════════════════════════════════════════════════════════
# Phase 5: Simulate per-pair on test set
# ═════════════════════════════════════════════════════════════════════════════
p("\n[5/6] Simulating per-pair trading strategies...")

unique_pairs = np.unique(pair_ids)
strat_names = ["S1: Vol-Target", "S2: Regime K-Means", "S3: Anomaly Filter", "Buy & Hold"]
per_pair_results = {s: [] for s in strat_names}  # list of (sharpe, cumret, maxdd) per pair

test_pair_ids = pair_ids[~is_train]
test_returns = actual_returns[~is_train]
test_emb_scaled = Xte  # already scaled

for pid in unique_pairs:
    # Test-set mask for this pair
    pmask = test_pair_ids == pid
    if pmask.sum() < 3:
        continue

    pair_ret = test_returns[pmask]
    pair_emb = test_emb_scaled[pmask]

    # --- S1: Vol-Target ---
    pred_vol = clf_vol.predict(pair_emb)
    sizes = np.where(pred_vol == 1, 0.3, 1.0)  # high-vol -> reduce
    s1_rets = sizes * pair_ret
    # Transaction cost when size changes
    for i in range(1, len(sizes)):
        if sizes[i] != sizes[i - 1]:
            s1_rets[i] -= TX_COST
    per_pair_results["S1: Vol-Target"].append((
        compute_sharpe(s1_rets), compute_cum_return(s1_rets), compute_max_drawdown(s1_rets),
    ))

    # --- S2: Regime K-Means ---
    cluster_ids = kmeans.predict(pair_emb)
    # Size proportional to historical Sharpe (0 if negative)
    sizes_km = np.array([max(regime_sharpe[c], 0) / max_pos_sharpe for c in cluster_ids])
    s2_rets = sizes_km * pair_ret
    for i in range(1, len(sizes_km)):
        if sizes_km[i] != sizes_km[i - 1]:
            s2_rets[i] -= TX_COST
    per_pair_results["S2: Regime K-Means"].append((
        compute_sharpe(s2_rets), compute_cum_return(s2_rets), compute_max_drawdown(s2_rets),
    ))

    # --- S3: Anomaly Filter ---
    dists = np.linalg.norm(pair_emb - centroid, axis=1)
    normal_mask = dists <= threshold  # normal regime -> trade
    s3_rets = np.where(normal_mask, pair_ret, 0.0)
    s3_sizes = normal_mask.astype(float)
    for i in range(1, len(s3_sizes)):
        if s3_sizes[i] != s3_sizes[i - 1]:
            s3_rets[i] -= TX_COST
    per_pair_results["S3: Anomaly Filter"].append((
        compute_sharpe(s3_rets), compute_cum_return(s3_rets), compute_max_drawdown(s3_rets),
    ))

    # --- Buy & Hold ---
    per_pair_results["Buy & Hold"].append((
        compute_sharpe(pair_ret), compute_cum_return(pair_ret), compute_max_drawdown(pair_ret),
    ))

p("  Simulated %d pairs" % len(per_pair_results["Buy & Hold"]))

# ═════════════════════════════════════════════════════════════════════════════
# Phase 6: Report median + IQR per-pair + JSON export
# ═════════════════════════════════════════════════════════════════════════════

def fmt_median_iqr(values):
    """Format as 'median [p25, p75]'."""
    med = np.median(values)
    p25, p75 = np.percentile(values, [25, 75])
    return med, p25, p75


p("\n" + "=" * 75)
p("  BACKTEST v2 RESULTS — Latent-State Strategies")
p("=" * 75)
p("  Model     : %.1fM params (d=%d, step %d)" % (n_params / 1e6, d_model, latest))
p("  Data      : %d pairs, %d test windows" % (n_pairs_ok, n_test))
p("  Source    : %s (pre-tokenized)" % ARRAYRECORD_DIR)
p("  Horizon   : %d min context -> %d min prediction" % (SEQ_LEN, FUTURE_LEN))
p("  Tx cost   : %.2f%% round-trip" % (TX_COST * 100))
p("  Vol acc   : %.1f%%" % (best_vol["bal_acc"] * 100))
p("  K-Means   : K=%d regimes, Sharpes=%s" % (
    K_REGIMES, {c: "%.1f" % s for c, s in regime_sharpe.items()}))
p("  Anomaly   : p%d threshold=%.2f" % (ANOMALY_PERCENTILE, threshold))
p("-" * 75)
p("  %-24s %16s %16s %16s" % ("Strategy", "Sharpe", "CumRet (%)", "MaxDD (%)"))
p("-" * 75)

results = {}
for name in strat_names:
    pairs_data = per_pair_results[name]
    sharpes = np.array([r[0] for r in pairs_data])
    cumrets = np.array([r[1] for r in pairs_data])
    maxdds = np.array([r[2] for r in pairs_data])

    sh_med, sh_25, sh_75 = fmt_median_iqr(sharpes)
    cr_med, cr_25, cr_75 = fmt_median_iqr(cumrets * 100)
    dd_med, dd_25, dd_75 = fmt_median_iqr(maxdds * 100)

    p("  %-24s %5.2f [%5.2f,%5.2f] %5.1f [%5.1f,%5.1f] %5.1f [%5.1f,%5.1f]" % (
        name,
        sh_med, sh_25, sh_75,
        cr_med, cr_25, cr_75,
        dd_med, dd_25, dd_75,
    ))

    results[name] = {
        "sharpe_median": round(float(sh_med), 3),
        "sharpe_p25": round(float(sh_25), 3),
        "sharpe_p75": round(float(sh_75), 3),
        "cumret_median": round(float(np.median(cumrets)), 5),
        "cumret_p25": round(float(np.percentile(cumrets, 25)), 5),
        "cumret_p75": round(float(np.percentile(cumrets, 75)), 5),
        "maxdd_median": round(float(np.median(maxdds)), 5),
        "maxdd_p25": round(float(np.percentile(maxdds, 25)), 5),
        "maxdd_p75": round(float(np.percentile(maxdds, 75)), 5),
        "n_pairs": len(pairs_data),
    }

p("-" * 75)

# Comparisons vs B&H
bh_sharpe = results["Buy & Hold"]["sharpe_median"]
for sname in ["S1: Vol-Target", "S2: Regime K-Means", "S3: Anomaly Filter"]:
    s_sharpe = results[sname]["sharpe_median"]
    s_dd = results[sname]["maxdd_median"]
    bh_dd = results["Buy & Hold"]["maxdd_median"]
    p("  %s vs B&H: Sharpe %.2f vs %.2f (%s), MaxDD %.1f%% vs %.1f%%" % (
        sname.split(":")[0].strip(), s_sharpe, bh_sharpe,
        "BEATS" if s_sharpe > bh_sharpe else "LOSES",
        s_dd * 100, bh_dd * 100,
    ))

elapsed = time.time() - t_global
p("\n  Total time: %.1fs" % elapsed)
p("=" * 75)

# Export JSON
output = {
    "version": "v2_latent_state",
    "model": {
        "params_M": round(n_params / 1e6, 1),
        "d_model": d_model,
        "checkpoint_step": int(latest),
        "config": os.environ.get("SCALE_CONFIG", ""),
    },
    "data": {
        "n_pairs": n_pairs_ok,
        "arrayrecord_dir": ARRAYRECORD_DIR,
        "seq_len": SEQ_LEN,
        "future_len": FUTURE_LEN,
        "n_test_windows": n_test,
    },
    "strategies_config": {
        "vol_classifier_bal_acc": round(best_vol["bal_acc"], 4),
        "vol_classifier_C": best_vol["C"],
        "kmeans_K": K_REGIMES,
        "kmeans_regime_sharpes": {str(c): round(s, 3) for c, s in regime_sharpe.items()},
        "anomaly_percentile": ANOMALY_PERCENTILE,
        "anomaly_threshold": round(float(threshold), 3),
    },
    "tx_cost_roundtrip": TX_COST,
    "aggregation": "per-pair median [IQR]",
    "strategies": results,
    "elapsed_seconds": round(elapsed, 1),
}

os.makedirs("results", exist_ok=True)
out_path = "results/backtest_v2_latent.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
p("  Results saved to %s" % out_path)
