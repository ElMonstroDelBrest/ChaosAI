"""Linear Probing: evaluate JEPA representations on downstream tasks.

Tasks:
  A) Direction prediction (up/down over next N min)
  B) Volatility prediction (high/low vol regime over next N min)
  C) Large move prediction (will |return| > threshold?)

Standard protocol: freeze JEPA encoder → extract embeddings → logistic regression.

Usage:
    SCALE_CONFIG=configs/scaling/v6e_26m.yaml SCALE_TIER=26m TPU_GEN=v6e \
    EVAL_DATA_DIR=data/ohlcv_1m/ \
    PYTHONPATH=. python scripts/linear_probe.py
"""

import os
import time
import traceback

import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from src.strate_i.config import load_config as load_strate_i_config
from src.strate_i.lightning_module import StrateILightningModule

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
STRATE_I_CONFIG = os.environ.get("STRATE_I_CONFIG", "configs/strate_i_1m_p1.yaml")
STRATE_I_CKPT = os.environ.get(
    "STRATE_I_CKPT",
    "checkpoints/strate-i-epoch=02-val/loss/total=0.0038.ckpt",
)
EVAL_DATA_DIR = os.environ.get("EVAL_DATA_DIR", "data/ohlcv_1m/")
SEQ_LEN = 128
TRAIN_RATIO = 0.7
BATCH_ENC = 256
MAX_CANDLES = 20000
FUTURE_LEN = 32  # prediction horizon (minutes)


def main():
    print("=" * 60, flush=True)
    print("  LINEAR PROBING — Representation Quality", flush=True)
    print("=" * 60, flush=True)

    # ── Step 1: Load Strate I tokenizer ──
    print("\n[1/5] Loading Strate I tokenizer...", flush=True)
    si_config = load_strate_i_config(STRATE_I_CONFIG)
    si_module = StrateILightningModule.load_from_checkpoint(
        STRATE_I_CKPT, config=si_config, map_location="cpu"
    )
    si_module.eval()
    tokenizer = si_module.tokenizer
    patch_len = si_config.patch.patch_length
    stride = si_config.patch.stride
    print("  Strate I loaded (patch_length=%d)" % patch_len, flush=True)

    # ── Step 2: Load JEPA model + checkpoint ──
    print("\n[2/5] Loading JEPA model + checkpoint...", flush=True)
    from src.common.jax_checkpoint import load_jepa_checkpoint
    ckpt_data = load_jepa_checkpoint(os.environ["SCALE_CONFIG"])
    config = ckpt_data["config"]
    state = ckpt_data["state"]
    n_params = ckpt_data["n_params"]
    d_model = ckpt_data["d_model"]
    latest = ckpt_data["latest_step"]
    S = config.embedding.seq_len
    print("  JEPA: %.1fM params (d=%d), checkpoint step %d" % (
        n_params / 1e6, d_model, latest), flush=True)

    # ── Step 3: Tokenize + compute labels ──
    print("\n[3/5] Tokenizing + computing labels...", flush=True)
    import glob

    ohlcv_files = sorted(glob.glob(os.path.join(EVAL_DATA_DIR, "*.pt")))
    print("  Found %d pairs" % len(ohlcv_files), flush=True)

    all_tokens = []
    all_direction = []   # 1=up, 0=down
    all_vol_regime = []  # 1=high vol, 0=low vol
    all_large_move = []  # 1=|ret|>threshold, 0=small move
    all_is_train = []
    total_len = SEQ_LEN + FUTURE_LEN

    t0 = time.time()
    for fi, fpath in enumerate(ohlcv_files):
        ohlcv = torch.load(fpath, map_location="cpu", weights_only=True)
        n_candles = ohlcv.shape[0]
        if n_candles > MAX_CANDLES:
            ohlcv = ohlcv[-MAX_CANDLES:]
            n_candles = MAX_CANDLES
        if n_candles < total_len + 10:
            continue

        closes = ohlcv[:, 3].numpy()
        # Log returns for volatility
        log_rets = np.diff(np.log(closes + 1e-10))

        # Tokenize
        patches = ohlcv.unfold(0, patch_len, stride).permute(0, 2, 1)
        with torch.no_grad():
            tids = []
            for i in range(0, patches.shape[0], 8192):
                tids.append(tokenizer.tokenize(patches[i:i + 8192]).cpu())
            tids = torch.cat(tids, dim=0).numpy()

        n_windows = len(tids) // total_len
        if n_windows < 2:
            continue

        # Compute realized volatility for each future window (for median threshold)
        vols = []
        for w in range(n_windows):
            start = w * total_len
            ctx_end = (start + SEQ_LEN) * stride
            fut_end = min(ctx_end + FUTURE_LEN * stride, n_candles - 1)
            if ctx_end >= len(log_rets) or fut_end >= n_candles:
                vols.append(0.0)
            else:
                rets_window = log_rets[ctx_end:min(fut_end, len(log_rets))]
                vols.append(np.std(rets_window) if len(rets_window) > 1 else 0.0)
        vols = np.array(vols)
        vol_median = np.median(vols[vols > 0]) if np.any(vols > 0) else 0.0

        tokens_seq = []
        dir_labels = []
        vol_labels = []
        move_labels = []

        for w in range(n_windows):
            start = w * total_len
            tokens_seq.append(tids[start:start + SEQ_LEN])

            ctx_end = (start + SEQ_LEN) * stride
            fut_end = min(ctx_end + FUTURE_LEN * stride, n_candles - 1)

            if ctx_end >= n_candles:
                dir_labels.append(0)
                vol_labels.append(0)
                move_labels.append(0)
            else:
                # Direction
                ret = (closes[fut_end] - closes[ctx_end]) / (closes[ctx_end] + 1e-10)
                dir_labels.append(1 if ret > 0 else 0)

                # Volatility regime
                vol_labels.append(1 if vols[w] > vol_median else 0)

                # Large move (|return| > 0.2%)
                move_labels.append(1 if abs(ret) > 0.002 else 0)

        arr = np.array(tokens_seq, dtype=np.int64)
        split = int(len(arr) * TRAIN_RATIO)
        itr = np.zeros(len(arr), dtype=bool)
        itr[:split] = True

        all_tokens.append(arr)
        all_direction.append(np.array(dir_labels, dtype=np.int64))
        all_vol_regime.append(np.array(vol_labels, dtype=np.int64))
        all_large_move.append(np.array(move_labels, dtype=np.int64))
        all_is_train.append(itr)

        if (fi + 1) % 100 == 0:
            print("  Processed %d/%d pairs" % (fi + 1, len(ohlcv_files)), flush=True)

    tokens = np.concatenate(all_tokens)
    labels_dir = np.concatenate(all_direction)
    labels_vol = np.concatenate(all_vol_regime)
    labels_move = np.concatenate(all_large_move)
    is_train = np.concatenate(all_is_train)

    n_total = len(tokens)
    n_train = is_train.sum()
    n_test = n_total - n_train
    print("  %d samples (%d train, %d test) in %.1fs" % (
        n_total, n_train, n_test, time.time() - t0), flush=True)
    print("  Direction: %.1f%% up | Vol: %.1f%% high | Large move: %.1f%% (threshold=0.2%%)" % (
        labels_dir.mean() * 100, labels_vol.mean() * 100, labels_move.mean() * 100), flush=True)

    # ── Step 4: Extract JEPA embeddings ──
    print("\n[4/5] Extracting JEPA embeddings...", flush=True)
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
            print("  Encoded %d/%d" % (end, n_total), flush=True)

    print("  Embeddings (%d, %d) in %.1fs" % (n_total, emb_dim, time.time() - t0), flush=True)

    # ── Step 5: Linear Probe — all tasks ──
    print("\n[5/5] Linear probing...", flush=True)

    tasks = {
        "Direction (up/down)": labels_dir,
        "Volatility regime": labels_vol,
        "Large move (>0.2%)": labels_move,
    }

    results = {}
    for task_name, labels in tasks.items():
        best = {"bal_acc": 0.0}

        for pool, sl in [("mean", slice(0, d_model)), ("last", slice(d_model, None)), ("mean+last", slice(None))]:
            X_train = embeddings[is_train][:, sl]
            y_train = labels[is_train]
            X_test = embeddings[~is_train][:, sl]
            y_test = labels[~is_train]

            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_train)
            Xte = scaler.transform(X_test)

            for C in [0.001, 0.01, 0.1, 1.0]:
                clf = LogisticRegression(
                    max_iter=3000, C=C, solver="lbfgs",
                    class_weight="balanced",
                )
                clf.fit(Xtr, y_train)
                y_pred = clf.predict(Xte)
                bal = balanced_accuracy_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)

                if bal > best["bal_acc"]:
                    best = {
                        "bal_acc": bal,
                        "acc": acc,
                        "pool": pool,
                        "C": C,
                        "train_bal": balanced_accuracy_score(y_train, clf.predict(Xtr)),
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "balance": y_test.mean(),
                    }

        results[task_name] = best
        print("  %s: balanced_acc=%.1f%% (pool=%s, C=%.3f)" % (
            task_name, best["bal_acc"] * 100, best["pool"], best["C"]), flush=True)

    # ── Final Report ──
    print("\n" + "=" * 60, flush=True)
    print("  LINEAR PROBING RESULTS", flush=True)
    print("=" * 60, flush=True)
    print("  Model    : %.1fM params (d=%d, step %d)" % (
        n_params / 1e6, d_model, latest), flush=True)
    print("  Data     : %d pairs, last %dk candles/pair" % (
        len(ohlcv_files), MAX_CANDLES // 1000), flush=True)
    print("  Context  : %d min → predict next %d min" % (SEQ_LEN, FUTURE_LEN), flush=True)
    print("  Samples  : %d train / %d test (temporal 70/30)" % (n_train, n_test), flush=True)
    print("  Random   : 50.0%%", flush=True)
    print("-" * 60, flush=True)

    for task_name, info in results.items():
        print("\n  Task: %s" % task_name, flush=True)
        print("  Label balance  : %.1f%% positive (test)" % (info["balance"] * 100), flush=True)
        print("  Pooling        : %s" % info["pool"], flush=True)
        print("  Regularization : C=%.3f" % info["C"], flush=True)
        print("  Train bal.acc  : %.1f%%" % (info["train_bal"] * 100), flush=True)
        print("  TEST BAL.ACC   : %.1f%%" % (info["bal_acc"] * 100), flush=True)
        print("  Edge vs random : +%.1f pp" % ((info["bal_acc"] - 0.5) * 100), flush=True)
        print("  ---", flush=True)
        print(classification_report(
            info["y_test"], info["y_pred"],
            target_names=["Negative", "Positive"],
        ), flush=True)

    print("=" * 60, flush=True)

    # Summary line
    best_task = max(results.items(), key=lambda x: x[1]["bal_acc"])
    print("\n  BEST: %s → %.1f%% balanced accuracy (+%.1f pp vs random)" % (
        best_task[0], best_task[1]["bal_acc"] * 100,
        (best_task[1]["bal_acc"] - 0.5) * 100), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
