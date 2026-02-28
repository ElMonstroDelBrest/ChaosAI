"""Out-of-sample evaluation of Fin-JEPA on fresh (unseen) data.

Loads a trained checkpoint, runs a single pass over eval ArrayRecords,
reports loss breakdown + comparison with training loss.

Usage:
    SCALE_CONFIG=configs/scaling/v6e_26m.yaml \
    SCALE_TIER=26m \
    TPU_GEN=v6e \
    EVAL_DATA_DIR=data/arrayrecord_eval/ \
    PYTHONPATH=. python scripts/eval_oos.py
"""

import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.jax_v6.data.grain_loader import create_dataloader
from src.jax_v6.training.sharding import shard_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("eval_oos")

# ── Config ──
from src.common.jax_checkpoint import load_jepa_checkpoint

scale = os.environ.get("SCALE_TIER", "26m")
tpu_gen = os.environ.get("TPU_GEN", "v6e")
eval_data_dir = os.environ.get("EVAL_DATA_DIR", "data/arrayrecord_eval/")
ckpt_dir = os.environ.get(
    "CKPT_DIR",
    os.path.join(os.getcwd(), f"checkpoints/jax_{tpu_gen}/{scale}"),
)
train_final_loss = float(os.environ.get("TRAIN_LOSS", "1644"))

n_devices = len(jax.devices())
device_kind = jax.devices()[0].device_kind
log.info("=== OOS Eval | %d chips (%s) | scale=%s ===", n_devices, device_kind, scale)
log.info("Eval data: %s", eval_data_dir)
log.info("Checkpoint: %s", ckpt_dir)

# ── Load JEPA checkpoint ──
ckpt_data = load_jepa_checkpoint(os.environ["SCALE_CONFIG"], tpu_gen=tpu_gen, scale_tier=scale)
config = ckpt_data["config"]
state = ckpt_data["state"]
mesh = ckpt_data["mesh"]
n_params = ckpt_data["n_params"]
latest = ckpt_data["latest_step"]
log.info("Model: %s (%.1fM params)", f"{n_params:,}", n_params / 1e6)
log.info("Checkpoint restored (step %d)", latest)

# ── Eval DataLoader ──
# Use split="val" + val_ratio=1.0 → all pairs go to val → single epoch
# Small batch for eval (eval data may be much smaller than training data)
eval_batch = min(config.training.batch_size, 64)
eval_loader = create_dataloader(
    eval_data_dir,
    split="val",
    batch_size=eval_batch,
    seq_len=config.embedding.seq_len,
    mask_ratio=0.5,
    val_ratio=1.0,  # ALL pairs → val (single pass)
    worker_count=0,
    prefetch_buffer_size=64,
)
log.info("Eval batch size: %d", eval_batch)

# ── Eval function (no grad, deterministic) ──
@jax.jit
def eval_fn(params, target_params, batch, rng):
    outputs = model.apply(
        {"params": params},
        batch,
        target_params=target_params,
        key=rng,
        deterministic=True,
    )
    return {
        "loss": outputs["loss"],
        "invariance": outputs["invariance"],
        "variance": outputs["variance"],
        "covariance": outputs["covariance"],
        "cfm_loss": outputs["cfm_loss"],
    }


# ── Eval Loop ──
log.info("=== Running eval on %s ===", eval_data_dir)
metrics_acc = {"loss": [], "invariance": [], "variance": [], "covariance": [], "cfm_loss": []}
rng = jax.random.PRNGKey(0)
t0 = time.time()

for i, batch in enumerate(eval_loader):
    batch = {k: jnp.array(v) for k, v in batch.items() if not isinstance(v, (str, bytes))}
    batch = shard_batch(batch, mesh)

    rng, step_rng = jax.random.split(rng)
    metrics = eval_fn(state.params, state.target_params, batch, step_rng)

    for k in metrics_acc:
        metrics_acc[k].append(float(metrics[k]))

    if (i + 1) % 5 == 0:
        log.info(
            "batch %d | loss %.1f | inv %.1f | var %.1f | cov %.1f | cfm %.1f",
            i + 1,
            metrics_acc["loss"][-1],
            metrics_acc["invariance"][-1],
            metrics_acc["variance"][-1],
            metrics_acc["covariance"][-1],
            metrics_acc["cfm_loss"][-1],
        )

elapsed = time.time() - t0
n_batches = len(metrics_acc["loss"])

if n_batches == 0:
    log.error("No batches evaluated! Check eval data.")
    exit(1)

# ── Report ──
log.info("=" * 70)
log.info("  OUT-OF-SAMPLE EVALUATION RESULTS")
log.info("=" * 70)
log.info("  Data        : %s", eval_data_dir)
log.info("  Checkpoint  : step %d from %s", latest, ckpt_dir)
log.info("  Model       : %.1fM params, d=%d, L=%d",
         n_params / 1e6, config.mamba2.d_model, config.mamba2.n_layers)
log.info("  Batches     : %d | Batch size: %d | Time: %.1fs",
         n_batches, config.training.batch_size, elapsed)
log.info("-" * 70)
log.info("  %-14s %10s %10s %10s %10s", "Metric", "Mean", "Std", "Min", "Max")
log.info("-" * 70)
for k in ["loss", "invariance", "variance", "covariance", "cfm_loss"]:
    vals = np.array(metrics_acc[k])
    log.info("  %-14s %10.1f %10.1f %10.1f %10.1f",
             k, vals.mean(), vals.std(), vals.min(), vals.max())
log.info("-" * 70)
oos_loss = np.mean(metrics_acc["loss"])
ratio = oos_loss / train_final_loss if train_final_loss > 0 else float("inf")
log.info("  Training loss (final) : %.1f", train_final_loss)
log.info("  OOS loss (mean)       : %.1f", oos_loss)
log.info("  Ratio OOS/Train       : %.2fx", ratio)
if ratio < 1.2:
    log.info("  Verdict: GOOD — model generalizes well (OOS within 20%% of train)")
elif ratio < 1.5:
    log.info("  Verdict: OK — mild overfitting (OOS 20-50%% above train)")
else:
    log.info("  Verdict: OVERFIT — significant gap (OOS >50%% above train)")
log.info("=" * 70)
