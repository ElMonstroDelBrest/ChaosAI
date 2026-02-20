#!/bin/bash
# Launch script for Financial-IA on TPU v5p-32 (32 chips, 95 GB HBM each).
#
# Scaling ladder: 184M → 500M → 1.5B → 3B — preemptible TPU from TRC.
# Fork of launch_tpu_v6e.sh adapted for v5p multi-host topology.
#
# Usage:
#   ./scripts/launch_tpu_v5p.sh --scale=184m               # 184M baseline
#   ./scripts/launch_tpu_v5p.sh --scale=1.5b --resume       # resume from GCS
#   ./scripts/launch_tpu_v5p.sh --scale=3b --skip-create     # reuse existing VM
#   ./scripts/launch_tpu_v5p.sh --scale=500m --train-only    # skip data conversion
#
# Requirements:
#   - gcloud CLI authenticated with TRC v5p quota
#   - GCS bucket with source data or local data/tokens_v5/
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
TPU_NAME="${TPU_NAME:-fin-ia-v5p}"
ZONE="${ZONE:-europe-west4-a}"
TPU_TYPE="${TPU_TYPE:-v5p-32}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
GCS_BUCKET="${GCS_BUCKET:-gs://fin-ia-bucket}"
VERSION="${VERSION:-tpu-vm-v5p-rev47}"
REPO_DIR="/home/${USER}/Financial_IA"
VENV_DIR="${REPO_DIR}/.venv_tpu"
SKIP_CREATE=false
TRAIN_ONLY=false
RESUME=false
SCALE=""

for arg in "$@"; do
    case $arg in
        --skip-create) SKIP_CREATE=true ;;
        --train-only)  TRAIN_ONLY=true ;;
        --resume)      RESUME=true ;;
        --scale=*)     SCALE="${arg#*=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Validate --scale
if [ -z "$SCALE" ]; then
    echo "[ERROR] --scale is required. Options: s, m, l, xl, 184m, 500m, 1.5b, 3b"
    exit 1
fi

# Resolve config path — T-Shirt sizes (MXU-aligned) + legacy scaling ladder
case "$SCALE" in
    s|S|15m)     SCALE_CONFIG="configs/scaling/s_15m.yaml" ;;
    m|M|150m)    SCALE_CONFIG="configs/scaling/m_150m.yaml" ;;
    l|L|1b)      SCALE_CONFIG="configs/scaling/l_1b.yaml" ;;
    xl|XL|7b)    SCALE_CONFIG="configs/scaling/xl_7b.yaml" ;;
    184m)        SCALE_CONFIG="configs/scaling/184m.yaml" ;;
    500m)        SCALE_CONFIG="configs/scaling/500m.yaml" ;;
    1.5b)        SCALE_CONFIG="configs/scaling/1_5b.yaml" ;;
    3b)          SCALE_CONFIG="configs/scaling/3b.yaml" ;;
    *) echo "[ERROR] Invalid scale: $SCALE. Options: s, m, l, xl, 184m, 500m, 1.5b, 3b"; exit 1 ;;
esac

# Verify config exists locally
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -f "${LOCAL_REPO}/${SCALE_CONFIG}" ]; then
    echo "[ERROR] Config not found: ${LOCAL_REPO}/${SCALE_CONFIG}"
    exit 1
fi

echo "============================================================"
echo " Financial-IA — TPU v5p Scaling Ladder"
echo "============================================================"
echo "  Project:  ${PROJECT}"
echo "  Zone:     ${ZONE}"
echo "  TPU:      ${TPU_NAME} (${TPU_TYPE})"
echo "  Scale:    ${SCALE} (${SCALE_CONFIG})"
echo "  Resume:   ${RESUME}"
echo "  Bucket:   ${GCS_BUCKET}"
echo "  Date:     $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# ── Phase 0: Create TPU VM ─────────────────────────────────────────────────
if [ "$SKIP_CREATE" = false ]; then
    echo ""
    echo "=== Phase 0: Creating TPU VM ==="

    # Delete if exists (preemptible may have been reclaimed)
    if gcloud compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
        echo "  Existing TPU found. Deleting..."
        gcloud compute tpus tpu-vm delete "$TPU_NAME" \
            --zone="$ZONE" --project="$PROJECT" --quiet
    fi

    echo "  Creating ${TPU_TYPE} (preemptible)..."
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --accelerator-type="$TPU_TYPE" \
        --version="$VERSION" \
        --preemptible

    echo "  TPU VM created."
fi

# ── Phase 1: Upload code + setup environment on all workers ────────────────
echo ""
echo "=== Phase 1: Uploading code to TPU VM (all workers) ==="

echo "  Syncing ${LOCAL_REPO} -> TPU VM..."
gcloud compute tpus tpu-vm scp --recurse \
    "${LOCAL_REPO}/src" "${LOCAL_REPO}/scripts" "${LOCAL_REPO}/configs" \
    "${LOCAL_REPO}/pyproject.toml" \
    "${TPU_NAME}:~/Financial_IA/" \
    --zone="$ZONE" --project="$PROJECT" --worker=all

echo ""
echo "=== Phase 1b: Setting up environment (all workers) ==="

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" --worker=all \
    --command="$(cat <<'REMOTE_SETUP'
set -euo pipefail
echo "--- [TPU worker $(hostname)] Environment setup ---"

REPO_DIR="$HOME/Financial_IA"
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

# Ensure python3-venv + pip are installed
echo "  Installing python3-venv (if needed)..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip

# Create venv (recreate if broken)
VENV="$REPO_DIR/.venv_tpu"
if [ ! -f "$VENV/bin/activate" ]; then
    rm -rf "$VENV"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install JAX for TPU v5p
echo "  Installing JAX + TPU v5p support..."
pip install -U pip setuptools wheel
pip install -U \
    "jax[tpu]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install project deps
echo "  Installing project dependencies..."
pip install -U \
    flax \
    optax \
    orbax-checkpoint \
    diffrax \
    array-record \
    grain-nightly \
    tensorflow \
    dacite \
    pyyaml \
    tqdm

# Verify TPU access
echo "  Verifying TPU devices..."
python3 -c "
import jax
devices = jax.devices()
print(f'  Platform: {devices[0].platform}')
print(f'  Device count: {len(devices)}')
for d in devices:
    print(f'    {d}')
assert len(devices) >= 32, f'Expected >=32 TPU chips (v5p-32), got {len(devices)}'
assert devices[0].platform == 'tpu', f'Expected TPU, got {devices[0].platform}'
print('  TPU verification: OK')
"

echo "--- [TPU worker $(hostname)] Environment ready ---"
REMOTE_SETUP
)"

# ── Phase 2: Data conversion (ArrayRecord) ─────────────────────────────────
if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "=== Phase 2: Data conversion ==="

    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" --worker=all \
        --command="$(cat <<'REMOTE_DATA'
set -euo pipefail
REPO_DIR="$HOME/Financial_IA"
cd "$REPO_DIR"
source .venv_tpu/bin/activate

# Try to get ArrayRecord from GCS first (fast path)
if gsutil ls gs://fin-ia-bucket/data/arrayrecord/manifest.json &>/dev/null; then
    echo "  ArrayRecord found on GCS — syncing..."
    mkdir -p data/arrayrecord
    gsutil -m rsync -r gs://fin-ia-bucket/data/arrayrecord/ data/arrayrecord/
elif [ -f "data/arrayrecord/manifest.json" ]; then
    echo "  ArrayRecord manifest already exists locally — skipping."
else
    # Need to convert from .pt files
    if [ ! -d "data/tokens_v5" ] || [ -z "$(ls data/tokens_v5/*.pt 2>/dev/null)" ]; then
        if gsutil ls gs://fin-ia-bucket/data/tokens_v5/ &>/dev/null; then
            echo "  Syncing tokens from GCS..."
            mkdir -p data/tokens_v5
            gsutil -m rsync -r gs://fin-ia-bucket/data/tokens_v5/ data/tokens_v5/
        else
            echo "  [ERROR] No tokens found locally or on GCS."
            exit 1
        fi
    fi

    echo "  Converting .pt -> ArrayRecord..."
    python3 scripts/convert_pt_to_arrayrecord.py \
        --input data/tokens_v5/ \
        --output data/arrayrecord/ \
        --seq_len 128

    # Upload ArrayRecord to GCS for persistence across preemptions
    echo "  Syncing ArrayRecord to GCS..."
    gsutil -m rsync -r data/arrayrecord/ gs://fin-ia-bucket/data/arrayrecord/
fi

echo "--- [TPU worker $(hostname)] Data ready ---"
REMOTE_DATA
)"
fi

# ── Phase 3: Launch training ───────────────────────────────────────────────
echo ""
echo "=== Phase 3: Launching training (scale=${SCALE}) ==="

# Pass config and resume flag to remote
RESUME_FLAG="${RESUME}"
SCALE_TIER="${SCALE}"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" --worker=all \
    --command="$(cat <<REMOTE_TRAIN
set -euo pipefail
REPO_DIR="\$HOME/Financial_IA"
cd "\$REPO_DIR"
source .venv_tpu/bin/activate

# Export config path and scale tier for the training script
export SCALE_CONFIG="${SCALE_CONFIG}"
export SCALE_TIER="${SCALE_TIER}"
export RESUME="${RESUME_FLAG}"
export GCS_BUCKET="${GCS_BUCKET}"

# XLA compilation cache — persist across preemptions
export JAX_COMPILATION_CACHE_DIR="\$HOME/.jax_cache"
mkdir -p "\$JAX_COMPILATION_CACHE_DIR"

# Log topology for debugging
echo "--- [TPU worker \$(hostname)] Device topology ---"
python3 -c "
import jax
devices = jax.devices()
n = len(devices)
platform = devices[0].platform
print(f'  {n} {platform.upper()} chips detected')
print(f'  Process count: {jax.process_count()}')
print(f'  Process index: {jax.process_index()}')
"

# Resume from GCS checkpoint if available
CKPT_DIR="checkpoints/jax_v5p/\${SCALE_TIER}"
mkdir -p "\$CKPT_DIR"
if [ "\$RESUME" = "true" ]; then
    if gsutil ls "\${GCS_BUCKET}/checkpoints/jax_v5p/\${SCALE_TIER}/" &>/dev/null; then
        echo "  Restoring checkpoint from GCS..."
        gsutil -m rsync -r "\${GCS_BUCKET}/checkpoints/jax_v5p/\${SCALE_TIER}/" "\$CKPT_DIR/"
    else
        echo "  No checkpoint found on GCS — starting from scratch."
    fi
fi

# Sync XLA cache from GCS if available
if gsutil ls "\${GCS_BUCKET}/xla_cache/\${SCALE_TIER}/" &>/dev/null 2>&1; then
    echo "  Restoring XLA cache from GCS..."
    gsutil -m rsync -r "\${GCS_BUCKET}/xla_cache/\${SCALE_TIER}/" "\$JAX_COMPILATION_CACHE_DIR/" 2>/dev/null || true
fi

export JAX_TRACEBACK_FILTERING=off

# ── Production XLA flags for TPU v5p ──
# These tell the XLA compiler to be ultra-aggressive on kernel fusion,
# collective overlap, and memory management on the ICI torus.
export LIBTPU_INIT_ARGS=" \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_enable_latency_hiding_scheduler=true:2 \
  --xla_tpu_perform_spmd_cse_prevention=false \
  --xla_tpu_spmd_rng_bit_generator_unsafe=true \
  --xla_tpu_megacore_fusion_allow_ags=false \
  --xla_tpu_enable_aggressive_loop_fusion=true \
  --xla_tpu_enable_experimental_fusion_cost_model=true \
"

# Silence TensorFlow spam (Grain uses TF internally)
export TF_CPP_MIN_LOG_LEVEL=2

echo "--- [TPU worker \$(hostname)] Starting Fin-JEPA training (scale=\${SCALE_TIER}) ---"
echo "  LIBTPU_INIT_ARGS set (async collectives + LHS + aggressive fusion)"
nohup python3 -u -c "
import sys, os, logging, time, json
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger('train')

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from src.jax_v6.config import load_config
from src.jax_v6.training.sharding import create_mesh, shard_batch, shard_params
from src.jax_v6.training.train_state import create_train_state, create_checkpoint_manager
from src.jax_v6.training.train_step import train_step, eval_step
from src.jax_v6.training.preemption import PreemptionWatcher
from src.jax_v6.training.metrics import compute_mfu, compute_tokens_per_second
from src.jax_v6.jepa import FinJEPA
from src.jax_v6.data.grain_loader import create_dataloader

# ── Config ──
config = load_config(os.environ['SCALE_CONFIG'])
scale = os.environ['SCALE_TIER']
gcs_bucket = os.environ['GCS_BUCKET']
resume = os.environ.get('RESUME', 'false') == 'true'

log.info('Scale tier: %s', scale)
log.info('Config: d_model=%d, n_layers=%d, n_heads=%d, batch=%d, lr=%.1e, remat=%s',
    config.mamba2.d_model, config.mamba2.n_layers, config.mamba2.n_heads,
    config.training.batch_size, config.training.lr, config.mamba2.use_remat)

# ── Mesh ──
mesh = create_mesh()
n_devices = len(jax.devices())
log.info('Training on %d %s chips (process %d/%d)',
    n_devices, jax.devices()[0].platform.upper(),
    jax.process_index(), jax.process_count())
assert n_devices >= 8, f'Expected >=8 TPU chips, got {n_devices}'

# ── Model ──
model = FinJEPA.from_config(config)

# Estimate param count
B = config.training.batch_size // n_devices
S = config.embedding.seq_len
max_tgt = int(S * 0.5) + 8
dummy_batch = {
    'token_indices': jnp.zeros((B, S), dtype=jnp.int64),
    'weekend_mask': jnp.zeros((B, S), dtype=jnp.float32),
    'exo_clock': jnp.zeros((B, S, 2), dtype=jnp.float32),
    'block_mask': jnp.zeros((B, S), dtype=jnp.bool_),
    'target_positions': jnp.zeros((B, max_tgt), dtype=jnp.int64),
    'target_mask': jnp.ones((B, max_tgt), dtype=jnp.bool_),
}

key = jax.random.PRNGKey(42)
state = create_train_state(
    model, key, dummy_batch,
    lr=config.training.lr,
    weight_decay=config.training.weight_decay,
    tau_start=config.ema.tau_start,
    grad_clip=config.training.grad_clip,
    n_restarts=config.training.n_restarts,
)

n_params = sum(x.size for x in jax.tree.leaves(state.params))
log.info('Model params: %s (%.1fM)', f'{n_params:,}', n_params / 1e6)

state = shard_params(state, mesh)
log.info('TrainState sharded across %d chips', n_devices)

# ── Checkpoint manager (Orbax) ──
ckpt_dir = f'checkpoints/jax_v5p/{scale}'
ckpt_mgr = create_checkpoint_manager(ckpt_dir, max_to_keep=5)

# Restore if resuming
start_step = 0
if resume:
    latest = ckpt_mgr.latest_step()
    if latest is not None:
        log.info('Restoring checkpoint from step %d...', latest)
        state = ckpt_mgr.restore(latest, args=ocp.args.StandardRestore(state))
        start_step = latest
        log.info('Resumed from step %d', start_step)

# ── Data ──
train_loader = create_dataloader(
    config.data.arrayrecord_dir, split='train',
    batch_size=config.training.batch_size,
    seq_len=config.embedding.seq_len,
    worker_count=config.data.num_workers,
    prefetch_buffer_size=config.data.prefetch_buffer_size,
)
val_loader = create_dataloader(
    config.data.arrayrecord_dir, split='val',
    batch_size=config.training.batch_size,
    seq_len=config.embedding.seq_len,
    worker_count=config.data.num_workers,
    prefetch_buffer_size=config.data.prefetch_buffer_size,
)

# ── Preemption watcher ──
watcher = PreemptionWatcher()
watcher.start()

# ── Training loop ──
ckpt_interval = config.training.checkpoint_interval  # 250 steps
mfu_interval = 50
val_interval = 500

log.info('=== Training started (scale=%s, start_step=%d) ===', scale, start_step)
step = start_step
t0 = time.time()

for batch in train_loader:
    # Preemption check
    if watcher.should_stop():
        log.warning('PREEMPTION — saving emergency checkpoint at step %d', step)
        ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
        ckpt_mgr.wait_until_finished()
        import subprocess
        subprocess.run(['gsutil', '-m', 'rsync', '-r', ckpt_dir + '/',
            f'{gcs_bucket}/checkpoints/jax_v5p/{scale}/'], check=False)
        log.warning('Emergency checkpoint saved. Exiting.')
        break

    batch = {k: jnp.array(v) for k, v in batch.items() if not isinstance(v, (str, bytes))}
    batch = shard_batch(batch, mesh)
    state, metrics = train_step(state, batch, model)

    step += 1
    step_time = time.time() - t0

    # MFU + throughput logging
    if step % mfu_interval == 0:
        avg_step_time = step_time / mfu_interval
        mfu = compute_mfu(n_params, config.training.batch_size, S, avg_step_time, n_devices, peak_tflops=459.0)
        tps = compute_tokens_per_second(config.training.batch_size, S, avg_step_time)
        loss = float(metrics['loss'])
        grad_norm = float(metrics['grad_norm'])
        log.info(
            'step %d | loss %.4f | grad_norm %.2f | MFU %.1f%% | %.0f tok/s | %.2f s/step',
            step, loss, grad_norm, mfu * 100, tps, avg_step_time,
        )
        t0 = time.time()

    # Checkpoint
    if step % ckpt_interval == 0:
        log.info('Checkpointing at step %d...', step)
        ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
        ckpt_mgr.wait_until_finished()
        # Async GCS sync
        import subprocess
        subprocess.Popen(['gsutil', '-m', 'rsync', '-r', ckpt_dir + '/',
            f'{gcs_bucket}/checkpoints/jax_v5p/{scale}/'])
        log.info('Checkpoint saved (step %d)', step)

    # Validation
    if step % val_interval == 0:
        log.info('Running validation at step %d...', step)
        val_losses = []
        val_count = 0
        for val_batch in val_loader:
            val_batch = {k: jnp.array(v) for k, v in val_batch.items() if not isinstance(v, (str, bytes))}
            val_batch = shard_batch(val_batch, mesh)
            val_metrics = eval_step(state, val_batch, model)
            val_losses.append(float(val_metrics['loss']))
            val_count += 1
            if val_count >= 20:  # Cap validation to 20 batches
                break
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('nan')
        log.info('step %d | val_loss %.4f (over %d batches)', step, avg_val_loss, val_count)

watcher.stop()

# Final checkpoint
log.info('Saving final checkpoint at step %d...', step)
ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
ckpt_mgr.wait_until_finished()
import subprocess
subprocess.run(['gsutil', '-m', 'rsync', '-r', ckpt_dir + '/',
    f'{gcs_bucket}/checkpoints/jax_v5p/{scale}/'], check=False)

# Sync XLA cache to GCS
cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '')
if cache_dir and os.path.exists(cache_dir):
    subprocess.run(['gsutil', '-m', 'rsync', '-r', cache_dir + '/',
        f'{gcs_bucket}/xla_cache/{scale}/'], check=False)

log.info('=== Training complete (scale=%s, step=%d) ===', scale, step)
" > training_v5p.log 2>&1 &

TRAIN_PID=\$!
echo "  Training launched (PID: \$TRAIN_PID)"
echo "  Monitor: tail -f \$REPO_DIR/training_v5p.log"
echo "--- [TPU worker \$(hostname)] Launch complete ---"
REMOTE_TRAIN
)"

echo ""
echo "============================================================"
echo " Training launched on ${TPU_NAME} (${TPU_TYPE}, scale=${SCALE})"
echo ""
echo " Monitor logs:"
echo "   gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} \\"
echo "     -- tail -f ~/Financial_IA/training_v5p.log"
echo ""
echo " SSH into VM:"
echo "   gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}"
echo ""
echo " Delete when done:"
echo "   gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}"
echo "============================================================"
