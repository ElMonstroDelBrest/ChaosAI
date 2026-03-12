#!/bin/bash
# Launch script for Financial-IA on TPU v5e (v5 "efficient").
#
# Supports pod sizes: v5e-8, v5e-16, v5e-32, v5e-64.
# Auto-Sharder detects chip count and routes to optimal mesh:
#   v5e-8  -> (8,1) DP pur       | 1 host  | 128 GB HBM total
#   v5e-16 -> (8,2) leger FSDP   | 2 hosts | 256 GB HBM total
#   v5e-32 -> (8,4) FSDP modere  | 4 hosts | 512 GB HBM total
#   v5e-64 -> (8,8) tore 2D 1:1  | 8 hosts | 1024 GB HBM total
#
# v5e specs:
#   - HBM/chip     : 16 GB (SERRE — batch reduits vs v6e)
#   - Peak BF16    : ~197 TFLOPS/chip
#   - Interconnect : Tore 2D (similaire v6e)
#   - Version flag : v2-alpha-tpuv5-lite
#
# Usage:
#   ./scripts/launch_tpu_v5e.sh --scale=s                          # v5e-64 (defaut)
#   ./scripts/launch_tpu_v5e.sh --scale=s --pod=8                  # v5e-8 smoke test
#   ./scripts/launch_tpu_v5e.sh --scale=m --pod=32                 # v5e-32 medium run
#   ./scripts/launch_tpu_v5e.sh --scale=l --pod=64 --resume        # v5e-64 resume GCS
#   ./scripts/launch_tpu_v5e.sh --scale=s --pod=8 --skip-create    # reuse existing VM
#   ./scripts/launch_tpu_v5e.sh --scale=s --pod=8 --train-only     # skip data sync
#
# Env overrides:
#   TPU_NAME=my-tpu  ./scripts/launch_tpu_v5e.sh --scale=s --pod=8
#   ZONE=us-central1-a ./scripts/launch_tpu_v5e.sh --scale=s --pod=64
#
# Requirements:
#   - gcloud CLI authenticated with TRC v5e quota
#   - GCS bucket en europe-west4 (co-localise -> transfert gratuit)
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
ZONE="${ZONE:-europe-west4-b}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
GCS_BUCKET="${GCS_BUCKET:-gs://fin-ia-eu}"
VERSION="${VERSION:-v2-alpha-tpuv5-lite}"
SKIP_CREATE=false
TRAIN_ONLY=false
RESUME=false
SCALE=""
POD_SIZE=""

for arg in "$@"; do
    case $arg in
        --skip-create) SKIP_CREATE=true ;;
        --train-only)  TRAIN_ONLY=true ;;
        --resume)      RESUME=true ;;
        --scale=*)     SCALE="${arg#*=}" ;;
        --pod=*)       POD_SIZE="${arg#*=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── Validate inputs ───────────────────────────────────────────────────────
if [ -z "$SCALE" ]; then
    echo "[ERROR] --scale is required. Options: s, m, l"
    exit 1
fi

# Pod size: default 64, validate allowed values
if [ -z "$POD_SIZE" ]; then
    POD_SIZE=64
fi
case "$POD_SIZE" in
    8|16|32|64) ;; # OK
    *) echo "[ERROR] Invalid --pod=$POD_SIZE. Options: 8, 16, 32, 64"; exit 1 ;;
esac

# Derive TPU_TYPE and TPU_NAME from pod size
TPU_TYPE="${TPU_TYPE:-v5e-${POD_SIZE}}"
TPU_NAME="${TPU_NAME:-fin-ia-v5e}"

# ── Resolve config ────────────────────────────────────────────────────────
case "$SCALE" in
    s|S)    SCALE_CONFIG="configs/scaling/v5e_s.yaml" ;;
    m|M)    SCALE_CONFIG="configs/scaling/v5e_m.yaml" ;;
    l|L)    SCALE_CONFIG="configs/scaling/v5e_l.yaml" ;;
    *) echo "[ERROR] Invalid scale: $SCALE. Options: s, m, l"; exit 1 ;;
esac

LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -f "${LOCAL_REPO}/${SCALE_CONFIG}" ]; then
    echo "[ERROR] Config not found: ${LOCAL_REPO}/${SCALE_CONFIG}"
    exit 1
fi

# ── Pod topology info ─────────────────────────────────────────────────────
N_CHIPS="$POD_SIZE"
HBM_TOTAL=$((N_CHIPS * 16))

# Mesh shape (mirrors sharding.py _V6E_MESH_SHAPES — v5e uses same table)
case "$N_CHIPS" in
    8)  MESH="(8,1)  DP pur"         ; N_HOSTS=1 ;;
    16) MESH="(8,2)  leger FSDP"     ; N_HOSTS=2 ;;
    32) MESH="(8,4)  FSDP modere"    ; N_HOSTS=4 ;;
    64) MESH="(8,8)  tore 2D 1:1"    ; N_HOSTS=8 ;;
esac

# Worker flag: single-host (v5e-8) vs multi-host
if [ "$N_HOSTS" -eq 1 ]; then
    WORKER_FLAG="--worker=0"
else
    WORKER_FLAG="--worker=all"
fi

echo "============================================================"
echo " Financial-IA — TPU ${TPU_TYPE} (${N_HOSTS} host(s))"
echo "============================================================"
echo "  Project:  ${PROJECT}"
echo "  Zone:     ${ZONE}"
echo "  TPU:      ${TPU_NAME} (${TPU_TYPE})"
echo "  Scale:    ${SCALE} (${SCALE_CONFIG})"
echo "  Pod:      ${N_CHIPS} chips x 16 GB = ${HBM_TOTAL} GB HBM"
echo "  Mesh:     ${MESH}"
echo "  Resume:   ${RESUME}"
echo "  Bucket:   ${GCS_BUCKET}"
echo "  Date:     $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# ── Phase 0: Create TPU VM ─────────────────────────────────────────────────
if [ "$SKIP_CREATE" = false ]; then
    echo ""
    echo "=== Phase 0: Creating TPU VM (${TPU_TYPE}) ==="

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

    echo "  TPU VM created (${N_CHIPS} chips, ${N_HOSTS} host(s))."
fi

# ── Phase 1: Upload code + setup environment ──────────────────────────────
echo ""
echo "=== Phase 1: Uploading code to TPU VM (${N_HOSTS} host(s)) ==="

echo "  Syncing ${LOCAL_REPO} -> TPU VM..."
gcloud compute tpus tpu-vm scp --recurse \
    "${LOCAL_REPO}/src" "${LOCAL_REPO}/scripts" "${LOCAL_REPO}/configs" \
    "${LOCAL_REPO}/pyproject.toml" \
    "${TPU_NAME}:~/Financial_IA/" \
    --zone="$ZONE" --project="$PROJECT" ${WORKER_FLAG}

echo ""
echo "=== Phase 1b: Setting up environment (${N_HOSTS} host(s)) ==="

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" ${WORKER_FLAG} \
    --command="$(cat <<'REMOTE_SETUP'
set -euo pipefail
echo "--- [TPU worker $(hostname)] Environment setup ---"

REPO_DIR="$HOME/Financial_IA"
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

echo "  Installing python3-venv (if needed)..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip

VENV="$REPO_DIR/.venv_tpu"
if [ ! -f "$VENV/bin/activate" ]; then
    rm -rf "$VENV"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo "  Installing JAX + TPU support..."
pip install -U pip setuptools wheel
pip install -U \
    "jax[tpu]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

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

echo "  Verifying TPU devices..."
python3 -c "
import jax
devices = jax.devices()
n = len(devices)
kind = devices[0].device_kind
print(f'  Platform: {devices[0].platform}')
print(f'  Device count: {n}')
print(f'  Device kind: {kind}')
for d in devices[:4]:
    print(f'    {d}')
if n > 4:
    print(f'    ... ({n - 4} more)')
assert n >= 4, f'Expected >=4 TPU chips, got {n}'
assert devices[0].platform == 'tpu', f'Expected TPU, got {devices[0].platform}'
print(f'  TPU verification: OK ({n} chips)')
"

echo "--- [TPU worker $(hostname)] Environment ready ---"
REMOTE_SETUP
)"

# ── Phase 2: Data conversion (ArrayRecord) ─────────────────────────────────
if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "=== Phase 2: Data conversion ==="

    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" ${WORKER_FLAG} \
        --command="$(cat <<'REMOTE_DATA'
set -euo pipefail
REPO_DIR="$HOME/Financial_IA"
cd "$REPO_DIR"
source .venv_tpu/bin/activate

if gsutil ls gs://fin-ia-eu/data/arrayrecord/manifest.json &>/dev/null; then
    echo "  ArrayRecord found on GCS — syncing..."
    mkdir -p data/arrayrecord
    gsutil -m rsync -r gs://fin-ia-eu/data/arrayrecord/ data/arrayrecord/
elif [ -f "data/arrayrecord/manifest.json" ]; then
    echo "  ArrayRecord manifest already exists locally — skipping."
else
    if [ ! -d "data/tokens_v5" ] || [ -z "$(ls data/tokens_v5/*.pt 2>/dev/null)" ]; then
        if gsutil ls gs://fin-ia-eu/data/tokens_v5/ &>/dev/null; then
            echo "  Syncing tokens from GCS..."
            mkdir -p data/tokens_v5
            gsutil -m rsync -r gs://fin-ia-eu/data/tokens_v5/ data/tokens_v5/
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

    echo "  Syncing ArrayRecord to GCS..."
    gsutil -m rsync -r data/arrayrecord/ gs://fin-ia-eu/data/arrayrecord/
fi

echo "--- [TPU worker $(hostname)] Data ready ---"
REMOTE_DATA
)"
fi

# ── Phase 3: Launch training ───────────────────────────────────────────────
echo ""
echo "=== Phase 3: Launching training (${TPU_TYPE}, scale=${SCALE}) ==="

RESUME_FLAG="${RESUME}"
SCALE_TIER="${SCALE}"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" ${WORKER_FLAG} \
    --command="$(cat <<REMOTE_TRAIN
set -euo pipefail
REPO_DIR="\$HOME/Financial_IA"
cd "\$REPO_DIR"
source .venv_tpu/bin/activate

export SCALE_CONFIG="${SCALE_CONFIG}"
export SCALE_TIER="${SCALE_TIER}"
export RESUME="${RESUME_FLAG}"
export GCS_BUCKET="${GCS_BUCKET}"
export TPU_TYPE="${TPU_TYPE}"

export JAX_COMPILATION_CACHE_DIR="\$HOME/.jax_cache"
mkdir -p "\$JAX_COMPILATION_CACHE_DIR"

echo "--- [TPU worker \$(hostname)] Device topology ---"
python3 -c "
import jax
devices = jax.devices()
n = len(devices)
print(f'  {n} {devices[0].platform.upper()} chips ({devices[0].device_kind})')
print(f'  Process: {jax.process_index()}/{jax.process_count()}')
"

CKPT_DIR="checkpoints/jax_v5e/\${SCALE_TIER}"
mkdir -p "\$CKPT_DIR"
if [ "\$RESUME" = "true" ]; then
    if gsutil ls "\${GCS_BUCKET}/checkpoints/jax_v5e/\${SCALE_TIER}/" &>/dev/null; then
        echo "  Restoring checkpoint from GCS..."
        gsutil -m rsync -r "\${GCS_BUCKET}/checkpoints/jax_v5e/\${SCALE_TIER}/" "\$CKPT_DIR/"
    else
        echo "  No checkpoint found on GCS — starting from scratch."
    fi
fi

if gsutil ls "\${GCS_BUCKET}/xla_cache/v5e_\${SCALE_TIER}/" &>/dev/null 2>&1; then
    echo "  Restoring XLA cache from GCS..."
    gsutil -m rsync -r "\${GCS_BUCKET}/xla_cache/v5e_\${SCALE_TIER}/" "\$JAX_COMPILATION_CACHE_DIR/" 2>/dev/null || true
fi

export JAX_TRACEBACK_FILTERING=off

# ── XLA flags v5e (Tore 2D) — meme set que v6e, pas de megacore ──
export LIBTPU_INIT_ARGS=" \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_enable_latency_hiding_scheduler=true \
  --xla_tpu_spmd_rng_bit_generator_unsafe=true \
  --xla_tpu_enable_experimental_fusion_cost_model=true \
"

export TF_CPP_MIN_LOG_LEVEL=2

echo "--- [TPU worker \$(hostname)] Starting training (\${TPU_TYPE}, scale=\${SCALE_TIER}) ---"
nohup python3 -u -c "
import sys, os, logging, time
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

config = load_config(os.environ['SCALE_CONFIG'])
scale = os.environ['SCALE_TIER']
gcs_bucket = os.environ['GCS_BUCKET']
tpu_type = os.environ.get('TPU_TYPE', 'v5e-?')
resume = os.environ.get('RESUME', 'false') == 'true'

# ── Detect topology ──
n_devices = len(jax.devices())
device_kind = jax.devices()[0].device_kind
assert n_devices >= 4, f'Expected >=4 TPU chips, got {n_devices}'

# Auto-scale batch_size to actual chip count
# Config batch_size is written for 64 chips — scale proportionally
config_chips = 64  # reference chip count for config batch sizes
if n_devices != config_chips:
    original_batch = config.training.batch_size
    per_chip_batch = original_batch // config_chips
    adjusted_batch = per_chip_batch * n_devices
    log.info('Batch auto-scaled: %d (config, %d chips) -> %d (actual, %d chips), %d/chip',
        original_batch, config_chips, adjusted_batch, n_devices, per_chip_batch)
    object.__setattr__(config.training, 'batch_size', adjusted_batch)

log.info('=== %s | %d chips | scale=%s ===', tpu_type, n_devices, scale)
log.info('Config: d_model=%d, n_layers=%d, n_heads=%d, batch=%d, lr=%.1e, remat=%s',
    config.mamba2.d_model, config.mamba2.n_layers, config.mamba2.n_heads,
    config.training.batch_size, config.training.lr, config.mamba2.use_remat)

# ── Mesh — auto-detect via device_kind ──
mesh = create_mesh()

# ── Model ──
model = FinJEPA.from_config(config)

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
log.info('Model: %s (%.1fM params)', f'{n_params:,}', n_params / 1e6)
log.info('Per-chip: batch=%d, params=%.0f MB bf16', B, n_params * 2 / 1e6)

state = shard_params(state, mesh)

# ── Checkpoint manager ──
ckpt_dir = f'checkpoints/jax_v5e/{scale}'
ckpt_mgr = create_checkpoint_manager(ckpt_dir, max_to_keep=5)

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

watcher = PreemptionWatcher()
watcher.start()

ckpt_interval = config.training.checkpoint_interval
mfu_interval = 50
val_interval = 500
PEAK_TFLOPS = 197.0  # v5e bf16

log.info('=== Training started (%s, scale=%s, step=%d) ===', tpu_type, scale, start_step)
step = start_step
t0 = time.time()

for batch in train_loader:
    if watcher.should_stop():
        log.warning('PREEMPTION at step %d — saving...', step)
        ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
        ckpt_mgr.wait_until_finished()
        import subprocess
        subprocess.run(['gsutil', '-m', 'rsync', '-r', ckpt_dir + '/',
            f'{gcs_bucket}/checkpoints/jax_v5e/{scale}/'], check=False)
        break

    batch = {k: jnp.array(v) for k, v in batch.items() if not isinstance(v, (str, bytes))}
    batch = shard_batch(batch, mesh)
    state, metrics = train_step(state, batch, model)

    step += 1
    step_time = time.time() - t0

    if step % mfu_interval == 0:
        avg_step_time = step_time / mfu_interval
        mfu = compute_mfu(n_params, config.training.batch_size, S, avg_step_time, n_devices, peak_tflops=PEAK_TFLOPS)
        tps = compute_tokens_per_second(config.training.batch_size, S, avg_step_time)
        log.info(
            'step %d | loss %.4f | grad %.2f | MFU %.1f%% | %.0f tok/s | %.2fs',
            step, float(metrics['loss']), float(metrics['grad_norm']),
            mfu * 100, tps, avg_step_time,
        )
        t0 = time.time()

    if step % ckpt_interval == 0:
        log.info('Checkpoint step %d...', step)
        ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
        ckpt_mgr.wait_until_finished()
        import subprocess
        subprocess.Popen(['gsutil', '-m', 'rsync', '-r', ckpt_dir + '/',
            f'{gcs_bucket}/checkpoints/jax_v5e/{scale}/'])

    if step % val_interval == 0:
        val_losses = []
        for i, val_batch in enumerate(val_loader):
            if i >= 20: break
            val_batch = {k: jnp.array(v) for k, v in val_batch.items() if not isinstance(v, (str, bytes))}
            val_batch = shard_batch(val_batch, mesh)
            val_metrics = eval_step(state, val_batch, model)
            val_losses.append(float(val_metrics['loss']))
        avg_val = sum(val_losses) / len(val_losses) if val_losses else float('nan')
        log.info('step %d | val_loss %.4f (%d batches)', step, avg_val, len(val_losses))

watcher.stop()

log.info('Final checkpoint step %d...', step)
ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
ckpt_mgr.wait_until_finished()
import subprocess
subprocess.run(['gsutil', '-m', 'rsync', '-r', ckpt_dir + '/',
    f'{gcs_bucket}/checkpoints/jax_v5e/{scale}/'], check=False)

cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '')
if cache_dir and os.path.exists(cache_dir):
    subprocess.run(['gsutil', '-m', 'rsync', '-r', cache_dir + '/',
        f'{gcs_bucket}/xla_cache/v5e_{scale}/'], check=False)

log.info('=== Done (%s, scale=%s, step=%d) ===', tpu_type, scale, step)
" > training_v5e.log 2>&1 &

TRAIN_PID=\$!
echo "  Training launched (PID: \$TRAIN_PID)"
echo "  Monitor: tail -f \$HOME/Financial_IA/training_v5e.log"
echo "--- [TPU worker \$(hostname)] Launch complete ---"
REMOTE_TRAIN
)"

echo ""
echo "============================================================"
echo " Training launched on ${TPU_NAME} (${TPU_TYPE}, scale=${SCALE})"
echo ""
echo " Monitor logs:"
echo "   gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} \\"
echo "     -- tail -f ~/Financial_IA/training_v5e.log"
echo ""
echo " SSH into VM:"
echo "   gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}"
echo ""
echo " Delete when done:"
echo "   gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}"
echo "============================================================"
