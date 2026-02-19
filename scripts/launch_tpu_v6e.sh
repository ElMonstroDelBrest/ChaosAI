#!/bin/bash
# Launch script for Financial-IA on TPU v6e-8 (Trillium / CT6E).
#
# Creates a preemptible v6e-8 TPU VM, installs deps, converts data if needed,
# and launches Fin-JEPA training with GSPMD.
#
# Usage:
#   ./scripts/launch_tpu_v6e.sh                     # full pipeline
#   ./scripts/launch_tpu_v6e.sh --skip-create       # reuse existing VM
#   ./scripts/launch_tpu_v6e.sh --train-only        # skip data conversion
#
# Requirements:
#   - gcloud CLI authenticated with TPU v6e quota (ct6e-standard-8)
#   - GCS bucket with source data or local data/tokens_v5/
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
TPU_NAME="${TPU_NAME:-fin-ia-v6e}"
ZONE="${ZONE:-europe-west4-a}"
TPU_TYPE="${TPU_TYPE:-v6e-8}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
GCS_BUCKET="${GCS_BUCKET:-gs://fin-ia-bucket}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv6e}"
REPO_DIR="/home/${USER}/Financial_IA"
VENV_DIR="${REPO_DIR}/.venv_tpu"
SKIP_CREATE=false
TRAIN_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-create) SKIP_CREATE=true ;;
        --train-only)  TRAIN_ONLY=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "============================================================"
echo " Financial-IA — TPU v6e-8 (Trillium) Launch"
echo "============================================================"
echo "  Project:  ${PROJECT}"
echo "  Zone:     ${ZONE}"
echo "  TPU:      ${TPU_NAME} (${TPU_TYPE})"
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
        --runtime-version="$RUNTIME_VERSION" \
        --preemptible \
        --scopes=cloud-platform

    echo "  TPU VM created."
fi

# ── Phase 1: Setup environment on TPU VM ───────────────────────────────────
echo ""
echo "=== Phase 1: Setting up environment ==="

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="$(cat <<'REMOTE_SETUP'
set -euo pipefail
echo "--- [TPU] Environment setup ---"

# Clone or update repo
REPO_DIR="$HOME/Financial_IA"
if [ -d "$REPO_DIR/.git" ]; then
    echo "  Pulling latest..."
    cd "$REPO_DIR" && git pull --ff-only
else
    echo "  Cloning repo..."
    git clone https://github.com/$(git config --global user.name 2>/dev/null || echo 'user')/Financial_IA.git "$REPO_DIR" || {
        echo "  [WARN] Clone failed — assuming repo is pre-staged."
    }
fi
cd "$REPO_DIR"

# Create venv
VENV="$REPO_DIR/.venv_tpu"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install JAX for TPU v6e (Trillium)
# v6e requires the latest libtpu — use nightly or pinned compatible version.
echo "  Installing JAX + TPU v6e support..."
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
assert len(devices) >= 4, f'Expected >=4 TPU chips, got {len(devices)}'
assert devices[0].platform == 'tpu', f'Expected TPU, got {devices[0].platform}'
print('  TPU verification: OK')
"

echo "--- [TPU] Environment ready ---"
REMOTE_SETUP
)"

# ── Phase 2: Data conversion (ArrayRecord) ─────────────────────────────────
if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "=== Phase 2: Data conversion ==="

    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="$(cat <<'REMOTE_DATA'
set -euo pipefail
REPO_DIR="$HOME/Financial_IA"
cd "$REPO_DIR"
source .venv_tpu/bin/activate

# Sync data from GCS if local tokens_v5 is missing
if [ ! -d "data/tokens_v5" ] || [ -z "$(ls data/tokens_v5/*.pt 2>/dev/null)" ]; then
    echo "  Syncing tokens from GCS..."
    gsutil -m rsync -r gs://fin-ia-bucket/data/tokens_v5/ data/tokens_v5/
fi

# Convert to ArrayRecord (idempotent — skips if manifest exists and is fresh)
if [ -f "data/arrayrecord/manifest.json" ]; then
    echo "  ArrayRecord manifest already exists — skipping conversion."
else
    echo "  Converting .pt -> ArrayRecord..."
    python3 scripts/convert_pt_to_arrayrecord.py \
        --input data/tokens_v5/ \
        --output data/arrayrecord/ \
        --seq_len 128
fi

# Upload ArrayRecord to GCS for persistence across preemptions
echo "  Syncing ArrayRecord to GCS..."
gsutil -m rsync -r data/arrayrecord/ gs://fin-ia-bucket/data/arrayrecord/

echo "--- [TPU] Data ready ---"
REMOTE_DATA
)"
fi

# ── Phase 3: Launch training ───────────────────────────────────────────────
echo ""
echo "=== Phase 3: Launching training ==="

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="$(cat <<'REMOTE_TRAIN'
set -euo pipefail
REPO_DIR="$HOME/Financial_IA"
cd "$REPO_DIR"
source .venv_tpu/bin/activate

# Log topology for debugging
echo "--- [TPU] Device topology ---"
python3 -c "
import jax
devices = jax.devices()
n = len(devices)
platform = devices[0].platform
print(f'  {n} {platform.upper()} chips detected')
print(f'  Process count: {jax.process_count()}')
print(f'  Process index: {jax.process_index()}')
# Log local batch size
global_batch = 1024
local_batch = global_batch // n
print(f'  Global batch: {global_batch}, Local batch/chip: {local_batch}')
"

# Resume from GCS checkpoint if available
CKPT_DIR="checkpoints/jax_v6"
mkdir -p "$CKPT_DIR"
if gsutil ls gs://fin-ia-bucket/checkpoints/jax_v6/ &>/dev/null; then
    echo "  Restoring checkpoint from GCS..."
    gsutil -m rsync -r gs://fin-ia-bucket/checkpoints/jax_v6/ "$CKPT_DIR/"
fi

# Launch training with nohup (survives SSH disconnect)
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JAX_TRACEBACK_FILTERING=off

echo "--- [TPU] Starting Fin-JEPA training ---"
nohup python3 -u -c "
import sys, logging, time
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger('train')

import jax
import jax.numpy as jnp

from src.jax_v6.config import load_config
from src.jax_v6.training.sharding import create_mesh, shard_batch, shard_params
from src.jax_v6.training.train_state import create_train_state
from src.jax_v6.training.train_step import train_step, eval_step
from src.jax_v6.jepa import FinJEPA
from src.jax_v6.data.grain_loader import create_dataloader

# Load config
config = load_config('configs/strate_ii_jax.yaml') if __import__('pathlib').Path('configs/strate_ii_jax.yaml').exists() else None
if config is None:
    from src.jax_v6.config import StrateIIConfig
    config = StrateIIConfig()
    log.info('Using default config (no YAML found)')

# Setup mesh
mesh = create_mesh()
n_devices = len(jax.devices())
log.info('Training on %d %s chips', n_devices, jax.devices()[0].platform.upper())

# Create model + state
model = FinJEPA(config)
state = create_train_state(model, config, jax.random.PRNGKey(42))
state = shard_params(state, mesh)

# Create data loaders
train_loader = create_dataloader(
    config.data.arrayrecord_dir, split='train',
    batch_size=config.training.batch_size,
    seq_len=config.embedding.seq_len,
    worker_count=config.data.num_workers,
    prefetch_buffer_size=config.data.prefetch_buffer_size,
)

log.info('=== Training started ===')
step = 0
t0 = time.time()
for batch in train_loader:
    batch = {k: jnp.array(v) for k, v in batch.items() if not isinstance(v, str)}
    batch = shard_batch(batch, mesh)
    state, metrics = train_step(state, batch, model)

    step += 1
    if step % 100 == 0:
        elapsed = time.time() - t0
        loss = float(metrics['loss'])
        log.info('step %d | loss %.4f | %.1f steps/s', step, loss, 100 / elapsed)
        t0 = time.time()

    # Checkpoint every 1000 steps
    if step % 1000 == 0:
        log.info('Checkpointing at step %d...', step)
        import orbax.checkpoint as ocp
        mgr = ocp.CheckpointManager('checkpoints/jax_v6')
        mgr.save(step, args=ocp.args.StandardSave(state))
        # Sync to GCS
        import subprocess
        subprocess.run(['gsutil', '-m', 'rsync', '-r', 'checkpoints/jax_v6/', 'gs://fin-ia-bucket/checkpoints/jax_v6/'], check=False)

log.info('=== Training complete at step %d ===', step)
" > training_v6e.log 2>&1 &

TRAIN_PID=$!
echo "  Training launched (PID: $TRAIN_PID)"
echo "  Monitor: tail -f $REPO_DIR/training_v6e.log"
echo "  Or:      gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE -- tail -f ~/Financial_IA/training_v6e.log"
echo "--- [TPU] Launch complete ---"
REMOTE_TRAIN
)"

echo ""
echo "============================================================"
echo " Training launched on ${TPU_NAME} (${TPU_TYPE})"
echo ""
echo " Monitor logs:"
echo "   gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} \\"
echo "     -- tail -f ~/Financial_IA/training_v6e.log"
echo ""
echo " SSH into VM:"
echo "   gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}"
echo ""
echo " Delete when done:"
echo "   gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}"
echo "============================================================"
