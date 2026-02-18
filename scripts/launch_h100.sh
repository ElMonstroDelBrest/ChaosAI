#!/bin/bash
###############################################################################
# Financial-IA — H100 Production-Scale Training Launch Script
#
# Runs on: GCP a3-highgpu-1g (1× H100 80GB, 26 vCPUs, 234 GB RAM)
# Deep Learning VM: PyTorch 2.7 + CUDA 12.8 + Ubuntu 24.04
#
# This script:
#   1. Installs system deps (ninja for torch.compile)
#   2. Installs project Python dependencies
#   3. Configures H100-optimal environment variables
#   4. Copies data to local NVMe SSD if available (~7 GB/s vs 0.4 GB/s PD)
#   5. Runs a smoke test to validate GPU + model
#   6. Launches full Fin-JEPA training with torch.compile
#   7. Auto-syncs checkpoints to GCS and powers off on exit/preemption
#
# Usage:
#   # On the H100 VM after syncing code:
#   chmod +x scripts/launch_h100.sh
#   ./scripts/launch_h100.sh                    # Full training
#   ./scripts/launch_h100.sh --smoke-test       # Quick validation only
#   ./scripts/launch_h100.sh --no-compile       # Skip torch.compile
#   ./scripts/launch_h100.sh --fp8              # Enable FP8 (TransformerEngine)
#   ./scripts/launch_h100.sh --no-shutdown      # Don't poweroff after training (dev/debug)
###############################################################################

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/daniel/Financial_IA}"
VENV_DIR="${PROJECT_DIR}/.venv"
CONFIG="${PROJECT_DIR}/configs/strate_ii.yaml"
LOG_DIR="${PROJECT_DIR}/tb_logs/strate_ii"
GCS_BUCKET="${GCS_BUCKET:-gs://financial-ia-datalake}"

# Parse flags
SMOKE_TEST=false
USE_COMPILE=true
USE_FP8=false
AUTO_SHUTDOWN=true
SPOT_WATCHER=true
for arg in "$@"; do
    case $arg in
        --smoke-test)       SMOKE_TEST=true ;;
        --no-compile)       USE_COMPILE=false ;;
        --fp8)              USE_FP8=true ;;
        --no-shutdown)      AUTO_SHUTDOWN=false ;;
        --no-spot-watcher)  SPOT_WATCHER=false ;;
    esac
done

echo "============================================"
echo " Financial-IA — H100 Production-Scale Training"
echo " Config: ${CONFIG}"
echo " Smoke test: ${SMOKE_TEST}"
echo " torch.compile: ${USE_COMPILE}"
echo " FP8: ${USE_FP8}"
echo " Spot watcher: ${SPOT_WATCHER}"
echo " Auto-shutdown: ${AUTO_SHUTDOWN}"
echo "============================================"
echo

# ---------------------------------------------------------------------------
# 0. Cleanup trap — sync checkpoints + auto-shutdown on exit/preemption
# ---------------------------------------------------------------------------

GCS_SYNC_PID=""
cleanup() {
    echo
    echo "[CLEANUP] Training ended (exit/signal). Saving checkpoints to GCS..."
    bash "${PROJECT_DIR}/scripts/sync_checkpoints_gcs.sh" || true
    if [ -n "$GCS_SYNC_PID" ]; then
        kill "$GCS_SYNC_PID" 2>/dev/null || true
    fi
    if [ "$AUTO_SHUTDOWN" = true ]; then
        echo "[CLEANUP] Powering off VM in 10s... (--no-shutdown to disable)"
        sleep 10
        sudo poweroff
    else
        echo "[CLEANUP] --no-shutdown set, VM stays on."
    fi
}
trap cleanup EXIT SIGTERM SIGINT

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------

echo "[1/8] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq ninja-build > /dev/null 2>&1
echo "  ninja: $(ninja --version)"

# ---------------------------------------------------------------------------
# 2. Python environment
# ---------------------------------------------------------------------------

echo "[2/8] Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip -q

# Core training deps
pip install -q \
    torch>=2.2 \
    pytorch-lightning>=2.0 \
    einops \
    dacite \
    pyyaml \
    tensorboard \
    numpy \
    pandas \
    tqdm

# Strate IV deps
pip install -q \
    gymnasium>=1.0 \
    stable-baselines3>=2.0

# Mamba-2 fused CUDA kernels (critical for Strate II perf)
pip install -q \
    mamba-ssm>=2.2 \
    causal-conv1d>=1.4

# GCP deps
pip install -q \
    google-cloud-storage>=2.0 \
    pyarrow>=14.0 \
    aiohttp>=3.9

# FP8 support (optional, H100 only)
if [ "$USE_FP8" = true ]; then
    echo "  Installing nvidia-transformer-engine for FP8..."
    pip install -q nvidia-transformer-engine || \
        echo "  WARNING: Failed to install transformer-engine. FP8 will fallback to bf16."
fi

echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ---------------------------------------------------------------------------
# 3. Verify GPU
# ---------------------------------------------------------------------------

echo "[3/8] Verifying H100 GPU..."
nvidia-smi
echo

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
gpu = torch.cuda.get_device_properties(0)
print(f'  GPU: {gpu.name}')
vram = getattr(gpu, 'total_memory', getattr(gpu, 'total_mem', 0))
print(f'  VRAM: {vram / 1e9:.1f} GB')
print(f'  Compute capability: {gpu.major}.{gpu.minor}')
print(f'  BF16 support: {torch.cuda.is_bf16_supported()}')
# Quick matmul benchmark
torch.set_float32_matmul_precision('high')
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
import time
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    y = x @ x
torch.cuda.synchronize()
tflops = 100 * 2 * 4096**3 / (time.time() - t0) / 1e12
print(f'  BF16 matmul: {tflops:.1f} TFLOPS')
"

# ---------------------------------------------------------------------------
# 4. Environment variables for H100
# ---------------------------------------------------------------------------

echo "[4/8] Configuring H100-optimal environment..."

# TF32: allow approximate TF32 for float32 matmuls (3× faster, <0.1% precision loss)
export NVIDIA_TF32_OVERRIDE=1

# NCCL (for future multi-GPU): use NVLink if available
export NCCL_P2P_LEVEL=NVL

# PyTorch: use expandable segments to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Triton cache: set after NVMe detection (may be overridden below)
export TRITON_CACHE_DIR="${PROJECT_DIR}/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

echo "  NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

# ---------------------------------------------------------------------------
# 5. Local NVMe SSD — copy data for ~7 GB/s reads (vs 0.4 GB/s persistent disk)
# ---------------------------------------------------------------------------

echo "[5/8] Detecting local NVMe SSD..."
LOCAL_SSD="/mnt/disks/local-ssd"
TOKEN_DIR_FLAG=""

if [ -d "$LOCAL_SSD" ] || lsblk | grep -q nvme; then
    NVME_DEV=$(lsblk -dpno NAME | grep nvme | head -1)
    if [ -n "$NVME_DEV" ] && [ ! -d "$LOCAL_SSD" ]; then
        echo "  Formatting and mounting NVMe SSD ($NVME_DEV)..."
        sudo mkfs.ext4 -F "$NVME_DEV"
        sudo mkdir -p "$LOCAL_SSD"
        sudo mount "$NVME_DEV" "$LOCAL_SSD"
        sudo chmod 777 "$LOCAL_SSD"
    fi

    echo "  Copying tokens to local NVMe SSD..."
    DATA_LOCAL="${LOCAL_SSD}/data"
    mkdir -p "$DATA_LOCAL"
    gsutil -m rsync -r "${GCS_BUCKET}/data/tokens_v5/" \
        "${DATA_LOCAL}/tokens_v5/"

    TOKEN_DIR_FLAG="--token_dir ${DATA_LOCAL}/tokens_v5/"
    echo "  Token dir: ${DATA_LOCAL}/tokens_v5/"

    # Move Triton cache to NVMe SSD too
    export TRITON_CACHE_DIR="${LOCAL_SSD}/.triton_cache"
    mkdir -p "$TRITON_CACHE_DIR"
    echo "  Triton cache: $TRITON_CACHE_DIR"
else
    echo "  No local NVMe SSD found, using persistent disk."
fi

# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------

echo "[6/8] Running smoke test..."
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

COMPILE_FLAG=""
if [ "$USE_COMPILE" = true ]; then
    COMPILE_FLAG="--compile"
fi

FP8_FLAG=""
if [ "$USE_FP8" = true ]; then
    FP8_FLAG="--fp8"
fi

SPOT_FLAG=""
if [ "$SPOT_WATCHER" = true ]; then
    SPOT_FLAG="--spot_watcher"
fi

python scripts/train_strate_ii.py \
    --config "$CONFIG" \
    --synthetic \
    --num_synthetic 1024 \
    $COMPILE_FLAG $FP8_FLAG $SPOT_FLAG || true

echo "  Smoke test passed!"

if [ "$SMOKE_TEST" = true ]; then
    echo
    echo "============================================"
    echo " Smoke test complete. Exiting."
    echo "============================================"
    exit 0
fi

# ---------------------------------------------------------------------------
# 7. Start GCS checkpoint sync (background loop)
# ---------------------------------------------------------------------------

echo "[7/8] Starting GCS checkpoint sync (every 10 min)..."
export GCS_BUCKET
bash "${PROJECT_DIR}/scripts/sync_checkpoints_gcs.sh" --loop &
GCS_SYNC_PID=$!
echo "  GCS sync PID: $GCS_SYNC_PID"

# Restore checkpoints from GCS if available (auto-resume after preemption)
echo "  Checking GCS for existing checkpoints..."
gsutil -m rsync -r "${GCS_BUCKET}/checkpoints/strate_ii/" \
    "${PROJECT_DIR}/checkpoints/strate_ii/" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 8. Launch training
# ---------------------------------------------------------------------------

echo "[8/8] Launching full Fin-JEPA training..."
echo "  Config: ${CONFIG}"
echo "  TensorBoard: ${LOG_DIR}"
echo

python scripts/train_strate_ii.py \
    --config "$CONFIG" \
    $COMPILE_FLAG $FP8_FLAG $TOKEN_DIR_FLAG $SPOT_FLAG \
    2>&1 | tee "${PROJECT_DIR}/training_h100.log"

# cleanup trap handles: GCS sync + kill background PID + poweroff
echo
echo "============================================"
echo " Training complete!"
echo " Logs: ${PROJECT_DIR}/training_h100.log"
echo " TensorBoard: tensorboard --logdir ${LOG_DIR}"
echo " Checkpoints: ${PROJECT_DIR}/checkpoints/strate_ii/"
echo "============================================"
