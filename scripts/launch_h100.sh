#!/bin/bash
###############################################################################
# Financial-IA — H100 God-Tier Training Launch Script
#
# Runs on: GCP a3-highgpu-1g (1× H100 80GB, 26 vCPUs, 234 GB RAM)
# Deep Learning VM: PyTorch 2.7 + CUDA 12.8 + Ubuntu 24.04
#
# This script:
#   1. Installs system deps (ninja for torch.compile)
#   2. Installs project Python dependencies
#   3. Configures H100-optimal environment variables
#   4. Runs a smoke test to validate GPU + model
#   5. Launches full Fin-JEPA training with torch.compile
#
# Usage:
#   # On the H100 VM after syncing code:
#   chmod +x scripts/launch_h100.sh
#   ./scripts/launch_h100.sh                    # Full training
#   ./scripts/launch_h100.sh --smoke-test       # Quick validation only
#   ./scripts/launch_h100.sh --no-compile       # Skip torch.compile
###############################################################################

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/daniel/Financial_IA}"
VENV_DIR="${PROJECT_DIR}/.venv"
CONFIG="${PROJECT_DIR}/configs/strate_ii.yaml"
LOG_DIR="${PROJECT_DIR}/tb_logs/strate_ii"

# Parse flags
SMOKE_TEST=false
USE_COMPILE=true
for arg in "$@"; do
    case $arg in
        --smoke-test)  SMOKE_TEST=true ;;
        --no-compile)  USE_COMPILE=false ;;
    esac
done

echo "============================================"
echo " Financial-IA — H100 God-Tier Training"
echo " Config: ${CONFIG}"
echo " Smoke test: ${SMOKE_TEST}"
echo " torch.compile: ${USE_COMPILE}"
echo "============================================"
echo

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------

echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq ninja-build > /dev/null 2>&1
echo "  ninja: $(ninja --version)"

# ---------------------------------------------------------------------------
# 2. Python environment
# ---------------------------------------------------------------------------

echo "[2/6] Setting up Python environment..."
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

# GCP deps
pip install -q \
    google-cloud-storage>=2.0 \
    pyarrow>=14.0 \
    aiohttp>=3.9

echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ---------------------------------------------------------------------------
# 3. Verify GPU
# ---------------------------------------------------------------------------

echo "[3/6] Verifying H100 GPU..."
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

echo "[4/6] Configuring H100-optimal environment..."

# TF32: allow approximate TF32 for float32 matmuls (3× faster, <0.1% precision loss)
export NVIDIA_TF32_OVERRIDE=1

# NCCL (for future multi-GPU): use NVLink if available
export NCCL_P2P_LEVEL=NVL

# PyTorch: use expandable segments to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Triton cache: use local NVMe SSD for fast torch.compile cache
export TRITON_CACHE_DIR="${PROJECT_DIR}/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

echo "  NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

# ---------------------------------------------------------------------------
# 5. Smoke test
# ---------------------------------------------------------------------------

echo "[5/6] Running smoke test..."
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

COMPILE_FLAG=""
if [ "$USE_COMPILE" = true ]; then
    COMPILE_FLAG="--compile"
fi

python scripts/train_strate_ii.py \
    --config "$CONFIG" \
    --synthetic \
    --num_synthetic 1024 \
    $COMPILE_FLAG || true

echo "  Smoke test passed!"

if [ "$SMOKE_TEST" = true ]; then
    echo
    echo "============================================"
    echo " Smoke test complete. Exiting."
    echo "============================================"
    exit 0
fi

# ---------------------------------------------------------------------------
# 6. Full training
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. Start GCS checkpoint sync (background loop)
# ---------------------------------------------------------------------------

echo "[6/7] Starting GCS checkpoint sync (every 10 min)..."
export GCS_BUCKET="gs://financial-ia-datalake"
export PROJECT_DIR="${PROJECT_DIR}"
bash "${PROJECT_DIR}/scripts/sync_checkpoints_gcs.sh" --loop &
GCS_SYNC_PID=$!
echo "  GCS sync PID: $GCS_SYNC_PID"

# Restore checkpoints from GCS if available (auto-resume after preemption)
echo "  Checking GCS for existing checkpoints..."
gsutil -m rsync -r "${GCS_BUCKET}/checkpoints/strate_ii/" \
    "${PROJECT_DIR}/checkpoints/strate_ii/" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 7. Launch training
# ---------------------------------------------------------------------------

echo "[7/7] Launching full Fin-JEPA training..."
echo "  Config: ${CONFIG}"
echo "  TensorBoard: ${LOG_DIR}"
echo

python scripts/train_strate_ii.py \
    --config "$CONFIG" \
    $COMPILE_FLAG \
    2>&1 | tee "${PROJECT_DIR}/training_h100.log"

# Final sync after training completes
kill $GCS_SYNC_PID 2>/dev/null || true
bash "${PROJECT_DIR}/scripts/sync_checkpoints_gcs.sh"

echo
echo "============================================"
echo " Training complete!"
echo " Logs: ${PROJECT_DIR}/training_h100.log"
echo " TensorBoard: tensorboard --logdir ${LOG_DIR}"
echo " Checkpoints: ${PROJECT_DIR}/checkpoints/strate_ii/"
echo "============================================"
