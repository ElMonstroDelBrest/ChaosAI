#!/bin/bash
# Startup script for Financial-IA GPU Training VM (Deep Learning VM / H100)
set -euo pipefail

LOG="/var/log/financial-ia-setup.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Financial-IA Training VM Setup ==="
echo "Date: $(date -u)"

# ── Phase 0: Wait for NVIDIA kernel module ────────────────────────────────
# GCP DLVM loads nvidia.ko asynchronously via systemd. If training starts
# before the driver is ready, mamba-ssm silently falls back to CPU scan.
# H100 (Hopper) also requires nvidia-fabricmanager for NVLink/NVSwitch.
echo "=== Waiting for NVIDIA driver (max 5 min) ==="
TIMEOUT=300
ELAPSED=0
until nvidia-smi &>/dev/null; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "[FATAL] nvidia-smi unavailable after ${TIMEOUT}s. Halting VM to avoid Spot cost waste."
        sudo poweroff
        exit 1
    fi
    echo "  [${ELAPSED}s] Waiting for nvidia.ko..."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "  Driver ready after ${ELAPSED}s."

# H100: ensure Fabric Manager is active (required for NVLink peer access)
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "H100"; then
    echo "=== H100 detected — checking nvidia-fabricmanager ==="
    if ! systemctl is-active --quiet nvidia-fabricmanager 2>/dev/null; then
        echo "  Starting nvidia-fabricmanager..."
        sudo systemctl start nvidia-fabricmanager
    fi
    systemctl is-active nvidia-fabricmanager && echo "  nvidia-fabricmanager: OK"
fi

echo "=== GPU Status ==="
nvidia-smi

# ── Phase 1: Install dependencies ─────────────────────────────────────────
# Deep Learning VM already has conda/pip/PyTorch/CUDA
echo "=== Installing project dependencies ==="
pip install --upgrade pip
pip install \
    google-cloud-storage \
    google-cloud-logging \
    pytorch-lightning \
    gymnasium \
    stable-baselines3 \
    einops \
    dacite \
    pyyaml \
    tensorboard \
    pyarrow \
    pandas \
    tqdm

# ── Phase 2: Verify Mamba CUDA kernels ────────────────────────────────────
# mamba-ssm falls back silently to pure-PyTorch scan if CUDA kernels fail.
# A silent fallback would cut training throughput by ~10x without any error.
echo "=== Verifying Mamba-SSM CUDA kernels ==="
python3 - <<'PYEOF'
import sys
try:
    import mamba_ssm
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("  mamba_ssm fused CUDA scan: OK")
except ImportError as e:
    print(f"  [FATAL] Mamba CUDA kernel not available: {e}", file=sys.stderr)
    print("  Install with: pip install mamba-ssm causal-conv1d", file=sys.stderr)
    sys.exit(1)
PYEOF

echo "=== Setup complete ==="
