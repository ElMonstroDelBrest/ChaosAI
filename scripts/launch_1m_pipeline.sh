#!/bin/bash
# Full pipeline: 1-minute data → Strate I (PyTorch/CPU) → Strate II (JAX/TPU)
#
# Runs entirely on TPU v6e VM:
#   Phase 0: Create TPU VM (or reuse existing)
#   Phase 1: Upload code + install deps (JAX + PyTorch CPU)
#   Phase 2: Download 1m parquets from GCS → convert to .pt
#   Phase 3: Train Strate I tokenizer (PyTorch on CPU, ~100K params)
#   Phase 4: Pretokenize → ArrayRecords
#   Phase 5: Train Strate II 30M (JAX on TPU)
#
# Usage:
#   ./scripts/launch_1m_pipeline.sh                    # full pipeline
#   ./scripts/launch_1m_pipeline.sh --skip-create      # reuse existing VM
#   ./scripts/launch_1m_pipeline.sh --train-only       # skip data prep (ArrayRecords ready)
#   ./scripts/launch_1m_pipeline.sh --resume           # resume Strate II from checkpoint
#
# Prereqs:
#   - 1m parquets on GCS: gs://fin-ia-eu/data/raw/1m_parquet/
#   - gcloud CLI authenticated with TRC v6e quota
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
ZONE="${ZONE:-europe-west4-a}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
GCS_BUCKET="${GCS_BUCKET:-gs://fin-ia-eu}"
VERSION="${VERSION:-v2-alpha-tpuv6e}"
TPU_TYPE="${TPU_TYPE:-v6e-8}"
TPU_NAME="${TPU_NAME:-fin-ia-v6e}"
SCALE_CONFIG="configs/scaling/v6e_30m.yaml"
SCALE_TIER="30m"
SKIP_CREATE=false
TRAIN_ONLY=false
RESUME=false
MAX_PAIRS=50             # Strate I trains on subset for speed (50 pairs ≈ 3.8M patches)

for arg in "$@"; do
    case $arg in
        --skip-create) SKIP_CREATE=true ;;
        --train-only)  TRAIN_ONLY=true ;;
        --resume)      RESUME=true ;;
        --all-pairs)   MAX_PAIRS=0 ;;    # Train Strate I on all 432 pairs
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
N_CHIPS="${TPU_TYPE##*-}"
HBM_TOTAL=$((N_CHIPS * 32))

echo "============================================================"
echo " Financial-IA — 1m Pipeline (${TPU_TYPE})"
echo "============================================================"
echo "  Project:  ${PROJECT}"
echo "  Zone:     ${ZONE}"
echo "  TPU:      ${TPU_NAME} (${TPU_TYPE})"
echo "  Scale:    ${SCALE_TIER} (${SCALE_CONFIG})"
echo "  HBM:      ${N_CHIPS} chips × 32 GB = ${HBM_TOTAL} GB"
echo "  Resume:   ${RESUME}"
echo "  Bucket:   ${GCS_BUCKET}"
echo "  Date:     $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# Worker flag
if [ "$N_CHIPS" -le 8 ]; then
    WORKER_FLAG="--worker=0"
else
    WORKER_FLAG="--worker=all"
fi

# ── Phase 0: Create TPU VM ───────────────────────────────────────────────
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

    echo "  TPU VM created."
fi

# ── Phase 1: Upload code + install deps ──────────────────────────────────
echo ""
echo "=== Phase 1: Uploading code + installing deps ==="

gcloud compute tpus tpu-vm scp --recurse \
    "${LOCAL_REPO}/src" "${LOCAL_REPO}/scripts" "${LOCAL_REPO}/configs" \
    "${LOCAL_REPO}/pyproject.toml" \
    "${TPU_NAME}:~/Financial_IA/" \
    --zone="$ZONE" --project="$PROJECT" ${WORKER_FLAG}

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" ${WORKER_FLAG} \
    --command="$(cat <<'REMOTE_SETUP'
set -euo pipefail
echo "--- [$(hostname)] Environment setup ---"

REPO_DIR="$HOME/Financial_IA"
cd "$REPO_DIR"

sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip

VENV="$REPO_DIR/.venv_tpu"
if [ ! -f "$VENV/bin/activate" ]; then
    rm -rf "$VENV"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo "  Installing JAX + TPU..."
pip install -U pip setuptools wheel
pip install -U \
    "jax[tpu]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "  Installing JAX ecosystem..."
pip install -U \
    flax optax orbax-checkpoint diffrax \
    array-record grain-nightly tensorflow \
    dacite pyyaml tqdm

echo "  Installing PyTorch CPU (for Strate I)..."
pip install -U \
    torch --index-url https://download.pytorch.org/whl/cpu
pip install -U \
    pytorch-lightning pandas pyarrow

echo "  Verifying TPU..."
python3 -c "
import jax
n = len(jax.devices())
print(f'  TPU: {n} chips ({jax.devices()[0].device_kind})')
assert n >= 4
" || echo "  [WARN] TPU verification failed (may be in use). Continuing..."

echo "  Verifying PyTorch..."
python3 -c "
import torch; print(f'  PyTorch: {torch.__version__} (CPU)')
import pytorch_lightning; print(f'  Lightning: {pytorch_lightning.__version__}')
"

echo "--- [$(hostname)] Environment ready ---"
REMOTE_SETUP
)"

# ── Phase 2-4: Data preparation (parquets → tokens → ArrayRecords) ──────
if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "=== Phase 2-4: Data preparation (1m parquets → ArrayRecords) ==="

    MAX_PAIRS_ARG="${MAX_PAIRS}"
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" --worker=0 \
        --command="$(cat <<REMOTE_DATA
set -euo pipefail
REPO_DIR="\$HOME/Financial_IA"
cd "\$REPO_DIR"
source .venv_tpu/bin/activate
export PYTHONPATH="\$REPO_DIR"

# ── Phase 2: Download 1m parquets from GCS → convert to .pt ──
echo ""
echo "=== Phase 2: Parquets → .pt ==="

if [ -f "data/ohlcv_1m/.done" ]; then
    echo "  OHLCV .pt files already exist — skipping."
else
    mkdir -p data/raw/1m_parquet
    echo "  Downloading 1m parquets from GCS (~23 GB)..."
    gcloud storage rsync -r ${GCS_BUCKET}/data/raw/1m_parquet/ data/raw/1m_parquet/
    echo "  Downloaded: \$(ls data/raw/1m_parquet/*.parquet 2>/dev/null | wc -l) files"

    echo "  Converting parquets → .pt..."
    python3 scripts/convert_parquet_to_pt.py \
        --input_dir data/raw/1m_parquet \
        --output_dir data/ohlcv_1m

    touch data/ohlcv_1m/.done
    echo "  Done: \$(ls data/ohlcv_1m/*.pt 2>/dev/null | wc -l) .pt files"

    # Free parquet disk space
    rm -rf data/raw/1m_parquet
    echo "  Cleaned up parquets."
fi

# ── Phase 3: Train Strate I tokenizer (PyTorch/CPU) ──
echo ""
echo "=== Phase 3: Training Strate I (CPU) ==="

if [ -f "checkpoints/strate_i_1m_best.ckpt" ]; then
    echo "  Strate I checkpoint exists — skipping."
else
    # Optionally use subset for speed
    MAX_PAIRS=${MAX_PAIRS_ARG}
    if [ "\$MAX_PAIRS" -gt 0 ]; then
        echo "  Creating subset (\$MAX_PAIRS pairs) for Strate I training..."
        mkdir -p data/ohlcv_1m_subset
        ls -S data/ohlcv_1m/*.pt | head -\$MAX_PAIRS | while read f; do
            ln -sf "\$(realpath "\$f")" "data/ohlcv_1m_subset/\$(basename "\$f")"
        done
        # Temporarily override data path
        sed 's|data/ohlcv_1m/|data/ohlcv_1m_subset/|' configs/strate_i_1m.yaml > /tmp/strate_i_1m_subset.yaml
        STRATE_I_CONFIG="/tmp/strate_i_1m_subset.yaml"
        echo "  Subset: \$(ls data/ohlcv_1m_subset/*.pt | wc -l) pairs"
    else
        STRATE_I_CONFIG="configs/strate_i_1m.yaml"
        echo "  Training on all pairs."
    fi

    echo "  Starting Strate I training..."
    python3 scripts/train_strate_i.py --config "\$STRATE_I_CONFIG" 2>&1 | tail -50

    # Find best checkpoint
    BEST_CKPT=\$(ls -t checkpoints/strate-i-*.ckpt 2>/dev/null | head -1)
    if [ -z "\$BEST_CKPT" ]; then
        echo "  [ERROR] No Strate I checkpoint produced!"
        exit 1
    fi
    cp "\$BEST_CKPT" checkpoints/strate_i_1m_best.ckpt
    echo "  Best checkpoint: \$BEST_CKPT → checkpoints/strate_i_1m_best.ckpt"
fi

# ── Phase 4: Pretokenize → ArrayRecords ──
echo ""
echo "=== Phase 4: Pretokenize + ArrayRecords ==="

if [ -f "data/arrayrecord_1m/manifest.json" ]; then
    echo "  ArrayRecords already exist — skipping."
else
    echo "  Pretokenizing all 432 pairs..."
    python3 scripts/pretokenize.py \
        --strate_i_config configs/strate_i_1m.yaml \
        --checkpoint checkpoints/strate_i_1m_best.ckpt \
        --data_dir data/ohlcv_1m/ \
        --output_dir data/tokens_1m/ \
        --seq_len 128 \
        --apathy_percentile 10.0 2>&1 | tail -20

    N_TOKENS=\$(ls data/tokens_1m/*.pt 2>/dev/null | wc -l)
    echo "  Tokens: \$N_TOKENS sequences"

    echo "  Converting to ArrayRecords..."
    python3 scripts/convert_pt_to_arrayrecord.py \
        --input data/tokens_1m/ \
        --output data/arrayrecord_1m/ \
        --seq_len 128

    echo "  Uploading ArrayRecords to GCS..."
    gcloud storage rsync -r data/arrayrecord_1m/ ${GCS_BUCKET}/data/arrayrecord_1m/

    echo "  Done. ArrayRecords on GCS."
fi

echo "--- Data preparation complete ---"
REMOTE_DATA
)"
fi

# ── Phase 5: Launch Strate II training (JAX/TPU) ────────────────────────
echo ""
echo "=== Phase 5: Launching Strate II 30M training (JAX/TPU) ==="

RESUME_FLAG="${RESUME}"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" ${WORKER_FLAG} \
    --command="$(cat <<REMOTE_TRAIN
set -euo pipefail
REPO_DIR="\$HOME/Financial_IA"
cd "\$REPO_DIR"
source .venv_tpu/bin/activate

export PYTHONPATH="\$REPO_DIR"
export SCALE_CONFIG="${SCALE_CONFIG}"
export SCALE_TIER="${SCALE_TIER}"
export RESUME="${RESUME_FLAG}"
export GCS_BUCKET="${GCS_BUCKET}"
export TPU_TYPE="${TPU_TYPE}"
export TPU_GEN="v6e"
export JAX_PLATFORMS=tpu

export JAX_COMPILATION_CACHE_DIR="\$HOME/.jax_cache"
mkdir -p "\$JAX_COMPILATION_CACHE_DIR"

# ── XLA flags v6e ──
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

echo "--- [TPU \$(hostname)] Device topology ---"
python3 -c "
import jax
devices = jax.devices()
n = len(devices)
print(f'  {n} {devices[0].platform.upper()} chips ({devices[0].device_kind})')
print(f'  Process: {jax.process_index()}/{jax.process_count()}')
"

# Restore checkpoint if resuming
CKPT_DIR="checkpoints/jax_v6e/${SCALE_TIER}"
mkdir -p "\$CKPT_DIR"
if [ "${RESUME_FLAG}" = "true" ]; then
    if gsutil ls "${GCS_BUCKET}/checkpoints/jax_v6e/${SCALE_TIER}/" &>/dev/null; then
        echo "  Restoring checkpoint from GCS..."
        gsutil -m rsync -r "${GCS_BUCKET}/checkpoints/jax_v6e/${SCALE_TIER}/" "\$CKPT_DIR/"
    fi
fi

# Restore XLA cache
if gsutil ls "${GCS_BUCKET}/xla_cache/v6e_${SCALE_TIER}/" &>/dev/null 2>&1; then
    echo "  Restoring XLA cache from GCS..."
    gsutil -m rsync -r "${GCS_BUCKET}/xla_cache/v6e_${SCALE_TIER}/" "\$JAX_COMPILATION_CACHE_DIR/" 2>/dev/null || true
fi

echo "--- Starting Strate II (30M, ${TPU_TYPE}) ---"
nohup python3 -u scripts/run_training.py > training_1m_30m.log 2>&1 &

TRAIN_PID=\$!
echo "  Training launched (PID: \$TRAIN_PID)"
echo "  Monitor: tail -f ~/Financial_IA/training_1m_30m.log"
echo "--- Launch complete ---"
REMOTE_TRAIN
)"

echo ""
echo "============================================================"
echo " Pipeline launched on ${TPU_NAME} (${TPU_TYPE})"
echo ""
echo " Monitor logs:"
echo "   gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} \\"
echo "     -- tail -f ~/Financial_IA/training_1m_30m.log"
echo ""
echo " SSH into VM:"
echo "   gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}"
echo ""
echo " Delete when done:"
echo "   gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}"
echo "============================================================"
