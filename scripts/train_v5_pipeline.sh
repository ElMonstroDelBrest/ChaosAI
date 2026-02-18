#!/bin/bash
###############################################################################
# Financial-IA v5 — Full Training Pipeline (H100)
#
# Expands from 15 pairs / 2 yrs to ~299 pairs / 2-6 yrs.
# Retrains everything from scratch with 10-20x more data.
#
# Phases:
#   1. Download all Binance Futures pairs + convert parquet -> .pt
#   2. Re-train Strate I (VQ-VAE) + re-tokenize 299 pairs
#   3. Train Strate II (Fin-JEPA) from scratch, 300 epochs
#   4. Precompute trajectory buffer (3000 multiverse + 2000 historical)
#   5. Train PPO (2M steps, 8x SubprocVecEnv, larger policy)
#   6. Demo + verification + GCS sync
#
# Usage:
#   chmod +x scripts/train_v5_pipeline.sh
#   nohup ./scripts/train_v5_pipeline.sh > training_v5.log 2>&1 &
#
#   # Skip data download (already done locally):
#   ./scripts/train_v5_pipeline.sh --skip-download
#
#   # Skip to specific phase:
#   ./scripts/train_v5_pipeline.sh --start-phase 3
###############################################################################

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/Financial_IA}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

LOG="training_v5.log"
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Parse flags
SKIP_DOWNLOAD=false
START_PHASE=1
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --start-phase=*) START_PHASE="${arg#*=}" ;;
        --start-phase) shift; START_PHASE="$1" ;;
    esac
done

# Ensure 'python' exists
PYTHON=$(which python3 2>/dev/null || which python)

log "=== Financial-IA v5 Pipeline Starting ==="
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Start phase: ${START_PHASE}"
log "Skip download: ${SKIP_DOWNLOAD}"

# ─── Phase 1: Data Expansion ────────────────────────────────────────────────

if [ "$START_PHASE" -le 1 ]; then
    log "=== Phase 1a: Download all Binance Futures pairs ==="

    if [ "$SKIP_DOWNLOAD" = true ]; then
        log "  Skipping download (--skip-download)"
    else
        $PYTHON scripts/download_massive_data.py \
            --interval 1h \
            --output_dir data/raw \
            --max_concurrent 12 \
            2>&1 | tee -a "$LOG"
        log "  Download complete: $(ls data/raw/1h/*.parquet 2>/dev/null | wc -l) parquet files"
    fi

    log "=== Phase 1b: Convert parquet -> .pt tensors ==="
    $PYTHON scripts/convert_parquet_to_pt.py \
        --input_dir data/raw/1h \
        --output_dir data/ohlcv_v5 \
        2>&1 | tee -a "$LOG"
    log "  Conversion complete: $(ls data/ohlcv_v5/*.pt 2>/dev/null | wc -l) .pt files"
fi

# ─── Phase 2: Re-train Strate I + Re-tokenize ──────────────────────────────

if [ "$START_PHASE" -le 2 ]; then
    log "=== Phase 2a: Re-train Strate I (VQ-VAE) on expanded dataset ==="
    $PYTHON scripts/train_strate_i.py \
        --config configs/strate_i_binance.yaml \
        2>&1 | tee -a "$LOG"
    log "  Strate I training complete."

    # Find best Strate I checkpoint (named strate-i-epoch=*, NOT in strate_ii/ dir)
    STRATE_I_CKPT=$(find checkpoints -maxdepth 1 -name 'strate-i-*' -name '*.ckpt' \
        -printf '%f\t%p\n' 2>/dev/null | sort | tail -1 | cut -f2)
    if [ -z "${STRATE_I_CKPT:-}" ]; then
        STRATE_I_CKPT=$(find checkpoints -maxdepth 1 -name 'strate-i-*' -name '*.ckpt' | head -1)
    fi
    log "  Best Strate I checkpoint: $STRATE_I_CKPT"

    log "=== Phase 2b: Tokenize 299 pairs (seq_len=128) ==="
    $PYTHON scripts/pretokenize.py \
        --strate_i_config configs/strate_i_binance.yaml \
        --checkpoint "$STRATE_I_CKPT" \
        --data_dir data/ohlcv_v5 \
        --output_dir data/tokens_v5/ \
        --seq_len 128 \
        2>&1 | tee -a "$LOG"
    log "  Tokenization complete: $(ls data/tokens_v5/*.pt 2>/dev/null | wc -l) token files"
fi

# ─── Phase 3: Train Strate II (Fin-JEPA) from scratch ──────────────────────

if [ "$START_PHASE" -le 3 ]; then
    log "=== Phase 3: Train Strate II (Fin-JEPA) — 300 epochs ==="
    $PYTHON scripts/train_strate_ii.py \
        --config configs/strate_ii.yaml \
        --compile \
        --no_resume \
        2>&1 | tee -a "$LOG"
    log "  Strate II training complete."
fi

# Find best Strate I and II checkpoints (needed for phases 4+)
STRATE_I_CKPT=$(find checkpoints -maxdepth 1 -name 'strate-i-*' -name '*.ckpt' \
    -printf '%f\t%p\n' 2>/dev/null | sort | tail -1 | cut -f2)
if [ -z "${STRATE_I_CKPT:-}" ]; then
    STRATE_I_CKPT=$(find checkpoints -maxdepth 1 -name 'strate-i-*' -name '*.ckpt' | head -1)
fi

STRATE_II_BEST=$(find checkpoints/strate_ii -name '*.ckpt' ! -name 'last*' \
    -printf '%f\t%p\n' 2>/dev/null | sort | head -1 | cut -f2)
if [ -z "${STRATE_II_BEST:-}" ]; then
    STRATE_II_BEST="checkpoints/strate_ii/last.ckpt"
fi
log "  Strate I checkpoint: ${STRATE_I_CKPT:-NOT FOUND}"
log "  Strate II checkpoint: ${STRATE_II_BEST:-NOT FOUND}"

# ─── Phase 4: Precompute trajectory buffer ──────────────────────────────────

if [ "$START_PHASE" -le 4 ]; then
    log "=== Phase 4a: Precompute 3000 multiverse episodes ==="
    $PYTHON scripts/precompute_trajectories.py \
        --strate_i_checkpoint "$STRATE_I_CKPT" \
        --strate_ii_checkpoint "$STRATE_II_BEST" \
        --token_dir data/tokens_v5/ \
        --ohlcv_dir data/ohlcv_v5/ \
        --output_dir data/trajectory_buffer_v5_multi/ \
        --n_episodes 3000 \
        --stratified \
        2>&1 | tee -a "$LOG"
    log "  Multiverse buffer: $(ls data/trajectory_buffer_v5_multi/*.pt 2>/dev/null | wc -l) episodes"

    log "=== Phase 4b: Precompute 2000 historical episodes ==="
    $PYTHON scripts/precompute_trajectories.py \
        --historical \
        --strate_i_checkpoint "$STRATE_I_CKPT" \
        --strate_ii_checkpoint "$STRATE_II_BEST" \
        --token_dir data/tokens_v5/ \
        --ohlcv_dir data/ohlcv_v5/ \
        --output_dir data/trajectory_buffer_v5_hist/ \
        --n_episodes 2000 \
        --stratified \
        2>&1 | tee -a "$LOG"
    log "  Historical buffer: $(ls data/trajectory_buffer_v5_hist/*.pt 2>/dev/null | wc -l) episodes"

    log "=== Phase 4c: Merge buffers into data/trajectory_buffer_v5/ ==="
    mkdir -p data/trajectory_buffer_v5

    # Copy multiverse episodes (episode_00000 .. episode_02999)
    cp data/trajectory_buffer_v5_multi/*.pt data/trajectory_buffer_v5/

    # Rename historical episodes with offset to avoid collision
    OFFSET=$(ls data/trajectory_buffer_v5_multi/*.pt 2>/dev/null | wc -l)
    for f in data/trajectory_buffer_v5_hist/episode_*.pt; do
        [ -f "$f" ] || continue
        base=$(basename "$f")
        # Extract the episode number
        num=$(echo "$base" | sed 's/episode_\([0-9]*\)\.pt/\1/')
        new_num=$((10#$num + OFFSET))
        new_name=$(printf "episode_%05d.pt" "$new_num")
        cp "$f" "data/trajectory_buffer_v5/$new_name"
    done
    log "  Merged buffer: $(ls data/trajectory_buffer_v5/*.pt 2>/dev/null | wc -l) total episodes"
fi

# ─── Phase 5: Train PPO ────────────────────────────────────────────────────

if [ "$START_PHASE" -le 5 ]; then
    log "=== Phase 5: Train PPO (2M steps, 8x SubprocVecEnv) ==="
    $PYTHON scripts/train_strate_iv.py \
        --config configs/strate_iv.yaml \
        --buffer_dir data/trajectory_buffer_v5/ \
        --total_timesteps 2000000 \
        --n_envs 8 \
        --no_resume \
        2>&1 | tee -a "$LOG"
    log "  PPO training complete."
fi

# ─── Phase 6: Verification + Demo + GCS sync ───────────────────────────────

log "=== Phase 6: Verification ==="

# Check file counts
log "  OHLCV files:     $(ls data/ohlcv_v5/*.pt 2>/dev/null | wc -l)"
log "  Token files:     $(ls data/tokens_v5/*.pt 2>/dev/null | wc -l)"
log "  Buffer episodes: $(ls data/trajectory_buffer_v5/*.pt 2>/dev/null | wc -l)"

# Check best model + vecnormalize
BEST_DIR="tb_logs/strate_iv_v5/best_model"
if [ -f "$BEST_DIR/best_model.zip" ] && [ -f "$BEST_DIR/vecnormalize.pkl" ]; then
    log "  best_model.zip + vecnormalize.pkl present in $BEST_DIR"
else
    log "  WARNING: Missing files in $BEST_DIR!"
    ls -la "$BEST_DIR/" 2>/dev/null || true
fi

# Run demo
log "  Running demo..."
$PYTHON scripts/demo_results.py \
    --model_path "$BEST_DIR/best_model.zip" \
    --buffer_dir data/trajectory_buffer_v5/ \
    --output outputs/demo_v5.png \
    --n_demos 5 \
    2>&1 | tee -a "$LOG" || log "  Demo failed (non-critical)"

log "  Demo plots saved to outputs/"

# GCS sync
log "=== Syncing to GCS ==="
export TRAIN_VERSION=v5
bash scripts/sync_checkpoints_gcs.sh 2>&1 | tee -a "$LOG" || log "  GCS sync failed (non-critical)"

log "=== v5 Pipeline Complete ==="
log "Outputs:"
log "  Model:        $BEST_DIR/best_model.zip"
log "  VecNormalize: $BEST_DIR/vecnormalize.pkl"
log "  Buffer:       data/trajectory_buffer_v5/ ($(ls data/trajectory_buffer_v5/*.pt 2>/dev/null | wc -l) episodes)"
log "  Demos:        outputs/demo_v5*.png"
log "  Log:          $LOG"
