#!/bin/bash
# Scaling Ladder — wrapper for TPU v5p-32 training across model sizes.
#
# Validates the --scale argument, resolves the YAML config,
# optionally displays config (--dry-run), then delegates to launch_tpu_v5p.sh.
#
# Usage:
#   ./scripts/scaling_ladder.sh --scale=184m              # baseline
#   ./scripts/scaling_ladder.sh --scale=1.5b              # sweet spot
#   ./scripts/scaling_ladder.sh --scale=3b --resume        # resume after preemption
#   ./scripts/scaling_ladder.sh --scale=500m --dry-run     # show config, don't launch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SCALE=""
RESUME=false
DRY_RUN=false
EXTRA_ARGS=()

for arg in "$@"; do
    case $arg in
        --scale=*)   SCALE="${arg#*=}" ;;
        --resume)    RESUME=true ;;
        --dry-run)   DRY_RUN=true ;;
        --skip-create|--train-only) EXTRA_ARGS+=("$arg") ;;
        *) echo "[ERROR] Unknown arg: $arg"; echo "Usage: $0 --scale={184m,500m,1.5b,3b} [--resume] [--dry-run] [--skip-create] [--train-only]"; exit 1 ;;
    esac
done

# Validate --scale
if [ -z "$SCALE" ]; then
    echo "[ERROR] --scale is required."
    echo ""
    echo "Available tiers:"
    echo "  184m  — 184M params, d_model=1024, 24 layers  (validation / baseline)"
    echo "  500m  — 505M params, d_model=1536, 30 layers  (phase transition)"
    echo "  1.5b  — 1.5B params, d_model=2560, 32 layers  (sweet spot)"
    echo "  3b    — 3.1B params, d_model=3072, 48 layers  (full scale)"
    echo ""
    echo "Usage: $0 --scale={184m,500m,1.5b,3b} [--resume] [--dry-run]"
    exit 1
fi

# Resolve config YAML
case "$SCALE" in
    184m)  CONFIG_PATH="configs/scaling/184m.yaml" ;;
    500m)  CONFIG_PATH="configs/scaling/500m.yaml" ;;
    1.5b)  CONFIG_PATH="configs/scaling/1_5b.yaml" ;;
    3b)    CONFIG_PATH="configs/scaling/3b.yaml" ;;
    *)
        echo "[ERROR] Invalid scale: '$SCALE'"
        echo "Valid options: 184m, 500m, 1.5b, 3b"
        exit 1
        ;;
esac

FULL_CONFIG="${REPO_DIR}/${CONFIG_PATH}"
if [ ! -f "$FULL_CONFIG" ]; then
    echo "[ERROR] Config file not found: ${FULL_CONFIG}"
    exit 1
fi

echo "============================================================"
echo " Scaling Ladder — ${SCALE}"
echo "============================================================"
echo "  Config:  ${CONFIG_PATH}"
echo "  Resume:  ${RESUME}"
echo "  Dry run: ${DRY_RUN}"
echo "============================================================"

# Validate config is loadable
echo ""
echo "=== Validating config ==="
cd "$REPO_DIR"
python3 -c "
from src.jax_v6.config import load_config
c = load_config('${CONFIG_PATH}')
m = c.mamba2
t = c.training
e = c.ema
print(f'  d_model:    {m.d_model}')
print(f'  n_layers:   {m.n_layers}')
print(f'  n_heads:    {m.n_heads}')
print(f'  use_remat:  {m.use_remat}')
print(f'  batch_size: {t.batch_size}')
print(f'  lr:         {t.lr}')
print(f'  wd:         {t.weight_decay}')
print(f'  ckpt_every: {t.checkpoint_interval} steps')
print(f'  tau_start:  {e.tau_start}')
print(f'  predictor:  hidden_dim={c.predictor.hidden_dim}')
print('  Config: OK')
"
if [ $? -ne 0 ]; then
    echo "[ERROR] Config validation failed!"
    exit 1
fi

# Dry run — stop here
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "=== Dry run — config is valid. Not launching TPU. ==="
    echo ""
    echo "To launch:"
    echo "  $0 --scale=${SCALE}$([ \"$RESUME\" = true ] && echo ' --resume')"
    exit 0
fi

# Delegate to launch script
echo ""
echo "=== Launching TPU v5p-32 ==="

LAUNCH_ARGS=(--scale="$SCALE")
if [ "$RESUME" = true ]; then
    LAUNCH_ARGS+=(--resume)
fi
LAUNCH_ARGS+=("${EXTRA_ARGS[@]}")

exec "${SCRIPT_DIR}/launch_tpu_v5p.sh" "${LAUNCH_ARGS[@]}"
