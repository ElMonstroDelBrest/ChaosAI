#!/bin/bash
set -e
cd ~/Financial_IA
export PYTHONPATH=$PWD

echo "============================================="
echo "  Out-of-Sample Evaluation Pipeline"
echo "============================================="
echo ""

# ── Step 1: Download fresh data ──
echo "[1/3] Downloading fresh 1min candles (last 2 days, 50 pairs)..."
python scripts/download_eval_data.py \
    --days 2 \
    --output_dir data/ohlcv_eval/
echo ""
echo "Downloaded $(ls data/ohlcv_eval/*.pt 2>/dev/null | wc -l) pairs"
echo ""

# ── Step 2: Tokenize with Strate I ──
echo "[2/3] Tokenizing with Strate I (patch_length=1)..."
rm -rf data/arrayrecord_eval/
python scripts/pretokenize_to_arrayrecord.py \
    --strate_i_config configs/strate_i_1m_p1.yaml \
    --checkpoint "checkpoints/strate-i-epoch=02-val/loss/total=0.0038.ckpt" \
    --data_dir data/ohlcv_eval/ \
    --output_dir data/arrayrecord_eval/ \
    --seq_len 128
echo ""

# ── Step 3: Eval ──
echo "[3/3] Running OOS evaluation with 26M checkpoint..."
export SCALE_CONFIG=configs/scaling/v6e_26m.yaml
export SCALE_TIER=26m
export TPU_GEN=v6e
export TPU_TYPE=v6e-8
export JAX_PLATFORMS=tpu
export TF_CPP_MIN_LOG_LEVEL=2
export EVAL_DATA_DIR=data/arrayrecord_eval/
export TRAIN_LOSS=1644

# Apply orbax patch if needed
ORBAX_FILE=~/.local/lib/python3.10/site-packages/orbax/checkpoint/_src/serialization/replica_slices.py
if grep -q "with jax.sharding.set_mesh" "$ORBAX_FILE" 2>/dev/null; then
    echo "Applying orbax patch..."
    sed -i 's/with jax.sharding.set_mesh(mesh):/if True:  # patched: skip set_mesh (JAX 0.6.x compat)/g' "$ORBAX_FILE"
fi

python scripts/eval_oos.py

echo ""
echo "============================================="
echo "  Evaluation complete"
echo "============================================="
