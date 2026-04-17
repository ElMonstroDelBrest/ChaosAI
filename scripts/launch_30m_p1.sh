#!/bin/bash
set -e
cd ~/Financial_IA

echo "=== Strate II (30M) training with patch_length=1 data ==="
echo "Config: configs/scaling/v6e_30m.yaml"
echo "Data: data/arrayrecord_1m_p1/ (4.06M records)"
echo ""

export PYTHONPATH=$PWD
export SCALE_CONFIG=configs/scaling/v6e_30m.yaml
export SCALE_TIER=30m
export TPU_TYPE=v6e-8
export TPU_GEN=v6e
export JAX_PLATFORMS=tpu
export TF_CPP_MIN_LOG_LEVEL=2
export DISABLE_CKPT=false

# Apply orbax patch if needed (JAX 0.6.x compat)
ORBAX_FILE=~/.local/lib/python3.10/site-packages/orbax/checkpoint/_src/serialization/replica_slices.py
if grep -q "with jax.sharding.set_mesh" "$ORBAX_FILE" 2>/dev/null; then
    echo "Applying orbax patch..."
    sed -i 's/with jax.sharding.set_mesh(mesh):/if True:  # patched: skip set_mesh (JAX 0.6.x compat)/g' "$ORBAX_FILE"
    echo "Patched $ORBAX_FILE"
fi

nohup python scripts/run_training.py \
    > train_30m_p1.log 2>&1 &

sleep 2
PID=$(pgrep -f "run_training.py" | head -1)
echo "PID: $PID"
echo "Log: train_30m_p1.log"
echo "Monitor: tail -f train_30m_p1.log"
