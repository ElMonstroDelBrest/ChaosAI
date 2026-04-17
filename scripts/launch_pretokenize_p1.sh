#!/bin/bash
set -e
cd ~/Financial_IA

# Kill any existing retokenization
pkill -f pretokenize_to_arrayrecord.py 2>/dev/null || true
sleep 1
rm -rf data/arrayrecord_1m_p1/

echo "Starting retokenization with patch_length=1..."
export PYTHONPATH=$PWD
nohup python scripts/pretokenize_to_arrayrecord.py \
    --strate_i_config configs/strate_i_1m_p1.yaml \
    --checkpoint "checkpoints/strate-i-epoch=02-val/loss/total=0.0038.ckpt" \
    --data_dir data/ohlcv_1m/ \
    --output_dir data/arrayrecord_1m_p1/ \
    --seq_len 128 \
    > pretokenize_p1.log 2>&1 &

sleep 2
echo "PID: $(pgrep -f pretokenize_to_arrayrecord | head -1)"
echo "Log: pretokenize_p1.log"
