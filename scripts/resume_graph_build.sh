#!/bin/bash
# Resume graph build from where it stopped.
# Resume-safe: skips existing .pt files.
#
# Usage: bash scripts/resume_graph_build.sh
# Monitor: tail -f /tmp/build_graphs.log

cd /home/daniel/Documents/Financial_IA

echo "Current: $(ls data/onchain/graphs/eth/*.pt 2>/dev/null | wc -l) / $(ls data/onchain/raw/eth/*.parquet 2>/dev/null | wc -l) graphs"

nohup bash -c '
cd /home/daniel/Documents/Financial_IA
while true; do
    RAW=$(ls data/onchain/raw/eth/*.parquet 2>/dev/null | wc -l)
    BUILT=$(ls data/onchain/graphs/eth/*.pt 2>/dev/null | wc -l)
    TODO=$((RAW - BUILT))
    echo "[$(date +%H:%M)] raw=$RAW graphs=$BUILT todo=$TODO"
    if [ "$TODO" -gt 100 ]; then
        PYTHONPATH=. .venv/bin/python scripts/build_graphs.py --chain eth --workers 12 2>&1 | tail -3
    elif [ "$TODO" -gt 0 ]; then
        PYTHONPATH=. .venv/bin/python scripts/build_graphs.py --chain eth --workers 8 2>&1 | tail -3
    else
        echo "All graphs built!"
        break
    fi
    sleep 30
done
' > /tmp/build_graphs.log 2>&1 &

echo "Launched (PID: $!). Monitor: tail -f /tmp/build_graphs.log"
