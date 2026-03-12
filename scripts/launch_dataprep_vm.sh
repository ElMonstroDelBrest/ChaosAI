#!/usr/bin/env bash
set -euo pipefail
#
# Launch a Spot VM for high-speed bulk data download + preprocessing.
# Optimized for max bandwidth: n2-standard-8, pd-ssd, europe-west4-a.
#
# Pipeline:
#   1. Download Binance Vision (futures + spot 1min) — aiohttp 512 connections
#   2. Download yfinance (stocks, ETFs, forex, commodities)
#   3. Convert all ZIPs to parquet
#   4. Upload to GCS (free — same region)
#   5. Delete VM
#
# Cost: ~$0.26/hour (n2-standard-8 Spot) × ~2h = ~$0.52 total
#
# Usage:
#   bash scripts/launch_dataprep_vm.sh              # create VM
#   bash scripts/launch_dataprep_vm.sh run           # create + auto-run pipeline
#   bash scripts/launch_dataprep_vm.sh delete         # cleanup

PROJECT="financial-ai-487700"
ZONE="europe-west4-a"
VM_NAME="fin-ia-dataprep-v2"
BUCKET="gs://fin-ia-eu"
MACHINE="n2-standard-8"   # 8 vCPU, 32 GB RAM — good for 512 async connections + parquet conversion
DISK_SIZE="200GB"          # ~25 GB downloads + ~80 GB CSV + ~40 GB parquet + margin
DISK_TYPE="pd-balanced"    # Good enough I/O (quota limit 250 GB SSD in europe-west4)

# ─── Delete ───────────────────────────────────────────────────
if [[ "${1:-}" == "delete" ]]; then
    echo "=== Deleting VM $VM_NAME ==="
    gcloud compute instances delete "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT" --quiet
    echo "VM deleted. Cost stops now."
    exit 0
fi

echo "=== Creating Spot VM: $VM_NAME ($MACHINE, $DISK_SIZE $DISK_TYPE) ==="
echo "=== Zone: $ZONE (co-located with GCS $BUCKET — FREE transfer) ==="
echo "=== Estimated cost: ~\$0.26/hour (Spot) ==="

gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --machine-type="$MACHINE" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size="$DISK_SIZE" \
    --boot-disk-type="$DISK_TYPE" \
    --scopes=storage-rw,logging-write \
    --metadata=startup-script='#!/bin/bash
set -e
echo "=== [$(date)] Startup: installing tools ==="
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git aria2

# Python deps for the downloader
pip3 install -q aiohttp pandas pyarrow tqdm yfinance

echo "=== [$(date)] Tools installed ==="
echo "READY" > /tmp/vm_ready
'

echo ""
echo "=== VM created! Wait ~90s for startup, then: ==="
echo ""
echo "  # SSH in:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "  # Or auto-run the full pipeline (from your local machine):"
echo "  bash scripts/launch_dataprep_vm.sh run"
echo ""

# ─── Auto-run pipeline ───────────────────────────────────────
if [[ "${1:-}" == "run" ]]; then
    echo "=== Waiting for VM startup... ==="
    for i in $(seq 1 30); do
        if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" \
            --command="test -f /tmp/vm_ready && echo OK" 2>/dev/null | grep -q OK; then
            echo "VM ready!"
            break
        fi
        echo "  Waiting... ($i/30)"
        sleep 5
    done

    echo "=== Uploading download script ==="
    gcloud compute scp scripts/download_bulk_free.py \
        "$VM_NAME:~/download_bulk_free.py" \
        --zone="$ZONE" --project="$PROJECT"

    echo "=== Running full pipeline ==="
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --command="
set -e
mkdir -p /data/raw

echo '=== Phase 1: Binance Vision (futures + spot 1min, 512 connections) ==='
python3 ~/download_bulk_free.py \
    --output /data/raw/ \
    --phases binance \
    --workers 512 \
    --interval 1m \
    --start 2017-01 --end 2026-02 \
    --to-parquet --parquet-workers 8

echo '=== Phase 2: yfinance (stocks, ETFs, forex, commodities) ==='
python3 ~/download_bulk_free.py \
    --output /data/raw/ \
    --phases yfinance

echo '=== Phase 3: Upload to GCS (FREE — same region) ==='
# Parquet files (compact, ready for pipeline)
gsutil -m -o 'GSUtil:parallel_composite_upload_threshold=50M' \
    cp -r /data/raw/futures_parquet/ $BUCKET/data/raw/futures_1m_parquet/
gsutil -m -o 'GSUtil:parallel_composite_upload_threshold=50M' \
    cp -r /data/raw/spot_parquet/ $BUCKET/data/raw/spot_1m_parquet/
gsutil -m cp -r /data/raw/yfinance_parquet/ $BUCKET/data/raw/yfinance_parquet/

# Manifest
gsutil cp /data/raw/download_manifest.json $BUCKET/data/raw/download_manifest.json

echo '=== Phase 4: Summary ==='
echo 'Futures parquet:'
ls /data/raw/futures_parquet/ | wc -l
echo 'Spot parquet:'
ls /data/raw/spot_parquet/ | wc -l
echo 'yfinance parquet:'
ls /data/raw/yfinance_parquet/ | wc -l
du -sh /data/raw/*/

echo '=== ALL DONE — delete VM with: bash scripts/launch_dataprep_vm.sh delete ==='
"
fi
