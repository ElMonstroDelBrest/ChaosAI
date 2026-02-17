#!/bin/bash
###############################################################################
# Financial-IA — GCS Checkpoint Sync (Cron-friendly)
#
# Syncs local checkpoints and logs to GCS bucket every run.
# Designed to run as a cron job every 10 minutes on Spot VMs.
#
# Setup (on the VM):
#   chmod +x scripts/sync_checkpoints_gcs.sh
#
#   # Run once manually:
#   ./scripts/sync_checkpoints_gcs.sh
#
#   # Setup cron (every 10 min):
#   (crontab -l 2>/dev/null; echo "*/10 * * * * /home/daniel/Financial_IA/scripts/sync_checkpoints_gcs.sh >> /tmp/gcs_sync.log 2>&1") | crontab -
#
#   # Or run in background loop:
#   ./scripts/sync_checkpoints_gcs.sh --loop &
###############################################################################

set -euo pipefail

BUCKET="${GCS_BUCKET:-gs://financial-ia-datalake}"
PROJECT_DIR="${PROJECT_DIR:-/home/daniel/Financial_IA}"
SYNC_INTERVAL=600  # 10 minutes (for --loop mode)

echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] GCS checkpoint sync starting..."

sync_to_gcs() {
    # Sync Strate II checkpoints
    if [ -d "${PROJECT_DIR}/checkpoints/strate_ii" ]; then
        gsutil -m rsync -r \
            "${PROJECT_DIR}/checkpoints/strate_ii/" \
            "${BUCKET}/checkpoints/strate_ii/" \
            2>&1 | tail -3
        echo "  Synced strate_ii checkpoints"
    fi

    # Sync Strate IV checkpoints
    if [ -d "${PROJECT_DIR}/tb_logs/strate_iv" ]; then
        gsutil -m rsync -r \
            "${PROJECT_DIR}/tb_logs/strate_iv/" \
            "${BUCKET}/checkpoints/strate_iv/" \
            2>&1 | tail -3
        echo "  Synced strate_iv checkpoints"
    fi

    # Sync TensorBoard logs (for remote monitoring)
    if [ -d "${PROJECT_DIR}/tb_logs" ]; then
        gsutil -m rsync -r \
            "${PROJECT_DIR}/tb_logs/" \
            "${BUCKET}/tb_logs/" \
            2>&1 | tail -3
        echo "  Synced TensorBoard logs"
    fi

    # Sync training log
    if [ -f "${PROJECT_DIR}/training_h100.log" ]; then
        gsutil cp "${PROJECT_DIR}/training_h100.log" \
            "${BUCKET}/logs/training_h100.log" \
            2>/dev/null
    fi

    echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] Sync complete."
}

# Main logic
if [ "${1:-}" = "--loop" ]; then
    echo "Running in loop mode (every ${SYNC_INTERVAL}s)..."
    while true; do
        sync_to_gcs || echo "  WARNING: sync failed, will retry"
        sleep "$SYNC_INTERVAL"
    done
else
    sync_to_gcs
fi
