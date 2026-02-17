#!/bin/bash
###############################################################################
# Financial-IA — Sync code to GCP VM
#
# Usage:
#   ./scripts/sync_gcp.sh training    # Sync to training VM
#   ./scripts/sync_gcp.sh ingest      # Sync to ingest VM
#   ./scripts/sync_gcp.sh training pull  # Pull logs/checkpoints back
###############################################################################

set -euo pipefail

ROLE="${1:-training}"
DIRECTION="${2:-push}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/opt/financial-ia"

# Get VM name and zone from Terraform
cd "$PROJECT_DIR/infra"
VM_NAME="financial-ia-${ROLE}"
ZONE=$(terraform output -raw 2>/dev/null | grep -oP 'zone=\K[^ ]+' || echo "us-central1-a")

# Get external IP
VM_IP=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$ZONE" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)

if [ -z "$VM_IP" ]; then
    echo "Error: Could not find IP for $VM_NAME in zone $ZONE"
    exit 1
fi

echo "VM: $VM_NAME ($VM_IP)"

EXCLUDES=(
    --exclude '.venv/'
    --exclude '__pycache__/'
    --exclude '*.pt'
    --exclude '*.ckpt'
    --exclude '*.zip'
    --exclude '/data/'
    --exclude 'outputs/'
    --exclude 'lightning_logs/'
    --exclude 'tb_logs/'
    --exclude '.git/'
    --exclude 'infra/.terraform/'
    --exclude 'infra/tfplan'
    --exclude 'infra/*.tfstate*'
)

if [ "$DIRECTION" = "push" ]; then
    echo "Pushing code → $VM_NAME:$REMOTE_DIR/"
    rsync -avz --progress \
        "${EXCLUDES[@]}" \
        "$PROJECT_DIR/" \
        "daniel@$VM_IP:$REMOTE_DIR/"
    echo "Done. SSH: ssh daniel@$VM_IP"

elif [ "$DIRECTION" = "pull" ]; then
    echo "Pulling logs/checkpoints ← $VM_NAME"
    mkdir -p "$PROJECT_DIR/tb_logs" "$PROJECT_DIR/checkpoints"

    rsync -avz --progress \
        "daniel@$VM_IP:$REMOTE_DIR/tb_logs/" \
        "$PROJECT_DIR/tb_logs/"

    rsync -avz --progress \
        "daniel@$VM_IP:$REMOTE_DIR/checkpoints/" \
        "$PROJECT_DIR/checkpoints/"

    echo "Done."
else
    echo "Usage: $0 <training|ingest> [push|pull]"
    exit 1
fi
