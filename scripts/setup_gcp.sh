#!/bin/bash
###############################################################################
# Financial-IA — GCP Setup & Deployment
#
# This script:
#   1. Authenticates with GCP (Application Default Credentials)
#   2. Enables required APIs
#   3. Initializes and applies Terraform
#   4. Prints connection info
#
# Usage:
#   chmod +x scripts/setup_gcp.sh
#   ./scripts/setup_gcp.sh <PROJECT_ID>
#
# Prerequisites:
#   - gcloud CLI installed (https://cloud.google.com/sdk/docs/install)
#   - terraform installed (https://developer.hashicorp.com/terraform/install)
#   - SSH key at ~/.ssh/id_ed25519.pub
###############################################################################

set -euo pipefail

# Ensure ~/.local/bin is in PATH (for user-installed terraform)
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

if [ $# -lt 1 ]; then
    echo "Usage: $0 <GCP_PROJECT_ID>"
    echo "Example: $0 my-financial-ia-project"
    exit 1
fi

PROJECT_ID="$1"
REGION="us-central1"
ZONE="us-central1-a"

echo "============================================"
echo " Financial-IA — GCP Setup"
echo " Project: $PROJECT_ID"
echo " Region:  $REGION"
echo "============================================"
echo

# ---------------------------------------------------------------------------
# 1. Authentication
# ---------------------------------------------------------------------------

echo "[1/5] Authenticating with GCP..."
gcloud auth login --quiet 2>/dev/null || true
gcloud auth application-default login --quiet 2>/dev/null || true
gcloud config set project "$PROJECT_ID"
gcloud config set compute/region "$REGION"
gcloud config set compute/zone "$ZONE"
echo "  Done."

# ---------------------------------------------------------------------------
# 2. Enable required APIs
# ---------------------------------------------------------------------------

echo "[2/5] Enabling GCP APIs..."
gcloud services enable \
    compute.googleapis.com \
    storage.googleapis.com \
    iam.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com \
    --quiet
echo "  Done."

# ---------------------------------------------------------------------------
# 3. Terraform init + plan + apply
# ---------------------------------------------------------------------------

echo "[3/5] Initializing Terraform..."
cd "$(dirname "$0")/../infra"

terraform init -upgrade

echo "[4/5] Planning infrastructure..."
terraform plan \
    -var="project_id=$PROJECT_ID" \
    -var="region=$REGION" \
    -var="zone=$ZONE" \
    -out=tfplan

echo
read -p "Apply this plan? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[5/5] Applying infrastructure..."
    terraform apply tfplan
else
    echo "Aborted."
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. Print connection info
# ---------------------------------------------------------------------------

echo
echo "============================================"
echo " Infrastructure deployed!"
echo "============================================"
terraform output
echo
echo "Quick commands:"
echo "  # SSH to ingest VM:"
echo "  $(terraform output -raw ssh_ingest)"
echo
echo "  # SSH to training VM:"
echo "  $(terraform output -raw ssh_training)"
echo
echo "  # Sync code to training VM:"
echo "  ./scripts/sync_gcp.sh training"
echo
echo "  # Run data ingestion from ingest VM:"
echo "  python scripts/gcp_ingest.py --bucket $(terraform output -raw bucket_url | sed 's|gs://||')"
echo
