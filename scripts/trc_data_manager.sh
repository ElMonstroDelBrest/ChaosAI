#!/bin/bash
###############################################################################
# trc_data_manager.sh — Data Lake Orchestrator for TRC TPU Training
#
# Manages the Google Drive (30 TB Cold) ↔ GCS (Hot, paid) ↔ TPU pipeline.
# Goal: pay GCS only during active training, bypass Drive API rate limits.
#
# Commands:
#   stage   — Drive → GCS (high-perf rclone, 64 parallel transfers)
#   backup  — GCS checkpoints → tar.gz → Drive (avoids Orbax small-file API spam)
#   cleanup — Wipe GCS data+checkpoints (FinOps zero-billing guarantee)
#   status  — Show current GCS usage and estimated monthly cost
#
# Usage:
#   ./scripts/trc_data_manager.sh stage [--dry-run]
#   ./scripts/trc_data_manager.sh backup [--step XXXX | --latest] [--scale s|m|l|xl]
#   ./scripts/trc_data_manager.sh cleanup [--force]
#   ./scripts/trc_data_manager.sh status
#
# Requires: rclone (with Drive remote configured), gsutil, gcloud
###############################################################################
set -euo pipefail

# ── Configuration (override via environment) ─────────────────────────────────

GCS_BUCKET="${GCS_BUCKET:-gs://fin-ia-bucket}"
DRIVE_REMOTE="${DRIVE_REMOTE:-drive}"
DRIVE_ROOT="${DRIVE_ROOT:-ChaosAI_DataLake}"
TPU_ZONE="${TPU_ZONE:-europe-west4-a}"

# Drive paths (inside DRIVE_ROOT)
DRIVE_ARRAYRECORD="${DRIVE_REMOTE}:${DRIVE_ROOT}/03_training_ready/arrayrecords_v5"
DRIVE_CHECKPOINTS="${DRIVE_REMOTE}:${DRIVE_ROOT}/04_checkpoints"

# GCS paths
GCS_DATA="${GCS_BUCKET}/data/arrayrecord"
GCS_CHECKPOINTS="${GCS_BUCKET}/checkpoints/jax_v6"
GCS_XLA_CACHE="${GCS_BUCKET}/xla_cache"

# rclone tuning — saturate bandwidth, avoid Drive API timeouts
RCLONE_TRANSFERS=64
RCLONE_CHECKERS=64
RCLONE_CHUNK_SIZE="128M"
RCLONE_DRIVE_PACER_MIN_SLEEP="10ms"

# Local paths
LOCAL_CKPT_DIR="checkpoints/jax_v6"
TMP_TAR_DIR="/tmp/chaosai_backup"

# ── Helpers ──────────────────────────────────────────────────────────────────

ts() { date -u "+%Y-%m-%d %H:%M:%S UTC"; }
log()  { echo "[$(ts)] $*"; }
warn() { echo "[$(ts)] WARNING: $*" >&2; }
die()  { echo "[$(ts)] FATAL: $*" >&2; exit 1; }

# Strip gs:// prefix for gsutil commands that need bare bucket name
bucket_name() { echo "${GCS_BUCKET}" | sed 's|gs://||'; }

usage() {
    cat <<'USAGE'
Usage: trc_data_manager.sh <command> [options]

Commands:
  stage     Stage training data from Google Drive → GCS
            Options: --dry-run  (preview without copying)

  backup    Archive checkpoints from GCS/local → Google Drive
            Options: --step XXXX   (specific step number)
                     --latest      (most recent checkpoint, default)
                     --scale SIZE  (s|m|l|xl|184m|500m|1_5b|3b, default: auto-detect)

  cleanup   Wipe all data and checkpoints from GCS (FinOps zero-billing)
            Options: --force  (skip confirmation prompt)

  status    Show GCS bucket usage and estimated monthly cost

Environment variables (override defaults):
  GCS_BUCKET     (default: gs://fin-ia-bucket)
  DRIVE_REMOTE   (default: drive)
  DRIVE_ROOT     (default: ChaosAI_DataLake)
  TPU_ZONE       (default: europe-west4-a)
USAGE
    exit 1
}

# ── Region Safety Check ──────────────────────────────────────────────────────

check_region_match() {
    log "Verifying GCS bucket region matches TPU zone..."

    # Extract bucket location (e.g., EUROPE-WEST4)
    local bucket_location
    bucket_location=$(gsutil ls -L -b "${GCS_BUCKET}" 2>/dev/null \
        | grep -i "Location constraint" \
        | awk '{print tolower($NF)}') || true

    if [[ -z "$bucket_location" ]]; then
        warn "Could not determine bucket location. Verify manually!"
        warn "  gsutil ls -L -b ${GCS_BUCKET}"
        return 0
    fi

    # Extract region from zone (europe-west4-a → europe-west4)
    local tpu_region="${TPU_ZONE%-*}"

    if [[ "$bucket_location" != "$tpu_region" ]]; then
        die "REGION MISMATCH — bucket '${GCS_BUCKET}' is in '${bucket_location}' but TPU zone is '${TPU_ZONE}' (region: ${tpu_region}).
  Inter-region egress will be billed on EVERY Grain batch load!
  Fix: recreate bucket in ${tpu_region} or change TPU_ZONE.
  To force anyway: set SKIP_REGION_CHECK=1"
    fi

    log "  Bucket region: ${bucket_location} ✓ (matches TPU zone ${TPU_ZONE})"
}

# ── Command: stage ───────────────────────────────────────────────────────────

cmd_stage() {
    local dry_run=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) dry_run=true; shift ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    log "═══════════════════════════════════════════════════════════════"
    log "STAGE: Google Drive → GCS (high-performance transfer)"
    log "  Source: ${DRIVE_ARRAYRECORD}/"
    log "  Dest:   ${GCS_DATA}/"
    log "  Mode:   ${RCLONE_TRANSFERS} parallel transfers, ${RCLONE_CHUNK_SIZE} chunks"
    log "═══════════════════════════════════════════════════════════════"

    # Safety: region check (skip if SKIP_REGION_CHECK=1)
    if [[ "${SKIP_REGION_CHECK:-0}" != "1" ]]; then
        check_region_match
    fi

    # Verify rclone remote exists
    if ! rclone listremotes 2>/dev/null | grep -q "^${DRIVE_REMOTE}:$"; then
        die "rclone remote '${DRIVE_REMOTE}:' not found. Run: rclone config"
    fi

    # Verify source exists on Drive
    log "Checking Drive source..."
    if ! rclone lsd "${DRIVE_ARRAYRECORD}/" &>/dev/null; then
        # lsd might fail on files-only dirs, try lsf
        if ! rclone lsf "${DRIVE_ARRAYRECORD}/" --max-depth 1 2>/dev/null | head -1 | grep -q .; then
            die "Source not found on Drive: ${DRIVE_ARRAYRECORD}/"
        fi
    fi

    local rclone_flags=(
        --transfers "$RCLONE_TRANSFERS"
        --checkers "$RCLONE_CHECKERS"
        --drive-chunk-size "$RCLONE_CHUNK_SIZE"
        --drive-pacer-min-sleep "$RCLONE_DRIVE_PACER_MIN_SLEEP"
        --drive-acknowledge-abuse      # skip Drive virus scan prompt on large files
        --fast-list                    # reduce API calls (use ListR)
        --stats 30s                    # progress every 30s
        --stats-one-line               # compact progress
        --log-level INFO
        -P                             # real-time progress bar
    )

    if $dry_run; then
        rclone_flags+=(--dry-run)
        log "DRY RUN — no files will be copied"
    fi

    log "Starting rclone copy..."
    local t_start
    t_start=$(date +%s)

    rclone copy \
        "${DRIVE_ARRAYRECORD}/" \
        "${GCS_DATA}/" \
        "${rclone_flags[@]}"

    local t_end
    t_end=$(date +%s)
    local elapsed=$(( t_end - t_start ))

    log "Stage complete in ${elapsed}s"

    # Verify shard count on GCS
    local shard_count
    shard_count=$(gsutil ls "${GCS_DATA}/*.arrayrecord" 2>/dev/null | wc -l) || true
    log "ArrayRecord shards on GCS: ${shard_count}"

    if [[ "$shard_count" -eq 0 ]] && ! $dry_run; then
        warn "No .arrayrecord files found on GCS after staging!"
    fi
}

# ── Command: backup ──────────────────────────────────────────────────────────

cmd_backup() {
    local target_step="latest"
    local scale_tier="auto"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --step)   target_step="$2"; shift 2 ;;
            --latest) target_step="latest"; shift ;;
            --scale)  scale_tier="$2"; shift 2 ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    log "═══════════════════════════════════════════════════════════════"
    log "BACKUP: Checkpoints → tar.gz → Google Drive"
    log "═══════════════════════════════════════════════════════════════"

    # Auto-detect scale tier from existing checkpoints
    if [[ "$scale_tier" == "auto" ]]; then
        # Look for scale dirs in local checkpoints
        if [[ -d "${LOCAL_CKPT_DIR}" ]]; then
            local detected
            detected=$(ls -1d "${LOCAL_CKPT_DIR}"/step_* 2>/dev/null | head -1 || true)
            if [[ -n "$detected" ]]; then
                scale_tier="default"
            fi
        fi
        # Fallback: check GCS
        if [[ "$scale_tier" == "auto" ]]; then
            scale_tier="default"
            log "  Scale tier auto-detect: using 'default'"
        fi
    fi

    # Normalize scale tier names
    case "$scale_tier" in
        s|S|15m)   scale_tier="s_15m" ;;
        m|M|150m)  scale_tier="m_150m" ;;
        l|L|1b)    scale_tier="l_1b" ;;
        xl|XL|7b)  scale_tier="xl_7b" ;;
    esac

    local drive_dest="${DRIVE_CHECKPOINTS}/${scale_tier}"

    # Find checkpoint directory
    local ckpt_source=""

    if [[ "$target_step" == "latest" ]]; then
        # Find highest step number locally
        if [[ -d "${LOCAL_CKPT_DIR}" ]]; then
            ckpt_source=$(ls -1d "${LOCAL_CKPT_DIR}"/step_* 2>/dev/null \
                | sort -t_ -k2 -n | tail -1 || true)
        fi
        # If not local, try GCS
        if [[ -z "$ckpt_source" ]]; then
            log "No local checkpoints found, syncing latest from GCS..."
            local latest_gcs
            latest_gcs=$(gsutil ls -d "${GCS_CHECKPOINTS}/step_*" 2>/dev/null \
                | sort -t_ -k2 -n | tail -1 || true)
            if [[ -n "$latest_gcs" ]]; then
                target_step=$(echo "$latest_gcs" | grep -oP 'step_\K[0-9]+')
                mkdir -p "${LOCAL_CKPT_DIR}/step_${target_step}"
                gsutil -m rsync -r "${latest_gcs}" "${LOCAL_CKPT_DIR}/step_${target_step}/"
                ckpt_source="${LOCAL_CKPT_DIR}/step_${target_step}"
            fi
        fi
    else
        ckpt_source="${LOCAL_CKPT_DIR}/step_${target_step}"
        if [[ ! -d "$ckpt_source" ]]; then
            # Try GCS
            log "Step ${target_step} not found locally, pulling from GCS..."
            mkdir -p "$ckpt_source"
            gsutil -m rsync -r "${GCS_CHECKPOINTS}/step_${target_step}/" "$ckpt_source/" \
                || die "Checkpoint step_${target_step} not found on GCS either"
        fi
    fi

    if [[ -z "$ckpt_source" ]] || [[ ! -d "$ckpt_source" ]]; then
        die "No checkpoint found (local: ${LOCAL_CKPT_DIR}/step_*, GCS: ${GCS_CHECKPOINTS}/step_*)"
    fi

    # Extract step number from path
    local step_num
    step_num=$(basename "$ckpt_source" | grep -oP 'step_\K[0-9]+' || echo "unknown")

    log "  Checkpoint: ${ckpt_source}"
    log "  Step:       ${step_num}"
    log "  Drive dest: ${drive_dest}/"

    # Count files — Orbax generates thousands of small files per chip
    local file_count
    file_count=$(find "$ckpt_source" -type f | wc -l)
    local total_size
    total_size=$(du -sh "$ckpt_source" | cut -f1)
    log "  Files: ${file_count} (${total_size} total)"
    log "  Archiving to avoid Drive API rate limits (${file_count} files → 1 tar.gz)..."

    # Tar archive — single file for Drive, avoids API spam
    mkdir -p "$TMP_TAR_DIR"
    local tar_name="checkpoint_step_${step_num}_$(date -u +%Y%m%d_%H%M%S).tar.gz"
    local tar_path="${TMP_TAR_DIR}/${tar_name}"

    tar -czf "$tar_path" -C "$(dirname "$ckpt_source")" "$(basename "$ckpt_source")"

    local tar_size
    tar_size=$(du -sh "$tar_path" | cut -f1)
    log "  Archive: ${tar_name} (${tar_size})"

    # Upload to Drive via rclone
    log "  Uploading to Drive..."
    rclone copy \
        "$tar_path" \
        "${drive_dest}/" \
        --transfers 4 \
        --drive-chunk-size "$RCLONE_CHUNK_SIZE" \
        --drive-pacer-min-sleep "$RCLONE_DRIVE_PACER_MIN_SLEEP" \
        -P

    log "  Backup complete: ${drive_dest}/${tar_name}"

    # Cleanup temp tar
    rm -f "$tar_path"
    log "  Temp archive cleaned"
}

# ── Command: cleanup ─────────────────────────────────────────────────────────

cmd_cleanup() {
    local force=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force) force=true; shift ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    log "═══════════════════════════════════════════════════════════════"
    log "CLEANUP: Wipe GCS data + checkpoints (FinOps zero-billing)"
    log "═══════════════════════════════════════════════════════════════"

    # Show what will be deleted
    log "Paths to delete:"
    local paths_to_delete=(
        "${GCS_BUCKET}/data/"
        "${GCS_CHECKPOINTS}/"
        "${GCS_XLA_CACHE}/"
    )

    local total_objects=0
    for path in "${paths_to_delete[@]}"; do
        local count
        count=$(gsutil ls -r "$path" 2>/dev/null | wc -l) || count=0
        log "  ${path}  (${count} objects)"
        total_objects=$(( total_objects + count ))
    done

    if [[ "$total_objects" -eq 0 ]]; then
        log "Bucket is already clean. Nothing to delete."
        log "GCS storage cost: \$0/month"
        return 0
    fi

    # Show current storage usage
    local bucket_size
    bucket_size=$(gsutil du -s "${GCS_BUCKET}" 2>/dev/null | awk '{print $1}') || bucket_size=0
    local bucket_size_gb=$(( bucket_size / 1073741824 ))
    local monthly_cost
    monthly_cost=$(echo "scale=2; ${bucket_size_gb} * 0.02" | bc 2>/dev/null || echo "?")

    log ""
    log "Current GCS usage: ${bucket_size_gb} GB (~\$${monthly_cost}/month)"
    log "Total objects to delete: ${total_objects}"
    log ""

    # Confirmation
    if ! $force; then
        echo -n "[$(ts)] Are you sure you want to delete ALL data from GCS? (y/N): "
        read -r confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            log "Cleanup cancelled."
            return 0
        fi
    fi

    # Delete
    log "Deleting..."
    for path in "${paths_to_delete[@]}"; do
        if gsutil ls "$path" &>/dev/null; then
            gsutil -m rm -rf "$path" || warn "Failed to fully clean: $path"
            log "  Deleted: $path"
        else
            log "  Skipped (empty): $path"
        fi
    done

    log ""
    log "═══════════════════════════════════════════════════════════════"
    log "Cleanup complete. GCS storage cost: \$0/month"
    log "═══════════════════════════════════════════════════════════════"
}

# ── Command: status ──────────────────────────────────────────────────────────

cmd_status() {
    log "═══════════════════════════════════════════════════════════════"
    log "GCS BUCKET STATUS: ${GCS_BUCKET}"
    log "═══════════════════════════════════════════════════════════════"

    # Bucket location
    local location
    location=$(gsutil ls -L -b "${GCS_BUCKET}" 2>/dev/null \
        | grep -i "Location constraint" \
        | awk '{print $NF}') || location="unknown"
    local storage_class
    storage_class=$(gsutil ls -L -b "${GCS_BUCKET}" 2>/dev/null \
        | grep -i "Storage class" \
        | awk '{print $NF}') || storage_class="unknown"

    log "  Location:      ${location}"
    log "  Storage class: ${storage_class}"
    log "  TPU zone:      ${TPU_ZONE}"

    # Region match check
    local tpu_region="${TPU_ZONE%-*}"
    local bucket_loc_lower
    bucket_loc_lower=$(echo "$location" | tr '[:upper:]' '[:lower:]')
    if [[ "$bucket_loc_lower" == "$tpu_region" ]]; then
        log "  Region match:  YES (zero egress)"
    else
        warn "Region mismatch! Bucket=${location}, TPU=${TPU_ZONE} — egress fees apply!"
    fi

    log ""

    # Per-prefix breakdown
    local prefixes=("data/arrayrecord" "data/tokens_v5" "checkpoints" "xla_cache")
    for prefix in "${prefixes[@]}"; do
        local size_info
        size_info=$(gsutil du -s "${GCS_BUCKET}/${prefix}/" 2>/dev/null) || size_info="0 "
        local bytes
        bytes=$(echo "$size_info" | awk '{print $1}')
        local gb
        gb=$(echo "scale=2; ${bytes:-0} / 1073741824" | bc 2>/dev/null || echo "0")
        local cost
        cost=$(echo "scale=3; ${gb} * 0.02" | bc 2>/dev/null || echo "0")
        log "  ${prefix}/:  ${gb} GB  (~\$${cost}/month)"
    done

    log ""

    # Total
    local total_info
    total_info=$(gsutil du -s "${GCS_BUCKET}" 2>/dev/null) || total_info="0 "
    local total_bytes
    total_bytes=$(echo "$total_info" | awk '{print $1}')
    local total_gb
    total_gb=$(echo "scale=2; ${total_bytes:-0} / 1073741824" | bc 2>/dev/null || echo "0")
    local total_cost
    total_cost=$(echo "scale=2; ${total_gb} * 0.02" | bc 2>/dev/null || echo "0")

    log "  TOTAL: ${total_gb} GB — estimated \$${total_cost}/month (Standard storage)"
    log ""

    # Drive backup status
    log "Drive backups (${DRIVE_CHECKPOINTS}/):"
    rclone lsf "${DRIVE_CHECKPOINTS}/" --max-depth 2 2>/dev/null \
        | head -20 \
        | while read -r line; do log "  ${line}"; done \
        || log "  (no backups found or Drive remote not configured)"
}

# ── Main ─────────────────────────────────────────────────────────────────────

[[ $# -lt 1 ]] && usage

COMMAND="$1"
shift

case "$COMMAND" in
    stage)   cmd_stage   "$@" ;;
    backup)  cmd_backup  "$@" ;;
    cleanup) cmd_cleanup "$@" ;;
    status)  cmd_status  "$@" ;;
    -h|--help|help) usage ;;
    *) die "Unknown command: $COMMAND (use: stage, backup, cleanup, status)" ;;
esac
