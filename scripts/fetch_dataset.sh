#!/usr/bin/env bash
# YOLO team side — OPTIONAL inspection tool. Pulls the current sheep-pose
# dataset from the labeling pod down to the YOLO repo so labels can be
# eyeballed locally (open the JPEGs, sanity-check keypoint placements).
#
# NOT the primary handoff path. Training runs on the pod — see:
#   scripts/train_on_pod.sh           (kick off training on the 4090)
#   scripts/sync_weights_from_pod.sh  (pull trained weights down)
#
# Use this script when you want to scroll through the actual labeled
# frames, OR as a backup if you need the dataset off the pod for any
# reason. Running this isn't required for the train loop.
#
# One command:
#   ./fetch_dataset.sh
#
# Does:
#   1. Triggers a batch export on the pod so the dataset is up to date
#      with every labeled video_id (no need for the labeler to manually
#      curl /api/export/keypoints per-video).
#   2. rsync's the dataset directory to the local destination.
#   3. Prints counts (train/val images + labels on disk).
#
# Config: put pod connection info in ~/.sheep-yolo.env OR pass on CLI.
#
#   POD_IP=38.65.239.23
#   POD_SSH_PORT=27921
#   SSH_KEY=~/.ssh/id_ed25519           # defaults to this
#   DATASET=sheep-pose-v0.1               # default dataset name
#   LOCAL_DATASETS_DIR=~/dev/lorewood-advisors/sheep-yolo/datasets   # default
#
# CLI override order: args > env > ~/.sheep-yolo.env defaults.

set -e

# Load config file if present
if [ -f "$HOME/.sheep-yolo.env" ]; then
  set -a
  # shellcheck disable=SC1090,SC1091
  . "$HOME/.sheep-yolo.env"
  set +a
fi

POD_IP="${POD_IP:-}"
POD_SSH_PORT="${POD_SSH_PORT:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
DATASET="${DATASET:-sheep-pose-v0.1}"
LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-$HOME/dev/lorewood-advisors/sheep-yolo/datasets}"

# Positional overrides
POD_IP="${1:-$POD_IP}"
POD_SSH_PORT="${2:-$POD_SSH_PORT}"
DATASET="${3:-$DATASET}"

if [ -z "$POD_IP" ] || [ -z "$POD_SSH_PORT" ]; then
  cat >&2 <<EOF
Missing pod SSH info. Either put in ~/.sheep-yolo.env:

  POD_IP=38.65.239.23
  POD_SSH_PORT=27921

Or pass on CLI:
  $0 <pod-ip> <pod-ssh-port> [dataset-name]
EOF
  exit 1
fi

# Ensure the parent datasets/ exists on a fresh sheep-yolo clone.
mkdir -p "$LOCAL_DATASETS_DIR"
LOCAL_DIR="${LOCAL_DATASETS_DIR%/}/${DATASET}"
mkdir -p "$LOCAL_DIR"

# Step 1: tell the pod to batch-export all labeled videos into the dataset.
echo "[fetch] Triggering batch export on pod for dataset=${DATASET}..."
EXPORT_RESP=$(ssh -p "$POD_SSH_PORT" -i "$SSH_KEY" "root@$POD_IP" \
  "curl -s -X POST 'http://localhost:8000/api/export/keypoints/all?dataset=${DATASET}'")

echo "[fetch] Pod export summary:"
echo "$EXPORT_RESP" | python3 -m json.tool 2>/dev/null || echo "$EXPORT_RESP"

# Step 2: rsync the dataset directory down to local.
POD_EXPORTS="/workspace/SamSeesSheep/data/labels/exports/${DATASET}/"
echo ""
echo "[fetch] rsync'ing ${POD_IP}:${POD_EXPORTS} → ${LOCAL_DIR}/"
rsync -avz --delete \
  -e "ssh -p $POD_SSH_PORT -i $SSH_KEY" \
  "root@${POD_IP}:${POD_EXPORTS}" \
  "${LOCAL_DIR}/"

echo ""
echo "[fetch] Done. Dataset is at ${LOCAL_DIR}"
echo ""
echo "[fetch] Counts:"
echo "  train images: $(find "$LOCAL_DIR/train/images" -type f 2>/dev/null | wc -l)"
echo "  train labels: $(find "$LOCAL_DIR/train/labels" -type f 2>/dev/null | wc -l)"
echo "  val images:   $(find "$LOCAL_DIR/val/images" -type f 2>/dev/null | wc -l)"
echo "  val labels:   $(find "$LOCAL_DIR/val/labels" -type f 2>/dev/null | wc -l)"
echo ""
echo "[fetch] To train (on the pod, not locally — see scripts/train_on_pod.sh):"
echo "  ./scripts/train_on_pod.sh"
