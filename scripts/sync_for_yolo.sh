#!/usr/bin/env bash
# YOLO team side — pull the current sheep-pose dataset from the labeling
# pod down to the local training machine. One command:
#
#   ./sync_for_yolo.sh
#
# Does:
#   1. Triggers a batch export on the pod so the dataset is up to date
#      with every labeled video_id (no need for the labeler to manually
#      curl /api/export/keypoints per-video)
#   2. rsync's the dataset directory to the local destination
#   3. Prints the Ultralytics yolo train command, ready to copy-paste
#
# Config: put pod connection info in ~/.sheep-yolo.env OR pass on CLI:
#
#   POD_IP=38.65.239.23
#   POD_SSH_PORT=27921
#   SSH_KEY=~/.ssh/id_ed25519           # defaults to this
#   DATASET=sheep-pose-v0                # default dataset name
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
DATASET="${DATASET:-sheep-pose-v0}"
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

LOCAL_DIR="${LOCAL_DATASETS_DIR%/}/${DATASET}"
mkdir -p "$LOCAL_DIR"

# Step 1: tell the pod to batch-export all labeled videos into the dataset.
# The endpoint reads every data/labels/*/review.json and appends results.
echo "[sync] Triggering batch export on pod..."
EXPORT_RESP=$(ssh -p "$POD_SSH_PORT" -i "$SSH_KEY" "root@$POD_IP" \
  "curl -s -X POST 'http://localhost:8000/api/export/keypoints/all?dataset=${DATASET}'")

echo "[sync] Pod export summary:"
echo "$EXPORT_RESP" | python3 -m json.tool 2>/dev/null || echo "$EXPORT_RESP"

# Step 2: rsync the dataset directory down to local.
POD_EXPORTS="/workspace/SamSeesSheep/data/labels/exports/${DATASET}/"
echo ""
echo "[sync] rsync'ing ${POD_IP}:${POD_EXPORTS} → ${LOCAL_DIR}/"
rsync -avz --delete \
  -e "ssh -p $POD_SSH_PORT -i $SSH_KEY" \
  "root@${POD_IP}:${POD_EXPORTS}" \
  "${LOCAL_DIR}/"

echo ""
echo "[sync] Done. Dataset is at ${LOCAL_DIR}"
echo ""
echo "[sync] Counts:"
echo "  train images: $(find "$LOCAL_DIR/train/images" -type f 2>/dev/null | wc -l)"
echo "  train labels: $(find "$LOCAL_DIR/train/labels" -type f 2>/dev/null | wc -l)"
echo "  val images:   $(find "$LOCAL_DIR/val/images" -type f 2>/dev/null | wc -l)"
echo "  val labels:   $(find "$LOCAL_DIR/val/labels" -type f 2>/dev/null | wc -l)"
echo ""
echo "[sync] Ready to train. From ${LOCAL_DIR}:"
echo ""
echo "  cd ${LOCAL_DIR}"
echo "  yolo train data=data.yaml model=yolo26n-pose.pt \\"
echo "    epochs=100 imgsz=640 batch=8 workers=0"
echo ""
echo "[sync] data.yaml uses path: . so train/val resolve relative to the yaml."
echo "[sync] The cd above ensures Ultralytics picks it up correctly."
