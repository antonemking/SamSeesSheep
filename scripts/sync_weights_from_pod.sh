#!/usr/bin/env bash
# YOLO team side — rsync trained best.pt from the pod down to the
# sheep-yolo repo's weights/ dir. Runs AFTER scripts/train_on_pod.sh
# finishes. Only the ~10 MB weights cross the wire — the dataset
# stays on the pod.
#
# One command:
#   ./sync_weights_from_pod.sh                    # uses defaults
#   ./sync_weights_from_pod.sh sheep-pose-v0.2    # override dataset
#
# Does:
#   1. rsync's /workspace/runs/pose/<DATASET>.run/weights/best.pt from
#      the pod to ${LOCAL_WEIGHTS_DIR}/<DATASET>.pt
#   2. Also grabs last.pt (last epoch's weights) for completeness.
#   3. Prints the path you'd set as YOLOE_MODEL for the σ-benchmark run.
#
# Config via ~/.sheep-yolo.env or CLI. Defaults:
#
#   DATASET=sheep-pose-v0.1
#   LOCAL_WEIGHTS_DIR=~/dev/lorewood-advisors/sheep-yolo/weights

set -e

if [ -f "$HOME/.sheep-yolo.env" ]; then
  set -a
  # shellcheck disable=SC1090,SC1091
  . "$HOME/.sheep-yolo.env"
  set +a
fi

POD_IP="${POD_IP:-}"
POD_SSH_PORT="${POD_SSH_PORT:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
DATASET="${1:-${DATASET:-sheep-pose-v0.1}}"
LOCAL_WEIGHTS_DIR="${LOCAL_WEIGHTS_DIR:-$HOME/dev/lorewood-advisors/sheep-yolo/weights}"

POD_IP="${2:-$POD_IP}"
POD_SSH_PORT="${3:-$POD_SSH_PORT}"

if [ -z "$POD_IP" ] || [ -z "$POD_SSH_PORT" ]; then
  cat >&2 <<EOF
Missing pod SSH info. Either put in ~/.sheep-yolo.env:

  POD_IP=38.65.239.23
  POD_SSH_PORT=27921

Or pass on CLI:
  $0 [dataset-name] <pod-ip> <pod-ssh-port>
EOF
  exit 1
fi

mkdir -p "$LOCAL_WEIGHTS_DIR"

RUN_NAME="${DATASET}.run"
REMOTE_BEST="/workspace/runs/pose/${RUN_NAME}/weights/best.pt"
REMOTE_LAST="/workspace/runs/pose/${RUN_NAME}/weights/last.pt"
LOCAL_BEST="${LOCAL_WEIGHTS_DIR%/}/${DATASET}.pt"
LOCAL_LAST="${LOCAL_WEIGHTS_DIR%/}/${DATASET}.last.pt"

echo "[weights] Pulling best.pt from pod..."
rsync -avz \
  -e "ssh -p $POD_SSH_PORT -i $SSH_KEY" \
  "root@${POD_IP}:${REMOTE_BEST}" \
  "${LOCAL_BEST}"

echo "[weights] Pulling last.pt (optional, for resume/compare)..."
rsync -avz \
  -e "ssh -p $POD_SSH_PORT -i $SSH_KEY" \
  "root@${POD_IP}:${REMOTE_LAST}" \
  "${LOCAL_LAST}" || echo "[weights] last.pt not present, skipping."

echo ""
echo "[weights] Local files:"
ls -lh "${LOCAL_BEST}" "${LOCAL_LAST}" 2>/dev/null || true
echo ""
echo "[weights] Use in sheep-yolo's σ-benchmark:"
echo "  export YOLOE_MODEL=${LOCAL_BEST}"
echo "  # then run your motionless-sheep σ benchmark against \$YOLOE_MODEL"
