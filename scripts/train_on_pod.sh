#!/usr/bin/env bash
# YOLO team side — kick off YOLO-pose training on the pod's 4090.
# Synchronous: stays attached until training finishes (~10 min on 4090
# at the current dataset scale). Trained weights end up at
# /workspace/runs/pose/<DATASET>.run/weights/best.pt on the pod; then
# you pull them with scripts/sync_weights_from_pod.sh.
#
# One command to start:
#   ./train_on_pod.sh
#
# Does:
#   1. Triggers a batch export on the pod so the dataset reflects every
#      labeled video_id (no need for the labeler to pre-export manually).
#   2. SSHes the pod, cds into the dataset dir, runs `uv run yolo train`.
#      `path: .` in data.yaml + the cd means train/val paths resolve
#      correctly no matter where the dataset directory lives.
#   3. Streams training logs live to your terminal.
#   4. Prints the path to the weights once done.
#
# Config via ~/.sheep-yolo.env (same file fetch_dataset.sh uses) or CLI.
# Defaults mirror the command the YOLO team hand-verified:
#
#   EPOCHS=100 IMGSZ=640 BATCH=8 MODEL=yolo26n-pose.pt
#
# Override any of those in .sheep-yolo.env or export them before calling.

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
DATASET="${DATASET:-sheep-pose-v0.1}"
MODEL="${MODEL:-yolo26n-pose.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-8}"

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

POD_IP="${1:-$POD_IP}"
POD_SSH_PORT="${2:-$POD_SSH_PORT}"
DATASET="${3:-$DATASET}"

# Namespace the run dir by dataset name so sheep-pose-v0.1.run and
# sheep-pose-v0.2.run don't clobber each other. Pod-side runs live at
# /workspace/runs/pose/<RUN_NAME>/.
RUN_NAME="${DATASET}.run"

echo "[train] Pod: ${POD_IP}:${POD_SSH_PORT}"
echo "[train] Dataset: ${DATASET}"
echo "[train] Run: ${RUN_NAME}"
echo "[train] Model: ${MODEL} · epochs=${EPOCHS} · imgsz=${IMGSZ} · batch=${BATCH}"
echo ""

# Step 1: make sure the dataset on the pod reflects all current reviews.
echo "[train] Triggering batch export on pod..."
EXPORT_RESP=$(ssh -p "$POD_SSH_PORT" -i "$SSH_KEY" "root@$POD_IP" \
  "curl -s -X POST 'http://localhost:8000/api/export/keypoints/all?dataset=${DATASET}'")

echo "[train] Export summary:"
echo "$EXPORT_RESP" | python3 -m json.tool 2>/dev/null || echo "$EXPORT_RESP"
echo ""

# Step 2: run yolo train on the pod, synchronous, logs streaming through SSH.
# uv run picks up the sheep-seg venv (ultralytics is a listed dep).
echo "[train] Starting training on pod's 4090. This streams live..."
echo "[train] Pod working dir: /workspace/SamSeesSheep/data/labels/exports/${DATASET}"
echo ""

ssh -t -p "$POD_SSH_PORT" -i "$SSH_KEY" "root@$POD_IP" bash <<REMOTE
set -e
cd /workspace/SamSeesSheep
source \$HOME/.local/bin/env 2>/dev/null || true

# Pre-flight: validate the exported dataset before burning GPU time.
# validate_dataset.py exits non-zero on structural/format errors; set -e
# aborts the whole training run in that case.
echo "[train] Validating dataset ${DATASET}..."
uv run python scripts/validate_dataset.py --dataset ${DATASET}

cd data/labels/exports/${DATASET}
mkdir -p /workspace/runs/pose

# exist_ok=True is the default in Ultralytics 8.4 when name conflicts.
# Removing the prior run dir first so per-epoch logs don't append weirdly.
rm -rf /workspace/runs/pose/${RUN_NAME}

uv run --project /workspace/SamSeesSheep yolo train \
  data=data.yaml \
  model=${MODEL} \
  project=/workspace/runs/pose \
  name=${RUN_NAME} \
  epochs=${EPOCHS} \
  imgsz=${IMGSZ} \
  batch=${BATCH} \
  workers=0
REMOTE

echo ""
echo "[train] Training finished."
echo "[train] Weights on pod: /workspace/runs/pose/${RUN_NAME}/weights/best.pt"
echo ""
echo "[train] Pull weights to local with:"
echo "  ./scripts/sync_weights_from_pod.sh ${DATASET}"
