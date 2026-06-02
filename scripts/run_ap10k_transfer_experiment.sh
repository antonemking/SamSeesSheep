#!/usr/bin/env bash
# LOCAL launcher for the AP-10K transfer experiment. Mirrors train_on_pod.sh:
# reads ~/.sheep-yolo.env for pod SSH info, then runs the pod-side driver
# (scripts/ap10k_transfer_experiment.sh) on the pod's GPU, streaming logs.
#
# The heavy lifting — AP-10K convert, 3 training runs, eval — all happens on
# the pod. Only the final comparison.json comes back here.
#
# Usage:
#   HELDOUT=IMG_1234 ./scripts/run_ap10k_transfer_experiment.sh
#   ./scripts/run_ap10k_transfer_experiment.sh <pod-ip> <pod-ssh-port> <heldout-video-id>
#
# All pod-side knobs (DATASET, AP10K_ROOT, MODEL, *_EPOCHS, IMGSZ, BATCH)
# pass straight through from your environment.

set -e

if [ -f "$HOME/.sheep-yolo.env" ]; then
  set -a; . "$HOME/.sheep-yolo.env"; set +a
fi

POD_IP="${1:-${POD_IP:-}}"
POD_SSH_PORT="${2:-${POD_SSH_PORT:-}}"
HELDOUT="${3:-${HELDOUT:-}}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [ -z "$POD_IP" ] || [ -z "$POD_SSH_PORT" ] || [ -z "$HELDOUT" ]; then
  cat >&2 <<EOF
Missing config. Need pod SSH info (~/.sheep-yolo.env or CLI) and a held-out clip.

  POD_IP=...        POD_SSH_PORT=...     (in ~/.sheep-yolo.env)
  HELDOUT=IMG_1234  ./scripts/run_ap10k_transfer_experiment.sh

  or: $0 <pod-ip> <pod-ssh-port> <heldout-video-id>
EOF
  exit 1
fi

# Forward the optional pod-side knobs only if set locally, so the pod-side
# defaults apply otherwise.
FWD=""
for v in DATASET AP10K_ROOT MODEL AP10K_EPOCHS FT_EPOCHS IMGSZ BATCH; do
  if [ -n "${!v:-}" ]; then FWD="$FWD $v=${!v}"; fi
done

echo "[run] Pod: ${POD_IP}:${POD_SSH_PORT}  heldout=${HELDOUT}"
echo "[run] Forwarding:${FWD:- (pod defaults)}"

ssh -t -p "$POD_SSH_PORT" -i "$SSH_KEY" "root@$POD_IP" bash <<REMOTE
set -e
cd /workspace/SamSeesSheep
source \$HOME/.local/bin/env 2>/dev/null || true
git pull --ff-only 2>/dev/null || echo "[run] (skipping git pull)"
HELDOUT=${HELDOUT} ${FWD} bash scripts/ap10k_transfer_experiment.sh
REMOTE

LOCAL_OUT="sheep-yolo/artifacts/ap10k-transfer-comparison.json"
mkdir -p "$(dirname "$LOCAL_OUT")"
echo ""
echo "[run] Pulling comparison.json back..."
rsync -avz -e "ssh -p $POD_SSH_PORT -i $SSH_KEY" \
  "root@${POD_IP}:/workspace/runs/ap10k-transfer/comparison.json" \
  "$LOCAL_OUT" && echo "[run] -> $LOCAL_OUT" || echo "[run] (no comparison.json yet)"
