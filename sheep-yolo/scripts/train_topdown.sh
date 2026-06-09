#!/usr/bin/env bash
# Train the TOP-DOWN crop-pose model — LOCAL (no pod needed).
#
# Runs on the laptop's GTX 1660 Ti (6 GB). The crop dataset is a few hundred
# ~256 px single-instance head crops, so a nano pose model trains in minutes
# and fits in 6 GB comfortably. SAM is the only thing that needs the pod;
# this stage does not touch it.
#
# Prereq: scripts/crop_export.py has produced the crops dataset.
#
#   bash scripts/train_topdown.sh
#   EPOCHS=200 BATCH=16 bash scripts/train_topdown.sh   # override
#
# Output: weights/_topdown/<RUN>/weights/best.pt
# Promote a winner with:
#   cp weights/_topdown/<RUN>/weights/best.pt weights/sheep-pose-v0.7-topdown-yolo26n.pt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHEEP_YOLO="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SHEEP_YOLO/.." && pwd)"

DATA="${DATA:-$REPO_ROOT/data/labels/exports/sheep-pose-v0.4-crops/data.yaml}"
MODEL="${MODEL:-yolo26n-pose.pt}"     # COCO-pose init; only the backbone transfers (5-kpt head is reinitialized)
IMGSZ="${IMGSZ:-256}"                 # crop model trains/serves at 256 — keep bench_topdown in sync
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-32}"                  # drop to 16 if the 1660 Ti OOMs
RUN="${RUN:-v0.4-crops}"

if [ ! -f "$DATA" ]; then
  echo "Missing dataset: $DATA" >&2
  echo "Run first:  uv run --project \"$SHEEP_YOLO\" python scripts/crop_export.py" >&2
  exit 1
fi

echo "[topdown] data:  $DATA"
echo "[topdown] model: $MODEL · imgsz=$IMGSZ · epochs=$EPOCHS · batch=$BATCH"
echo "[topdown] run:   weights/_topdown/$RUN"
echo ""

cd "$SHEEP_YOLO"
uv run --project "$SHEEP_YOLO" yolo train \
  data="$DATA" \
  model="$MODEL" \
  project="$SHEEP_YOLO/weights/_topdown" \
  name="$RUN" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  device=0 \
  workers=4 \
  exist_ok=True

echo ""
echo "[topdown] done -> $SHEEP_YOLO/weights/_topdown/$RUN/weights/best.pt"
echo "[topdown] bench: uv run --project \"$SHEEP_YOLO\" python scripts/bench_topdown.py \\"
echo "                   --pose weights/_topdown/$RUN/weights/best.pt"
