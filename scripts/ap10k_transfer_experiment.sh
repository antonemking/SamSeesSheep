#!/usr/bin/env bash
# POD-SIDE driver for the AP-10K transfer experiment. Runs ON the pod (it
# needs the GPU and the on-volume sheep dataset). Launch it from your laptop
# with scripts/run_ap10k_transfer_experiment.sh, or run it directly after
# SSHing into the pod.
#
# Answers one question: does pretraining the backbone on AP-10K (animal-body
# pose, arXiv 2108.12617) beat the stock COCO-human-pose backbone for our
# 5-keypoint sheep_head task?
#
#   Arm A (baseline):   yolo26n-pose.pt            -> fine-tune on sheep train
#   Arm B (experiment): yolo26n-pose -> AP-10K     -> fine-tune on sheep train
#
# Both arms train on the SAME train set (every clip except the held-out one)
# and are scored on the SAME held-out clip with ground-truth keypoints.
#
# NOTE on the 17->5 keypoint mismatch: AP-10K has 17 whole-body keypoints,
# our schema has 5 head/ear keypoints. Ultralytics loads the AP-10K weights
# by shape-intersection, so Stage 2 keeps the backbone + neck and
# reinitializes the pose head. The experiment therefore measures *backbone*
# transfer, which is exactly the intended comparison. See docs/AP10K_TRANSFER.md.
#
# Config via env vars (or ~/.sheep-yolo.env if present):
#   DATASET     sheep export dataset name   (default sheep-pose-v0.4)
#   HELDOUT     video_id to hold out as test  (REQUIRED)
#   AP10K_ROOT  untarred AP-10K dir           (default /workspace/ap-10k)
#   MODEL       base pose model              (default yolo26n-pose.pt)
#   AP10K_EPOCHS  Stage-1 epochs             (default 50)
#   FT_EPOCHS     Stage-2 epochs             (default 100)
#   IMGSZ         (default 640)   BATCH (default 8)
#
# Usage on the pod:
#   HELDOUT=IMG_1234 bash scripts/ap10k_transfer_experiment.sh

set -euo pipefail

REPO="/workspace/SamSeesSheep"
RUNS="/workspace/runs/ap10k-transfer"
AP10K_YOLO="/workspace/ap10k-yolo"
EXP_DATA_DIR="/workspace/ap10k-exp/sheep-heldout"

if [ -f "$HOME/.sheep-yolo.env" ]; then
  set -a; . "$HOME/.sheep-yolo.env"; set +a
fi

DATASET="${DATASET:-sheep-pose-v0.4}"
HELDOUT="${HELDOUT:-}"
AP10K_ROOT="${AP10K_ROOT:-/workspace/ap-10k}"
MODEL="${MODEL:-yolo26n-pose.pt}"
AP10K_EPOCHS="${AP10K_EPOCHS:-50}"
FT_EPOCHS="${FT_EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-8}"

if [ -z "$HELDOUT" ]; then
  echo "ERROR: set HELDOUT=<video_id> (the clip to leave out as the test set)." >&2
  echo "       List candidates with: ls $REPO/data/labels" >&2
  exit 1
fi
if [ ! -d "$AP10K_ROOT/annotations" ]; then
  cat >&2 <<EOF
ERROR: AP-10K not found at $AP10K_ROOT (no annotations/ dir).
Download + untar the official AP-10K release there first, e.g.:
  mkdir -p $AP10K_ROOT && cd $AP10K_ROOT
  # from https://github.com/AlexTheBad/AP-10K (Google Drive link)
  #   ap-10k.tar.gz -> data/ + annotations/
  tar xzf ap-10k.tar.gz
Then re-run. Override the location with AP10K_ROOT=...
EOF
  exit 1
fi

cd "$REPO"
mkdir -p "$RUNS"
RUN_BASE() { uv run --project "$REPO" "$@"; }

echo "================ AP-10K transfer experiment ================"
echo " dataset=$DATASET  heldout=$HELDOUT  model=$MODEL"
echo " stage1 epochs=$AP10K_EPOCHS  stage2 epochs=$FT_EPOCHS  imgsz=$IMGSZ batch=$BATCH"
echo "============================================================"

# ---- Step 0: refresh the sheep export so every reviewed clip is included.
echo "[0/5] Triggering batch export of sheep dataset..."
curl -s -X POST "http://localhost:8000/api/export/keypoints/all?dataset=${DATASET}" \
  | python3 -m json.tool 2>/dev/null || echo "  (export endpoint not reachable — using existing export)"

# ---- Step 1: build the leave-one-clip-out sheep split.
echo "[1/5] Building held-out split (test = $HELDOUT)..."
rm -rf "$EXP_DATA_DIR"
RUN_BASE python sheep-yolo/scripts/prep_experiment_split.py \
  --export-dir "$REPO/data/labels/exports/${DATASET}" \
  --heldout "$HELDOUT" \
  --out "$EXP_DATA_DIR"

# ---- Step 2: convert AP-10K to YOLO-pose (skip if already done).
if [ -f "$AP10K_YOLO/data.yaml" ]; then
  echo "[2/5] AP-10K already converted at $AP10K_YOLO (skipping). rm it to rebuild."
else
  echo "[2/5] Converting AP-10K -> YOLO-pose..."
  RUN_BASE python sheep-yolo/scripts/prep_ap10k.py \
    --ap10k-root "$AP10K_ROOT" --out "$AP10K_YOLO"
fi

# ---- Step 3: Stage-1 — pretrain the base model on AP-10K (17-kpt animal pose).
AP10K_BEST="$RUNS/ap10k-pretrain/weights/best.pt"
if [ -f "$AP10K_BEST" ]; then
  echo "[3/5] AP-10K pretrain already done ($AP10K_BEST). rm the run dir to redo."
else
  echo "[3/5] Stage 1: pretraining $MODEL on AP-10K (~10K imgs)..."
  rm -rf "$RUNS/ap10k-pretrain"
  RUN_BASE yolo train data="$AP10K_YOLO/data.yaml" model="$MODEL" \
    project="$RUNS" name=ap10k-pretrain \
    epochs="$AP10K_EPOCHS" imgsz="$IMGSZ" batch="$BATCH" workers=0
fi

# ---- Step 4: Stage-2 — two fine-tunes on the IDENTICAL sheep train set.
echo "[4/5] Stage 2A: baseline fine-tune ($MODEL -> sheep)..."
rm -rf "$RUNS/baseline"
RUN_BASE yolo train data="$EXP_DATA_DIR/data.yaml" model="$MODEL" \
  project="$RUNS" name=baseline \
  epochs="$FT_EPOCHS" imgsz="$IMGSZ" batch="$BATCH" workers=0

echo "[4/5] Stage 2B: AP-10K-pretrained fine-tune (backbone -> sheep)..."
rm -rf "$RUNS/ap10k-ft"
RUN_BASE yolo train data="$EXP_DATA_DIR/data.yaml" model="$AP10K_BEST" \
  project="$RUNS" name=ap10k-ft \
  epochs="$FT_EPOCHS" imgsz="$IMGSZ" batch="$BATCH" workers=0

# ---- Step 5: score both arms on the held-out clip.
echo "[5/5] Scoring both arms on held-out clip $HELDOUT..."
RUN_BASE python sheep-yolo/scripts/eval_transfer_experiment.py \
  --data "$EXP_DATA_DIR/data.yaml" \
  --baseline   "$RUNS/baseline/weights/best.pt" \
  --experiment "$RUNS/ap10k-ft/weights/best.pt" \
  --out "$RUNS/comparison.json" \
  --imgsz "$IMGSZ"

echo ""
echo "[done] Weights:"
echo "   baseline (A):   $RUNS/baseline/weights/best.pt"
echo "   AP-10K-ft (B):  $RUNS/ap10k-ft/weights/best.pt"
echo "   comparison:     $RUNS/comparison.json"
