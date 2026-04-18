"""Export reviewed keypoints to YOLO-pose training format.

Reads data/labels/{video_id}/review.json, applies the YOLO team's
v-flag mapping, and writes an Ultralytics-compatible dataset under
data/labels/exports/{dataset}/.

THE V-FLAG MAPPING (echoed from sheep-yolo/sheep-seg-conversation/LOG.md
so it's auditable in code, not just docs):

  sheep-seg internal v  →  YOLO-pose .txt output
  0 (mask missing)      →  "0 0 0"    (skip; not in training signal)
  1 (SAM auto, unreviewed) → "0 0 0"  (skip; machine guess, not ground truth)
  2 (human-reviewed)    →  "<px> <py> 2"   (labeled + visible)

Escape hatch: ?pseudo=true promotes v=1 → YOLO v=2 at export time.
Use only for "does it train at all?" prototype runs — training on
unreviewed machine guesses teaches the model to trust SAM's mask
geometry as ground truth, which defeats the σ benchmark.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from backend.config import LABELS_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["export"])

EXPORTS_DIR = LABELS_DIR / "exports"


def _hash_split(video_id: str, frame_idx: int, val_ratio: float) -> str:
    """Deterministic train/val assignment from (video_id, frame_idx).

    Same frame lands in the same split across re-runs, so re-exporting
    an updated review doesn't silently move training samples to val.
    """
    h = hashlib.md5(f"{video_id}:{frame_idx}".encode()).hexdigest()
    bucket = int(h[:8], 16) % 10000 / 10000.0
    return "val" if bucket < val_ratio else "train"


def _yolo_label_line(
    bbox: dict,
    keypoints: list[dict],
    frame_w: int,
    frame_h: int,
    pseudo: bool,
) -> str | None:
    """Build one YOLO-pose label line; return None if no usable keypoints.

    YOLO-pose format per instance:
        class bx by bw bh  k0x k0y k0v  k1x k1y k1v  ...  k4x k4y k4v
    where bx/by/bw/bh are center-xywh normalized to [0, 1].
    """
    labeled = 0
    parts_kp: list[str] = []
    for kp in keypoints:
        v = int(kp.get("v", 0))
        if v == 2 or (pseudo and v == 1):
            kx = float(kp["x"]) / frame_w
            ky = float(kp["y"]) / frame_h
            parts_kp.extend([f"{kx:.6f}", f"{ky:.6f}", "2"])
            labeled += 1
        else:
            # v=0 and (non-pseudo) v=1 both become "0 0 0" per the mapping.
            parts_kp.extend(["0.000000", "0.000000", "0"])

    if labeled == 0:
        return None

    # Our head_bbox is top-left + wh; YOLO wants center-xywh normalized.
    bx = (bbox["x"] + bbox["w"] / 2) / frame_w
    by = (bbox["y"] + bbox["h"] / 2) / frame_h
    bw = bbox["w"] / frame_w
    bh = bbox["h"] / frame_h
    return " ".join([
        "0",  # single class: sheep_head
        f"{bx:.6f}", f"{by:.6f}", f"{bw:.6f}", f"{bh:.6f}",
        *parts_kp,
    ])


def _write_data_yaml(dataset_dir: Path) -> None:
    """Write the 6-field data.yaml the YOLO team specified.

    path: absolute, so Ultralytics can find train/ and val/ regardless
    of what working directory the trainer runs from. Override if
    training happens on a different filesystem.
    """
    yaml_text = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 1\n"
        f"names: ['sheep_head']\n"
        f"kpt_shape: [5, 3]\n"
        f"flip_idx: [0, 2, 1, 4, 3]\n"
    )
    (dataset_dir / "data.yaml").write_text(yaml_text)


@router.post("/export/keypoints")
async def export_keypoints(
    video_id: str = Query(..., description="Source review to export from"),
    dataset: str = Query(
        "sheep-pose-v0",
        description="Output dataset directory name under data/labels/exports/",
    ),
    pseudo: bool = Query(
        False,
        description="Promote v=1 → YOLO v=2. Prototype-only; NOT for σ benchmarks.",
    ),
    val_split: float = Query(
        0.2, ge=0.0, le=0.5,
        description="Fraction of frames to hold out for validation.",
    ),
) -> dict:
    """Write a YOLO-pose dataset slice from one video's review state.

    Safe to call multiple times with different video_ids targeting the
    same dataset name — each call appends its frames, and each frame
    lands in a deterministic split based on hash(video_id + frame_idx).
    """
    # Path traversal guard — both inputs eventually hit the filesystem.
    for name, val in (("video_id", video_id), ("dataset", dataset)):
        if "/" in val or ".." in val or not val:
            raise HTTPException(status_code=400, detail=f"Invalid {name}")

    review_path = LABELS_DIR / video_id / "review.json"
    if not review_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No review state for video {video_id}. Run /api/analyze/video first.",
        )
    review = json.loads(review_path.read_text())

    dataset_dir = EXPORTS_DIR / dataset
    for split in ("train", "val"):
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    frames_src = LABELS_DIR / video_id / "frames"
    exported = {"train": 0, "val": 0}
    skipped_no_labels = 0
    skipped_no_bbox = 0
    skipped_missing_image = 0

    for f in review.get("frames", []):
        idx = int(f["frame_idx"])
        bbox = f.get("head_bbox")
        keypoints = f.get("keypoints") or []
        frame_w = f.get("frame_width")
        frame_h = f.get("frame_height")

        if bbox is None or frame_w is None or frame_h is None:
            skipped_no_bbox += 1
            continue

        label_line = _yolo_label_line(
            bbox, keypoints, frame_w, frame_h, pseudo=pseudo,
        )
        if label_line is None:
            skipped_no_labels += 1
            continue

        split = _hash_split(video_id, idx, val_split)
        stem = f"{video_id}_{idx:04d}"
        img_src = frames_src / f"frame{idx:04d}.jpg"
        if not img_src.exists():
            skipped_missing_image += 1
            continue

        img_dst = dataset_dir / split / "images" / f"{stem}.jpg"
        lbl_dst = dataset_dir / split / "labels" / f"{stem}.txt"
        shutil.copyfile(img_src, img_dst)
        lbl_dst.write_text(label_line + "\n")
        exported[split] += 1

    # Always (re)write data.yaml so path stays current if the directory moves.
    _write_data_yaml(dataset_dir)

    result = {
        "ok": True,
        "video_id": video_id,
        "dataset": dataset,
        "dataset_path": str(dataset_dir.resolve()),
        "pseudo": pseudo,
        "val_split": val_split,
        "exported": exported,
        "skipped": {
            "no_labeled_keypoints": skipped_no_labels,
            "missing_bbox_or_dims":  skipped_no_bbox,
            "missing_source_image":  skipped_missing_image,
        },
    }
    logger.info(
        "Export video=%s dataset=%s pseudo=%s → train=%d val=%d "
        "skipped_no_labels=%d skipped_no_bbox=%d skipped_missing_img=%d",
        video_id, dataset, pseudo, exported["train"], exported["val"],
        skipped_no_labels, skipped_no_bbox, skipped_missing_image,
    )
    return result
