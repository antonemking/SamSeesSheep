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


def _hash_bucket(video_id: str, frame_idx: int) -> float:
    """Deterministic [0.0, 1.0) bucket for (video_id, frame_idx)."""
    h = hashlib.md5(f"{video_id}:{frame_idx}".encode()).hexdigest()
    return int(h[:8], 16) % 10000 / 10000.0


def _split_frames(
    video_id: str, frame_indices: list[int], val_ratio: float,
) -> tuple[set[int], set[int]]:
    """Split frame indices into (train, val) sets deterministically.

    Uses hash bucketing per frame. Guarantees val is non-empty when
    there are at least 2 frames and val_ratio > 0 — previously, small
    label sets could randomly all bucket to train and leave val empty,
    which crashes Ultralytics at train time (the YOLO team's "empty
    val/images crashes loader" callout on 2026-04-18).
    """
    buckets = [(i, _hash_bucket(video_id, i)) for i in frame_indices]
    train = {i for i, b in buckets if b >= val_ratio}
    val = {i for i, b in buckets if b < val_ratio}

    if val_ratio > 0 and not val and len(frame_indices) >= 2:
        # Promote the single highest-bucket train frame to val. Deterministic
        # because buckets are fixed; keeps the promotion reproducible.
        promote = max(buckets, key=lambda t: t[1])[0]
        train.discard(promote)
        val.add(promote)
    return train, val


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


def _write_data_yaml(dataset_dir: Path, val_has_data: bool) -> None:
    """Write the 6-field data.yaml the YOLO team specified.

    path: "." (relative) so the dataset is portable — same yaml works
    on the pod, on the YOLO team's training machine, anywhere it gets
    rsync'd. Ultralytics resolves train/val relative to the yaml's
    own location when path is ".".

    When val is empty (e.g., val_split=0 or only one exported frame),
    point `val:` at train/images to keep the loader happy. At this
    scale v0.1 σ-benchmark doesn't use held-out val anyway.
    """
    val_path = "val/images" if val_has_data else "train/images"
    yaml_text = (
        f"path: .\n"
        f"train: train/images\n"
        f"val: {val_path}\n"
        f"nc: 1\n"
        f"names: ['sheep_head']\n"
        f"kpt_shape: [5, 3]\n"
        f"flip_idx: [0, 2, 1, 4, 3]\n"
    )
    (dataset_dir / "data.yaml").write_text(yaml_text)


def _export_one_video(
    video_id: str, dataset_dir: Path, pseudo: bool, val_split: float,
) -> dict:
    """Core per-video export routine, shared by the single and batch endpoints."""
    review_path = LABELS_DIR / video_id / "review.json"
    if not review_path.exists():
        return {
            "video_id": video_id,
            "exported": {"train": 0, "val": 0},
            "skipped_reason": "no review.json",
        }
    review = json.loads(review_path.read_text())

    for split in ("train", "val"):
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    frames_src = LABELS_DIR / video_id / "frames"

    # First pass: figure out which frames ARE exportable (have ≥1 instance
    # with labeled keypoints + bbox + source image). In multi-subject mode
    # each frame's .txt gets N lines (one per instance with usable kps).
    # A frame is skipped only if NO instance has any usable keypoint.
    exportable: dict[int, dict] = {}
    skipped_no_labels = 0
    skipped_no_bbox = 0
    skipped_missing_image = 0
    total_instances_written = 0

    for f in review.get("frames", []):
        idx = int(f["frame_idx"])
        frame_w = f.get("frame_width")
        frame_h = f.get("frame_height")
        if frame_w is None or frame_h is None:
            skipped_no_bbox += 1
            continue

        label_lines: list[str] = []
        for inst in f.get("instances") or []:
            bbox = inst.get("head_bbox")
            keypoints = inst.get("keypoints") or []
            if bbox is None:
                continue
            line = _yolo_label_line(
                bbox, keypoints, frame_w, frame_h, pseudo=pseudo,
            )
            if line is not None:
                label_lines.append(line)

        if not label_lines:
            skipped_no_labels += 1
            continue

        img_src = frames_src / f"frame{idx:04d}.jpg"
        if not img_src.exists():
            skipped_missing_image += 1
            continue

        exportable[idx] = {"label_lines": label_lines, "img_src": img_src}
        total_instances_written += len(label_lines)

    if not exportable:
        return {
            "video_id": video_id,
            "exported": {"train": 0, "val": 0},
            "skipped_no_labels": skipped_no_labels,
            "skipped_no_bbox": skipped_no_bbox,
            "skipped_missing_image": skipped_missing_image,
            "skipped_reason": "no reviewed keypoints (v=2) on any instance",
        }

    # Second pass: deterministic split with min-1-val guarantee when possible.
    train_idxs, val_idxs = _split_frames(
        video_id, list(exportable.keys()), val_ratio=val_split,
    )

    # Clean stale files from any previous export of THIS video_id so a re-run
    # (e.g., after more review work) rebalances cleanly instead of leaving
    # orphans in the opposite split.
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            for p in (dataset_dir / split / sub).glob(f"{video_id}_*"):
                p.unlink()

    exported = {"train": 0, "val": 0}
    for idx, payload in exportable.items():
        split = "val" if idx in val_idxs else "train"
        stem = f"{video_id}_{idx:04d}"
        img_dst = dataset_dir / split / "images" / f"{stem}.jpg"
        lbl_dst = dataset_dir / split / "labels" / f"{stem}.txt"
        shutil.copyfile(payload["img_src"], img_dst)
        lbl_dst.write_text("\n".join(payload["label_lines"]) + "\n")
        exported[split] += 1

    return {
        "video_id": video_id,
        "exported": exported,
        "instances_written": total_instances_written,
        "skipped_no_labels": skipped_no_labels,
        "skipped_no_bbox": skipped_no_bbox,
        "skipped_missing_image": skipped_missing_image,
    }


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
    """Write a YOLO-pose dataset slice from one video's review state."""
    for name, val in (("video_id", video_id), ("dataset", dataset)):
        if "/" in val or ".." in val or not val:
            raise HTTPException(status_code=400, detail=f"Invalid {name}")

    dataset_dir = EXPORTS_DIR / dataset
    per_video = _export_one_video(video_id, dataset_dir, pseudo, val_split)

    if per_video["exported"]["train"] + per_video["exported"]["val"] == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No exportable frames for {video_id}: "
                f"{per_video.get('skipped_reason', 'unknown')}. "
                "Review at least one frame in /label/{video_id}."
            ),
        )

    dataset_val_has_data = any((dataset_dir / "val" / "images").glob("*.jpg"))
    _write_data_yaml(dataset_dir, val_has_data=dataset_val_has_data)

    result = {
        "ok": True,
        "video_id": video_id,
        "dataset": dataset,
        "dataset_path": str(dataset_dir.resolve()),
        "pseudo": pseudo,
        "val_split": val_split,
        "exported": per_video["exported"],
        "skipped": {
            "no_labeled_keypoints": per_video.get("skipped_no_labels", 0),
            "missing_bbox_or_dims":  per_video.get("skipped_no_bbox", 0),
            "missing_source_image":  per_video.get("skipped_missing_image", 0),
        },
    }
    logger.info(
        "Export video=%s dataset=%s pseudo=%s → train=%d val=%d",
        video_id, dataset, pseudo,
        per_video["exported"]["train"], per_video["exported"]["val"],
    )
    return result


@router.post("/export/keypoints/all")
async def export_all_keypoints(
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
    """Batch-export every labeled video into the dataset in one call.

    Walks data/labels/ for every review.json, runs the per-video exporter
    on each, aggregates results. Saves the hand-off a round-trip per video.

    Call from the laptop/YOLO side:
        curl -X POST 'http://pod:8000/api/export/keypoints/all?dataset=sheep-pose-v0'
    """
    if "/" in dataset or ".." in dataset or not dataset:
        raise HTTPException(status_code=400, detail="Invalid dataset")

    dataset_dir = EXPORTS_DIR / dataset
    per_video_results = []
    total = {"train": 0, "val": 0}

    for video_dir in sorted(LABELS_DIR.iterdir()):
        if not video_dir.is_dir() or video_dir.name == "exports":
            continue
        if not (video_dir / "review.json").exists():
            continue
        r = _export_one_video(video_dir.name, dataset_dir, pseudo, val_split)
        per_video_results.append(r)
        total["train"] += r["exported"]["train"]
        total["val"] += r["exported"]["val"]

    dataset_val_has_data = any((dataset_dir / "val" / "images").glob("*.jpg"))
    _write_data_yaml(dataset_dir, val_has_data=dataset_val_has_data)

    logger.info(
        "Batch export dataset=%s → %d videos, train=%d val=%d",
        dataset, len(per_video_results), total["train"], total["val"],
    )

    return {
        "ok": True,
        "dataset": dataset,
        "dataset_path": str(dataset_dir.resolve()),
        "pseudo": pseudo,
        "val_split": val_split,
        "total_exported": total,
        "videos": per_video_results,
    }
