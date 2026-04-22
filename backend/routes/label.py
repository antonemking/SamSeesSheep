"""Keypoint labeling routes — read and mutate review.json per video.

Schema v2 (multi-subject): each frame has an `instances[]` array; each
instance has its own `obj_id`, `head_bbox`, and `keypoints[5]`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel as PydanticBaseModel, Field

from backend.config import LABELS_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["label"])


class Keypoint(PydanticBaseModel):
    x: float
    y: float
    v: int = Field(ge=0, le=2)


class InstanceUpdate(PydanticBaseModel):
    obj_id: int
    keypoints: list[Keypoint]


class FrameUpdate(PydanticBaseModel):
    instances: list[InstanceUpdate]


def _review_path(video_id: str) -> Path:
    # Prevent path traversal — video_id is expected to be a short uuid stem.
    if "/" in video_id or ".." in video_id or not video_id:
        raise HTTPException(status_code=400, detail="Invalid video_id")
    return LABELS_DIR / video_id / "review.json"


def _load_review(video_id: str) -> dict:
    p = _review_path(video_id)
    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No labeling state for video {video_id}. Run /api/analyze/video first.",
        )
    return json.loads(p.read_text())


def _write_review(video_id: str, payload: dict) -> None:
    p = _review_path(video_id)
    p.write_text(json.dumps(payload, indent=2))


def _frame_has_reviewed_kp(frame: dict) -> bool:
    """A frame counts as reviewed if any of its instances has any v=2 kp."""
    for inst in frame.get("instances") or []:
        if any(k.get("v") == 2 for k in inst.get("keypoints", [])):
            return True
    return False


@router.get("/labels")
async def list_labeled_videos() -> dict:
    """List all videos with a review.json, with review progress per video.

    Powers the 'Resume labeling' list on the dashboard so reviewers can
    jump back into an existing session without remembering the video_id.
    """
    results = []
    for video_dir in sorted(LABELS_DIR.iterdir()):
        if not video_dir.is_dir() or video_dir.name == "exports":
            continue
        review_file = video_dir / "review.json"
        if not review_file.exists():
            continue
        try:
            data = json.loads(review_file.read_text())
        except json.JSONDecodeError:
            continue
        frames = data.get("frames", [])
        reviewed = sum(1 for f in frames if _frame_has_reviewed_kp(f))
        results.append({
            "video_id": data.get("video_id", video_dir.name),
            "n_frames": len(frames),
            "n_reviewed": reviewed,
            "n_obj_ids": len(data.get("obj_ids") or []),
            "updated_at": review_file.stat().st_mtime,
        })
    # Most recently updated first
    results.sort(key=lambda r: -r["updated_at"])
    return {"videos": results}


@router.get("/label/{video_id}")
async def get_review_state(video_id: str) -> dict:
    """Return the full review.json for the labeling UI to render."""
    return _load_review(video_id)


@router.post("/label/{video_id}/frame/{frame_idx}")
async def update_frame_keypoints(
    video_id: str, frame_idx: int, update: FrameUpdate,
) -> dict:
    """Replace the per-instance keypoints for one frame.

    The exporter trusts v=2 only. Dots the reviewer moves arrive with v=2;
    unplaced keypoints the user explicitly skipped stay at v=0. See
    sheep-yolo/sheep-seg-conversation/LOG.md for the mapping.

    Payload shape (schema v2):
        {"instances": [{"obj_id": 5, "keypoints": [{x,y,v}, ... 5 items]},
                        ...]}

    Every instance already present in review.json for this frame must be
    represented in the payload, even if its keypoints haven't changed.
    """
    review = _load_review(video_id)
    frames = review.get("frames", [])
    if not (0 <= frame_idx < len(frames)):
        raise HTTPException(
            status_code=400,
            detail=f"frame_idx {frame_idx} out of range for {len(frames)} frames",
        )

    existing_instances = frames[frame_idx].get("instances") or []
    existing_by_oid = {int(inst.get("obj_id")): inst for inst in existing_instances}
    incoming_by_oid = {inst.obj_id: inst for inst in update.instances}

    if set(incoming_by_oid) != set(existing_by_oid):
        raise HTTPException(
            status_code=400,
            detail=(
                f"obj_id set mismatch: existing={sorted(existing_by_oid)}, "
                f"incoming={sorted(incoming_by_oid)}"
            ),
        )

    new_instances = []
    for oid, inst in existing_by_oid.items():
        incoming = incoming_by_oid[oid]
        original_kps = inst.get("keypoints", [])
        if len(incoming.keypoints) != len(original_kps):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"obj_id {oid}: expected {len(original_kps)} keypoints, "
                    f"got {len(incoming.keypoints)}"
                ),
            )
        new_kps = []
        for i, kp in enumerate(incoming.keypoints):
            auto_v = original_kps[i].get("auto_v", 0) if i < len(original_kps) else 0
            new_kps.append({
                "x": float(kp.x),
                "y": float(kp.y),
                "v": int(kp.v),
                "auto_v": auto_v,
            })
        new_instances.append({
            "obj_id": oid,
            "head_bbox": inst.get("head_bbox"),
            "head_confidence": inst.get("head_confidence"),
            "keypoints": new_kps,
        })

    frames[frame_idx]["instances"] = new_instances
    _write_review(video_id, review)

    reviewed = sum(1 for f in frames if _frame_has_reviewed_kp(f))
    return {"ok": True, "frame_idx": frame_idx, "frames_with_reviews": reviewed}
