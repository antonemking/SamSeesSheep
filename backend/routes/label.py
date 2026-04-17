"""Keypoint labeling routes — read and mutate review.json per video."""

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


class FrameUpdate(PydanticBaseModel):
    keypoints: list[Keypoint]


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


@router.get("/label/{video_id}")
async def get_review_state(video_id: str) -> dict:
    """Return the full review.json for the labeling UI to render."""
    return _load_review(video_id)


@router.post("/label/{video_id}/frame/{frame_idx}")
async def update_frame_keypoints(
    video_id: str, frame_idx: int, update: FrameUpdate,
) -> dict:
    """Replace the keypoints array for one frame.

    The exporter trusts v=2 only. Dots the reviewer moves should arrive
    with v=2; unplaced keypoints the user explicitly skipped should stay
    at v=0. See sheep-yolo/sheep-seg-conversation/LOG.md for the mapping.
    """
    review = _load_review(video_id)
    frames = review.get("frames", [])
    if not (0 <= frame_idx < len(frames)):
        raise HTTPException(
            status_code=400,
            detail=f"frame_idx {frame_idx} out of range for {len(frames)} frames",
        )
    expected = len(frames[frame_idx].get("keypoints", []))
    if len(update.keypoints) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} keypoints, got {len(update.keypoints)}",
        )
    # Preserve auto_v from the original record; caller only sends x/y/v.
    original_kps = frames[frame_idx].get("keypoints", [])
    new_kps = []
    for i, kp in enumerate(update.keypoints):
        auto_v = original_kps[i].get("auto_v", 0) if i < len(original_kps) else 0
        new_kps.append({
            "x": float(kp.x),
            "y": float(kp.y),
            "v": int(kp.v),
            "auto_v": auto_v,
        })
    frames[frame_idx]["keypoints"] = new_kps
    _write_review(video_id, review)

    reviewed = sum(
        1 for f in frames
        if any(k.get("v") == 2 for k in f.get("keypoints", []))
    )
    return {"ok": True, "frame_idx": frame_idx, "frames_with_reviews": reviewed}
