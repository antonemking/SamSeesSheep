"""Video analysis route — hands off to the YOLO pipeline."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel as PydanticBaseModel

from backend.config import MAX_FRAMES, UPLOAD_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])


class VideoAnalyzeRequest(PydanticBaseModel):
    video_id: str
    click_x: float | None = None
    click_y: float | None = None
    max_frames: int = MAX_FRAMES
    # "parts" = YOLOE head/ear/nose (ear angle). "whole" = YOLO26-seg COCO
    # sheep class only (no ear angle — just stable whole-animal tracking).
    mode: str = "parts"
    # "bytetrack.yaml" (fast, spatial-only) or "botsort.yaml" (appearance ReID,
    # costs more per frame but recovers IDs better through occlusion).
    tracker: str | None = None


@router.post("/analyze/video")
async def analyze_video_endpoint(req: VideoAnalyzeRequest) -> dict:
    all_matches = list(UPLOAD_DIR.glob(f"{req.video_id}.*"))
    matches = [
        p for p in all_matches
        if "_frame0" not in p.stem and p.suffix.lower() in {
            ".mp4", ".mov", ".webm", ".m4v", ".avi"
        }
    ]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Video {req.video_id} not found")
    video_path = matches[0]

    click_point = None
    if req.click_x is not None and req.click_y is not None:
        click_point = (float(req.click_x), float(req.click_y))

    from backend.pipeline.video import analyze_video, analyze_video_whole

    try:
        if req.mode == "whole":
            result = analyze_video_whole(
                video_path, click_point_norm=click_point,
                max_frames=req.max_frames, tracker=req.tracker,
            )
        else:
            result = analyze_video(
                video_path, click_point_norm=click_point,
                max_frames=req.max_frames, tracker=req.tracker,
            )
    except Exception as e:
        logger.exception("YOLO analysis failed")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {e}")

    result["video_id"] = req.video_id
    result["video_url"] = f"/uploads/{video_path.name}"
    return result
