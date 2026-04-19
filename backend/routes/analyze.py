"""Video analysis routes — SAM 3 Video tracking and ear angles."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel as PydanticBaseModel

from backend.config import UPLOAD_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])


class VideoAnalyzeRequest(PydanticBaseModel):
    video_id: str
    subject: str = "sheep"
    max_frames: int = 30
    click_x: float | None = None
    click_y: float | None = None


@router.post("/analyze/video")
async def analyze_video_endpoint(req: VideoAnalyzeRequest) -> dict:
    """Run SAM 3 Video tracking across a clip to get per-frame ear angles."""
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
        click_point = (req.click_x, req.click_y)

    from backend.pipeline.video import (
        analyze_video as _run,
        unload_video_model,
        _full_pipeline_enabled,
    )
    import gc, torch

    # Retry cascade adapts to hardware.
    # - Full pipeline (24GB+): start at the requested count (up to 60),
    #   step down to 30, 15. Fewer OOM expected, so cascade is wide.
    # - Survival (6GB): start at 20, step down to 10, 5. Every clip is
    #   close to the OOM ceiling so we need aggressive fallbacks.
    if _full_pipeline_enabled():
        attempt_frames = [min(req.max_frames, 60), 30, 15]
    else:
        attempt_frames = [min(req.max_frames, 20), 10, 5]
    # Deduplicate while preserving order (req.max_frames=10 gives [10,10,5])
    seen = set()
    attempt_frames = [n for n in attempt_frames if not (n in seen or seen.add(n))]
    last_err = None
    for idx, n in enumerate(attempt_frames):
        try:
            result = _run(
                video_path, subject=req.subject, max_frames=n,
                click_point=click_point,
            )
            if n < req.max_frames:
                result["retry_reduced_frames_from"] = req.max_frames
            break
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(
                "Video OOM at max_frames=%d — unloading model before next retry", n,
            )
            # Drop the model entirely — empty_cache can't undo allocator
            # fragmentation left behind by the failed run.
            unload_video_model()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            last_err = e
        except Exception as e:
            logger.exception("Video analysis failed")
            raise HTTPException(status_code=500, detail=f"Video analysis failed: {e}")
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed (OOM after retries): {last_err}",
        )

    result["video_id"] = req.video_id
    result["video_url"] = f"/uploads/{video_path.name}"
    return result
