"""Video upload route."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile
from PIL import Image

from backend.config import UPLOAD_DIR

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload_video")
async def upload_video(file: UploadFile) -> dict:
    """Save the video and a JPEG of frame 0 so the UI can show a click-to-select photo."""
    video_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".webm", ".m4v", ".avi"}:
        suffix = ".mp4"
    save_path = UPLOAD_DIR / f"{video_id}{suffix}"
    contents = await file.read()
    save_path.write_bytes(contents)

    first_frame_url = None
    first_frame_w = first_frame_h = None
    try:
        import imageio.v3 as iio
        frames = iio.imread(str(save_path), plugin="pyav")
        if len(frames) > 0:
            first_frame = Image.fromarray(frames[0])
            first_frame_w, first_frame_h = first_frame.size
            if max(first_frame_w, first_frame_h) > 1024:
                scale = 1024 / max(first_frame_w, first_frame_h)
                first_frame = first_frame.resize(
                    (int(first_frame_w * scale), int(first_frame_h * scale)),
                    Image.LANCZOS,
                )
            frame_path = UPLOAD_DIR / f"{video_id}_frame0.jpg"
            first_frame.convert("RGB").save(frame_path, "JPEG", quality=88)
            first_frame_url = f"/uploads/{frame_path.name}"
    except Exception:
        pass

    return {
        "video_id": video_id,
        "filename": save_path.name,
        "size_kb": len(contents) // 1024,
        "first_frame_url": first_frame_url,
        "first_frame_width": first_frame_w,
        "first_frame_height": first_frame_h,
    }
