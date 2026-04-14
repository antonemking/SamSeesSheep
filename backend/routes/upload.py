"""Photo upload routes."""

from __future__ import annotations

import io
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile
from PIL import Image

from backend.config import UPLOAD_DIR
from backend.models import PhotoAnalysis

router = APIRouter(prefix="/api", tags=["upload"])


async def save_upload(file: UploadFile) -> tuple[str, Path, int, int]:
    """Save an uploaded file. Converts HEIC to JPEG for browser/OpenCV compat."""
    photo_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename or "photo.jpg").suffix.lower() or ".jpg"
    contents = await file.read()

    # Convert HEIC/HEIF to JPEG
    if suffix in {".heic", ".heif"}:
        img = Image.open(io.BytesIO(contents))
        save_path = UPLOAD_DIR / f"{photo_id}.jpg"
        img.convert("RGB").save(save_path, "JPEG", quality=92)
        width, height = img.size
    else:
        save_path = UPLOAD_DIR / f"{photo_id}{suffix}"
        save_path.write_bytes(contents)
        img = Image.open(save_path)
        width, height = img.size

    return photo_id, save_path, width, height


@router.post("/upload")
async def upload_photos(files: list[UploadFile]) -> list[PhotoAnalysis]:
    """Upload one or more sheep photos for analysis."""
    results = []
    for file in files:
        photo_id, path, width, height = await save_upload(file)
        analysis = PhotoAnalysis(
            photo_id=photo_id,
            filename=path.name,
            upload_time=datetime.now(),
            image_width=width,
            image_height=height,
        )
        results.append(analysis)
    return results


@router.post("/upload_video")
async def upload_video(file: UploadFile) -> dict:
    """Upload a video for frame-by-frame analysis."""
    import uuid as _uuid
    video_id = str(_uuid.uuid4())[:8]
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".webm", ".m4v", ".avi"}:
        suffix = ".mp4"
    save_path = UPLOAD_DIR / f"{video_id}{suffix}"
    contents = await file.read()
    save_path.write_bytes(contents)
    return {
        "video_id": video_id,
        "filename": save_path.name,
        "size_kb": len(contents) // 1024,
    }
