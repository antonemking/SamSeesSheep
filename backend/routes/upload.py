"""Video upload + import routes."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile
from PIL import Image

from backend.config import UPLOAD_DIR

router = APIRouter(prefix="/api", tags=["upload"])


def _extract_first_frame(save_path: Path, video_id: str) -> tuple[str | None, int | None, int | None]:
    """Save the video's first frame as a JPEG thumbnail. Returns (url, w, h)."""
    try:
        import imageio.v3 as iio
        frames = iio.imread(str(save_path), plugin="pyav")
    except Exception:
        return None, None, None
    if len(frames) == 0:
        return None, None, None
    first_frame = Image.fromarray(frames[0])
    w, h = first_frame.size
    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        first_frame = first_frame.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS,
        )
    frame_path = UPLOAD_DIR / f"{video_id}_frame0.jpg"
    first_frame.convert("RGB").save(frame_path, "JPEG", quality=88)
    return f"/uploads/{frame_path.name}", w, h


@router.post("/upload_video")
async def upload_video(file: UploadFile) -> dict:
    """Upload a video for frame-by-frame analysis via multipart body.

    Also saves the first frame so the UI can show a click-to-track selector.
    """
    video_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".webm", ".m4v", ".avi"}:
        suffix = ".mp4"
    save_path = UPLOAD_DIR / f"{video_id}{suffix}"
    contents = await file.read()
    save_path.write_bytes(contents)

    first_frame_url, first_frame_w, first_frame_h = _extract_first_frame(
        save_path, video_id,
    )

    return {
        "video_id": video_id,
        "filename": save_path.name,
        "size_kb": len(contents) // 1024,
        "first_frame_url": first_frame_url,
        "first_frame_width": first_frame_w,
        "first_frame_height": first_frame_h,
    }


@router.post("/import_video")
async def import_video(
    path: str = Query(..., description="Absolute path to a video file on the server filesystem"),
) -> dict:
    """Register a video that's already on the pod's filesystem.

    Lets push_clip.sh avoid HTTP multipart uploads entirely: scp the file to
    the pod, then call this endpoint with the path. Originally a workaround for
    RunPod's HTTP/2 proxy dropping large uploads with ERR_HTTP2_PROTOCOL_ERROR;
    still the preferred path on Vast (direct port, but big browser uploads are
    flaky regardless). Same downstream effect as /upload_video — creates a
    video_id, copies the file into UPLOAD_DIR, extracts the first-frame thumbnail.

    Call from the pod itself (localhost) OR from the browser — either works.
    """
    src = Path(path)
    if not src.is_absolute():
        raise HTTPException(status_code=400, detail="path must be absolute")
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {src}")

    video_id = str(uuid.uuid4())[:8]
    suffix = src.suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".webm", ".m4v", ".avi"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension {suffix}. Expected mp4/mov/webm/m4v/avi.",
        )
    dst = UPLOAD_DIR / f"{video_id}{suffix}"
    shutil.copyfile(src, dst)

    first_frame_url, first_frame_w, first_frame_h = _extract_first_frame(
        dst, video_id,
    )

    return {
        "video_id": video_id,
        "filename": dst.name,
        "size_kb": dst.stat().st_size // 1024,
        "first_frame_url": first_frame_url,
        "first_frame_width": first_frame_w,
        "first_frame_height": first_frame_h,
        "imported_from": str(src),
    }
