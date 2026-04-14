"""Analysis routes — run the segmentation + ear angle pipeline."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from pydantic import BaseModel as PydanticBaseModel

from backend.config import RESULTS_DIR, SAMPLE_DIR, UPLOAD_DIR
from backend.models import EUPResult, NarrativeResult, PhotoAnalysis, SessionState
from fastapi.responses import Response

from backend.pipeline.depth import build_mesh_from_photo
from backend.pipeline.mesh3d import render_mesh_views, render_turntable
from backend.pipeline.ear_angle import extract_ear_angles, _decode_mask
from backend.pipeline.eup import compute_eup
from backend.pipeline.narrative import generate_narrative
from backend.pipeline.segment import segment_sheep_at_point, segment_sheep, segment_sheep_multipoint, segment_sheep_sam3

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])

# In-memory session state (single-user demo)
_session = SessionState()


def get_session() -> SessionState:
    """Get the current session state."""
    return _session


def reset_session():
    """Reset session state."""
    global _session
    _session = SessionState()


@router.post("/analyze/batch")
async def analyze_batch() -> SessionState:
    """Run the full pipeline on all uploaded photos."""
    photo_files = sorted(UPLOAD_DIR.glob("*"))
    photo_files = [f for f in photo_files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"}]

    if not photo_files:
        raise HTTPException(status_code=404, detail="No photos uploaded")

    reset_session()

    for photo_path in photo_files:
        image = cv2.imread(str(photo_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        photo_id = photo_path.stem

        seg_result = segment_sheep(image, photo_id)
        ear_result = extract_ear_angles(seg_result)

        analysis = PhotoAnalysis(
            photo_id=photo_id,
            filename=photo_path.name,
            image_width=w,
            image_height=h,
            segmentation=seg_result,
            ear_angles=ear_result,
        )
        _session.photos.append(analysis)

    _session.eup = compute_eup(_session.photos)
    return _session


class ClickRequest(PydanticBaseModel):
    photo_id: str
    x: float  # 0-1 normalized
    y: float  # 0-1 normalized


@router.post("/analyze/click")
async def analyze_click(req: ClickRequest) -> PhotoAnalysis:
    """Run SAM with a user-clicked point on the animal's face."""
    # Look in uploads first, then sample/
    matches = list(UPLOAD_DIR.glob(f"{req.photo_id}.*"))
    if not matches:
        matches = list(SAMPLE_DIR.glob(f"{req.photo_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Photo {req.photo_id} not found")

    photo_path = matches[0]
    image = cv2.imread(str(photo_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    h, w = image.shape[:2]
    px = int(req.x * w)
    py = int(req.y * h)

    seg_result = segment_sheep_at_point(image, req.photo_id, px, py)
    ear_result = extract_ear_angles(seg_result)

    analysis = PhotoAnalysis(
        photo_id=req.photo_id,
        filename=photo_path.name,
        image_width=w,
        image_height=h,
        image_url=f"/uploads/{photo_path.name}" if "uploads" in str(photo_path) else f"/sample/{photo_path.name}",
        segmentation=seg_result,
        ear_angles=ear_result,
    )

    _session.photos = [p for p in _session.photos if p.photo_id != req.photo_id]
    _session.photos.append(analysis)
    _session.eup = compute_eup(_session.photos)

    return analysis


class AutoSegmentRequest(PydanticBaseModel):
    photo_id: str
    subject: str = "sheep"  # or "goat"


@router.post("/analyze/auto")
async def analyze_auto(req: AutoSegmentRequest) -> PhotoAnalysis:
    """Auto-segment using SAM 3 with text prompts — no clicks needed."""
    matches = list(UPLOAD_DIR.glob(f"{req.photo_id}.*"))
    if not matches:
        matches = list(SAMPLE_DIR.glob(f"{req.photo_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Photo {req.photo_id} not found")

    photo_path = matches[0]
    image = cv2.imread(str(photo_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    h, w = image.shape[:2]
    seg_result = segment_sheep_sam3(image, req.photo_id, subject=req.subject)
    ear_result = extract_ear_angles(seg_result)

    analysis = PhotoAnalysis(
        photo_id=req.photo_id,
        filename=photo_path.name,
        image_width=w,
        image_height=h,
        image_url=f"/uploads/{photo_path.name}" if "uploads" in str(photo_path) else f"/sample/{photo_path.name}",
        segmentation=seg_result,
        ear_angles=ear_result,
    )

    _session.photos = [p for p in _session.photos if p.photo_id != req.photo_id]
    _session.photos.append(analysis)
    _session.eup = compute_eup(_session.photos)

    return analysis


class PointLabel(PydanticBaseModel):
    x: float  # 0-1 normalized
    y: float
    label: str  # "face", "left_ear", "right_ear"


class MultiClickRequest(PydanticBaseModel):
    photo_id: str
    points: list[PointLabel]


@router.post("/analyze/multiclick")
async def analyze_multiclick(req: MultiClickRequest) -> PhotoAnalysis:
    """Run SAM with face center + ear tip points for precise segmentation."""
    matches = list(UPLOAD_DIR.glob(f"{req.photo_id}.*"))
    if not matches:
        matches = list(SAMPLE_DIR.glob(f"{req.photo_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Photo {req.photo_id} not found")

    photo_path = matches[0]
    image = cv2.imread(str(photo_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    h, w = image.shape[:2]

    face_pt = None
    ear_pts = []
    eye_pts = []
    nose_pt = None

    for p in req.points:
        px, py = int(p.x * w), int(p.y * h)
        if p.label == "face":
            face_pt = (px, py)
        elif p.label in ("left_ear", "right_ear", "ear1", "ear2"):
            ear_pts.append((px, py))
        elif p.label in ("left_eye", "right_eye", "eye1", "eye2"):
            eye_pts.append((px, py))
        elif p.label == "nose":
            nose_pt = (px, py)

    if face_pt is None:
        raise HTTPException(status_code=400, detail="Face point is required")

    # Auto-assign left/right by x-position (smaller x = viewer's left)
    left_ear_pt = None
    right_ear_pt = None
    if len(ear_pts) >= 2:
        ear_pts.sort(key=lambda pt: pt[0])
        left_ear_pt = ear_pts[0]
        right_ear_pt = ear_pts[1]
    elif len(ear_pts) == 1:
        if ear_pts[0][0] < face_pt[0]:
            left_ear_pt = ear_pts[0]
        else:
            right_ear_pt = ear_pts[0]

    left_eye_pt = None
    right_eye_pt = None
    if len(eye_pts) >= 2:
        eye_pts.sort(key=lambda pt: pt[0])
        left_eye_pt = eye_pts[0]
        right_eye_pt = eye_pts[1]
    elif len(eye_pts) == 1:
        if eye_pts[0][0] < face_pt[0]:
            left_eye_pt = eye_pts[0]
        else:
            right_eye_pt = eye_pts[0]

    seg_result = segment_sheep_multipoint(
        image, req.photo_id, face_pt,
        left_ear_pt, right_ear_pt,
        left_eye_pt, right_eye_pt,
        nose_pt,
    )
    ear_result = extract_ear_angles(seg_result)

    analysis = PhotoAnalysis(
        photo_id=req.photo_id,
        filename=photo_path.name,
        image_width=w,
        image_height=h,
        image_url=f"/uploads/{photo_path.name}" if "uploads" in str(photo_path) else f"/sample/{photo_path.name}",
        segmentation=seg_result,
        ear_angles=ear_result,
    )

    _session.photos = [p for p in _session.photos if p.photo_id != req.photo_id]
    _session.photos.append(analysis)
    _session.eup = compute_eup(_session.photos)

    return analysis


class MeshRequest(PydanticBaseModel):
    photo_id: str


@router.post("/analyze/mesh")
async def generate_mesh(req: MeshRequest):
    """Generate a 3D GLB mesh from depth estimation + SAM mask."""
    # Find the photo
    matches = list(UPLOAD_DIR.glob(f"{req.photo_id}.*"))
    if not matches:
        matches = list(SAMPLE_DIR.glob(f"{req.photo_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Photo {req.photo_id} not found")

    photo_path = matches[0]
    image = cv2.imread(str(photo_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    # Get head mask from session
    photo_analysis = next((p for p in _session.photos if p.photo_id == req.photo_id), None)
    if photo_analysis is None or photo_analysis.segmentation is None:
        raise HTTPException(status_code=400, detail="Run segmentation first (click on face)")

    head_mask_b64 = photo_analysis.segmentation.masks.get("head")
    if not head_mask_b64:
        raise HTTPException(status_code=400, detail="No head mask found")

    head_mask = _decode_mask(head_mask_b64)
    mask_uint8 = (head_mask.astype(np.uint8) * 255)

    glb_bytes = build_mesh_from_photo(image, mask_uint8)
    if glb_bytes is None:
        raise HTTPException(status_code=500, detail="Depth mesh generation failed")

    # Save GLB to results directory for Three.js to load
    glb_filename = f"{req.photo_id}_mesh.glb"
    glb_path = RESULTS_DIR / glb_filename
    glb_path.write_bytes(glb_bytes)
    glb_url = f"/results/{glb_filename}"

    # Render turntable animation (24 frames, works without WebGL)
    turntable = render_turntable(glb_bytes)

    # Static views as additional fallback
    views = render_mesh_views(glb_bytes)

    return {
        "turntable": turntable,
        "views": views,
        "glb_url": glb_url,
        "photo_id": req.photo_id,
    }


@router.post("/analyze/{photo_id}")
async def analyze_photo(photo_id: str) -> PhotoAnalysis:
    """Run segmentation + ear angle extraction on a single photo."""
    matches = list(UPLOAD_DIR.glob(f"{photo_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Photo {photo_id} not found")

    photo_path = matches[0]
    image = cv2.imread(str(photo_path))
    if image is None:
        raise HTTPException(status_code=400, detail=f"Could not read image")

    h, w = image.shape[:2]
    seg_result = segment_sheep(image, photo_id)
    ear_result = extract_ear_angles(seg_result)

    analysis = PhotoAnalysis(
        photo_id=photo_id,
        filename=photo_path.name,
        image_width=w,
        image_height=h,
        segmentation=seg_result,
        ear_angles=ear_result,
    )

    _session.photos = [p for p in _session.photos if p.photo_id != photo_id]
    _session.photos.append(analysis)
    _session.eup = compute_eup(_session.photos)

    return analysis


@router.post("/narrative")
async def generate_narrative_endpoint(context: str = "") -> NarrativeResult:
    """Generate a Claude API narrative from current session results."""
    if _session.eup is None:
        raise HTTPException(status_code=400, detail="No analysis results. Run /analyze/batch first.")

    narrative = generate_narrative(_session.eup, additional_context=context)
    _session.narrative = narrative
    return narrative
