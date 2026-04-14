"""Ear angle extraction from segmentation masks.

Computes the angle of each ear relative to the dorsal head axis
using geometric analysis of the segmentation mask contours.

Published literature reference:
- McLennan et al. (2019): SPFES ear-position action unit
- Reefmann et al. (2009): ear posture taxonomy (forward, backward, asymmetric, passive)
- Boissy et al. (2011): ear posture as emotional valence indicator
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from backend.config import EAR_DOWN_THRESHOLD_DEG, EAR_UP_THRESHOLD_DEG
from backend.models import EarAngleResult, EarPosition, SegmentationResult

logger = logging.getLogger(__name__)


def _decode_mask(b64_png: str) -> np.ndarray:
    """Decode a base64 PNG mask to a binary numpy array."""
    data = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.array(img) > 127


def _compute_mask_angle(mask: np.ndarray) -> Optional[tuple[float, float]]:
    """Compute the major axis angle of a mask region.

    Uses PCA on the mask's nonzero coordinates to find the principal axis,
    then falls back to fitEllipse or minAreaRect if PCA is unreliable.

    Returns:
        (angle_degrees, quality) where angle is relative to horizontal
        (0 = horizontal, 90 = vertical, positive = counter-clockwise)
        and quality is 0-1 indicating measurement confidence.
    """
    coords = np.where(mask)
    if len(coords[0]) < 10:
        return None

    # Points as (x, y) for geometric analysis
    points = np.column_stack((coords[1], coords[0])).astype(np.float64)
    n_points = len(points)

    # Method 1: PCA on mask coordinates
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Major axis is the eigenvector with largest eigenvalue
    major_axis = eigenvectors[:, -1]

    # Angle of major axis relative to horizontal
    angle_rad = np.arctan2(major_axis[1], major_axis[0])
    angle_deg = np.degrees(angle_rad)

    # Quality metric: elongation ratio (more elongated = better angle estimate)
    if eigenvalues[0] > 0:
        elongation = eigenvalues[-1] / eigenvalues[0]
    else:
        elongation = 1.0

    # Quality: higher elongation + more points = better
    quality = min(1.0, (elongation / 10.0) * min(1.0, n_points / 100.0))

    # Method 2: Try fitEllipse for refinement if enough points
    if n_points >= 5:
        try:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 5:
                    ellipse = cv2.fitEllipse(largest)
                    # ellipse returns ((cx, cy), (w, h), angle)
                    # OpenCV angle is clockwise from horizontal
                    ellipse_angle = -ellipse[2]  # Convert to CCW convention
                    ellipse_w, ellipse_h = ellipse[1]
                    if ellipse_w > 0:
                        ellipse_elongation = ellipse_h / ellipse_w
                        if ellipse_elongation > 1.5:
                            angle_deg = ellipse_angle
                            quality = min(1.0, quality + 0.2)
        except cv2.error:
            pass

    return angle_deg, quality


def _compute_head_midline(head_mask: np.ndarray) -> Optional[float]:
    """Estimate the dorsal midline angle of the head (fallback).

    Uses PCA on the head mask. For frontal photos this gives the
    left-right axis across the face, not the actual nose-to-poll axis.
    Use _compute_anatomical_midline() when nose + ears are available.
    """
    result = _compute_mask_angle(head_mask)
    if result is None:
        return None
    angle, _ = result
    return angle


def _mask_centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
    """Return (cx, cy) centroid of a mask, or None if empty."""
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    return float(coords[1].mean()), float(coords[0].mean())


def _compute_anatomical_midline(
    nose_mask: Optional[np.ndarray],
    left_ear_mask: Optional[np.ndarray],
    right_ear_mask: Optional[np.ndarray],
) -> Optional[tuple[float, tuple[float, float]]]:
    """Compute the dorsal head axis from nose + ear landmarks.

    Returns (angle_deg, head_center) where:
    - angle_deg is the angle of the axis pointing from nose toward the
      between-ears midpoint (the "up" direction of the head)
    - head_center is (cx, cy) of the midpoint between ears, used as a
      reference for ear-base disambiguation

    The axis direction:
    - In a frontal photo with head upright: angle ~ +90 (points up the image)
    - In a profile photo with nose forward-right: angle ~ 0 or 180
    """
    nose_c = _mask_centroid(nose_mask) if nose_mask is not None else None
    left_c = _mask_centroid(left_ear_mask) if left_ear_mask is not None else None
    right_c = _mask_centroid(right_ear_mask) if right_ear_mask is not None else None

    if nose_c is None:
        return None

    # Ear midpoint (use whatever ears we have)
    if left_c and right_c:
        ear_mid = ((left_c[0] + right_c[0]) / 2, (left_c[1] + right_c[1]) / 2)
    elif left_c:
        ear_mid = left_c
    elif right_c:
        ear_mid = right_c
    else:
        return None

    # Vector from nose to ear-midpoint = "head up" direction
    dx = ear_mid[0] - nose_c[0]
    dy = ear_mid[1] - nose_c[1]
    # In image coordinates, y increases downward — flip for math convention
    angle_rad = np.arctan2(-dy, dx)
    angle_deg = float(np.degrees(angle_rad))
    return angle_deg, ear_mid


def _compute_ear_direction(
    ear_mask: np.ndarray, head_center: tuple[float, float]
) -> Optional[tuple[float, float]]:
    """Compute the ear's pointing direction (base → tip) as a unit vector.

    Base = point on the ear mask closest to head center.
    Tip = point on the ear mask farthest from head center.
    Direction = tip - base, normalized.
    """
    coords = np.where(ear_mask)
    if len(coords[0]) < 10:
        return None

    xs = coords[1].astype(np.float64)
    ys = coords[0].astype(np.float64)
    hx, hy = head_center

    dists = np.sqrt((xs - hx) ** 2 + (ys - hy) ** 2)
    base_idx = int(np.argmin(dists))
    tip_idx = int(np.argmax(dists))

    bx, by = xs[base_idx], ys[base_idx]
    tx, ty = xs[tip_idx], ys[tip_idx]

    dx, dy = tx - bx, ty - by
    mag = np.sqrt(dx * dx + dy * dy)
    if mag < 1e-6:
        return None
    return float(dx / mag), float(dy / mag)


def _classify_ear_position(angle_relative_to_head: float) -> EarPosition:
    """Classify ear position based on angle relative to dorsal head axis.

    Thresholds derived from SPFES literature:
    - Up/Alert: ear rotated > 30° above the head axis (forward and up)
    - Neutral: ear between -10° and 30° relative to head axis
    - Down/Back: ear rotated > 10° below the head axis
    """
    if angle_relative_to_head > EAR_UP_THRESHOLD_DEG:
        return EarPosition.UP
    elif angle_relative_to_head < EAR_DOWN_THRESHOLD_DEG:
        return EarPosition.DOWN
    else:
        return EarPosition.NEUTRAL


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-90, 90] range.

    PCA eigenvectors have a 180-degree direction ambiguity — the axis
    can point either way. Since ear angles in the SPFES context are
    always in roughly [-60, +60], anything outside ±90 is a PCA flip.
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


def extract_ear_angles(
    segmentation: SegmentationResult,
) -> EarAngleResult:
    """Extract ear angles from segmentation masks.

    When nose + ears are all available (SAM 3), uses anatomical midline:
      1. Head "up" axis from nose → ear-midpoint
      2. For each ear, compute (base → tip) direction vector
      3. Measure ear angle relative to head-horizontal (perpendicular to head-up)
    Falls back to PCA + head-mask midline if landmarks are missing.
    """
    result = EarAngleResult(photo_id=segmentation.photo_id)

    # Decode available masks once
    masks_data = {}
    for k in ("head", "left_ear", "right_ear", "nose"):
        if k in segmentation.masks:
            masks_data[k] = _decode_mask(segmentation.masks[k])

    anatomical = _compute_anatomical_midline(
        masks_data.get("nose"),
        masks_data.get("left_ear"),
        masks_data.get("right_ear"),
    )

    if anatomical is not None:
        # Preferred path: use anatomical midline + directed ear vectors
        head_up_angle, head_center = anatomical
        # Head-horizontal axis is perpendicular to head-up axis
        head_horizontal_angle = head_up_angle - 90.0
        result.head_midline_angle_deg = head_up_angle

        for side, mask_key in (("left", "left_ear"), ("right", "right_ear")):
            ear_mask = masks_data.get(mask_key)
            if ear_mask is None:
                continue
            direction = _compute_ear_direction(ear_mask, head_center)
            if direction is None:
                continue
            dx, dy = direction
            # Angle of ear direction in image (standard math convention: y up)
            ear_angle_world = float(np.degrees(np.arctan2(-dy, dx)))
            # Angle relative to head-horizontal: + = up, - = down
            rel = _normalize_angle(ear_angle_world - head_horizontal_angle)
            # For left ear (viewer's left, negative x direction from center),
            # flip sign so "up" means forward/up regardless of side
            if side == "left":
                rel = -rel + 180 if rel > 0 else -rel - 180
                rel = _normalize_angle(rel)
            setattr(result, f"{side}_ear_angle_deg", rel)
            setattr(result, f"{side}_ear_position", _classify_ear_position(rel))
            result.measurement_quality = max(result.measurement_quality, 0.9)
        return result

    # Fallback: PCA on ear masks + head-mask major axis as midline
    head_midline = None
    if "head" in masks_data:
        head_midline = _compute_head_midline(masks_data["head"])
        if head_midline is not None:
            result.head_midline_angle_deg = head_midline

    for side, mask_key in (("left", "left_ear"), ("right", "right_ear")):
        ear_mask = masks_data.get(mask_key)
        if ear_mask is None:
            continue
        r = _compute_mask_angle(ear_mask)
        if r is None:
            continue
        angle, quality = r
        if head_midline is not None:
            rel = _normalize_angle(angle - head_midline)
        else:
            rel = _normalize_angle(angle)
        setattr(result, f"{side}_ear_angle_deg", rel)
        setattr(result, f"{side}_ear_position", _classify_ear_position(rel))
        result.measurement_quality = max(result.measurement_quality, quality)

    return result
