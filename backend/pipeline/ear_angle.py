"""Ear angle extraction from segmentation masks.

Computes the angle of each ear relative to the dorsal head axis
using geometric analysis of the segmentation mask contours.

Published literature reference:
- McLennan et al. (2019): SPFES ear-position action unit
- Reefmann et al. (2009): ear posture taxonomy (forward, backward, asymmetric, passive)
- Boissy et al. (2011): ear posture as emotional valence indicator
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from backend.config import EAR_DOWN_THRESHOLD_DEG, EAR_UP_THRESHOLD_DEG
from backend.models import EarPosition

logger = logging.getLogger(__name__)


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

    Returns (angle_deg, head_center) where angle is the direction from
    the nose toward the ear-midpoint — the "head up" direction. Unlike
    the head-mask PCA version, this is unambiguous (no sign flip) and
    doesn't wobble frame-to-frame with mask shape noise, because both
    endpoints are anatomical landmarks. Preferred when nose is detected.
    """
    nose_c = _mask_centroid(nose_mask) if nose_mask is not None else None
    if nose_c is None:
        return None

    left_c = _mask_centroid(left_ear_mask) if left_ear_mask is not None else None
    right_c = _mask_centroid(right_ear_mask) if right_ear_mask is not None else None
    if left_c and right_c:
        ear_mid = ((left_c[0] + right_c[0]) / 2, (left_c[1] + right_c[1]) / 2)
    elif left_c:
        ear_mid = left_c
    elif right_c:
        ear_mid = right_c
    else:
        return None

    dx = ear_mid[0] - nose_c[0]
    dy = ear_mid[1] - nose_c[1]
    # Image coords: y increases downward; flip for math convention
    angle_deg = float(np.degrees(np.arctan2(-dy, dx)))
    return angle_deg, ear_mid


def _compute_head_midline_pca(
    head_mask: np.ndarray,
    left_ear_mask: Optional[np.ndarray] = None,
    right_ear_mask: Optional[np.ndarray] = None,
) -> Optional[tuple[float, tuple[float, float]]]:
    """Compute dorsal head axis from head mask alone (no nose required).

    PCA on the head mask gives an undirected long axis. The ear-midpoint
    position relative to the head centroid resolves which end of that axis
    is "up" (the poll / dorsal side). Falls back to image-vertical "up"
    when neither ear is available.

    Returns (head_up_angle_deg, head_center) matching the shape of
    _compute_anatomical_midline. head_center is the ear-midpoint (or, if
    no ears, the head centroid) so _compute_ear_direction can resolve
    base vs. tip the same way.
    """
    head_c = _mask_centroid(head_mask)
    if head_c is None:
        return None

    coords = np.where(head_mask)
    if len(coords[0]) < 20:
        return None
    points = np.column_stack((coords[1], coords[0])).astype(np.float64)
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, -1]  # (dx, dy) in image coords, undirected

    left_c = _mask_centroid(left_ear_mask) if left_ear_mask is not None else None
    right_c = _mask_centroid(right_ear_mask) if right_ear_mask is not None else None
    if left_c and right_c:
        ear_mid = ((left_c[0] + right_c[0]) / 2, (left_c[1] + right_c[1]) / 2)
    elif left_c:
        ear_mid = left_c
    elif right_c:
        ear_mid = right_c
    else:
        # No ears to disambiguate — assume head-up = image-up
        return 90.0, head_c

    # Flip axis if it points away from the ears (ears mark the dorsal end)
    ear_dx = ear_mid[0] - head_c[0]
    ear_dy = ear_mid[1] - head_c[1]
    if ear_dx * major[0] + ear_dy * major[1] < 0:
        major = -major

    # Image coords: y increases downward; flip for math convention
    angle_deg = float(np.degrees(np.arctan2(-major[1], major[0])))
    return angle_deg, ear_mid


def _extract_ear_keypoints(
    ear_mask: np.ndarray, head_center: tuple[float, float]
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """Return (base_xy, tip_xy) pixel coordinates on the ear mask.

    base = point on the ear mask closest to head_center
    tip  = point on the ear mask farthest from head_center

    Matches the YOLO-pose keypoint derivation spec: pass the head-mask
    centroid as head_center for the canonical ear-base/tip keypoints
    (slots 1-4 in the [nose, L-base, R-base, L-tip, R-tip] schema).
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
    return (
        (float(xs[base_idx]), float(ys[base_idx])),
        (float(xs[tip_idx]),  float(ys[tip_idx])),
    )


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


