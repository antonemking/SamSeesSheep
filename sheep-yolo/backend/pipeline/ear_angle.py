"""Ear-angle extraction from binary segmentation masks.

Mask-in, angle-out — works the same whether the masks came from SAM 3 (v1) or
from YOLOE-seg (v2). Kept geometrically identical to sheep-seg v1 so that the
two pipelines are directly comparable on the same clips.

Published references:
- McLennan et al. 2019: SPFES ear-position action unit
- Reefmann et al. 2009: ear posture taxonomy (forward/back/asymmetric/passive)
- Boissy et al. 2011: ear posture as emotional valence indicator
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from backend.config import EAR_DOWN_THRESHOLD_DEG, EAR_UP_THRESHOLD_DEG
from backend.models import EarPosition

logger = logging.getLogger(__name__)


def _mask_centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    return float(coords[1].mean()), float(coords[0].mean())


def compute_anatomical_midline(
    nose_mask: Optional[np.ndarray],
    left_ear_mask: Optional[np.ndarray],
    right_ear_mask: Optional[np.ndarray],
) -> Optional[tuple[float, tuple[float, float]]]:
    """Dorsal head axis from nose + ear landmarks. Nose → ear-midpoint is the
    "head up" direction — unambiguous, no PCA sign flip, lower frame-to-frame
    wobble than a head-mask PCA axis."""
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
    angle_deg = float(np.degrees(np.arctan2(-dy, dx)))
    return angle_deg, ear_mid


def compute_head_midline_pca(
    head_mask: np.ndarray,
    left_ear_mask: Optional[np.ndarray] = None,
    right_ear_mask: Optional[np.ndarray] = None,
) -> Optional[tuple[float, tuple[float, float]]]:
    """Fallback when nose isn't detected: PCA on head mask, disambiguated
    toward whichever end the ear-midpoint sits on."""
    head_c = _mask_centroid(head_mask)
    if head_c is None:
        return None

    coords = np.where(head_mask)
    if len(coords[0]) < 20:
        return None
    points = np.column_stack((coords[1], coords[0])).astype(np.float64)
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, -1]

    left_c = _mask_centroid(left_ear_mask) if left_ear_mask is not None else None
    right_c = _mask_centroid(right_ear_mask) if right_ear_mask is not None else None
    if left_c and right_c:
        ear_mid = ((left_c[0] + right_c[0]) / 2, (left_c[1] + right_c[1]) / 2)
    elif left_c:
        ear_mid = left_c
    elif right_c:
        ear_mid = right_c
    else:
        return 90.0, head_c

    ear_dx = ear_mid[0] - head_c[0]
    ear_dy = ear_mid[1] - head_c[1]
    if ear_dx * major[0] + ear_dy * major[1] < 0:
        major = -major

    angle_deg = float(np.degrees(np.arctan2(-major[1], major[0])))
    return angle_deg, ear_mid


def compute_ear_direction(
    ear_mask: np.ndarray, head_center: tuple[float, float]
) -> Optional[tuple[float, float]]:
    """Base → tip unit vector. Base = mask point closest to head center;
    tip = mask point farthest from it."""
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


def classify_ear_position(angle_relative_to_head: float) -> EarPosition:
    if angle_relative_to_head > EAR_UP_THRESHOLD_DEG:
        return EarPosition.UP
    if angle_relative_to_head < EAR_DOWN_THRESHOLD_DEG:
        return EarPosition.DOWN
    return EarPosition.NEUTRAL


def normalize_angle(angle: float) -> float:
    """Collapse to [-90, 90]. PCA and arctan2 produce 180° flips that are
    meaningless in the SPFES window (ear angles live in roughly [-60, +60])."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


def ear_angle_from_masks(
    head_mask: Optional[np.ndarray],
    left_ear_mask: Optional[np.ndarray],
    right_ear_mask: Optional[np.ndarray],
    nose_mask: Optional[np.ndarray],
) -> dict:
    """End-to-end: four optional boolean masks → angles dict.

    Returns a dict with any subset of:
      left_ear_angle_deg, right_ear_angle_deg,
      left_ear_position, right_ear_position,
      head_midline_angle_deg, midline_source.
    Keys are omitted when the input doesn't permit computing them.
    """
    out: dict = {}
    if head_mask is None:
        return out
    if left_ear_mask is None and right_ear_mask is None:
        return out

    midline = compute_anatomical_midline(nose_mask, left_ear_mask, right_ear_mask)
    source = "nose"
    if midline is None:
        midline = compute_head_midline_pca(head_mask, left_ear_mask, right_ear_mask)
        source = "head_pca" if midline is not None else None
    if midline is None:
        return out

    head_up_angle, head_center = midline
    head_horizontal = head_up_angle - 90.0
    out["head_midline_angle_deg"] = head_up_angle
    out["midline_source"] = source

    for side, mask in (("left", left_ear_mask), ("right", right_ear_mask)):
        if mask is None:
            continue
        direction = compute_ear_direction(mask, head_center)
        if direction is None:
            continue
        dx, dy = direction
        world_angle = float(np.degrees(np.arctan2(-dy, dx)))
        rel = normalize_angle(world_angle - head_horizontal)
        # Mirror left-side so "up" is positive on both sides.
        if side == "left":
            rel = normalize_angle(-rel + 180) if rel > 0 else normalize_angle(-rel - 180)
        out[f"{side}_ear_angle_deg"] = rel
        out[f"{side}_ear_position"] = classify_ear_position(rel).value

    return out
