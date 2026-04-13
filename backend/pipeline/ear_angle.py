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
    """Estimate the dorsal midline angle of the head.

    The head midline is approximated as the major axis of the head mask,
    which should align roughly with the nose-to-poll axis.
    """
    result = _compute_mask_angle(head_mask)
    if result is None:
        return None
    angle, _ = result
    return angle


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

    Computes the angle of each ear relative to the head's dorsal midline,
    then classifies each ear as up/neutral/down per SPFES thresholds.
    """
    result = EarAngleResult(photo_id=segmentation.photo_id)

    # Get head midline
    head_midline = None
    if "head" in segmentation.masks:
        head_mask = _decode_mask(segmentation.masks["head"])
        head_midline = _compute_head_midline(head_mask)
        if head_midline is not None:
            result.head_midline_angle_deg = head_midline

    # Process left ear
    if "left_ear" in segmentation.masks:
        left_mask = _decode_mask(segmentation.masks["left_ear"])
        left_result = _compute_mask_angle(left_mask)
        if left_result is not None:
            angle, quality = left_result
            if head_midline is not None:
                relative_angle = _normalize_angle(angle - head_midline)
            else:
                relative_angle = _normalize_angle(angle)
            result.left_ear_angle_deg = relative_angle
            result.left_ear_position = _classify_ear_position(relative_angle)
            result.measurement_quality = max(result.measurement_quality, quality)

    # Process right ear
    if "right_ear" in segmentation.masks:
        right_mask = _decode_mask(segmentation.masks["right_ear"])
        right_result = _compute_mask_angle(right_mask)
        if right_result is not None:
            angle, quality = right_result
            if head_midline is not None:
                relative_angle = _normalize_angle(angle - head_midline)
            else:
                relative_angle = _normalize_angle(angle)
            result.right_ear_angle_deg = relative_angle
            result.right_ear_position = _classify_ear_position(relative_angle)
            result.measurement_quality = max(result.measurement_quality, quality)

    return result
