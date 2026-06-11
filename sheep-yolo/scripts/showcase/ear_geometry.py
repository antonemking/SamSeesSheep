"""Shared sheep-head keypoint geometry for showcase renderers."""
from __future__ import annotations

from typing import Iterable

import numpy as np

# YOLO-pose keypoint order used by SamSeesSheep.
# 0 nose, 1 left ear base, 2 right ear base, 3 left ear tip, 4 right ear tip.
SKEL = [(0, 1), (0, 2), (1, 3), (2, 4)]
KPT_TH = 0.4


def signed_angle_deg(midline: np.ndarray, ear_vector: np.ndarray) -> float:
    """Signed angle between muzzle midline and an ear vector in degrees."""
    return float(
        np.degrees(
            np.arctan2(
                midline[0] * ear_vector[1] - midline[1] * ear_vector[0],
                midline[0] * ear_vector[0] + midline[1] * ear_vector[1],
            )
        )
    )


def ear_angles(kpts: np.ndarray, kpt_th: float = KPT_TH) -> tuple[float, float]:
    """Return ``(left_angle, right_angle)`` for one 5x3 keypoint array.

    Values are absolute degrees between each ear and the muzzle midline. Missing
    or low-confidence points return ``np.nan`` for that ear. Nose and both ear
    bases are required because they define the head axis.
    """
    nose, lb, rb, lt, rt = kpts
    if min(nose[2], lb[2], rb[2]) <= kpt_th:
        return np.nan, np.nan
    ear_midpoint = (lb[:2] + rb[:2]) / 2
    muzzle_midline = nose[:2] - ear_midpoint
    left = abs(signed_angle_deg(muzzle_midline, lt[:2] - lb[:2])) if lt[2] > kpt_th else np.nan
    right = abs(signed_angle_deg(muzzle_midline, rt[:2] - rb[:2])) if rt[2] > kpt_th else np.nan
    return left, right


def mean_ear_angle(kpts: np.ndarray, kpt_th: float = KPT_TH) -> float:
    """Mean readable ear angle for one sheep, or ``np.nan`` if neither ear reads."""
    left, right = ear_angles(kpts, kpt_th=kpt_th)
    vals = [v for v in (left, right) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def nanmedian_baseline(series: Iterable[np.ndarray]) -> float:
    """Median across several angle arrays, ignoring NaNs."""
    arrays = [np.asarray(a, dtype=float).ravel() for a in series if len(a)]
    if not arrays:
        return np.nan
    merged = np.concatenate(arrays)
    return float(np.nanmedian(merged)) if np.isfinite(merged).any() else np.nan
