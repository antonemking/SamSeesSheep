"""Confidence-aware `supervision` drawing helpers for sheep showcase renders.

Tracking stays in Ultralytics. This module is drawing-only and should be used by
optional artifact renderers, not the core FastAPI inference path. Run CLIs that
import this under ``uv run --with supervision ...`` if supervision is not already
installed in the sheep-yolo environment.
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv  # type: ignore[reportMissingImports]

try:  # script execution: python scripts/showcase/foo.py
    from .ear_geometry import KPT_TH, SKEL
except ImportError:  # pragma: no cover
    from ear_geometry import KPT_TH, SKEL  # type: ignore

SV_SKEL = [(a + 1, b + 1) for a, b in SKEL]  # supervision edges are 1-based


def skeleton(scene: np.ndarray, kpts: np.ndarray, color: sv.Color, *, kpt_th: float = KPT_TH,
             vrad: int = 5, thickness: int = 2) -> np.ndarray:
    """Draw one 5-point sheep-head skeleton with SamSeesSheep confidence filtering."""
    xy = kpts[:, :2].copy().astype(float)
    conf = kpts[:, 2].astype(float)
    xy[conf <= kpt_th] = 0
    sv_kpts = sv.KeyPoints(xy=xy[None, ...], confidence=conf[None, ...])
    scene = sv.EdgeAnnotator(color=color, thickness=thickness, edges=SV_SKEL).annotate(scene, sv_kpts)
    for (x, y), score in zip(kpts[:, :2], conf):
        if score > kpt_th:
            cv2.circle(scene, (int(x), int(y)), vrad, (color.b, color.g, color.r), -1)
    return scene


def boxes_labels(scene: np.ndarray, xyxy: list | np.ndarray, class_ids: list | np.ndarray,
                 labels: list[str], palette: sv.ColorPalette, *, thickness: int = 3,
                 roundness: float = 0.5, text_scale: float = 0.55) -> np.ndarray:
    """Draw rounded boxes and color-coded label chips."""
    if len(xyxy) == 0:
        return scene
    detections = sv.Detections(xyxy=np.asarray(xyxy, dtype=float), class_id=np.asarray(class_ids, dtype=int))
    scene = sv.RoundBoxAnnotator(
        color=palette,
        color_lookup=sv.ColorLookup.CLASS,
        thickness=thickness,
        roundness=roundness,
    ).annotate(scene, detections)
    scene = sv.LabelAnnotator(
        color=palette,
        color_lookup=sv.ColorLookup.CLASS,
        text_color=sv.Color.WHITE,
        text_scale=text_scale,
        text_padding=6,
        border_radius=6,
        smart_position=True,
    ).annotate(scene, detections, labels=labels)
    return scene


def faint_flock(scene: np.ndarray, kpts: np.ndarray, *, kpt_th: float = KPT_TH,
                color: tuple[int, int, int] = (185, 185, 185)) -> np.ndarray:
    """Thin gray skeleton for unnamed background sheep."""
    pts = {j: tuple(kpts[j][:2].astype(int)) for j in range(5) if kpts[j][2] > kpt_th}
    for a, b in SKEL:
        if a in pts and b in pts:
            cv2.line(scene, pts[a], pts[b], color, 1)
    return scene
