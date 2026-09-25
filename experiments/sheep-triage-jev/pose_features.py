"""Per-track pose summaries from SamSeesSheep YOLO-pose output.

Keypoint order is v0.7 (no eye keypoints):

    0 nose, 1 L_ear_base, 2 R_ear_base, 3 L_ear_tip, 4 R_ear_tip

Left/right is image-space, matching the labeling schema. Ear angle is the
absolute angle between the nose-to-ear-midline vector and the ear base-to-tip
vector — the same geometry as ``sheep-yolo/scripts/render_ekg.py``. The
summary is a geometric description of a track. It is not a pain or welfare
score.
"""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Mapping, Sequence

KPT_NAMES: tuple[str, ...] = (
    "nose",
    "L_ear_base",
    "R_ear_base",
    "L_ear_tip",
    "R_ear_tip",
)
NOSE, L_BASE, R_BASE, L_TIP, R_TIP = range(5)

Point = Sequence[float]  # x, y, conf
Box = Sequence[float]  # xyxy


def _ang(midline: tuple[float, float], ear: tuple[float, float]) -> float:
    cross = midline[0] * ear[1] - midline[1] * ear[0]
    dot = midline[0] * ear[0] + midline[1] * ear[1]
    return math.degrees(math.atan2(cross, dot))


def _ok(pt: Point, kpt_conf: float) -> bool:
    return len(pt) >= 3 and pt[2] > kpt_conf


def ear_angles(
    kpts: Sequence[Point], kpt_conf: float
) -> tuple[float | None, float | None]:
    """Return (left_deg, right_deg). None when that ear cannot be measured."""
    if len(kpts) < 5:
        return None, None
    nose, lb, rb, lt, rt = (kpts[i] for i in range(5))
    if not (_ok(nose, kpt_conf) and _ok(lb, kpt_conf) and _ok(rb, kpt_conf)):
        return None, None
    mid = ((lb[0] + rb[0]) / 2.0, (lb[1] + rb[1]) / 2.0)
    midline = (nose[0] - mid[0], nose[1] - mid[1])
    if midline[0] * midline[0] + midline[1] * midline[1] < 1e-6:
        return None, None

    def _one(base: Point, tip: Point) -> float | None:
        if not _ok(tip, kpt_conf):
            return None
        vec = (tip[0] - base[0], tip[1] - base[1])
        if vec[0] * vec[0] + vec[1] * vec[1] < 1e-6:
            return None
        return abs(_ang(midline, vec))

    return _one(lb, lt), _one(rb, rt)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.pstdev(values))


def _round(value: float | None, ndigits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def summarize_track(
    track_id: int,
    observations: list[dict],
    *,
    n_source_frames: int,
    fps: float,
    kpt_conf: float,
) -> dict:
    """Collapse one track's boxes and keypoints into a JSON-ready summary."""
    observations = sorted(observations, key=lambda obs: obs["frame"])
    n = len(observations)
    left: list[float] = []
    right: list[float] = []
    vis = [0] * len(KPT_NAMES)
    widths: list[float] = []
    heights: list[float] = []
    centroids: list[tuple[float, float]] = []

    for obs in observations:
        kpts = obs["kpts"]
        for i, name in enumerate(KPT_NAMES):
            if i < len(kpts) and _ok(kpts[i], kpt_conf):
                vis[i] += 1
        la, ra = ear_angles(kpts, kpt_conf)
        if la is not None:
            left.append(la)
        if ra is not None:
            right.append(ra)
        x1, y1, x2, y2 = obs["box"]
        widths.append(float(x2) - float(x1))
        heights.append(float(y2) - float(y1))
        centroids.append(((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0))

    path = 0.0
    for (x0, y0), (x1, y1) in zip(centroids, centroids[1:]):
        path += math.hypot(x1 - x0, y1 - y0)
    cxs = [c[0] for c in centroids]
    cys = [c[1] for c in centroids]
    frames = [int(obs["frame"]) for obs in observations]
    gaps = 0
    for a, b in zip(frames, frames[1:]):
        if b - a > 1:
            gaps += b - a - 1

    facing = 0
    for obs in observations:
        kpts = obs["kpts"]
        if (
            len(kpts) >= 3
            and _ok(kpts[NOSE], kpt_conf)
            and _ok(kpts[L_BASE], kpt_conf)
            and _ok(kpts[R_BASE], kpt_conf)
        ):
            facing += 1

    return {
        "track_id": int(track_id),
        "n_frames": n,
        "frame_first": frames[0],
        "frame_last": frames[-1],
        "gap_frames": gaps,
        "coverage": _round(n / n_source_frames if n_source_frames else 0.0),
        "duration_s": _round(n / fps if fps else 0.0),
        "keypoint_visibility": {
            name: _round(vis[i] / n if n else 0.0) for i, name in enumerate(KPT_NAMES)
        },
        "facing_fraction": _round(facing / n if n else 0.0),
        "ear_angle_deg": {
            "left_median": _round(_median(left)),
            "left_std": _round(_std(left)),
            "right_median": _round(_median(right)),
            "right_std": _round(_std(right)),
            "n_left": len(left),
            "n_right": len(right),
        },
        "motion": {
            "centroid_path_px": _round(path),
            "centroid_std_x_px": _round(_std(cxs)),
            "centroid_std_y_px": _round(_std(cys)),
            "bbox_median_w_px": _round(_median(widths)),
            "bbox_median_h_px": _round(_median(heights)),
        },
    }


def group_frames(per_frame: Iterable[Mapping[int, dict]]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for frame_idx, tracks in enumerate(per_frame):
        for track_id, obs in tracks.items():
            grouped.setdefault(int(track_id), []).append(
                {
                    "frame": frame_idx,
                    "box": list(obs["box"]),
                    "kpts": [list(pt) for pt in obs["kpts"]],
                }
            )
    return grouped


def summarize_frames(
    per_frame: Sequence[Mapping[int, dict]],
    *,
    fps: float,
    kpt_conf: float,
    min_frames: int,
) -> tuple[list[dict], list[int]]:
    """Return (kept summaries, dropped track ids)."""
    grouped = group_frames(per_frame)
    kept: list[dict] = []
    dropped: list[int] = []
    n_source = len(per_frame)
    for track_id in sorted(grouped):
        obs = grouped[track_id]
        if len(obs) < min_frames:
            dropped.append(track_id)
            continue
        kept.append(
            summarize_track(
                track_id,
                obs,
                n_source_frames=n_source,
                fps=fps,
                kpt_conf=kpt_conf,
            )
        )
    return kept, dropped


def state_for_jev(summary: dict) -> dict:
    """Text/JSON state for one TypeSafe Jev call. No image, no secrets."""
    return {
        "task": (
            "Triage one tracked sheep from pose numbers only. "
            "No image is attached. These are geometric measurements "
            "from a five-keypoint sheep-head model."
        ),
        "keypoint_order": list(KPT_NAMES),
        "field_notes": {
            "ear_angle_deg": (
                "Absolute degrees between the nose-to-ear-midline vector "
                "and that ear's base-to-tip vector. left_std / right_std "
                "are the spread across frames where the ear was measurable."
            ),
            "keypoint_visibility": "Fraction of this track's frames where that keypoint confidence cleared the threshold.",
            "facing_fraction": "Fraction of frames with nose and both ear bases visible.",
            "motion": "Centroid path and spread are in source-frame pixels.",
            "coverage": "Fraction of source frames that contain this track id.",
        },
        "track": summary,
    }


def _kpt(x: float, y: float, conf: float = 1.0) -> list[float]:
    return [x, y, conf]


def _head(cx: float, cy: float, *, left_along_midline: bool, right_conf: float) -> dict:
    """Schematic head. A horizontal ear tip is 90°; one along the midline is 0°."""
    nose = _kpt(cx, cy - 60)
    lb = _kpt(cx - 30, cy)
    rb = _kpt(cx + 30, cy)
    if left_along_midline:
        lt = _kpt(cx - 30, cy - 60)
    else:
        lt = _kpt(cx - 80, cy)
    rt = _kpt(cx + 80, cy, right_conf)
    box = [cx - 90, cy - 80, cx + 90, cy + 40]
    return {"box": box, "kpts": [nose, lb, rb, lt, rt]}


def synthetic_frames() -> tuple[list[dict[int, dict]], float]:
    """Two tracks, no weights and no video.

    Track 1 is a still head with both ears out (stable 90°).
    Track 7 alternates that ear between 90° and 0°, walks sideways, and
    never shows the other ear. Track 99 is a one-frame flicker so the
    min-frames filter has something to drop.
    """
    frames: list[dict[int, dict]] = []
    for i in range(8):
        steady = _head(400, 300, left_along_midline=False, right_conf=1.0)
        jumpy = _head(
            80 + i * 40,
            500,
            left_along_midline=(i % 2 == 1),
            right_conf=0.0,
        )
        frame: dict[int, dict] = {1: steady, 7: jumpy}
        if i == 0:
            frame[99] = _head(10, 10, left_along_midline=False, right_conf=1.0)
        frames.append(frame)
    return frames, 30.0
