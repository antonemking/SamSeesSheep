"""Tracking cache utilities for SamSeesSheep showcase renderers.

The cache stores the expensive Ultralytics tracking pass once, then renderers can
iterate quickly over hand-picked windows/subjects. This replaces the old
``/tmp/per_frame_IMG_3651.pkl`` convention used by black-sheep-reid.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

TrackMap = dict[int, tuple[np.ndarray, np.ndarray]]  # track_id -> (xyxy, kpts)


def resolve_path(path: str | Path, base: Path = ROOT) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else base / p


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch/ultralytics tensor-like values to numpy without hard typing."""
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def frame_tracks(per_frame: list[TrackMap] | dict[int, TrackMap], frame_idx: int) -> TrackMap:
    """Return tracks for a frame from either the old list cache or new dict cache."""
    if isinstance(per_frame, dict):
        return per_frame.get(frame_idx, {})
    if 0 <= frame_idx < len(per_frame):
        return per_frame[frame_idx]
    return {}


def track_video(
    clip: Path,
    weights: Path,
    *,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
    imgsz: int | None = None,
    device: str | int | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run YOLO-pose tracking and return a pickle-friendly cache payload."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs: dict[str, Any] = {
        "source": str(clip),
        "tracker": tracker,
        "stream": True,
        "conf": conf,
        "verbose": False,
    }
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    if device is not None:
        kwargs["device"] = device

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    per_frame: list[TrackMap] = []
    brightness: dict[int, list[float]] = {}
    present: dict[int, list[int]] = {}

    for frame_idx, result in enumerate(model.track(**kwargs)):
        if max_frames is not None and frame_idx >= max_frames:
            break
        tracks: TrackMap = {}
        if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
            gray = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2GRAY)
            ids = _to_numpy(result.boxes.id).astype(int)
            xyxy = _to_numpy(result.boxes.xyxy)
            kpts = _to_numpy(result.keypoints.data)
            for j, tid in enumerate(ids):
                track_id = int(tid)
                box = xyxy[j].astype(float)
                kp = kpts[j].astype(float)
                tracks[track_id] = (box, kp)
                x1, y1, x2, y2 = [int(v) for v in box]
                crop = gray[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size:
                    brightness.setdefault(track_id, []).append(float(crop.mean()))
                present.setdefault(track_id, []).append(frame_idx)
        per_frame.append(tracks)

    return {
        "version": 1,
        "clip": str(clip),
        "weights": str(weights),
        "tracker": tracker,
        "conf": conf,
        "imgsz": imgsz,
        "device": device,
        "fps": float(fps),
        "width": width,
        "height": height,
        "per_frame": per_frame,
        "brightness": brightness,
        "present": present,
    }


def save_cache(payload: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(payload))


def load_cache(cache_path: Path) -> dict[str, Any]:
    payload = pickle.loads(cache_path.read_bytes())
    # Old black-sheep-reid caches were tuples: (per_frame, brightness, present).
    if isinstance(payload, tuple) and len(payload) == 3:
        per_frame, brightness, present = payload
        return {
            "version": 0,
            "clip": None,
            "weights": None,
            "tracker": None,
            "conf": None,
            "fps": 30.0,
            "per_frame": per_frame,
            "brightness": brightness,
            "present": present,
        }
    return cast(dict[str, Any], payload)


def top_tracks(
    payload: dict[str, Any],
    *,
    frame_start: int = 0,
    frame_end: int | None = None,
    limit: int = 6,
    min_frames: int = 15,
) -> list[int]:
    """Pick the most persistent track IDs inside a frame window."""
    present = payload.get("present") or {}
    if frame_end is None:
        frame_end = len(payload.get("per_frame", []))
    scored: list[tuple[int, int]] = []
    for raw_tid, frames in present.items():
        tid = int(raw_tid)
        n = sum(1 for f in frames if frame_start <= int(f) < frame_end)
        if n >= min_frames:
            scored.append((n, tid))
    return [tid for _, tid in sorted(scored, reverse=True)[:limit]]


def darkest_persistent_track(
    payload: dict[str, Any],
    *,
    exclude: set[int] | None = None,
    min_frames: int = 15,
) -> int:
    """Select the lowest-median-brightness persistent track.

    This is a heuristic for the black ewe demo, not learned cross-video ReID.
    """
    exclude = exclude or set()
    brightness = payload.get("brightness") or {}
    cand = []
    for raw_tid, vals in brightness.items():
        tid = int(raw_tid)
        if tid in exclude or len(vals) < min_frames:
            continue
        cand.append((float(np.median(vals)), len(vals), tid))
    if not cand:
        raise ValueError(f"no track with at least {min_frames} brightness samples")
    cand.sort()
    return int(cand[0][2])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a reusable YOLO-pose tracking cache for showcase renderers.")
    p.add_argument("--clip", required=True, help="video path, relative to sheep-yolo/ or absolute")
    p.add_argument("--weights", default="weights/sheep-pose-v0.7-yolo26n.pt", help="YOLO-pose weights")
    p.add_argument("--output", help="cache path; default artifacts/cache/<clip-stem>-tracks.pkl")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--device")
    p.add_argument("--max-frames", type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clip = resolve_path(args.clip)
    weights = resolve_path(args.weights)
    output = resolve_path(args.output) if args.output else ARTIFACTS / "cache" / f"{clip.stem}-tracks.pkl"
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    payload = track_video(
        clip,
        weights,
        tracker=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        max_frames=args.max_frames,
    )
    save_cache(payload, output)
    print(f"wrote {output}  ({len(payload['per_frame'])} frames, {len(payload['present'])} tracks)")


if __name__ == "__main__":
    main()
