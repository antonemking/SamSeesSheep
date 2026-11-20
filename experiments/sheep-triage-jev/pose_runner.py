"""YOLO-pose tracking. Ultralytics is imported only when a clip is tracked."""

from __future__ import annotations

from pathlib import Path


def frame_from_result(result) -> dict[int, dict]:
    """Pull ``{track_id: {box, kpts}}`` out of one Ultralytics track result.

    ``kpts`` is a list of ``[x, y, conf]`` in v0.7 order. Frames with no
    track ids contribute an empty dict.
    """
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or getattr(boxes, "id", None) is None:
        return {}
    ids = boxes.id.cpu().numpy().astype(int)
    xy = boxes.xyxy.cpu().numpy()
    kp = keypoints.data.cpu().numpy()
    out: dict[int, dict] = {}
    n = min(len(ids), len(xy), len(kp))
    for j in range(n):
        row = kp[j]
        if row.shape[-1] < 3:
            raise RuntimeError(
                "expected YOLO-pose keypoints as (x, y, conf); "
                f"got last dimension {row.shape[-1]}"
            )
        out[int(ids[j])] = {
            "box": [float(v) for v in xy[j].tolist()],
            "kpts": [[float(v) for v in pt] for pt in row.tolist()],
        }
    return out


def read_fps(clip: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(clip))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if fps <= 1.0:
        return 30.0
    return fps


def track_clip(
    weights: Path,
    clip: Path,
    *,
    conf: float,
    max_frames: int | None,
    tracker: str,
    imgsz: int | None,
) -> list[dict[int, dict]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs: dict = {
        "source": str(clip),
        "tracker": tracker,
        "stream": True,
        "conf": conf,
        "verbose": False,
        "save": False,
    }
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    frames: list[dict[int, dict]] = []
    for result in model.track(**kwargs):
        frames.append(frame_from_result(result))
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames
