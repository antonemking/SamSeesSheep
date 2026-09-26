"""Decode a clip the way a person saw it: upright.

Phones store the sensor's pixels plus a display rotation in the container (the
video track's display matrix); an iPhone held upside down or in portrait
writes 180 or 90. Players apply it. Whether OpenCV's ``VideoCapture`` does
depends on its build and backend, so one .MOV can decode upright on one machine
and upside down on another. Tracking, the replay and the thumbnails all read
frames here instead: PyAV decodes the stored pixels (it never rotates them) and
the rotation the container asks for is applied once, before anything sees them.

``tracks.json`` records that rotation as ``frame_rotation``. Runs made before
it was recorded were tracked through OpenCV's default decoder, so their
coordinates are turned to match by asking OpenCV how it turns the clip here.

PyAV, NumPy and OpenCV are imported only when a clip is read.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TURNS = (0, 90, 180, 270)
PROBE_FRAMES = 10  # frames compared when working out what OpenCV did to an old run


@dataclass(frozen=True)
class Orientation:
    rotation: int  # clockwise degrees players turn the stored picture: 0, 90, 180 or 270
    width: int  # stored picture, before turning
    height: int


def _av() -> Any:
    try:
        import av
    except ImportError:
        raise SystemExit(
            "Reading a clip needs PyAV. Run it in the sheep-yolo env:\n"
            "  uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py ..."
        ) from None
    return av


def _clockwise(frame: Any) -> int:
    """PyAV reports the display matrix counterclockwise, in -180..180."""
    if not hasattr(frame, "rotation"):
        raise SystemExit("This PyAV is too old to read a clip's rotation. Update the sheep-yolo env (uv sync).")
    return round(-float(frame.rotation) / 90) * 90 % 360


def orientation(clip: Path) -> Orientation:
    av = _av()
    with av.open(str(clip)) as container:
        for frame in container.decode(video=0):
            return Orientation(_clockwise(frame), frame.width, frame.height)
    raise SystemExit(f"could not read frames from {clip}")


def turn(img: Any, degrees_cw: int) -> Any:
    """Turn an image clockwise by 0, 90, 180 or 270 degrees."""
    import numpy as np

    if degrees_cw % 360 == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=-(degrees_cw // 90)))


def upright_frames(clip: Path, *, max_frames: int | None = None) -> Generator[Any, None, None]:
    """BGR frames of ``clip``, each turned the way the container says to show it."""
    av = _av()
    with av.open(str(clip)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if max_frames is not None and i >= max_frames:
                return
            yield turn(frame.to_ndarray(format="bgr24"), _clockwise(frame))


def opencv_rotation(clip: Path) -> int:
    """Clockwise degrees OpenCV's default ``VideoCapture`` turns ``clip`` on this machine.

    Found by matching its frames against the stored ones, because backends
    differ in whether they apply the container's rotation and in whether they
    report doing so.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(clip))
    theirs: list[Any] = []
    try:
        while len(theirs) < PROBE_FRAMES:
            ok, img = cap.read()
            if not ok:
                break
            theirs.append(img.astype(np.int16))
    finally:
        cap.release()
    av = _av()
    stored: list[Any] = []
    with av.open(str(clip)) as container:
        for frame in container.decode(video=0):
            if len(stored) == len(theirs):
                break
            stored.append(frame.to_ndarray(format="bgr24").astype(np.int16))
    if not theirs or not stored:
        return 0

    def mismatch(degrees_cw: int) -> float:
        total = 0.0
        for a, b in zip(theirs, stored):
            b = turn(b, degrees_cw)
            if a.shape != b.shape:
                return float("inf")
            total += float(np.abs(a - b).mean())
        return total

    return min(TURNS, key=mismatch)


def _turn_point(x: float, y: float, degrees_cw: int, width: float, height: float) -> tuple[float, float]:
    if degrees_cw == 90:
        return height - y, x
    if degrees_cw == 180:
        return width - x, height - y
    if degrees_cw == 270:
        return y, width - x
    return x, y


def turn_frames(
    frames: list[dict[int, dict]], degrees_cw: int, width: float, height: float
) -> list[dict[int, dict]]:
    """Turn per-frame boxes and keypoints clockwise inside a ``width`` × ``height`` picture."""
    degrees_cw %= 360
    if degrees_cw == 0:
        return frames
    out: list[dict[int, dict]] = []
    for frame in frames:
        turned: dict[int, dict] = {}
        for tid, obs in frame.items():
            x1, y1, x2, y2 = obs["box"]
            ax, ay = _turn_point(x1, y1, degrees_cw, width, height)
            bx, by = _turn_point(x2, y2, degrees_cw, width, height)
            kpts = []
            for pt in obs["kpts"]:
                px, py = _turn_point(pt[0], pt[1], degrees_cw, width, height)
                kpts.append([px, py, *pt[2:]])
            turned[tid] = {**obs, "box": [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)], "kpts": kpts}
        out.append(turned)
    return out


def to_upright(
    frames: list[dict[int, dict]], clip: Path, recorded: int | None
) -> tuple[list[dict[int, dict]], int]:
    """A run's coordinates turned to match ``upright_frames(clip)``, and the rotation they are now in.

    ``recorded`` is the run's ``frame_rotation``; ``None`` means an older run,
    tracked through OpenCV's default decoder.
    """
    shape = orientation(clip)
    if recorded is None:
        recorded = opencv_rotation(clip)
    width, height = (shape.width, shape.height) if recorded in (0, 180) else (shape.height, shape.width)
    return turn_frames(frames, shape.rotation - recorded, width, height), shape.rotation


def recorded_rotation(tracks_doc: Mapping[str, Any]) -> int | None:
    value = tracks_doc.get("frame_rotation")
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
