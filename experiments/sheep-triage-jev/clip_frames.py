"""Decode a clip the way a person saw it: upright.

Phones store the sensor's pixels plus a display rotation in the container (the
video track's display matrix); an iPhone held upside down or in portrait
writes 180 or 90. Players apply it. Whether OpenCV's ``VideoCapture`` does
depends on its build and backend, so one .MOV can decode upright on one machine
and upside down on another. Tracking, the replay and the thumbnails all read
frames here instead: PyAV decodes the stored pixels (it never rotates them) and
the rotation the container asks for is applied once, before anything sees them.
The boxes and keypoints in ``frames.json`` are therefore in the same upright
frame as the replay's pixels.

``tracks.json`` records that rotation as ``frame_rotation``. A skeleton is only
drawn on a clip when the run's ``frame_rotation`` matches how this module turns
it now. Runs tracked before it was recorded went through OpenCV, which may have
handed the pose model the sheep upside down or sideways; their poses cannot be
trusted against the upright video, so drawing them is refused with a re-track
command instead.

PyAV and NumPy are imported only when a clip is read.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TURNS = (0, 90, 180, 270)


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


def recorded_rotation(tracks_doc: Mapping[str, Any]) -> int | None:
    value = tracks_doc.get("frame_rotation")
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def frame_mismatch(clip: Path, recorded: int | None) -> str | None:
    """Why a run's boxes and keypoints can't be drawn on ``upright_frames(clip)``, or None when they can."""
    rotation = orientation(clip).rotation
    if recorded == rotation or (recorded is None and rotation == 0):
        return None
    if recorded is None:
        return (
            f"{clip.name} is stored turned {rotation}° and this run was tracked before that was recorded, "
            "so its skeletons may be upside down or sideways against the video."
        )
    return f"this run was tracked on frames turned {recorded}°, but {clip.name} now reads as turned {rotation}°."


def retrack_hint(clip: Path) -> str:
    return (
        "Re-track the clip so the skeletons come from the same upright frames as the video:\n"
        "  uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py \\\n"
        f"    --clip {clip} --demo --out <new run folder>"
    )


def refusal(problem: str, clip: Path) -> str:
    return f"Not drawing skeletons: {problem}\n{retrack_hint(clip)}"


def require_same_frame(clip: Path, recorded: int | None) -> None:
    problem = frame_mismatch(clip, recorded)
    if problem:
        raise SystemExit(refusal(problem, clip))
