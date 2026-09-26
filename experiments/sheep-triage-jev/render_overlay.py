#!/usr/bin/env python3
"""Burn Look / Skip / Not checked into a replay of one run.

Reads a run folder written by ``run_pipeline.py --demo`` (``frames.json``,
``triage.json``, ``tracks.json``) and writes ``annotated.mp4``: pose boxes,
the ear-angle lines, and a Look / Skip / Not checked badge per track. Badges
come from ``triage.json`` only; the raw score is never drawn. It also saves
one small JPEG per sheep under ``thumbs/`` for the morning brief, cropped from
the source video (from the drawn replay when there is no video).

    uv run --project sheep-yolo python experiments/sheep-triage-jev/render_overlay.py \\
        experiments/sheep-triage-jev/runs/<name>/

OpenCV and NumPy are imported only when rendering. Clip frames come upright
from ``clip_frames`` (the container's rotation applied), and older runs'
coordinates are turned to match. The file is H.264 through PyAV
(``imageio[pyav]`` in the sheep-yolo env) so it seeks in a browser. If PyAV
cannot encode H.264 it falls back to OpenCV mp4v, which QuickTime and VLC play
but browsers may not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from clip_frames import recorded_rotation, to_upright, upright_frames
from glance_list import LOOK, SKIP, THUMBS_DIR, UNCHECKED, badge_for, clock, decisions_ran, thumb_path
from pose_features import ear_angles

VIDEO_NAME = "annotated.mp4"
FRAMES_NAME = "frames.json"
THUMB_SIZE = (320, 240)
THUMB_PAD = 1.35  # crop this much wider than the box, so the head is not cut at the edge

SKEL = [(0, 1), (0, 2)]  # nose to each ear base; ear base to tip is drawn per ear
UI_WIDTH = 1280.0  # line and text sizes are tuned for this output width
# BGR
BADGE_BG = {LOOK: (26, 159, 255), SKIP: (80, 175, 60), UNCHECKED: (170, 170, 170)}
BADGE_FG = {LOOK: (27, 29, 29), SKIP: (255, 255, 255), UNCHECKED: (27, 29, 29)}
KCOL = (255, 0, 255)
LEFT_COL = (180, 119, 31)  # #1f77b4, same as the ear-angle charts
RIGHT_COL = (189, 103, 148)  # #9467bd
MIDLINE_COL = (255, 255, 255)
FIELD_BG = (58, 66, 58)  # synthetic runs have no video to draw on


def save_frames(path: Path, frames: list[dict[int, dict]]) -> None:
    """Per-frame boxes and keypoints, so the overlay can be re-rendered later."""
    rows = []
    for frame in frames:
        rows.append(
            {
                str(tid): {
                    "box": [round(float(v), 1) for v in obs["box"]],
                    "kpts": [
                        [round(float(pt[0]), 1), round(float(pt[1]), 1), round(float(pt[2]), 3)]
                        for pt in obs["kpts"]
                    ],
                }
                for tid, obs in frame.items()
            }
        )
    path.write_text(json.dumps({"frames": rows}, separators=(",", ":")) + "\n")


def load_frames(path: Path) -> list[dict[int, dict]]:
    rows = json.loads(path.read_text())["frames"]
    return [{int(tid): obs for tid, obs in row.items()} for row in rows]


def badges_from_triage(triage: Mapping[str, Any]) -> dict[int, str]:
    return {int(row["track_id"]): badge_for(row.get("decision")) for row in triage.get("tracks", [])}


def header_parts(badges: Mapping[int, str], *, ran: bool | None = None) -> list[tuple[str, str]]:
    """(text, badge) segments for the top bar. All-Look runs show SKIP 0, never a fake Skip.

    ``ran`` says whether the check produced any decision; by default it is
    inferred from the badges, which cannot tell an all-cannot run from one
    that never ran.
    """
    values = list(badges.values())
    if ran is None:
        ran = any(v != UNCHECKED for v in values)
    if not ran:
        return [("NOT SORTED YET", UNCHECKED)]
    parts = [(f"LOOK {values.count(LOOK)}", LOOK), (f"SKIP {values.count(SKIP)}", SKIP)]
    if UNCHECKED in values:
        parts.append((f"NOT CHECKED {values.count(UNCHECKED)}", UNCHECKED))
    return parts


def thumb_frames(frames: list[dict[int, dict]], track_ids: set[int], kpt_conf: float) -> dict[int, int]:
    """Frame index to crop each sheep's thumbnail from.

    Prefers a frame with the face toward the camera (nose and both ear bases
    found), then the most keypoints found, then the frame nearest the middle
    of its time in view.
    """
    seen: dict[int, list[int]] = {}
    for idx, frame in enumerate(frames):
        for tid in frame:
            if tid in track_ids:
                seen.setdefault(tid, []).append(idx)
    picks: dict[int, int] = {}
    for tid, idxs in seen.items():
        mid = (idxs[0] + idxs[-1]) / 2

        def key(idx: int, tid: int = tid, mid: float = mid) -> tuple[bool, int, float]:
            found = [len(pt) >= 3 and pt[2] > kpt_conf for pt in frames[idx][tid]["kpts"][:5]]
            return (len(found) >= 3 and all(found[:3]), sum(found), -abs(idx - mid))

        picks[tid] = max(idxs, key=key)
    return picks


def crop_box(box: list[float], scale: float, width: int, height: int) -> tuple[int, int, int, int]:
    """A 4:3 crop around one box, padded and kept inside a ``width`` × ``height`` frame."""
    x1, y1, x2, y2 = (v * scale for v in box)
    w = max(x2 - x1, (y2 - y1) * 4 / 3, 8.0) * THUMB_PAD
    w = min(w, float(width), height * 4 / 3)
    h = w * 3 / 4
    left = min(max(0.0, (x1 + x2 - w) / 2), width - w)
    top = min(max(0.0, (y1 + y2 - h) / 2), height - h)
    return round(left), round(top), round(left + w), round(top + h)


def _save_thumb(cv2, img, box: list[float], scale: float, path: Path) -> None:
    left, top, right, bottom = crop_box(box, scale, img.shape[1], img.shape[0])
    crop = cv2.resize(img[top:bottom, left:right], THUMB_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])


class _Writer:
    """H.264 through PyAV when available, else OpenCV mp4v."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        self.codec = "h264"
        self._container = self._stream = self._cv = None
        self._n = 0
        if not self._open_h264(path, width, height, fps):
            import cv2

            self.codec = "mp4v"
            self._cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not self._cv.isOpened():
                raise SystemExit(f"could not open a video writer for {path}")

    def _open_h264(self, path: Path, width: int, height: int, fps: float) -> bool:
        try:
            import av
        except ImportError:
            print("PyAV not installed; writing mp4v (browsers may not play it)")
            return False
        try:
            container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        except av.error.FFmpegError as exc:
            print(f"PyAV could not open {path} ({exc}); writing mp4v")
            return False
        try:
            stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        except (ValueError, av.error.FFmpegError) as exc:
            container.close()
            print(f"PyAV has no H.264 encoder ({exc}); writing mp4v (browsers may not play it)")
            return False
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        self._av = av
        self._container, self._stream = container, stream
        return True

    def write(self, img) -> None:
        if self._cv is not None:
            self._cv.write(img)
            return
        frame = self._av.VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = self._n
        frame.time_base = 1 / self._stream.codec_context.framerate
        self._n += 1
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._cv is not None:
            self._cv.release()
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def _canvas_size(frames: list[dict[int, dict]]) -> tuple[int, int]:
    xs, ys = [640.0], [480.0]
    for frame in frames:
        for obs in frame.values():
            xs.append(obs["box"][2] + 40)
            ys.append(obs["box"][3] + 40)
    return int(max(xs)), int(max(ys))


def _text(cv2, img, text, org, scale, color, thickness):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _badge(cv2, img, text, x, y, badge, ui, top):
    scale, thick = 0.8 * ui, max(1, round(2 * ui))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = max(4, round(6 * ui))
    w, h = tw + 2 * pad, th + base + 2 * pad
    x = min(max(0, x), img.shape[1] - w)
    y0 = y - h if y - h >= top else y
    cv2.rectangle(img, (x, y0), (x + w, y0 + h), BADGE_BG[badge], -1)
    cv2.putText(
        img,
        text,
        (x + pad, y0 + pad + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        BADGE_FG[badge],
        thick,
        cv2.LINE_AA,
    )


def draw_track(
    cv2, img, obs: dict, track_id: int, badge: str, *, scale: float, kpt_conf: float, top: int
) -> None:
    ui = img.shape[1] / UI_WIDTH
    thick = max(2, round(3 * ui))
    thin = max(1, round(2 * ui))
    x1, y1, x2, y2 = (round(v * scale) for v in obs["box"])
    cv2.rectangle(img, (x1, y1), (x2, y2), BADGE_BG[badge], thick, cv2.LINE_AA)

    kpts = obs["kpts"]
    pts = {
        j: (round(pt[0] * scale), round(pt[1] * scale))
        for j, pt in enumerate(kpts[:5])
        if len(pt) >= 3 and pt[2] > kpt_conf
    }
    for a, b in SKEL:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], KCOL, thin, cv2.LINE_AA)
    if 0 in pts and 1 in pts and 2 in pts:
        mid = ((pts[1][0] + pts[2][0]) // 2, (pts[1][1] + pts[2][1]) // 2)
        cv2.line(img, mid, pts[0], MIDLINE_COL, thin, cv2.LINE_AA)
    left, right = ear_angles(kpts, kpt_conf)
    for base, tip, value, color in ((1, 3, left, LEFT_COL), (2, 4, right, RIGHT_COL)):
        if base not in pts or tip not in pts:
            continue
        cv2.line(img, pts[base], pts[tip], color, thick, cv2.LINE_AA)
        if value is not None:
            label = f"{value:.0f}"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, thin)
            gap = round(6 * ui)
            x = pts[tip][0] - tw - gap if pts[tip][0] < pts[base][0] else pts[tip][0] + gap
            _text(cv2, img, label, (x, pts[tip][1] - gap), 0.6 * ui, color, thin)
    for p in pts.values():
        cv2.circle(img, p, max(3, round(4 * ui)), KCOL, -1, cv2.LINE_AA)
    _badge(cv2, img, f"{badge.upper()}  #{track_id}", x1, y1, badge, ui, top)


def _draw_header(cv2, img, parts: list[tuple[str, str]], seconds: float, synthetic: bool) -> int:
    ui = img.shape[1] / UI_WIDTH
    h = round(44 * ui)
    band = img[:h].copy()
    band[:] = (20, 20, 20)
    img[:h] = cv2.addWeighted(band, 0.6, img[:h], 0.4, 0)
    thick = max(1, round(2 * ui))
    base_y = round(31 * ui)
    x = round(14 * ui)
    for text, badge in parts:
        _text(cv2, img, text, (x, base_y), 0.9 * ui, BADGE_BG[badge], thick)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * ui, thick)
        x += tw + round(28 * ui)
    right = clock(seconds) + ("   synthetic test, no video" if synthetic else "")
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * ui, thick)
    _text(cv2, img, right, (img.shape[1] - tw - round(14 * ui), base_y), 0.7 * ui, (230, 230, 230), thick)
    return h


def render(
    run_dir: Path,
    frames: list[dict[int, dict]],
    triage: Mapping[str, Any],
    *,
    clip: Path | None,
    fps: float,
    kpt_conf: float = 0.4,
    max_width: int = 1280,
) -> Path:
    """Write ``annotated.mp4`` into ``run_dir``. ``clip=None`` draws on a plain field.

    ``frames`` must be in the coordinates of ``upright_frames(clip)``; see ``clip_frames.to_upright``.
    """
    import cv2
    import numpy as np

    badges = badges_from_triage(triage)
    header = header_parts(badges, ran=decisions_ran(triage))
    source = None
    first = None
    if clip is not None:
        source = upright_frames(clip)
        first = next(source, None)
        if first is None:
            raise SystemExit(f"could not read frames from {clip}")
        src_h, src_w = first.shape[:2]
    else:
        src_w, src_h = _canvas_size(frames)
    scale = min(1.0, max_width / src_w) if max_width else 1.0
    out_w = max(2, int(src_w * scale) // 2 * 2)
    out_h = max(2, int(src_h * scale) // 2 * 2)
    scale = out_w / src_w

    out = run_dir / VIDEO_NAME
    thumbs_dir = run_dir / THUMBS_DIR
    thumbs_dir.mkdir(exist_ok=True)
    for stale in thumbs_dir.glob("sheep-*.jpg"):
        stale.unlink()
    due: dict[int, list[int]] = {}
    for tid, idx in thumb_frames(frames, set(badges), kpt_conf).items():
        due.setdefault(idx, []).append(tid)
    writer = _Writer(out, out_w, out_h, fps)
    written = n_thumbs = 0
    img: Any
    try:
        for idx, tracks in enumerate(frames):
            if source is None:
                img = np.full((out_h, out_w, 3), FIELD_BG, np.uint8)
            else:
                img = first if idx == 0 else next(source, None)
                if img is None:
                    break
                if img.shape[1] != out_w or img.shape[0] != out_h:
                    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
                # Real footage: crop before the overlay, so the thumbnail is the sheep itself.
                for tid in due.get(idx, ()):
                    _save_thumb(cv2, img, tracks[tid]["box"], scale, run_dir / thumb_path(tid))
                    n_thumbs += 1
            top = _draw_header(cv2, img, header, idx / fps, source is None)
            for tid in sorted(tracks):
                if tid in badges:  # dropped flickers have no decision and are not drawn
                    draw_track(
                        cv2, img, tracks[tid], tid, badges[tid], scale=scale, kpt_conf=kpt_conf, top=top
                    )
            if source is None:
                # No footage to crop: the drawn head is the only picture there is.
                for tid in due.get(idx, ()):
                    _save_thumb(cv2, img, tracks[tid]["box"], scale, run_dir / thumb_path(tid))
                    n_thumbs += 1
            writer.write(img)
            written += 1
    finally:
        writer.close()
        if source is not None:
            source.close()
    print(f"wrote {out}  ({written} frames, {out_w}x{out_h}, {writer.codec}) and {n_thumbs} thumbnails")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="Run folder written by run_pipeline.py --demo.")
    p.add_argument("--clip", type=Path, default=None, help="Source video, if it moved since the run.")
    p.add_argument("--max-width", type=int, default=1280)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    for name in (FRAMES_NAME, "tracks.json", "triage.json"):
        if not (args.run_dir / name).is_file():
            raise SystemExit(f"{name} not found in {args.run_dir}. Run run_pipeline.py with --demo first.")
    tracks_doc = json.loads((args.run_dir / "tracks.json").read_text())
    triage = json.loads((args.run_dir / "triage.json").read_text())
    clip = args.clip
    if clip is None and tracks_doc.get("clip") != "synthetic":
        clip = Path(tracks_doc["clip"])
    if clip is not None and not clip.is_file():
        raise SystemExit(f"clip not found: {clip}. Pass --clip.")
    frames = load_frames(args.run_dir / FRAMES_NAME)
    if clip is not None:
        frames, _ = to_upright(frames, clip, recorded_rotation(tracks_doc))
    render(
        args.run_dir,
        frames,
        triage,
        clip=clip,
        fps=tracks_doc["fps"],
        kpt_conf=tracks_doc.get("kpt_conf", 0.4),
        max_width=args.max_width,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
