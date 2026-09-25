#!/usr/bin/env python3
"""Burn Look / Skip into a replay of one run.

Reads a run folder written by ``run_pipeline.py --demo`` (``frames.json``,
``triage.json``, ``tracks.json``) and writes ``annotated.mp4``: pose boxes,
the ear-angle lines, and a Look / Skip badge per track. Badges come from
``triage.json`` only; the raw score is never drawn.

    uv run --project sheep-yolo python experiments/sheep-triage-jev/render_overlay.py \\
        experiments/sheep-triage-jev/runs/<name>/

OpenCV and NumPy are imported only when rendering. The file is H.264 through
PyAV (``imageio[pyav]`` in the sheep-yolo env) so it seeks in a browser. If
PyAV cannot encode H.264 it falls back to OpenCV mp4v, which QuickTime and VLC
play but browsers may not.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

from glance_list import LOOK, SKIP, UNCHECKED, badge_for, clock
from pose_features import ear_angles

VIDEO_NAME = "annotated.mp4"
FRAMES_NAME = "frames.json"

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


def badges_from_triage(triage: dict) -> dict[int, str]:
    return {int(row["track_id"]): badge_for(row.get("decision")) for row in triage.get("tracks", [])}


def header_parts(badges: dict[int, str]) -> list[tuple[str, str]]:
    """(text, badge) segments for the top bar. All-Look runs show SKIP 0, never a fake Skip."""
    values = list(badges.values())
    if values and all(v == UNCHECKED for v in values):
        return [("NOT SORTED YET", UNCHECKED)]
    return [(f"LOOK {values.count(LOOK)}", LOOK), (f"SKIP {values.count(SKIP)}", SKIP)]


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
    triage: dict,
    *,
    clip: Path | None,
    fps: float,
    kpt_conf: float = 0.4,
    max_width: int = 1280,
) -> Path:
    """Write ``annotated.mp4`` into ``run_dir``. ``clip=None`` draws on a plain field."""
    import cv2
    import numpy as np

    badges = badges_from_triage(triage)
    header = header_parts(badges)
    cap = None
    first = None
    if clip is not None:
        cap = cv2.VideoCapture(str(clip))
        ok, first = cap.read()
        if not ok:
            cap.release()
            raise SystemExit(f"could not read frames from {clip}")
        src_h, src_w = first.shape[:2]
    else:
        src_w, src_h = _canvas_size(frames)
    scale = min(1.0, max_width / src_w) if max_width else 1.0
    out_w = max(2, int(src_w * scale) // 2 * 2)
    out_h = max(2, int(src_h * scale) // 2 * 2)
    scale = out_w / src_w

    out = run_dir / VIDEO_NAME
    writer = _Writer(out, out_w, out_h, fps)
    written = 0
    try:
        for idx, tracks in enumerate(frames):
            if cap is None:
                img = np.full((out_h, out_w, 3), FIELD_BG, np.uint8)
            else:
                if idx == 0:
                    img = first
                else:
                    ok, img = cap.read()
                    if not ok:
                        break
                if img.shape[1] != out_w or img.shape[0] != out_h:
                    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
            top = _draw_header(cv2, img, header, idx / fps, cap is None)
            for tid in sorted(tracks):
                if tid in badges:  # dropped flickers have no decision and are not drawn
                    draw_track(
                        cv2, img, tracks[tid], tid, badges[tid], scale=scale, kpt_conf=kpt_conf, top=top
                    )
            writer.write(img)
            written += 1
    finally:
        writer.close()
        if cap is not None:
            cap.release()
    print(f"wrote {out}  ({written} frames, {out_w}x{out_h}, {writer.codec})")
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
    render(
        args.run_dir,
        load_frames(args.run_dir / FRAMES_NAME),
        triage,
        clip=clip,
        fps=tracks_doc["fps"],
        kpt_conf=tracks_doc.get("kpt_conf", 0.4),
        max_width=args.max_width,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
