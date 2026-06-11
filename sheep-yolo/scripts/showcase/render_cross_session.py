"""Render a cross-session per-animal ear-angle showcase.

This is the tracked, configurable version of the old
``black-sheep-reid/render_ekg_cross.py`` demo. It concatenates separate clips into
one visual record while preserving honest framing: this is re-acquisition with a
configured/manual identity rule, not learned biometric ReID.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from showcase.ear_geometry import ear_angles, nanmedian_baseline  # type: ignore
    from showcase.tracking_cache import (  # type: ignore
        ARTIFACTS,
        darkest_persistent_track,
        frame_tracks,
        load_cache,
        resolve_path,
        save_cache,
        track_video,
    )
else:
    from .ear_geometry import ear_angles, nanmedian_baseline
    from .tracking_cache import ARTIFACTS, darkest_persistent_track, frame_tracks, load_cache, resolve_path, save_cache, track_video

try:
    import supervision as sv  # type: ignore[reportMissingImports]
    if __package__ in (None, ""):
        from showcase import sv_annot  # type: ignore
    else:
        from . import sv_annot
except ImportError as exc:  # pragma: no cover
    raise SystemExit("render_cross_session needs supervision: run with `uv run --with supervision ...`") from exc

ROOT = Path(__file__).resolve().parents[2]
MAGENTA = sv.Color(r=255, g=0, b=255)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def session_cache_path(session: dict[str, Any], clip: Path) -> Path:
    if session.get("cache"):
        return resolve_path(session["cache"])
    return ARTIFACTS / "cache" / f"{clip.stem}-tracks.pkl"


def load_or_build_session(session: dict[str, Any], cfg: dict[str, Any], weights: Path) -> dict[str, Any]:
    clip = resolve_path(session["clip"])
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")
    cache = session_cache_path(session, clip)
    if cache.exists():
        payload = load_cache(cache)
    else:
        payload = track_video(
            clip,
            weights,
            tracker=str(cfg.get("tracker", "bytetrack.yaml")),
            conf=float(cfg.get("conf", 0.25)),
            imgsz=cfg.get("imgsz"),
            device=cfg.get("device"),
            max_frames=session.get("max_track_frames") or cfg.get("max_track_frames"),
        )
        save_cache(payload, cache)
        print(f"tracked + cached {clip.name}: {len(payload['per_frame'])} frames -> {cache}")
    return payload | {"clip_path": clip, "cache_path": cache}


def select_track(session: dict[str, Any], payload: dict[str, Any], used: set[int]) -> int:
    if "track_id" in session:
        return int(session["track_id"])
    selector = session.get("select", "darkest")
    if selector != "darkest":
        raise SystemExit(f"unsupported session selector {selector!r}; use track_id or select=darkest")
    return darkest_persistent_track(payload, exclude=set(session.get("exclude_track_ids", [])) | used,
                                    min_frames=int(session.get("min_frames", 15)))


def collect_segments(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    weights = resolve_path(cfg.get("weights", "weights/sheep-pose-v0.7-yolo26n.pt"))
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    segments = []
    used: set[int] = set()
    for session in cfg.get("sessions", []):
        payload = load_or_build_session(session, cfg, weights)
        track_id = select_track(session, payload, used)
        used.add(track_id)
        all_frames = [i for i in range(len(payload["per_frame"])) if track_id in frame_tracks(payload["per_frame"], i)]
        start = int(session.get("start_frame", all_frames[0] if all_frames else 0))
        max_frames = int(session.get("max_frames", cfg.get("max_frames_per_session", 140)))
        frames = [f for f in all_frames if f >= start][:max_frames]
        if not frames:
            print(f"!! {session.get('name', payload['clip_path'].stem)}: no frames for track {track_id}")
            continue
        left = np.array([ear_angles(frame_tracks(payload["per_frame"], f)[track_id][1])[0] for f in frames], dtype=float)
        right = np.array([ear_angles(frame_tracks(payload["per_frame"], f)[track_id][1])[1] for f in frames], dtype=float)
        segments.append({
            "name": session.get("name", payload["clip_path"].stem),
            "clip": payload["clip_path"],
            "payload": payload,
            "track_id": track_id,
            "frames": frames,
            "frame_set": set(frames),
            "left": left,
            "right": right,
        })
        print(f"{segments[-1]['name']}: track={track_id} frames={len(frames)} L~{np.nanmedian(left):.0f} R~{np.nanmedian(right):.0f}")
    return segments


def crop_subject(frame: np.ndarray, box: np.ndarray, kpts: np.ndarray, *, label: str, crop_side: int, panel: int) -> np.ndarray:
    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    x2 = min(frame.shape[1], max(crop_side, cx + crop_side // 2))
    y2 = min(frame.shape[0], max(crop_side, cy + crop_side // 2))
    x1, y1 = x2 - crop_side, y2 - crop_side
    crop = frame[y1:y2, x1:x2].copy()
    shifted = kpts.copy()
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1
    sv_annot.skeleton(crop, shifted, MAGENTA)
    shifted_box = [max(0, box[0] - x1), max(0, box[1] - y1), min(crop.shape[1], box[2] - x1), min(crop.shape[0], box[3] - y1)]
    sv_annot.boxes_labels(crop, [shifted_box], [0], [label], sv.ColorPalette(colors=[MAGENTA]), thickness=2, roundness=0.4, text_scale=0.5)
    return cv2.resize(crop, (panel, panel))


def render(cfg: dict[str, Any], *, still: bool = False, max_render_frames: int | None = None) -> Path:
    segments = collect_segments(cfg)
    if not segments:
        raise SystemExit("no renderable sessions")
    fps = float(cfg.get("fps", 30))
    panel = int(cfg.get("panel", 540))
    chart_w = int(cfg.get("chart_width", 760))
    crop_side = int(cfg.get("crop_side", 560))
    subject_label = str(cfg.get("subject_label", "black ewe"))
    baseline = nanmedian_baseline([s["left"] for s in segments] + [s["right"] for s in segments])
    starts: list[int] = []
    total = 0
    for s in segments:
        starts.append(total)
        total += len(s["frames"])
    if max_render_frames is not None:
        total = min(total, max_render_frames)

    def chart(global_i: int) -> np.ndarray:
        fig, ax = plt.subplots(figsize=(chart_w / 100, panel / 100), dpi=100)
        for si, s in enumerate(segments):
            g0 = starts[si]
            rel = global_i - g0
            if rel < 0:
                continue
            end = min(rel, len(s["frames"]) - 1)
            xs = np.arange(g0, g0 + end + 1)
            ax.plot(xs, s["left"][:end + 1], color="#1f77b4", lw=1.7)
            ax.plot(xs, s["right"][:end + 1], color="#9467bd", lw=1.7)
            if si > 0:
                ax.axvline(g0, color="0.8", lw=0.8, ls="--")
                ax.text(g0 + 2, 147, s["name"], fontsize=7, color="0.45")
        if not np.isnan(baseline):
            ax.axhline(baseline, color="0.55", lw=1.0, ls=":")
        ax.axvline(global_i, color="0.25", lw=1.2)
        ax.plot([], [], color="#1f77b4", lw=2, label="left ear")
        ax.plot([], [], color="#9467bd", lw=2, label="right ear")
        ax.set_xlim(0, max(1, total - 1))
        ax.set_ylim(*cfg.get("y_range", [60, 150]))
        ax.set_xticks([])
        ax.set_ylabel("ear angle |deg|")
        ax.set_title(cfg.get("title", "same ewe · cross-session measurement record"), fontsize=10.5)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.canvas.draw()
        canvas_rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
        buf = canvas_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    output = resolve_path(cfg.get("output", "artifacts/cross-session-ear-record.mp4"))
    if still:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    global_i = 0
    sample_canvas: np.ndarray | None = None
    for s in segments:
        cap = cv2.VideoCapture(str(s["clip"]))
        for local_i, frame_idx in enumerate(s["frames"]):
            if global_i >= total:
                break
            frame = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, raw = cap.read()
            if ok:
                frame = raw
            if frame is None:
                continue
            box, kpts = frame_tracks(s["payload"]["per_frame"], frame_idx)[s["track_id"]]
            crop = crop_subject(frame, box, kpts, label=subject_label, crop_side=crop_side, panel=panel)
            canvas = np.hstack([crop, chart(global_i)])
            if still:
                sample_canvas = canvas
            else:
                if writer is None:
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))  # type: ignore[attr-defined]
                writer.write(canvas)
            global_i += 1
        cap.release()
    if writer is not None:
        writer.release()
    if still and sample_canvas is not None:
        cv2.imwrite(str(output), sample_canvas)
    print(f"wrote {output}  ({global_i} frames, {global_i / fps:.1f}s)")
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="JSON config path")
    p.add_argument("--still", action="store_true", help="write a PNG still instead of an MP4")
    p.add_argument("--max-render-frames", type=int, help="smoke-test cap for rendering")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_path(args.config))
    render(cfg, still=args.still, max_render_frames=args.max_render_frames)


if __name__ == "__main__":
    main()
