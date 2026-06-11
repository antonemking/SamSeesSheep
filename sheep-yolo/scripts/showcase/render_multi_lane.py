"""Render polished multi-lane sheep ear-angle showcase videos.

This is the tracked, configurable version of the old
``black-sheep-reid/render_synced_pro.py`` demo. It can reuse an existing tracking
cache or build one from YOLO-pose weights, then renders selected track IDs as
parallel ear-angle lanes with face thumbnails.

Example:
    uv run --with supervision python scripts/showcase/render_multi_lane.py \
        --config scripts/showcase/configs/img_3651_6lane.json --still
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
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from showcase.ear_geometry import ear_angles  # type: ignore
    from showcase.tracking_cache import (  # type: ignore
        ARTIFACTS,
        frame_tracks,
        load_cache,
        resolve_path,
        save_cache,
        top_tracks,
        track_video,
    )
else:
    from .ear_geometry import ear_angles
    from .tracking_cache import ARTIFACTS, frame_tracks, load_cache, resolve_path, save_cache, top_tracks, track_video

try:
    import supervision as sv  # type: ignore[reportMissingImports]
    if __package__ in (None, ""):
        from showcase import sv_annot  # type: ignore
    else:
        from . import sv_annot
except ImportError as exc:  # pragma: no cover - exercised in user environment via CLI
    raise SystemExit("render_multi_lane needs supervision: run with `uv run --with supervision ...`") from exc

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(cfg: dict[str, Any], key: str, *, default: str | None = None) -> Path | None:
    value = cfg.get(key, default)
    return resolve_path(value) if value else None


def subject_specs(cfg: dict[str, Any], payload: dict[str, Any], f0: int, f1: int) -> list[dict[str, Any]]:
    configured = cfg.get("subjects") or []
    if configured:
        return [dict(s, track_id=int(s["track_id"])) for s in configured]
    track_ids = top_tracks(payload, frame_start=f0, frame_end=f1, limit=int(cfg.get("lanes", 6)), min_frames=int(cfg.get("min_track_frames", 15)))
    return [{"track_id": tid, "label": f"track {tid}"} for tid in track_ids]


def color_maps(track_ids: list[int]) -> tuple[dict[int, Any], dict[int, sv.Color], sv.ColorPalette]:
    cmap = plt.get_cmap("tab10")
    tab = cmap(np.linspace(0, 1, max(10, len(track_ids))))
    mpl = {tid: tuple(tab[i]) for i, tid in enumerate(track_ids)}
    svcol = {
        tid: sv.Color(r=int(tab[i][0] * 255), g=int(tab[i][1] * 255), b=int(tab[i][2] * 255))
        for i, tid in enumerate(track_ids)
    }
    return mpl, svcol, sv.ColorPalette(colors=[svcol[t] for t in track_ids])


def label_for(subjects: list[dict[str, Any]], tid: int) -> str:
    for s in subjects:
        if int(s["track_id"]) == tid:
            return str(s.get("label") or f"track {tid}")
    return f"track {tid}"


def angle_label(left: float, right: float) -> str:
    parts = []
    if not np.isnan(left):
        parts.append(f"L{left:.0f}")
    if not np.isnan(right):
        parts.append(f"R{right:.0f}")
    return " ".join(parts) if parts else "—"


def safe_read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def circular_thumbnail(clip: Path, payload: dict[str, Any], tid: int, f0: int, f1: int, *, size: int = 76) -> np.ndarray | None:
    best: tuple[float, int, np.ndarray] | None = None
    for f in range(f0, f1):
        tracks = frame_tracks(payload["per_frame"], f)
        if tid not in tracks:
            continue
        box, kpts = tracks[tid]
        left, right = ear_angles(kpts)
        if np.isnan(left) and np.isnan(right):
            continue
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        if best is None or area > best[0]:
            best = (area, f, box)
    if best is None:
        return None
    _, frame_idx, box = best
    cap = cv2.VideoCapture(str(clip))
    frame = safe_read_frame(cap, frame_idx)
    cap.release()
    if frame is None:
        return None
    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    side = max(8, int(max(box[2] - box[0], box[3] - box[1]) * 1.4))
    x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
    x2, y2 = min(frame.shape[1], x1 + side), min(frame.shape[0], y1 + side)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size))
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2 - 2, 255, -1)
    crop[mask == 0] = 245
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def render(cfg: dict[str, Any], *, still: bool = False, max_render_frames: int | None = None) -> Path:
    clip = resolve_config_path(cfg, "clip")
    weights = resolve_config_path(cfg, "weights", default="weights/sheep-pose-v0.7-yolo26n.pt")
    if clip is None:
        raise SystemExit("config missing clip")
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")

    cache = resolve_config_path(cfg, "cache") or ARTIFACTS / "cache" / f"{clip.stem}-tracks.pkl"
    if cache.exists():
        payload = load_cache(cache)
    else:
        if weights is None or not weights.exists():
            raise SystemExit(f"cache missing and weights not found: {weights}")
        payload = track_video(
            clip,
            weights,
            tracker=str(cfg.get("tracker", "bytetrack.yaml")),
            conf=float(cfg.get("conf", 0.25)),
            imgsz=cfg.get("imgsz"),
            device=cfg.get("device"),
            max_frames=cfg.get("max_track_frames"),
        )
        save_cache(payload, cache)
        print(f"tracked + cached {len(payload['per_frame'])} frames -> {cache}")

    window = cfg.get("window") or [0, len(payload["per_frame"])]
    f0, f1 = int(window[0]), int(window[1])
    subjects = subject_specs(cfg, payload, f0, f1)
    if not subjects:
        raise SystemExit("no subjects configured or found in cache")
    track_ids = [int(s["track_id"]) for s in subjects]
    labels = {int(s["track_id"]): str(s.get("label") or f"track {s['track_id']}") for s in subjects}
    mpl, svcol, palette = color_maps(track_ids)

    fps = float(cfg.get("fps") or payload.get("fps") or 30)
    height = int(cfg.get("height", 640))
    chart_w = int(cfg.get("chart_width", 900))
    n = max(0, f1 - f0)
    if max_render_frames is not None:
        n = min(n, max_render_frames)
        f1 = f0 + n
    ts = np.arange(n) / fps

    series_l = {tid: np.full(n, np.nan) for tid in track_ids}
    series_r = {tid: np.full(n, np.nan) for tid in track_ids}
    for i, frame_idx in enumerate(range(f0, f1)):
        tracks = frame_tracks(payload["per_frame"], frame_idx)
        for tid in track_ids:
            if tid in tracks:
                series_l[tid][i], series_r[tid][i] = ear_angles(tracks[tid][1])

    thumbs = {tid: circular_thumbnail(clip, payload, tid, f0, f1) for tid in track_ids}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "0.7", "axes.linewidth": 0.8})

    def chart(i: int) -> np.ndarray:
        fig, axes = plt.subplots(len(track_ids), 1, sharex=True, figsize=(chart_w / 100, height / 100), dpi=100)
        if len(track_ids) == 1:
            axes = [axes]
        fig.subplots_adjust(left=0.165, right=0.965, top=0.92, bottom=0.1, hspace=0.55)
        for ax, tid in zip(axes, track_ids):
            color = mpl[tid]
            ax.plot(ts[:i + 1], series_l[tid][:i + 1], color=color, lw=1.8, solid_capstyle="round")
            ax.plot(ts[:i + 1], series_r[tid][:i + 1], color=color, lw=1.2, ls="--", dash_capstyle="round")
            ax.axvline(ts[i], color="0.55", lw=0.8)
            ax.set_xlim(0, max(float(ts[-1]) if len(ts) else 0.01, 0.01))
            ax.set_ylim(*cfg.get("y_range", [60, 150]))
            ax.set_yticks(cfg.get("y_ticks", [90, 130]))
            ax.tick_params(labelsize=7, length=0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(axis="y", alpha=0.18)
            ax.set_title(labels[tid], loc="left", color=color, fontsize=9.5, fontweight="bold", pad=2)
            ax.text(0.995, 0.82, angle_label(series_l[tid][i], series_r[tid][i]), transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color=color)
            thumb = thumbs[tid]
            if thumb is not None:
                ab = AnnotationBbox(OffsetImage(thumb, zoom=0.5), (-0.135, 0.5), xycoords="axes fraction",
                                    frameon=False, box_alignment=(0.5, 0.5))
                ax.add_artist(ab)
        axes[0].annotate(cfg.get("title", "ear angle · multi-sheep readout  (— L, - - R)"),
                         xy=(0.5, 0.985), xycoords="figure fraction",
                         ha="center", va="top", fontsize=10.5, color="0.25")
        axes[-1].set_xlabel("time (s) · all lanes share one clock", fontsize=8.5)
        fig.canvas.draw()
        canvas_rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
        buf = canvas_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    def left_panel(frame_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(clip))
        frame = safe_read_frame(cap, frame_idx)
        cap.release()
        if frame is None:
            return np.zeros((height, int(height * 16 / 9), 3), dtype=np.uint8)
        tracks = frame_tracks(payload["per_frame"], frame_idx)
        for tid, (_, kpts) in tracks.items():
            if tid not in track_ids:
                sv_annot.faint_flock(frame, kpts)
        present = [tid for tid in track_ids if tid in tracks]
        if present:
            xyxy = []
            class_ids = []
            text = []
            for tid in present:
                box, kpts = tracks[tid]
                left, right = ear_angles(kpts)
                xyxy.append(box)
                class_ids.append(track_ids.index(tid))
                text.append(f"{label_for(subjects, tid)}  {angle_label(left, right)}")
            frame = sv_annot.boxes_labels(frame, xyxy, class_ids, text, palette)
            for tid in present:
                sv_annot.skeleton(frame, tracks[tid][1], svcol[tid])
        return cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))

    output = resolve_config_path(cfg, "output") or ARTIFACTS / f"synced-lanes-{clip.stem}.mp4"
    if still:
        output = output.with_suffix(".png")
        i = max(0, n - 1)
        canvas = np.hstack([left_panel(f0 + i), chart(i)])
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), canvas)
        print(f"wrote {output}")
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    for i, frame_idx in enumerate(range(f0, f1)):
        canvas = np.hstack([left_panel(frame_idx), chart(i)])
        if writer is None:
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))  # type: ignore[attr-defined]
        writer.write(canvas)
    if writer is not None:
        writer.release()
    print(f"wrote {output}  ({n} frames, {n / fps:.1f}s)")
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
