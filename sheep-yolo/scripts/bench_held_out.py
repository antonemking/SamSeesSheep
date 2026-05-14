"""Held-out 3-way benchmark: sheep-pose v0.2 vs v0.3 vs v0.4 on IMG_3651.

Runs all three models against test-clips/IMG_3651.MOV — a clip in NEITHER
training set (never pushed to the labeler, NCC < 0.23 vs every training video).

Outputs into ./artifacts:

  v0.2-on-IMG_3651.mp4, v0.3-on-IMG_3651.mp4, v0.4-on-IMG_3651.mp4
      Ultralytics-annotated inference, whole clip.

  v0.2-vs-v0.4-IMG_3651.mp4 + .png
      HERO 2-up for the LinkedIn post. v0.2 (98 instances) vs v0.4 (405).

  v0.2-vs-v0.3-vs-v0.4-IMG_3651.mp4 + .png
      Supplementary 3-up — shows the curve, not just the endpoints.

  bench_report-IMG_3651-3way.json
      σ + detection numbers for all 3 models.

Going forward, per the new "compare against last posted" policy, future
benchmarks are 2-way (latest vs last posted). This script is the one-off
bridge from the v0.3 post (which left a clean-held-out IOU outstanding)
to the v0.4 post.

Reuses pure helpers from bench_v02_v03.py to avoid duplication.
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import av  # PyAV — h264 (libx264) writer; cv2.VideoWriter's mp4v is grainy at typical bitrates
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from bench_v02_v03 import (  # noqa: E402
    _transcode,
    _overlay_kpts_multi,
    _compose_panel_zoom,
    _per_kpt,
    _per_kpt_residual,
    CLIPS,
    V02_BGR,
    V03_BGR,
    CONF,
)

V02 = ROOT / "weights" / "sheep-pose-v0.2-yolo26n.pt"
V03 = ROOT / "weights" / "sheep-pose-v0.3-yolo26n.pt"
V04 = ROOT / "weights" / "sheep-pose-v0.4-yolo26n.pt"
V04_BGR = (255, 0, 255)  # magenta — distinct from v0.2 orange and v0.3 green

SELECTED = sys.argv[1] if len(sys.argv) > 1 else "IMG_3651"
if SELECTED not in CLIPS:
    raise SystemExit(f"unknown clip {SELECTED!r}; pick one of {list(CLIPS)}")
CLIP = CLIPS[SELECTED]["path"]
WINDOW = CLIPS[SELECTED]["window"]
ROI = CLIPS[SELECTED]["roi"]


def in_roi(cx: float, cy: float) -> bool:
    return ROI[0] <= cx <= ROI[2] and ROI[1] <= cy <= ROI[3]


CACHE_DIR = ART / "_cache"


def _cache_path(model_path: Path, clip_path: Path) -> Path:
    return CACHE_DIR / f"{model_path.stem}__{clip_path.stem}.pkl"


def predict_clip(model_path: Path, label: str) -> tuple[Path, np.ndarray, list]:
    """Run inference. Caches (target_kpts, all_kpts) so re-render iterations
    don't pay the ~3 min inference cost per model. Cache invalidated by
    deleting CACHE_DIR or the specific .pkl file."""
    annotated = ART / f"{label}-on-{CLIP.stem}.mp4"
    cache = _cache_path(model_path, CLIP)
    if cache.exists() and annotated.exists():
        print(f"\n=== {label}  ({model_path.name}) [cached]")
        with open(cache, "rb") as f:
            d = pickle.load(f)
        return annotated, d["target"], d["all"]

    print(f"\n=== {label}  ({model_path.name}) ===")
    model = YOLO(str(model_path))
    out_name = f"{label}-on-{CLIP.stem}"
    results = model.predict(
        source=str(CLIP), conf=CONF, save=True,
        project=str(ART / "_ultra"), name=out_name, exist_ok=True,
        stream=True, verbose=False,
    )
    target_kpts_per_frame: list[np.ndarray] = []
    all_kpts_per_frame: list[np.ndarray] = []
    any_det = 0
    roi_det = 0
    n_frames = 0
    roi_cx = (ROI[0] + ROI[2]) / 2
    roi_cy = (ROI[1] + ROI[3]) / 2
    for r in results:
        n_frames += 1
        if r.boxes is None or len(r.boxes) == 0:
            target_kpts_per_frame.append(np.full((5, 3), np.nan))
            all_kpts_per_frame.append(np.zeros((0, 5, 3)))
            continue
        any_det += 1
        xywh = r.boxes.xywh.cpu().numpy()
        kpts_arr = r.keypoints.data.cpu().numpy()
        all_kpts_per_frame.append(kpts_arr)
        best = None
        best_d = float("inf")
        for k in range(len(xywh)):
            cx, cy = float(xywh[k, 0]), float(xywh[k, 1])
            if not in_roi(cx, cy):
                continue
            d = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
            if d < best_d:
                best_d = d
                best = k
        if best is None:
            target_kpts_per_frame.append(np.full((5, 3), np.nan))
            continue
        roi_det += 1
        target_kpts_per_frame.append(kpts_arr[best])

    kpts = np.stack(target_kpts_per_frame, axis=0)
    print(f"  frames={n_frames}  any-detection={any_det}  in-ROI={roi_det}")

    ultra_run_dir = ART / "_ultra" / out_name
    annotated = None
    for ext in (".mp4", ".avi"):
        for p in ultra_run_dir.glob(f"*{ext}"):
            annotated = p
            break
        if annotated:
            break
    if annotated is None:
        raise RuntimeError(f"no annotated video found in {ultra_run_dir}")
    final = ART / f"{label}-on-{CLIP.stem}.mp4"
    final.unlink(missing_ok=True)
    if annotated.suffix == ".mp4":
        final.symlink_to(annotated.resolve())
    else:
        _transcode(annotated, final)
    print(f"  annotated -> {final}")

    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump({"target": kpts, "all": all_kpts_per_frame}, f)
    return final, kpts, all_kpts_per_frame


# Keypoint indices match the v0.X training schema (see VALIDATION.md).
KPT_NOSE = 0
KPT_L_BASE = 1
KPT_R_BASE = 2
KPT_L_TIP = 3
KPT_R_TIP = 4

# matplotlib uses RGB 0-1; constants above are cv2 BGR 0-255.
V02_RGB = (V02_BGR[2] / 255, V02_BGR[1] / 255, V02_BGR[0] / 255)
V03_RGB = (V03_BGR[2] / 255, V03_BGR[1] / 255, V03_BGR[0] / 255)
V04_RGB = (V04_BGR[2] / 255, V04_BGR[1] / 255, V04_BGR[0] / 255)


def _safe_angle(midline_vec: np.ndarray, ear_vec: np.ndarray) -> float:
    """Signed angle (deg) between ear_vec and midline_vec. Positive when the
    ear tip lies on the snout-side of the head midline (alert/forward),
    negative when behind (drooping/back). Matches the SPFES convention used
    in the labeler observability chart."""
    cross = midline_vec[0] * ear_vec[1] - midline_vec[1] * ear_vec[0]
    dot = midline_vec[0] * ear_vec[0] + midline_vec[1] * ear_vec[1]
    return float(np.degrees(np.arctan2(cross, dot)))


def compute_ear_angles(kpt: np.ndarray) -> tuple[float, float]:
    """Return (L_ear_deg, R_ear_deg) for a single (5,3) keypoint array.

    Head midline runs from the midpoint of the ear bases toward the nose
    (= the snout-pointing axis of the head). The left/right ear vectors are
    (ear_base → ear_tip). Returns (NaN, NaN) when nose or both bases are
    missing — that's the floor on what we can measure with this keypoint
    set.
    """
    nose, lb, rb, lt, rt = kpt
    if nose[2] <= 0 or lb[2] <= 0 or rb[2] <= 0:
        return np.nan, np.nan
    mid = (lb[:2] + rb[:2]) / 2
    midline = nose[:2] - mid
    l_angle = _safe_angle(midline, lt[:2] - lb[:2]) if lt[2] > 0 else np.nan
    r_angle = _safe_angle(midline, rt[:2] - rb[:2]) if rt[2] > 0 else np.nan
    return l_angle, r_angle


def _residual_sigma_1d(x: np.ndarray, w: int = 7) -> float | None:
    """Std-dev of (x − rolling-median-w-of-x) over non-NaN entries.
    Matches the residual-σ definition used elsewhere in this bench."""
    ok = ~np.isnan(x)
    if ok.sum() < 8:
        return None
    v = x[ok]
    half = w // 2
    rmed = np.empty_like(v)
    for i in range(len(v)):
        rmed[i] = np.median(v[max(0, i - half): min(len(v), i + half + 1)])
    return float(np.std(v - rmed))


def ear_angle_series(target_kpts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (L, R) ear angle arrays over the σ window, NaN where the
    geometry is undefined (missing detection or kpt below confidence)."""
    a, b = WINDOW
    n = b - a
    L = np.empty(n)
    R = np.empty(n)
    for i in range(n):
        L[i], R[i] = compute_ear_angles(target_kpts[a + i])
    return L, R


def make_ear_angle_chart(targets: dict[str, np.ndarray], fps: float,
                          out_path: Path) -> Path:
    """Two-row matplotlib time series: left ear (top) and right ear (bottom).
    Each row overlays v0.2/v0.3/v0.4. The point is visual: v0.4's line is
    flat, v0.2's bounces. SPFES bands (green/amber/red) drawn behind for
    welfare context — they're not measurement thresholds in this clip, just
    a reading aid that matches the labeler dashboard."""
    a, b = WINDOW
    n = b - a
    t = np.arange(n) / fps
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    colors = {"v0.2": V02_RGB, "v0.3": V03_RGB, "v0.4": V04_RGB}
    series_pairs = {label: ear_angle_series(arr) for label, arr in targets.items()}
    for ax, ear_idx, ear_name in zip(axes, (0, 1), ("Left", "Right")):
        # SPFES bands (per labeler dashboard convention)
        ax.axhspan(-100, -10, color="red", alpha=0.05, zorder=0)
        ax.axhspan(-10, 30, color="orange", alpha=0.05, zorder=0)
        ax.axhspan(30, 200, color="green", alpha=0.05, zorder=0)
        for label, (L, R) in series_pairs.items():
            y = L if ear_idx == 0 else R
            sig = _residual_sigma_1d(y)
            sig_str = f"  σ={sig:.1f}°" if sig is not None else ""
            ax.plot(t, y, color=colors[label], linewidth=1.6, alpha=0.9,
                    label=f"{label}{sig_str}")
        ax.set_ylabel(f"{ear_name} ear angle (°)")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlim(0, float(t[-1]))
    axes[-1].set_xlabel(f"Time in motionless window (s)  ·  clip: {CLIP.stem}")
    fig.suptitle(
        "Ear angle on a stationary held-out sheep — same clip, three models\n"
        "Flatter = usable welfare signal · noisier = jitter masking the signal",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def predict_stock_baseline(stock_model: str = "yolo26n.pt",
                            label: str = "stock-yolo26n") -> tuple[Path, dict]:
    """Run stock COCO-trained YOLO on the clip — detection only, no kpts.

    This is the floor Post 2 promised to show: "off-the-shelf YOLO" applied
    to ear-angle measurement. The truthful answer is *stronger* than
    Post 2 framed it — stock yolo26n.pt has no keypoint head at all, so
    ear angle isn't noisy, it's literally unmeasurable. The annotated mp4
    shows boxes labeled `sheep`/`horse`/`cow` (closest COCO classes) and
    not a single dot on any animal's face.

    Caches the summary stats so re-renders skip the ~3 min inference cost.
    """
    annotated_final = ART / f"{label}-on-{CLIP.stem}.mp4"
    cache = CACHE_DIR / f"{label}__{CLIP.stem}.pkl"
    if cache.exists() and annotated_final.exists():
        print(f"\n=== {label}  ({stock_model}) [cached]")
        with open(cache, "rb") as f:
            return annotated_final, pickle.load(f)

    print(f"\n=== {label}  ({stock_model}) — detection only ===")
    model = YOLO(stock_model)
    out_name = f"{label}-on-{CLIP.stem}"
    # COCO indices for the closest classes to a sheep: 17=cat (no), 19=sheep,
    # 18=horse, 20=cow, 22=zebra. Restrict to ungulate-ish classes so YOLO
    # doesn't draw bird/person boxes on the pasture noise.
    sheep_class = 19
    n_frames = 0
    sheep_dets = 0
    any_dets = 0
    other_classes: dict[int, int] = {}
    results = model.predict(
        source=str(CLIP), conf=0.25, save=True,
        project=str(ART / "_ultra"), name=out_name, exist_ok=True,
        stream=True, verbose=False,
    )
    for r in results:
        n_frames += 1
        if r.boxes is None or len(r.boxes) == 0:
            continue
        any_dets += 1
        for c in r.boxes.cls.cpu().numpy().astype(int):
            if c == sheep_class:
                sheep_dets += 1
                break
        for c in r.boxes.cls.cpu().numpy().astype(int):
            if c != sheep_class:
                other_classes[int(c)] = other_classes.get(int(c), 0) + 1
    print(f"  frames={n_frames}  any-detection={any_dets}  sheep-class={sheep_dets}")

    ultra_run_dir = ART / "_ultra" / out_name
    annotated = None
    for ext in (".mp4", ".avi"):
        for p in ultra_run_dir.glob(f"*{ext}"):
            annotated = p
            break
        if annotated:
            break
    if annotated is None:
        raise RuntimeError(f"no annotated video found in {ultra_run_dir}")
    annotated_final.unlink(missing_ok=True)
    if annotated.suffix == ".mp4":
        annotated_final.symlink_to(annotated.resolve())
    else:
        _transcode(annotated, annotated_final)

    summary = {
        "model": stock_model,
        "n_frames": n_frames,
        "any_detection": any_dets,
        "sheep_class_detections": sheep_dets,
        "other_class_counts": other_classes,
        "keypoints_produced": 0,
        "ear_angle_measurable": False,
    }
    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(summary, f)
    return annotated_final, summary


def dedupe_detections(arr: np.ndarray, dist_px: float = 50.0) -> np.ndarray:
    """Merge YOLO-pose detections whose nose-kpt centers are < dist_px apart,
    keeping the one with higher detection confidence (kpt[0] visibility ~ conf).

    Background: Ultralytics inference uses IOU-based NMS on bounding boxes
    (default 0.7). On tighter v0.4-style detections the same sheep can produce
    two boxes with <70% overlap, so NMS leaves both. Each surviving detection
    drags its own 5 kpts — you see two noses + four ears stacked on one head.

    This second pass dedupes by kpt-distance rather than IOU, which is more
    robust to box tightness and is what the visualization actually cares about
    (the kpts are the demo). σ math is unaffected: we pick the in-ROI detection
    closest to ROI center, so duplicates wouldn't have entered the math
    anyway.

    Input: (N, 5, 3) — N detections, 5 kpts each, (x, y, conf).
    """
    if arr is None or len(arr) == 0:
        return arr
    keep_mask = np.ones(len(arr), dtype=bool)
    # Use mean of all visible kpt positions as the detection center; that's
    # more stable than the nose alone (which can be NaN/0-conf on partial dets).
    centers = []
    confs = []
    for det in arr:
        vis = det[:, 2] > 0
        if vis.any():
            centers.append(det[vis, :2].mean(axis=0))
            confs.append(float(det[vis, 2].mean()))
        else:
            centers.append(np.array([np.nan, np.nan]))
            confs.append(0.0)
    centers = np.asarray(centers)
    for i in range(len(arr)):
        if not keep_mask[i]:
            continue
        if np.isnan(centers[i, 0]):
            keep_mask[i] = False
            continue
        for j in range(i + 1, len(arr)):
            if not keep_mask[j] or np.isnan(centers[j, 0]):
                continue
            d = float(np.linalg.norm(centers[i] - centers[j]))
            if d < dist_px:
                # Keep the higher-confidence detection
                if confs[i] >= confs[j]:
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    break
    return arr[keep_mask]


def crop_from_n_models(all_kpts_per_model, window, src_w, src_h, pad=80):
    """Auto-fit crop to every keypoint detected by any model anywhere in the
    σ window. Pad ~half-head. The wide framing this produces keeps every
    in-frame sheep visible — the "model working on the whole flock" demo.

    (An earlier version tightened this to ROI ± fixed pad, thinking the
    auto-fit was responsible for the SXS graininess. It wasn't — the
    graininess was the cv2 mp4v codec at default low bitrate. PyAV +
    libx264 + CRF 18 fixes the video quality without losing the wide
    framing.)
    """
    a, b = window
    xs, ys = [], []
    for kpts_list in all_kpts_per_model:
        for i in range(a, b):
            arr = kpts_list[i]
            if arr is None or len(arr) == 0:
                continue
            xy = arr[:, :, :2]
            conf = arr[:, :, 2]
            valid = conf > 0
            if not valid.any():
                continue
            xs.extend(xy[..., 0][valid].tolist())
            ys.extend(xy[..., 1][valid].tolist())
    if not xs:
        return (0, 0, src_w, src_h)
    return (
        max(0, int(min(xs)) - pad),
        max(0, int(min(ys)) - pad),
        min(src_w, int(max(xs)) + pad),
        min(src_h, int(max(ys)) + pad),
    )


def _write_h264(out_path: Path, frame_iter, fps: float, w: int, h: int,
                crf: int = 18) -> None:
    """Stream BGR frames through PyAV → libx264 → mp4 at CRF 18.

    Replaces cv2.VideoWriter with mp4v fourcc (MPEG-4 Part 2 at ~22
    bits/pixel/sec at our dimensions — visibly grainy). CRF 18 is the
    "visually lossless" reference setting; expect ~50-100 bits/pixel/sec
    output. yuv420p for broad compatibility; libx264 wants even dimensions
    so the caller must pass even width/height.
    """
    assert w % 2 == 0 and h % 2 == 0, f"libx264/yuv420p need even dims, got {w}x{h}"
    container = av.open(str(out_path), mode="w")
    # PyAV needs an int/Fraction rate, not a float. cv2.CAP_PROP_FPS returns a
    # float (often 30.0); rounding to int is fine for our 30 fps source clips.
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "medium"}
    for frame_bgr in frame_iter:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def sigma_report_n(targets: dict[str, np.ndarray]) -> dict:
    a, b = WINDOW
    n = b - a
    has = {lbl: ~np.isnan(t[a:b, 0, 0]) for lbl, t in targets.items()}
    own, resid = {}, {}
    for lbl, t in targets.items():
        own[lbl] = _per_kpt(t[a:b], has[lbl])
        resid[lbl] = _per_kpt_residual(t[a:b])
    return {
        "window_frames": [a, b],
        "window_seconds": round(n / 30.0, 2),
        "detection_rate": {
            lbl: f"{int(has[lbl].sum())}/{n} ({has[lbl].sum()/n*100:.0f}%)"
            for lbl in targets
        },
        "own_frames_raw_sigma": own,
        "residual_sigma": resid,
    }


def make_n_way_sxs(labels, colors, captions, alls, out_path):
    cap = cv2.VideoCapture(str(CLIP))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx0, cy0, cx1, cy1 = crop_from_n_models(alls, WINDOW, src_w, src_h)
    crop_w = cx1 - cx0
    crop_h = cy1 - cy0
    panel_h = 720
    panel_w = int(round(panel_h * crop_w / crop_h))
    # libx264 + yuv420p requires even dimensions
    if panel_w % 2:
        panel_w += 1
    scale = panel_h / crop_h
    out_w = panel_w * len(labels)
    out_h = panel_h
    a, b = WINDOW

    def frame_iter():
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        for i in range(a, b):
            ok, frm = cap.read()
            if not ok:
                break
            cropped = cv2.resize(
                frm[cy0:cy1, cx0:cx1], (panel_w, panel_h),
                interpolation=cv2.INTER_CUBIC,
            )
            panels = []
            for color, caption, all_kpts in zip(colors, captions, alls):
                panel = _compose_panel_zoom(
                    cropped.copy(), all_kpts[i], scale, (cx0, cy0),
                    color, caption,
                )
                panels.append(panel)
            yield np.concatenate(panels, axis=1)

    _write_h264(out_path, frame_iter(), fps, out_w, out_h)
    cap.release()
    return out_path


def make_n_way_stills(labels, colors, targets, alls, out_path):
    cap = cv2.VideoCapture(str(CLIP))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a, b = WINDOW
    all_fired = [
        i for i in range(a, b)
        if all(not np.isnan(t[i, 0, 0]) for t in targets)
    ]
    if len(all_fired) >= 4:
        idx = np.linspace(0, len(all_fired) - 1, 4).astype(int)
        sample_frames = [all_fired[i] for i in idx]
    else:
        sample_frames = [a + (b - a) * f // 4 for f in (1, 2, 3)] + [b - 5]
    cx0, cy0, cx1, cy1 = crop_from_n_models(alls, WINDOW, src_w, src_h)
    crop_w = cx1 - cx0
    crop_h = cy1 - cy0
    cell_h = 540
    cell_w = int(round(cell_h * crop_w / crop_h))
    scale = cell_h / crop_h
    rows = []
    for label, all_kpts, color in zip(labels, alls, colors):
        row_cells = []
        for fi in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            cell = cv2.resize(
                frm[cy0:cy1, cx0:cx1], (cell_w, cell_h),
                interpolation=cv2.INTER_CUBIC,
            )
            _overlay_kpts_multi(cell, all_kpts[fi], scale, (cx0, cy0),
                                color, dot_r=12)
            cv2.rectangle(cell, (0, 0), (cell_w, 40), (30, 30, 30), -1)
            cv2.putText(cell, f"{label}  frame {fi}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            row_cells.append(cell)
        rows.append(np.concatenate(row_cells, axis=1))
    grid = np.concatenate(rows, axis=0)
    cap.release()
    cv2.imwrite(str(out_path), grid)
    return out_path


def main() -> None:
    assert CLIP.exists(), f"clip not found: {CLIP}"
    print(f"clip: {CLIP}")
    print(f"window: frames {WINDOW[0]}..{WINDOW[1]-1}  ROI: {ROI}")

    v02_mp4, v02_t, v02_all = predict_clip(V02, "v0.2")
    v03_mp4, v03_t, v03_all = predict_clip(V03, "v0.3")
    v04_mp4, v04_t, v04_all = predict_clip(V04, "v0.4")

    # Dedupe overlapping detections for visualization (NMS@0.7 sometimes leaves
    # two boxes on the same sheep — looks like double keypoints in the SXS).
    # target_kpts (the σ math input) is left alone because picking the closest
    # in-ROI detection already handles duplicates for the math.
    v02_all = [dedupe_detections(a) for a in v02_all]
    v03_all = [dedupe_detections(a) for a in v03_all]
    v04_all = [dedupe_detections(a) for a in v04_all]

    # Stock COCO baseline — the "off-the-shelf YOLO" Post 2 promised. Stock
    # yolo26n.pt has no keypoint head, so this is the literal floor: it can
    # find the sheep box but cannot measure ear angle at all.
    stock_mp4, stock_summary = predict_stock_baseline()

    sigmas = sigma_report_n({"v0.2": v02_t, "v0.3": v03_t, "v0.4": v04_t})

    # Ear-angle line chart — the visual Post 2 promised. v0.2 noisy → v0.4 flat.
    ear_angle_png = ART / f"ear_angle_lines-{CLIP.stem}.png"
    make_ear_angle_chart(
        {"v0.2": v02_t, "v0.3": v03_t, "v0.4": v04_t},
        fps=30.0,
        out_path=ear_angle_png,
    )

    # Per-model ear-angle residual σ (degrees) — the welfare-relevant scalar.
    ear_sigmas = {}
    for label, arr in {"v0.2": v02_t, "v0.3": v03_t, "v0.4": v04_t}.items():
        L, R = ear_angle_series(arr)
        ear_sigmas[label] = {
            "L_ear_residual_sigma_deg": _residual_sigma_1d(L),
            "R_ear_residual_sigma_deg": _residual_sigma_1d(R),
            "n_L_visible": int((~np.isnan(L)).sum()),
            "n_R_visible": int((~np.isnan(R)).sum()),
        }

    hero_sxs = ART / f"v0.2-vs-v0.4-{CLIP.stem}.mp4"
    make_n_way_sxs(
        labels=["v0.2", "v0.4"],
        colors=[V02_BGR, V04_BGR],
        captions=["v0.2  (98 instances, 3 vids)",
                  "v0.4  (405 instances, 8 vids)"],
        alls=[v02_all, v04_all],
        out_path=hero_sxs,
    )
    hero_stills = ART / f"v0.2-vs-v0.4-{CLIP.stem}.png"
    make_n_way_stills(
        labels=["v0.2", "v0.4"],
        colors=[V02_BGR, V04_BGR],
        targets=[v02_t, v04_t],
        alls=[v02_all, v04_all],
        out_path=hero_stills,
    )

    curve_sxs = ART / f"v0.2-vs-v0.3-vs-v0.4-{CLIP.stem}.mp4"
    make_n_way_sxs(
        labels=["v0.2", "v0.3", "v0.4"],
        colors=[V02_BGR, V03_BGR, V04_BGR],
        captions=["v0.2  (98 inst.)",
                  "v0.3  (313 inst.)",
                  "v0.4  (405 inst.)"],
        alls=[v02_all, v03_all, v04_all],
        out_path=curve_sxs,
    )
    curve_stills = ART / f"v0.2-vs-v0.3-vs-v0.4-{CLIP.stem}.png"
    make_n_way_stills(
        labels=["v0.2", "v0.3", "v0.4"],
        colors=[V02_BGR, V03_BGR, V04_BGR],
        targets=[v02_t, v03_t, v04_t],
        alls=[v02_all, v03_all, v04_all],
        out_path=curve_stills,
    )

    report = {
        "clip": str(CLIP.relative_to(ROOT)),
        "held_out": True,
        "training_set_v0.2": ["6f689b79", "7e53dfab", "d6873739"],
        "training_set_v0.3": ["6f689b79", "7e53dfab", "d6873739",
                              "00d25853", "0d1655d2", "c74474f9"],
        "training_set_v0.4": ["6f689b79", "7e53dfab", "d6873739",
                              "00d25853", "0d1655d2", "c74474f9",
                              "57ad3d2c", "92930772"],
        "conf_threshold": CONF,
        "annotated_mp4": {
            "v0.2": str(v02_mp4.relative_to(ROOT)),
            "v0.3": str(v03_mp4.relative_to(ROOT)),
            "v0.4": str(v04_mp4.relative_to(ROOT)),
        },
        "hero_v0.2_vs_v0.4_mp4": str(hero_sxs.relative_to(ROOT)),
        "hero_v0.2_vs_v0.4_png": str(hero_stills.relative_to(ROOT)),
        "curve_3way_mp4": str(curve_sxs.relative_to(ROOT)),
        "curve_3way_png": str(curve_stills.relative_to(ROOT)),
        "ear_angle_chart_png": str(ear_angle_png.relative_to(ROOT)),
        "stock_baseline": {
            "annotated_mp4": str(stock_mp4.relative_to(ROOT)),
            **stock_summary,
        },
        "sigmas": sigmas,
        "ear_angle_residual_sigma_deg": ear_sigmas,
    }
    res = sigmas["residual_sigma"]
    own = sigmas["own_frames_raw_sigma"]
    det = sigmas["detection_rate"]
    report["headline"] = {
        "detection": {
            "v0.2": det["v0.2"],
            "v0.3": det["v0.3"],
            "v0.4": det["v0.4"],
        },
        "residual_sigma_mean_px": {
            "v0.2": res["v0.2"]["mean_sigma"],
            "v0.3": res["v0.3"]["mean_sigma"],
            "v0.4": res["v0.4"]["mean_sigma"],
        },
        "raw_sigma_mean_px": {
            "v0.2": own["v0.2"]["mean_sigma"],
            "v0.3": own["v0.3"]["mean_sigma"],
            "v0.4": own["v0.4"]["mean_sigma"],
        },
        "ear_angle_residual_sigma_deg": {
            label: {
                "L": ear_sigmas[label]["L_ear_residual_sigma_deg"],
                "R": ear_sigmas[label]["R_ear_residual_sigma_deg"],
            }
            for label in ("v0.2", "v0.3", "v0.4")
        },
        "off_the_shelf_yolo": (
            f"stock {stock_summary['model']} produced "
            f"{stock_summary['keypoints_produced']} keypoints on this clip "
            f"({stock_summary['sheep_class_detections']}/{stock_summary['n_frames']} "
            f"frames flagged as COCO sheep). Ear angle is unmeasurable without "
            f"a keypoint head."
        ),
    }
    out_json = ART / f"bench_report-{CLIP.stem}-3way.json"
    out_json.write_text(json.dumps(report, indent=2))
    print("\n--- headline ---")
    print(json.dumps(report["headline"], indent=2))
    print(f"\nFull report -> {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
