"""v0.4 vs v0.5 held-out benchmark on IMG_3651.

Outputs into ./artifacts:
  v0.5-on-IMG_3651.mp4           Ultralytics-annotated inference
  v0.4-vs-v0.5-IMG_3651.mp4     2-up side-by-side
  v0.4-vs-v0.5-IMG_3651.png     4-frame still comparison
  ear_angle_lines-v04v05-IMG_3651.png   ear angle σ chart
  bench_report-IMG_3651-v04v05.json     headline numbers
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import av
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from bench_v02_v03 import (
    _transcode,
    _overlay_kpts_multi,
    _compose_panel_zoom,
    _per_kpt_residual,
    CLIPS,
    CONF,
)

V04_BGR = (255, 0, 255)   # magenta — matches bench_held_out.py

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
CACHE_DIR = ART / "_cache"

V04 = ROOT / "weights" / "sheep-pose-v0.4-yolo26n.pt"
V05 = ROOT / "weights" / "sheep-pose-v0.5-yolo26n.pt"
V05_BGR = (255, 165, 0)   # cyan-ish blue — distinct from v0.4 magenta

CLIP_KEY = "IMG_3651"
CLIP = CLIPS[CLIP_KEY]["path"]
WINDOW = CLIPS[CLIP_KEY]["window"]
ROI = CLIPS[CLIP_KEY]["roi"]

KPT_NOSE = 0
KPT_L_BASE = 1
KPT_R_BASE = 2
KPT_L_TIP = 3
KPT_R_TIP = 4

V04_RGB = (V04_BGR[2] / 255, V04_BGR[1] / 255, V04_BGR[0] / 255)
V05_RGB = (V05_BGR[2] / 255, V05_BGR[1] / 255, V05_BGR[0] / 255)


def in_roi(cx: float, cy: float) -> bool:
    return ROI[0] <= cx <= ROI[2] and ROI[1] <= cy <= ROI[3]


def _cache_path(model_path: Path) -> Path:
    return CACHE_DIR / f"{model_path.stem}__{CLIP.stem}.pkl"


def predict_clip(model_path: Path, label: str):
    annotated = ART / f"{label}-on-{CLIP.stem}.mp4"
    cache = _cache_path(model_path)
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
    target_kpts_per_frame = []
    all_kpts_per_frame = []
    any_det = roi_det = n_frames = 0
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
        best, best_d = None, float("inf")
        for k in range(len(xywh)):
            cx, cy = float(xywh[k, 0]), float(xywh[k, 1])
            if not in_roi(cx, cy):
                continue
            d = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
            if d < best_d:
                best_d, best = d, k
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
        raise RuntimeError(f"no annotated video in {ultra_run_dir}")
    final = ART / f"{label}-on-{CLIP.stem}.mp4"
    final.unlink(missing_ok=True)
    if annotated.suffix == ".mp4":
        final.symlink_to(annotated.resolve())
    else:
        _transcode(annotated, final)

    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump({"target": kpts, "all": all_kpts_per_frame}, f)
    return final, kpts, all_kpts_per_frame


def _safe_angle(midline_vec, ear_vec):
    cross = midline_vec[0] * ear_vec[1] - midline_vec[1] * ear_vec[0]
    dot = midline_vec[0] * ear_vec[0] + midline_vec[1] * ear_vec[1]
    return float(np.degrees(np.arctan2(cross, dot)))


def compute_ear_angles(kpt):
    nose, lb, rb, lt, rt = kpt
    if nose[2] <= 0 or lb[2] <= 0 or rb[2] <= 0:
        return np.nan, np.nan
    mid = (lb[:2] + rb[:2]) / 2
    midline = nose[:2] - mid
    l_angle = _safe_angle(midline, lt[:2] - lb[:2]) if lt[2] > 0 else np.nan
    r_angle = _safe_angle(midline, rt[:2] - rb[:2]) if rt[2] > 0 else np.nan
    return l_angle, r_angle


def _residual_sigma_1d(x, w=7):
    ok = ~np.isnan(x)
    if ok.sum() < 8:
        return None
    v = x[ok]
    half = w // 2
    rmed = np.empty_like(v)
    for i in range(len(v)):
        rmed[i] = np.median(v[max(0, i - half): min(len(v), i + half + 1)])
    return float(np.std(v - rmed))


def ear_angle_series(target_kpts):
    a, b = WINDOW
    n = b - a
    L, R = np.empty(n), np.empty(n)
    for i in range(n):
        L[i], R[i] = compute_ear_angles(target_kpts[a + i])
    return L, R


def make_ear_angle_chart(targets: dict, fps: float, out_path: Path):
    a, b = WINDOW
    n = b - a
    t = np.arange(n) / fps
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    colors = {"v0.4": V04_RGB, "v0.5": V05_RGB}
    series_pairs = {label: ear_angle_series(arr) for label, arr in targets.items()}
    for ax, ear_idx, ear_name in zip(axes, (0, 1), ("Left", "Right")):
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
        "Ear angle on a stationary held-out sheep — v0.4 vs v0.5\n"
        "Flatter = usable welfare signal · noisier = jitter masking the signal",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def crop_from_n_models(all_kpts_per_model, window, src_w, src_h, pad=80):
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


def _write_h264(out_path, frame_iter, fps, w, h, crf=18):
    assert w % 2 == 0 and h % 2 == 0
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "fast"}
    for bgr in frame_iter:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def make_sxs(labels, colors, alls, out_path):
    cap = cv2.VideoCapture(str(CLIP))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx0, cy0, cx1, cy1 = crop_from_n_models(alls, WINDOW, src_w, src_h)
    crop_w, crop_h = cx1 - cx0, cy1 - cy0
    panel_h = 720
    panel_w = int(round(panel_h * crop_w / crop_h))
    if panel_w % 2:
        panel_w += 1
    scale = panel_h / crop_h
    captions = labels

    def frame_iter():
        a, b = WINDOW
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        for i in range(a, b):
            ok, frm = cap.read()
            if not ok:
                break
            cropped = cv2.resize(frm[cy0:cy1, cx0:cx1], (panel_w, panel_h),
                                 interpolation=cv2.INTER_CUBIC)
            panels = [_compose_panel_zoom(cropped.copy(), alls[m][i], scale,
                                          (cx0, cy0), colors[m], captions[m])
                      for m in range(len(labels))]
            yield np.concatenate(panels, axis=1)

    _write_h264(out_path, frame_iter(), fps, panel_w * len(labels), panel_h)
    cap.release()
    return out_path


def make_stills(labels, colors, targets, alls, out_path):
    cap = cv2.VideoCapture(str(CLIP))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a, b = WINDOW
    all_fired = [i for i in range(a, b)
                 if all(not np.isnan(t[i, 0, 0]) for t in targets)]
    if len(all_fired) >= 4:
        idx = np.linspace(0, len(all_fired) - 1, 4).astype(int)
        sample_frames = [all_fired[i] for i in idx]
    else:
        sample_frames = [a + (b - a) * f // 4 for f in (1, 2, 3)] + [b - 5]
    cx0, cy0, cx1, cy1 = crop_from_n_models(alls, WINDOW, src_w, src_h)
    crop_w, crop_h = cx1 - cx0, cy1 - cy0
    cell_h, cell_w = 540, int(round(540 * crop_w / crop_h))
    scale = cell_h / crop_h
    rows = []
    for label, all_kpts, color in zip(labels, alls, colors):
        row = []
        for fi in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            if not ok:
                continue
            cropped = cv2.resize(frm[cy0:cy1, cx0:cx1], (cell_w, cell_h),
                                 interpolation=cv2.INTER_CUBIC)
            panel = _compose_panel_zoom(cropped, all_kpts[fi], scale,
                                        (cx0, cy0), color, label)
            row.append(panel)
        if row:
            rows.append(np.concatenate(row, axis=1))
    cap.release()
    if rows:
        grid = np.concatenate(rows, axis=0)
        cv2.imwrite(str(out_path), grid)
    return out_path


if __name__ == "__main__":
    _, v04_kpts, v04_all = predict_clip(V04, "v0.4")
    _, v05_kpts, v05_all = predict_clip(V05, "v0.5")

    cap = cv2.VideoCapture(str(CLIP))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # Ear angle chart
    chart_path = ART / f"ear_angle_lines-v04v05-{CLIP.stem}.png"
    make_ear_angle_chart({"v0.4": v04_kpts, "v0.5": v05_kpts}, fps, chart_path)
    print(f"\nEar angle chart -> {chart_path}")

    # SxS video
    sxs_path = ART / f"v0.4-vs-v0.5-{CLIP.stem}.mp4"
    make_sxs(
        ["v0.4", "v0.5"],
        [V04_BGR, V05_BGR],
        [v04_all, v05_all],
        sxs_path,
    )
    print(f"SxS video -> {sxs_path}")

    # Stills
    stills_path = ART / f"v0.4-vs-v0.5-{CLIP.stem}.png"
    make_stills(
        ["v0.4", "v0.5"],
        [V04_BGR, V05_BGR],
        [v04_kpts, v05_kpts],
        [v04_all, v05_all],
        stills_path,
    )
    print(f"Stills -> {stills_path}")

    # Sigma report
    a, b = WINDOW
    report = {}
    for label, kpts in [("v0.4", v04_kpts), ("v0.5", v05_kpts)]:
        L, R = ear_angle_series(kpts)
        sig_L = _residual_sigma_1d(L)
        sig_R = _residual_sigma_1d(R)
        n_roi = int(np.sum(~np.isnan(kpts[a:b, 0, 0])))
        report[label] = {
            "roi_detections": n_roi,
            "sigma_L": round(sig_L, 2) if sig_L else None,
            "sigma_R": round(sig_R, 2) if sig_R else None,
        }
    report_path = ART / f"bench_report-{CLIP.stem}-v04v05.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nBench report:\n{json.dumps(report, indent=2)}")
    print(f"\nSaved -> {report_path}")
