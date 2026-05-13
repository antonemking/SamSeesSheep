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

    sigmas = sigma_report_n({"v0.2": v02_t, "v0.3": v03_t, "v0.4": v04_t})

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
        "sigmas": sigmas,
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
    }
    out_json = ART / f"bench_report-{CLIP.stem}-3way.json"
    out_json.write_text(json.dumps(report, indent=2))
    print("\n--- headline ---")
    print(json.dumps(report["headline"], indent=2))
    print(f"\nFull report -> {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
