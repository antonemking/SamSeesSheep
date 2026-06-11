"""Top-down (detect -> crop -> keypoints) vs single-shot v0.5 on held-out IMG_3651.

Two modes:

  clip  (default) — the headline. Detector finds the head, we crop with the
        SAME margin as crop_export.py, run the crop-pose model on the zoom,
        map keypoints back to frame space, and compute residual ear-angle σ
        over the published motionless window. Numbers are directly comparable
        to LOR-101: v0.5 was σ_L 3.66° / σ_R 4.30°. The v0.5 single-shot
        baseline is reused from the bench cache pkl if present.

        Detector boxes are PREDICTED (IMG_3651 is held-out — no GT), so this
        is the realistic end-to-end read. Detector defaults to v0.5 weights.

  val   — the clean read. On our labeled v0.4 val split (GT boxes exist),
        compare keypoint error of top-down (GT crop) vs single-shot v0.5
        (full frame, detection IoU-matched to GT). Metric: NME = mean kpt
        pixel error / head-box diagonal. Isolates the keypoint gain from
        detector noise. This is the "does the zoom actually help" number.

  python bench_topdown.py --pose weights/_topdown/v0.4-crops/weights/best.pt
  python bench_topdown.py --mode val --pose <best.pt>
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Keep in lockstep with crop_export.py and train_topdown.sh.
MARGIN = 0.30
CROP_IMGSZ = 256
CONF = 0.25                       # detector conf, matches the published benches
KPT_NAMES = ["nose", "L_ear_base", "R_ear_base", "L_ear_tip", "R_ear_tip"]

ROOT = Path(__file__).resolve().parents[1]            # sheep-yolo/
REPO = ROOT.parent                                    # sheep-seg/
ART = ROOT / "artifacts"
CACHE_DIR = ART / "_cache"
ART.mkdir(exist_ok=True)

V05 = ROOT / "weights" / "sheep-pose-v0.5-yolo26n.pt"
VAL_DIR = Path.home() / "Backups/sheep-seg/labels/exports/sheep-pose-v0.4-yolo26n/val"

# Held-out clip config — copied verbatim from bench_v02_v03.py's CLIPS entry
# so the window/ROI exactly match the v0.4/v0.5/v0.6 numbers.
CLIP = REPO / "test-clips" / "IMG_3651.MOV"
WINDOW = (367, 522)              # 155 frames ≈ 5.1 s @ 30 fps
ROI = (1220, 80, 1560, 390)


def in_roi(cx, cy):
    return ROI[0] <= cx <= ROI[2] and ROI[1] <= cy <= ROI[3]


# ---------- ear-angle math (verbatim from bench_v04_v05_v06.py) ----------

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


def sigma_LR(target_kpts):
    L, R = ear_angle_series(target_kpts)
    return _residual_sigma_1d(L), _residual_sigma_1d(R)


# ---------- crop helper (mirror of crop_export.crop_instance geometry) ----

def expand_box(x1, y1, x2, y2, W, H, margin=MARGIN):
    bw, bh = x2 - x1, y2 - y1
    ex1 = int(max(0, round(x1 - margin * bw)))
    ey1 = int(max(0, round(y1 - margin * bh)))
    ex2 = int(min(W, round(x2 + margin * bw)))
    ey2 = int(min(H, round(y2 + margin * bh)))
    return ex1, ey1, ex2, ey2


def pose_on_crop(pose, img, box_xyxy):
    """Run the crop-pose model on the expanded crop; return (5,3) kpts in
    FRAME pixel coords, or None."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, W, H)
    if ex2 - ex1 < 8 or ey2 - ey1 < 8:
        return None
    crop = img[ey1:ey2, ex1:ex2]
    if crop.size == 0:
        return None
    pr = pose.predict(source=crop, imgsz=CROP_IMGSZ, conf=0.01, verbose=False)[0]
    if pr.keypoints is None or len(pr.keypoints) == 0:
        return None
    bi = 0
    if pr.boxes is not None and len(pr.boxes) > 1:
        bi = int(np.argmax(pr.boxes.conf.cpu().numpy()))
    kp = pr.keypoints.data.cpu().numpy()[bi].copy()     # (5,3) in crop px
    kp[:, 0] += ex1
    kp[:, 1] += ey1
    return kp


# ---------- clip mode -----------------------------------------------------

def predict_topdown_clip(detector_path, pose_path):
    det = YOLO(str(detector_path))
    pose = YOLO(str(pose_path))
    roi_cx = (ROI[0] + ROI[2]) / 2
    roi_cy = (ROI[1] + ROI[3]) / 2

    target = []
    n_frames = n_roi = 0
    for r in det.predict(source=str(CLIP), conf=CONF, stream=True, verbose=False):
        n_frames += 1
        if r.boxes is None or len(r.boxes) == 0:
            target.append(np.full((5, 3), np.nan))
            continue
        img = r.orig_img
        xyxy = r.boxes.xyxy.cpu().numpy()
        xywh = r.boxes.xywh.cpu().numpy()
        best, best_d = None, float("inf")
        for k in range(len(xywh)):
            cx, cy = float(xywh[k, 0]), float(xywh[k, 1])
            if not in_roi(cx, cy):
                continue
            d = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
            if d < best_d:
                best_d, best = d, k
        if best is None:
            target.append(np.full((5, 3), np.nan))
            continue
        kp = pose_on_crop(pose, img, xyxy[best])
        target.append(kp if kp is not None else np.full((5, 3), np.nan))
        if kp is not None:
            n_roi += 1
    return np.stack(target, axis=0), n_frames, n_roi


def load_v05_baseline():
    """Reuse the cached v0.5 single-shot target kpts if available."""
    cache = CACHE_DIR / "sheep-pose-v0.5-yolo26n__IMG_3651.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)["target"], "cache"
    # Fallback: recompute single-shot v0.5.
    model = YOLO(str(V05))
    roi_cx = (ROI[0] + ROI[2]) / 2
    roi_cy = (ROI[1] + ROI[3]) / 2
    target = []
    for r in model.predict(source=str(CLIP), conf=CONF, stream=True, verbose=False):
        if r.boxes is None or len(r.boxes) == 0:
            target.append(np.full((5, 3), np.nan))
            continue
        xywh = r.boxes.xywh.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy()
        best, best_d = None, float("inf")
        for k in range(len(xywh)):
            cx, cy = float(xywh[k, 0]), float(xywh[k, 1])
            if in_roi(cx, cy) and (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2 < best_d:
                best_d, best = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2, k
        target.append(kpts[best] if best is not None else np.full((5, 3), np.nan))
    return np.stack(target, axis=0), "recomputed"


def run_clip(pose_path, detector_path):
    print(f"clip: {CLIP.name}  window {WINDOW}  ROI {ROI}")
    print(f"detector: {Path(detector_path).name}   crop-pose: {Path(pose_path).name}\n")

    td, n_frames, n_roi = predict_topdown_clip(detector_path, pose_path)
    td_L, td_R = sigma_LR(td)
    v05, src = load_v05_baseline()
    v05_L, v05_R = sigma_LR(v05)

    a, b = WINDOW
    td_rate = int(np.sum(~np.isnan(td[a:b, 0, 0])))
    v05_rate = int(np.sum(~np.isnan(v05[a:b, 0, 0])))
    n = b - a

    def fmt(x):
        return f"{x:.2f}°" if x is not None else "  n/a"

    print(f"{'':16}{'v0.5 single-shot':>20}{'top-down':>14}")
    print(f"{'in-ROI frames':16}{f'{v05_rate}/{n}':>20}{f'{td_rate}/{n}':>14}")
    print(f"{'σ Left ear':16}{fmt(v05_L):>20}{fmt(td_L):>14}")
    print(f"{'σ Right ear':16}{fmt(v05_R):>20}{fmt(td_R):>14}")
    if td_L and td_R and v05_L and v05_R:
        print(f"{'σ avg':16}{fmt((v05_L+v05_R)/2):>20}{fmt((td_L+td_R)/2):>14}")
    print(f"\n(v0.5 baseline from {src})")

    out = ART / "bench_topdown-IMG_3651.json"
    out.write_text(json.dumps({
        "clip": CLIP.name, "window": list(WINDOW), "roi": list(ROI),
        "detector": Path(detector_path).name, "crop_pose": Path(pose_path).name,
        "margin": MARGIN, "crop_imgsz": CROP_IMGSZ,
        "v0.5_single_shot": {"sigma_L": v05_L, "sigma_R": v05_R,
                             "in_roi": f"{v05_rate}/{n}", "source": src},
        "top_down": {"sigma_L": td_L, "sigma_R": td_R,
                     "in_roi": f"{n_roi}/{n_frames}"},
    }, indent=2))
    print(f"report -> {out}")


# ---------- val mode (clean keypoint A/B with GT boxes) -------------------

def _parse_label_px(line, W, H):
    p = line.split()
    cx, cy, w, h = (float(v) for v in p[1:5])
    x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
    x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
    rest = [float(v) for v in p[5:]]
    kpts = np.array([[rest[i] * W, rest[i + 1] * H, rest[i + 2]]
                     for i in range(0, len(rest), 3)])
    return (x1, y1, x2, y2), kpts


def _nme(pred_kpts, gt_kpts, diag):
    """Mean kpt pixel error over GT-visible kpts, normalized by box diagonal."""
    errs = []
    for k in range(5):
        if gt_kpts[k, 2] <= 0:
            continue
        if pred_kpts is None or pred_kpts[k, 2] <= 0:
            errs.append(1.0)           # missed a visible kpt: full-diagonal penalty
            continue
        d = np.hypot(pred_kpts[k, 0] - gt_kpts[k, 0], pred_kpts[k, 1] - gt_kpts[k, 1])
        errs.append(d / diag)
    return float(np.mean(errs)) if errs else None


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run_val(pose_path):
    pose = YOLO(str(pose_path))
    ss = YOLO(str(V05))
    img_dir, lbl_dir = VAL_DIR / "images", VAL_DIR / "labels"
    td_nmes, ss_nmes_full = [], []
    ss_nmes_matched = []          # keypoint NME only where single-shot found the head
    n_inst = ss_matched = 0
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        lines = [l for l in lbl_path.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        img_path = next((img_dir / f"{lbl_path.stem}{e}"
                         for e in (".jpg", ".jpeg", ".png")
                         if (img_dir / f"{lbl_path.stem}{e}").exists()), None)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        ss_res = ss.predict(source=img, conf=CONF, verbose=False)[0]
        ss_boxes = (ss_res.boxes.xyxy.cpu().numpy() if ss_res.boxes is not None
                    and len(ss_res.boxes) else np.zeros((0, 4)))
        ss_kpts = (ss_res.keypoints.data.cpu().numpy() if ss_res.keypoints is not None
                   and len(ss_res.keypoints) else np.zeros((0, 5, 3)))
        for line in lines:
            box, gt = _parse_label_px(line, W, H)
            diag = np.hypot(box[2] - box[0], box[3] - box[1])
            if diag < 1:
                continue
            n_inst += 1
            # top-down: GT box -> crop -> pose (always "matched", GT box given)
            td_kp = pose_on_crop(pose, img, box)
            td_nmes.append(_nme(td_kp, gt, diag))
            # single-shot: best IoU match to GT box
            best, best_iou = None, 0.0
            for j in range(len(ss_boxes)):
                v = _iou(box, ss_boxes[j])
                if v > best_iou:
                    best_iou, best = v, j
            if best is not None and best_iou > 0.3:
                ss_matched += 1
                m = _nme(ss_kpts[best], gt, diag)
                ss_nmes_full.append(m)
                ss_nmes_matched.append(m)
            else:
                ss_nmes_full.append(_nme(None, gt, diag))   # full-diagonal miss penalty

    td_nmes = [x for x in td_nmes if x is not None]
    ss_full = [x for x in ss_nmes_full if x is not None]
    ss_match = [x for x in ss_nmes_matched if x is not None]
    print(f"val instances: {n_inst}   (source: {VAL_DIR})")
    print(f"single-shot head-detection rate: {ss_matched}/{n_inst} "
          f"({ss_matched/n_inst*100:.0f}%)\n")
    print(f"{'':34}{'NME (lower=better)':>20}")
    print(f"{'top-down (GT crop)':34}{np.mean(td_nmes):>20.4f}")
    print(f"{'single-shot, MATCHED only':34}{np.mean(ss_match):>20.4f}"
          f"   <- clean keypoint A/B")
    print(f"{'single-shot, incl. det misses':34}{np.mean(ss_full):>20.4f}"
          f"   <- end-to-end (penalizes misses)")
    clean = (np.mean(ss_match) - np.mean(td_nmes)) / np.mean(ss_match) * 100
    print(f"\nClean keypoint precision: top-down {clean:+.1f}% vs single-shot "
          f"({'better' if clean > 0 else 'worse'})")
    print("Detection misses inflate the end-to-end gap; the σ bench (clip mode)"
          " is the stability read.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["clip", "val"], default="clip")
    ap.add_argument("--pose", required=True, help="crop-pose best.pt")
    ap.add_argument("--detector", default=str(V05),
                    help="detector weights for clip mode (default v0.5)")
    args = ap.parse_args()
    pose = args.pose if Path(args.pose).is_absolute() else str(ROOT / args.pose)
    if args.mode == "clip":
        run_clip(pose, args.detector)
    else:
        run_val(pose)


if __name__ == "__main__":
    main()
