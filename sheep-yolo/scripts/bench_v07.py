"""v0.7 (pure in-distribution, 523 labels) vs v0.5 (curriculum) vs v0.4 — held-out IMG_3651.

The 2x2 question: does +118 of my own labels (v0.4 -> v0.7, pure recipe) beat the
AP-10k + hard-negatives curriculum (v0.5)? Same clip / window / ROI / residual
ear-angle math as the published v0.4/v0.5/v0.6 benches, so numbers are directly
comparable. v0.4 and v0.5 load from the bench cache; v0.7 is computed fresh.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
CACHE = ART / "_cache"
ART.mkdir(exist_ok=True)

CLIP = ROOT.parent / "test-clips" / "IMG_3651.MOV"
WINDOW = (367, 522)
ROI = (1220, 80, 1560, 390)
CONF = 0.25

MODELS = {
    "v0.4": ROOT / "weights" / "sheep-pose-v0.4-yolo26n.pt",
    "v0.5": ROOT / "weights" / "sheep-pose-v0.5-yolo26n.pt",
    "v0.7": ROOT / "weights" / "sheep-pose-v0.7-yolo26n.pt",
}


def in_roi(cx, cy):
    return ROI[0] <= cx <= ROI[2] and ROI[1] <= cy <= ROI[3]


def _safe_angle(m, e):
    cross = m[0] * e[1] - m[1] * e[0]
    dot = m[0] * e[0] + m[1] * e[1]
    return float(np.degrees(np.arctan2(cross, dot)))


def compute_ear_angles(kpt):
    nose, lb, rb, lt, rt = kpt
    if nose[2] <= 0 or lb[2] <= 0 or rb[2] <= 0:
        return np.nan, np.nan
    mid = (lb[:2] + rb[:2]) / 2
    midline = nose[:2] - mid
    l = _safe_angle(midline, lt[:2] - lb[:2]) if lt[2] > 0 else np.nan
    r = _safe_angle(midline, rt[:2] - rb[:2]) if rt[2] > 0 else np.nan
    return l, r


def _resid_sigma(x, w=7):
    ok = ~np.isnan(x)
    if ok.sum() < 8:
        return None
    v = x[ok]
    half = w // 2
    rmed = np.empty_like(v)
    for i in range(len(v)):
        rmed[i] = np.median(v[max(0, i - half): min(len(v), i + half + 1)])
    return float(np.std(v - rmed))


def sigma_LR(t):
    a, b = WINDOW
    n = b - a
    L, R = np.empty(n), np.empty(n)
    for i in range(n):
        L[i], R[i] = compute_ear_angles(t[a + i])
    return _resid_sigma(L), _resid_sigma(R)


def predict_target(model_path):
    cache = CACHE / f"{model_path.stem}__{CLIP.stem}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)["target"], "cache"
    model = YOLO(str(model_path))
    rcx, rcy = (ROI[0] + ROI[2]) / 2, (ROI[1] + ROI[3]) / 2
    target, allf = [], []
    for r in model.predict(source=str(CLIP), conf=CONF, stream=True, verbose=False):
        if r.boxes is None or len(r.boxes) == 0:
            target.append(np.full((5, 3), np.nan))
            allf.append(np.zeros((0, 5, 3)))
            continue
        xywh = r.boxes.xywh.cpu().numpy()
        kp = r.keypoints.data.cpu().numpy()
        allf.append(kp)
        best, bd = None, 1e18
        for k in range(len(xywh)):
            cx, cy = float(xywh[k, 0]), float(xywh[k, 1])
            if in_roi(cx, cy) and (cx - rcx) ** 2 + (cy - rcy) ** 2 < bd:
                bd = (cx - rcx) ** 2 + (cy - rcy) ** 2
                best = k
        target.append(kp[best] if best is not None else np.full((5, 3), np.nan))
    arr = np.stack(target, 0)
    CACHE.mkdir(exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump({"target": arr, "all": allf}, f)
    return arr, "computed"


def main():
    a, b = WINDOW
    results = {}
    print(f"clip: {CLIP.name}  window {WINDOW}  ROI {ROI}\n")
    print(f"{'model':6}{'σ Left':>10}{'σ Right':>10}{'σ avg':>9}{'in-ROI':>10}")
    for label, mp in MODELS.items():
        t, src = predict_target(mp)
        L, R = sigma_LR(t)
        rate = int(np.sum(~np.isnan(t[a:b, 0, 0])))
        avg = (L + R) / 2 if (L and R) else None
        results[label] = {"sigma_L": L, "sigma_R": R, "sigma_avg": avg,
                          "in_roi": f"{rate}/{b - a}", "source": src}
        print(f"{label:6}{L:>9.2f}°{R:>9.2f}°{avg:>8.2f}°{rate:>7}/{b - a}")
    (ART / "bench_v07-IMG_3651.json").write_text(json.dumps(results, indent=2))
    print("\nreport -> artifacts/bench_v07-IMG_3651.json")


if __name__ == "__main__":
    main()
