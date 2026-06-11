#!/usr/bin/env python3
"""Regenerate the Test_Clip_Morning v0.2/v0.3/v0.4 held-out benchmark JSON
from the cached per-frame keypoint predictions in ``artifacts/_cache``.

Background
----------
``bench_Test_Clip_Morning.json`` only stored the v0.7 ear-angle numbers.
The paper's Section 5.6 also reports v0.2/v0.3/v0.4 per-keypoint residual
sigma, ear-angle residual sigma, raw sigma, and detection rates on this
clip. Those numbers were computed from the pickle caches but never written
to a machine-readable artifact. This script closes that gap so every number
in Section 5.6 is backed by a traceable JSON file.

It reads ONLY the cached predictions (no inference, no GPU). The cache files
are produced by the archived ``bench_v02_v03.py`` /  ``bench_v04_v05*.py``
runs and live in ``artifacts/_cache/sheep-pose-vX-yolo26n__Test_Clip_Morning.pkl``.

Run from the ``sheep-yolo`` directory:
    python scripts/gen_bench_Test_Clip_Morning.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]            # sheep-yolo/
ART = ROOT / "artifacts"
CACHE = ART / "_cache"
sys.path.insert(0, str(ROOT / "scripts" / "archive"))

from bench_v02_v03 import CLIPS, KPT_NAMES, _per_kpt, _per_kpt_residual  # noqa: E402

CLIP_KEY = "Test_Clip_Morning"
A, B = CLIPS[CLIP_KEY]["window"]
N = B - A
ROI = CLIPS[CLIP_KEY]["roi"]

MODELS = {
    "v0.2": "sheep-pose-v0.2-yolo26n__Test_Clip_Morning.pkl",
    "v0.3": "sheep-pose-v0.3-yolo26n__Test_Clip_Morning.pkl",
    "v0.4": "sheep-pose-v0.4-yolo26n__Test_Clip_Morning.pkl",
}


def _safe_angle(midline, ear):
    cross = midline[0] * ear[1] - midline[1] * ear[0]
    dot = midline[0] * ear[0] + midline[1] * ear[1]
    return float(np.degrees(np.arctan2(cross, dot)))


def compute_ear_angles(kpt):
    nose, lb, rb, lt, rt = kpt
    if nose[2] <= 0 or lb[2] <= 0 or rb[2] <= 0:
        return np.nan, np.nan
    mid = (lb[:2] + rb[:2]) / 2
    midline = nose[:2] - mid
    la = _safe_angle(midline, lt[:2] - lb[:2]) if lt[2] > 0 else np.nan
    ra = _safe_angle(midline, rt[:2] - rb[:2]) if rt[2] > 0 else np.nan
    return la, ra


def _residual_sigma_1d(x, w=7):
    ok = ~np.isnan(x)
    if ok.sum() < 8:
        return None
    v = x[ok]
    half = w // 2
    rmed = np.array(
        [np.median(v[max(0, i - half): min(len(v), i + half + 1)]) for i in range(len(v))]
    )
    return float(np.std(v - rmed))


def main() -> int:
    report = {
        "clip": "test-clips/Test_Clip_Morning.mov",
        "held_out": True,
        "window_frames": [A, B],
        "window_seconds": round(N / 30.0, 2),
        "roi": list(ROI),
        "target_track_id": 215,
        "conf_threshold": 0.25,
        "running_median_window": 7,
        "method": (
            "residual sigma = std(kpt - rolling_median_7(kpt)); "
            "signed ear angle via head-midline geometry; cached predictions only"
        ),
        "detection_rate": {},
        "residual_sigma": {},
        "own_frames_raw_sigma": {},
        "ear_angle_residual_sigma_deg": {},
        "headline": {"residual_sigma_mean_px": {}, "raw_sigma_mean_px": {},
                     "ear_angle_residual_sigma_deg": {}},
    }

    for ver, fname in MODELS.items():
        kpts = pickle.load(open(CACHE / fname, "rb"))["target"]
        win = kpts[A:B]
        has = ~np.isnan(win[:, 0, 0])

        resid = _per_kpt_residual(win)
        raw = _per_kpt(win, has)

        L = np.array([compute_ear_angles(win[i])[0] for i in range(N)])
        R = np.array([compute_ear_angles(win[i])[1] for i in range(N)])
        sL, sR = _residual_sigma_1d(L), _residual_sigma_1d(R)

        report["detection_rate"][ver] = f"{int(has.sum())}/{N} ({has.sum() / N * 100:.0f}%)"
        report["residual_sigma"][ver] = resid
        report["own_frames_raw_sigma"][ver] = raw
        report["ear_angle_residual_sigma_deg"][ver] = {
            "L_ear_residual_sigma_deg": sL,
            "R_ear_residual_sigma_deg": sR,
            "n_L_visible": int((~np.isnan(L)).sum()),
            "n_R_visible": int((~np.isnan(R)).sum()),
        }
        report["headline"]["residual_sigma_mean_px"][ver] = resid["mean_sigma"]
        report["headline"]["raw_sigma_mean_px"][ver] = raw["mean_sigma"]
        report["headline"]["ear_angle_residual_sigma_deg"][ver] = {
            "L": round(sL, 2) if sL else None,
            "R": round(sR, 2) if sR else None,
        }

    # stock baseline (no keypoints)
    stock = pickle.load(open(CACHE / "stock-yolo26n__Test_Clip_Morning.pkl", "rb"))
    report["stock_baseline"] = {
        "model": "yolo26n.pt",
        "n_frames": stock.get("n_frames"),
        "any_class_detection_frames": stock.get("any_detection"),
        "sheep_class_detections": stock.get("sheep_class_detections"),
        "keypoints_produced": stock.get("keypoints_produced", 0),
        "ear_angle_measurable": stock.get("ear_angle_measurable", False),
    }

    out = ART / "bench_report-Test_Clip_Morning-v02v03v04.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["headline"], indent=2))
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
