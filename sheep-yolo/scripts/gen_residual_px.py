#!/usr/bin/env python3
"""Recompute mean residual-σ (px) on the HO-1 (IMG_3651) held-out window for
every model version, and save them to a committed JSON artifact.

The v0.2/v0.3/v0.4 values already live in ``bench_report-IMG_3651-3way.json``;
the v0.5/v0.6/v0.7 values are reported in the paper (Tables 4 and 11) but were
not previously saved to any committed artifact (the ``…-v04v05v0{6,7}.json``
files carry only ear-angle σ). This script closes that gap so
``verify_paper_claims.py`` can check them on a fresh clone.

Method matches the per-keypoint residual in ``gen_ear_angle_chart.py``:
    σ_px(kpt) = hypot( std(x − rollmed7(x)), std(y − rollmed7(y)) )  over visible samples,
    mean over the 5 keypoints.
It reproduces the committed v0.2/v0.3/v0.4 numbers exactly.

Usage (from sheep-yolo/, needs the _cache archive present):
    python scripts/gen_residual_px.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "_cache"
OUT = ROOT / "artifacts" / "bench_residual_px-IMG_3651-v05v06v07.json"
WINDOW = (367, 522)  # matches bench_report-IMG_3651-3way.json window_frames
KPTS = ["nose", "L_ear_base", "R_ear_base", "L_ear_tip", "R_ear_tip"]


def roll_med(v: np.ndarray, w: int = 7) -> np.ndarray:
    half = w // 2
    return np.array([np.median(v[max(0, i - half): min(len(v), i + half + 1)]) for i in range(len(v))])


def per_kpt_residual_px(kp: np.ndarray) -> dict:
    out = {}
    for k, name in enumerate(KPTS):
        ok = kp[:, k, 2] > 0
        if ok.sum() < 8:
            out[name] = None
            continue
        sx = np.std(kp[ok, k, 0] - roll_med(kp[ok, k, 0]))
        sy = np.std(kp[ok, k, 1] - roll_med(kp[ok, k, 1]))
        out[name] = round(float(np.hypot(sx, sy)), 2)
    return out


def main() -> int:
    a, b = WINDOW
    per_kpt, mean = {}, {}
    for ver in ["v0.4", "v0.5", "v0.6", "v0.7"]:
        f = CACHE / f"sheep-pose-{ver}-yolo26n__IMG_3651.pkl"
        kp = np.array(pickle.load(open(f, "rb"))["target"])[a:b]
        pk = per_kpt_residual_px(kp)
        per_kpt[ver] = pk
        mean[ver] = round(float(np.mean([v for v in pk.values() if v is not None])), 2)
    doc = {
        "clip": "test-clips/IMG_3651.MOV",
        "paper_id": "HO-1",
        "window_frames": list(WINDOW),
        "window_len": b - a,
        "method": ("residual sigma px per kpt = hypot(std(x - rollmed7(x)), "
                   "std(y - rollmed7(y))) on visible (conf>0) samples; mean over 5 "
                   "kpts. Recomputed from artifacts/_cache/*.pkl; reproduces the "
                   "committed v0.2/v0.3/v0.4 values in bench_report-IMG_3651-3way.json."),
        "residual_sigma_mean_px": mean,
        "per_kpt_residual_sigma_px": per_kpt,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"saved {OUT.name}: {mean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
