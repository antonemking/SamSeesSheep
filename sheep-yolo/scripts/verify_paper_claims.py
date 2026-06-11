#!/usr/bin/env python3
"""Verify that every headline number in the paper (docs/archive-paper.md and
its arXiv LaTeX source) is backed by a saved benchmark JSON artifact.

Run from the sheep-yolo directory:
    cd sheep-yolo
    python scripts/verify_paper_claims.py

This script does NOT run inference. It only reads saved JSON artifacts and
compares them against the values claimed in the paper. Every artifact it
reads can itself be regenerated from the cached predictions in
``artifacts/_cache`` via:
    python scripts/gen_bench_Test_Clip_Morning.py     # Test_Clip_Morning v0.2-v0.4
    python scripts/archive/bench_v02_v03.py IMG_3651  # IMG_3651 3-way
    python scripts/archive/bench_v04_v05_v06.py        # IMG_3651 v0.4/v0.5/v0.6
    python scripts/archive/bench_v04_v05_v07.py        # IMG_3651 v0.4/v0.5/v0.7

Exit code is 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ART = Path(__file__).resolve().parent.parent / "artifacts"


def check(claim: str, expected: float, actual: float, tol: float | None = None) -> bool:
    if tol is None:
        tol = max(abs(expected) * 0.02, 0.02)  # 2% or 0.02 absolute
    ok = abs(actual - expected) <= tol
    status = "PASS" if ok else f"FAIL (diff={actual - expected:+.3f})"
    print(f"  {status:6s}  {claim}: paper={expected}, artifact={actual:.3f}, tol={tol:.3f}")
    return ok


def load(name: str):
    p = ART / name
    if not p.exists():
        print(f"  MISSING ARTIFACT: {p}")
        return None
    return json.loads(p.read_text())


def main() -> int:
    failures = 0

    def fail_if(cond_ok: bool):
        nonlocal failures
        if not cond_ok:
            failures += 1

    # ── Test_Clip_Morning v0.7 (primary field, NOT the qa_roi variant) ──
    print("\nTest_Clip_Morning v0.7  [bench_Test_Clip_Morning.json]")
    data = load("bench_Test_Clip_Morning.json")
    if data:
        v07 = data["v0.7"]
        fail_if(check("sigma_left (deg)", 2.39, v07["sigma_left"]))
        fail_if(check("sigma_right (deg)", 3.29, v07["sigma_right"]))
        fail_if(check("sigma_avg (deg)", 2.84, v07["sigma_avg"]))
        # The paper uses the primary v0.7 field (2.84), not qa_roi (2.81).
        fail_if(check("primary field is 2.84 not qa 2.81", 2.84,
                      data["v0.7"]["sigma_avg"], tol=0.005))
        parts = v07.get("in_roi_target_detection_rate", "").split("/")
        if len(parts) == 2:
            fail_if(check("detection rate", 148 / 150, int(parts[0]) / int(parts[1]), tol=0.01))

    # ── Test_Clip_Morning v0.2/v0.3/v0.4 ───────────────────────────────
    print("\nTest_Clip_Morning v0.2/v0.3/v0.4  [bench_report-Test_Clip_Morning-v02v03v04.json]")
    data = load("bench_report-Test_Clip_Morning-v02v03v04.json")
    if data:
        resid = data["headline"]["residual_sigma_mean_px"]
        for ver, exp in {"v0.2": 6.73, "v0.3": 4.22, "v0.4": 3.85}.items():
            fail_if(check(f"{ver} residual sigma mean px", exp, resid[ver]))
        ang = data["headline"]["ear_angle_residual_sigma_deg"]
        for ver, (l, r) in {"v0.2": (5.37, 2.24), "v0.3": (2.49, 2.08),
                            "v0.4": (2.50, 2.72)}.items():
            fail_if(check(f"{ver} ear-angle L (deg)", l, ang[ver]["L"]))
            fail_if(check(f"{ver} ear-angle R (deg)", r, ang[ver]["R"]))

    # ── IMG_3651 v0.2/v0.3/v0.4 pixel + ear-angle sigma ────────────────
    print("\nIMG_3651 v0.2/v0.3/v0.4  [bench_report-IMG_3651-3way.json]")
    data = load("bench_report-IMG_3651-3way.json")
    if data:
        resid = data["headline"]["residual_sigma_mean_px"]
        for ver, exp in {"v0.2": 10.89, "v0.3": 8.90, "v0.4": 7.70}.items():
            fail_if(check(f"{ver} residual sigma mean px", exp, resid[ver]))
        raw = data["headline"]["raw_sigma_mean_px"]
        for ver, exp in {"v0.2": 49.44, "v0.3": 46.90, "v0.4": 46.60}.items():
            fail_if(check(f"{ver} raw sigma mean px", exp, raw[ver]))
        ang = data["headline"]["ear_angle_residual_sigma_deg"]
        for ver, (l, r) in {"v0.2": (6.71, 6.07), "v0.3": (4.82, 4.21),
                            "v0.4": (4.06, 4.09)}.items():
            fail_if(check(f"{ver} ear-angle L (deg)", l, ang[ver]["L"]))
            fail_if(check(f"{ver} ear-angle R (deg)", r, ang[ver]["R"]))
        stock = data["stock_baseline"]
        fail_if(check("stock sheep-class detections", 323, stock["sheep_class_detections"], tol=0))
        fail_if(check("stock keypoints produced", 0, stock["keypoints_produced"], tol=0))

    # ── IMG_3651 v0.6 ear-angle sigma (the regression case) ────────────
    print("\nIMG_3651 v0.6 (regression case)  [bench_report-IMG_3651-v04v05v06.json]")
    data = load("bench_report-IMG_3651-v04v05v06.json")
    if data:
        fail_if(check("v0.6 ear-angle L (deg)", 4.65, data["v0.6"]["sigma_L"]))
        fail_if(check("v0.6 ear-angle R (deg)", 3.55, data["v0.6"]["sigma_R"]))

    # ── IMG_3651 v0.4 / v0.5 / v0.7 ear-angle sigma ────────────────────
    print("\nIMG_3651 v0.4 / v0.5 / v0.7 ear-angle sigma  [bench_report-IMG_3651-v04v05v07.json]")
    data = load("bench_report-IMG_3651-v04v05v07.json")
    if data:
        for ver, (l, r) in {"v0.4": (4.06, 4.09), "v0.5": (3.66, 4.30),
                            "v0.7": (3.70, 4.46)}.items():
            fail_if(check(f"{ver} ear-angle L (deg)", l, data[ver]["sigma_L"]))
            fail_if(check(f"{ver} ear-angle R (deg)", r, data[ver]["sigma_R"]))

    print(f"\n{'=' * 56}")
    if failures == 0:
        print("OVERALL: PASS — every checked paper number matches an artifact.")
    else:
        print(f"OVERALL: {failures} FAILURE(S) — see details above.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
