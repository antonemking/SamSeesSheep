#!/usr/bin/env python3
"""Regenerate the ear-angle chart for a held-out clip from the cached
predictions, with a title that stays within the project's anti-overclaim
contract (VALIDATION.md): it describes measurement stability only, not any
downstream interpretation. The title reads
"Flatter = more stable measurement / noisier = keypoint jitter".

Usage (from sheep-yolo/):
    python scripts/gen_ear_angle_chart.py IMG_3651                      # default v0.2,v0.3,v0.4,v0.7
    python scripts/gen_ear_angle_chart.py IMG_3651 v0.2,v0.3,v0.4,v0.7
    python scripts/gen_ear_angle_chart.py Test_Clip_Morning v0.2,v0.3,v0.4

Note: only IMG_3651 has caches for every version (v0.2-v0.7); Test_Clip_Morning
has kpt caches for v0.2/v0.3/v0.4 only.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]            # sheep-yolo/
CACHE = ROOT / "artifacts" / "_cache"
sys.path.insert(0, str(ROOT / "scripts" / "archive"))
from bench_v02_v03 import CLIPS  # noqa: E402

COLORS = {
    "v0.2": (1.0, 0.647, 0.0),    # orange
    "v0.3": (0.0, 0.784, 0.314),  # green
    "v0.4": (1.0, 0.0, 1.0),      # magenta
    "v0.5": (0.12, 0.47, 0.85),   # blue
    "v0.6": (0.85, 0.10, 0.10),   # red
    "v0.7": (0.0, 0.60, 0.55),    # teal
}


def _safe_angle(midline, ear):
    cross = midline[0] * ear[1] - midline[1] * ear[0]
    dot = midline[0] * ear[0] + midline[1] * ear[1]
    return float(np.degrees(np.arctan2(cross, dot)))


def ear_angles(kpt):
    nose, lb, rb, lt, rt = kpt
    if nose[2] <= 0 or lb[2] <= 0 or rb[2] <= 0:
        return np.nan, np.nan
    mid = (lb[:2] + rb[:2]) / 2
    midline = nose[:2] - mid
    la = _safe_angle(midline, lt[:2] - lb[:2]) if lt[2] > 0 else np.nan
    ra = _safe_angle(midline, rt[:2] - rb[:2]) if rt[2] > 0 else np.nan
    return la, ra


def residual_sigma(x, w=7):
    ok = ~np.isnan(x)
    if ok.sum() < 8:
        return None
    v = x[ok]
    half = w // 2
    rmed = np.array([np.median(v[max(0, i - half): min(len(v), i + half + 1)]) for i in range(len(v))])
    return float(np.std(v - rmed))


def main(clip_key: str, versions: list[str]) -> int:
    a, b = CLIPS[clip_key]["window"]
    n = b - a
    fps = 30.0
    t = np.arange(n) / fps

    series = {}
    for ver in versions:
        f = CACHE / f"sheep-pose-{ver}-yolo26n__{clip_key}.pkl"
        if not f.exists():
            print(f"  skip {ver}: no cache for {clip_key} ({f.name})")
            continue
        kp = pickle.load(open(f, "rb"))["target"][a:b]
        L = np.array([ear_angles(kp[i])[0] for i in range(n)])
        R = np.array([ear_angles(kp[i])[1] for i in range(n)])
        series[ver] = (L, R)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, idx, name in zip(axes, (0, 1), ("Left", "Right")):
        ax.axhspan(-100, -10, color="red", alpha=0.05, zorder=0)
        ax.axhspan(-10, 30, color="orange", alpha=0.05, zorder=0)
        ax.axhspan(30, 200, color="green", alpha=0.05, zorder=0)
        for ver, (L, R) in series.items():
            y = L if idx == 0 else R
            sig = residual_sigma(y)
            label = f"{ver}  σ={sig:.1f}°" if sig is not None else ver
            ax.plot(t, y, color=COLORS[ver], linewidth=1.6, alpha=0.9, label=label)
        ax.set_ylabel(f"{name} ear angle (°)")
        ax.set_ylim(-130, 220)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlim(0, float(t[-1]))
    axes[-1].set_xlabel(f"Time in motionless window (s)  ·  clip: {clip_key}")
    fig.suptitle(
        f"Ear angle on a stationary held-out sheep — same clip, {len(series)} model versions\n"
        "Flatter = more stable measurement · noisier = keypoint jitter",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    tag = "".join(v.replace("v0.", "") for v in series)
    out = ROOT / "artifacts" / f"ear_angle_lines-{clip_key}-v{tag}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    clip = sys.argv[1] if len(sys.argv) > 1 else "IMG_3651"
    vers = (sys.argv[2] if len(sys.argv) > 2 else "v0.2,v0.3,v0.4,v0.7").split(",")
    sys.exit(main(clip, vers))
