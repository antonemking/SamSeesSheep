#!/usr/bin/env python3
"""Scatter of per-window ear-angle residual sigma vs within-window head motion,
for every fixed-length window slid across the target's detected span. The
hand-picked paper window is highlighted. Reads artifacts/window_sensitivity.json
(produced by window_sensitivity.py).

Shows that the reported noise floor is not an artifact of selecting the stillest
5 s: across all windows sigma_avg stays a small fraction of the SPFES 40 deg
band, and rises only modestly with head motion.

Run from sheep-yolo/:
    python scripts/gen_window_sensitivity_fig.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "window_sensitivity.json"
OUT = ROOT / "artifacts" / "window_sensitivity.png"

PANELS = [("HO-1_v0.4", "HO-1 (IMG_3651) · v0.4"),
          ("HO-1_v0.7", "HO-1 (IMG_3651) · v0.7"),
          ("HO-2_v0.4", "HO-2 (Test_Clip_Morning) · v0.4")]


def main() -> int:
    doc = json.loads(DATA.read_text())
    runs = {r["paper_id"] + "_" + r["model"]: r for r in doc["results"]}
    wins = doc["windows_by_run"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for ax, (key, title) in zip(axes, PANELS):
        r = runs[key]
        w = wins[key]
        mot = [x["motion_px"] for x in w]
        sig = [x["sigma_avg"] for x in w]
        ax.scatter(mot, sig, s=14, alpha=0.45, color="#4477aa",
                   edgecolors="none", label=f"all windows (n={len(w)})")
        ax.scatter([r["paper_window_motion_px"]], [r["paper_window_sigma_avg"]],
                   s=120, marker="*", color="#cc3311", zorder=5,
                   label=f"paper window ({r['paper_window_sigma_avg']:.2f}°, "
                         f"{r['paper_window_percentile']:.0f}th pct)")
        ax.axhline(r["sigma_avg_median"], color="#666", ls="--", lw=1,
                   label=f"median {r['sigma_avg_median']:.2f}°")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("within-window head motion (raw σ, px)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)
    axes[0].set_ylabel("ear-angle residual σ_avg (°)")
    fig.suptitle("Window-selection sensitivity: ear-angle noise floor vs window head motion "
                 "(each point = one 5 s window)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
