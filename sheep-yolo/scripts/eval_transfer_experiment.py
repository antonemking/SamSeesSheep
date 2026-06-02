"""Score the two arms of the AP-10K transfer experiment on the held-out clip.

Runs Ultralytics `val` for the baseline and the AP-10K-pretrained model
against the same held-out test split (defined by prep_experiment_split.py)
and writes a side-by-side comparison.

  Arm A (baseline):   yolo26n-pose.pt          -> fine-tune on sheep train
  Arm B (experiment): yolo26n-pose -> AP-10K   -> fine-tune on sheep train

Both arms output the 5-keypoint sheep_head schema, so they're scored with
identical metrics: pose mAP@50, pose mAP@50-95 (OKS-based), box mAP, plus
precision/recall.

Usage (on the pod):

    uv run python sheep-yolo/scripts/eval_transfer_experiment.py \
        --data /workspace/ap10k-exp/sheep-heldout/data.yaml \
        --baseline   /workspace/runs/ap10k-transfer/baseline/weights/best.pt \
        --experiment /workspace/runs/ap10k-transfer/ap10k-ft/weights/best.pt \
        --out /workspace/runs/ap10k-transfer/comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _eval_one(weights: Path, data: Path, imgsz: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # split='val' uses the data.yaml `val:` entry, which prep_experiment_split
    # points at the held-out test images.
    res = model.val(data=str(data), split="val", imgsz=imgsz, plots=False,
                    verbose=False)
    return {
        "weights": str(weights),
        "pose_map50_95": round(float(res.pose.map), 5),
        "pose_map50": round(float(res.pose.map50), 5),
        "box_map50_95": round(float(res.box.map), 5),
        "box_map50": round(float(res.box.map50), 5),
        "pose_precision": round(float(res.pose.mp), 5),
        "pose_recall": round(float(res.pose.mr), 5),
    }


def _pct(a: float, b: float) -> str:
    if b == 0:
        return "n/a"
    return f"{(a - b) / b * 100:+.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--experiment", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path("/workspace/runs/ap10k-transfer/comparison.json"))
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    print("[eval] Arm A — baseline (COCO-pose backbone -> sheep)")
    baseline = _eval_one(args.baseline, args.data, args.imgsz)
    print("[eval] Arm B — experiment (COCO-pose -> AP-10K -> sheep)")
    experiment = _eval_one(args.experiment, args.data, args.imgsz)

    report = {
        "data": str(args.data),
        "baseline": baseline,
        "experiment": experiment,
        "delta_vs_baseline": {
            k: _pct(experiment[k], baseline[k])
            for k in ("pose_map50_95", "pose_map50", "box_map50_95",
                      "pose_precision", "pose_recall")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Console table.
    metrics = [
        ("pose mAP@50-95", "pose_map50_95"),
        ("pose mAP@50", "pose_map50"),
        ("box  mAP@50-95", "box_map50_95"),
        ("pose precision", "pose_precision"),
        ("pose recall", "pose_recall"),
    ]
    print("\n" + "=" * 64)
    print(f"{'metric':<18}{'baseline (A)':>14}{'AP-10K (B)':>14}{'Δ B vs A':>14}")
    print("-" * 64)
    for label, key in metrics:
        a, b = baseline[key], experiment[key]
        print(f"{label:<18}{a:>14.4f}{b:>14.4f}{_pct(b, a):>14}")
    print("=" * 64)
    print(f"[eval] report -> {args.out}")
    print("[eval] Higher is better on every row. Positive Δ => AP-10K "
          "pretraining helped on the held-out clip.")


if __name__ == "__main__":
    main()
