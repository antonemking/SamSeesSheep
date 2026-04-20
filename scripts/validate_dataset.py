#!/usr/bin/env python3
"""Validate a YOLO-pose dataset exported by sheep-seg.

Runs structural + format + semantic checks before training so a bad dataset
doesn't silently burn GPU time. Called as a pre-flight inside
scripts/train_on_pod.sh; safe to run standalone for dataset inspection.

Usage:
    uv run python scripts/validate_dataset.py --dataset sheep-pose-v0.1
    uv run python scripts/validate_dataset.py --path /abs/path/to/dataset

Exit codes:
    0 — dataset is training-ready (warnings may still print)
    1 — critical errors; training should not proceed
    2 — bad invocation (missing directory, etc.)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml  # ultralytics pulls this in

EXPECTED_COLS = 20  # 1 class + 4 bbox + 5 keypoints × 3 fields
EXPECTED_KPT_SHAPE = [5, 3]
EXPECTED_FLIP_IDX = [0, 2, 1, 4, 3]
KP_NAMES = ["nose", "L-base", "R-base", "L-tip", "R-tip"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORTS_DIR = REPO_ROOT / "data" / "labels" / "exports"


class Report:
    """Collects errors / warnings / info; prints them with color on stdout."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.info.append(msg)

    def passed(self) -> bool:
        return not self.errors

    def render(self) -> None:
        for line in self.info:
            print(line)
        for w in self.warnings:
            print(f"\033[33m[warn]\033[0m  {w}")
        for e in self.errors:
            print(f"\033[31m[error]\033[0m {e}")
        print()
        if self.errors:
            print(
                f"\033[31mFAIL\033[0m — "
                f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
            )
        else:
            status = "PASS with warnings" if self.warnings else "PASS"
            print(f"\033[32m{status}\033[0m — ready to train")


def _validate_label_line(
    line: str, report: Report, fname: str
) -> list[int] | None:
    """Parse + validate one YOLO-pose label line. Returns placed keypoint indices."""
    parts = line.strip().split()
    if len(parts) != EXPECTED_COLS:
        report.error(
            f"{fname}: expected {EXPECTED_COLS} columns, got {len(parts)}"
        )
        return None

    try:
        values = [float(p) for p in parts]
    except ValueError:
        report.error(f"{fname}: non-numeric token in label line")
        return None

    cls = int(values[0])
    if cls != 0:
        report.error(
            f"{fname}: class={cls}, expected 0 (single-class sheep_head)"
        )

    # Bbox center-xywh normalized to [0, 1]
    for i, name in enumerate(("bx", "by", "bw", "bh"), start=1):
        v = values[i]
        if not 0 <= v <= 1:
            report.error(f"{fname}: {name}={v:.4f} out of [0, 1]")

    placed: list[int] = []
    for k in range(5):
        off = 5 + k * 3
        kx, ky, kv_float = values[off], values[off + 1], values[off + 2]
        kv = int(kv_float)
        if kv not in (0, 1, 2):
            report.error(
                f"{fname}: kp {k} ({KP_NAMES[k]}) v={kv}, expected 0/1/2"
            )
            continue
        if kv == 2:
            if not 0 <= kx <= 1 or not 0 <= ky <= 1:
                report.error(
                    f"{fname}: kp {k} ({KP_NAMES[k]}) placed but coords "
                    f"({kx:.4f}, {ky:.4f}) out of [0, 1]"
                )
            else:
                placed.append(k)
        elif kv == 0 and (kx != 0 or ky != 0):
            # Harmless but suggests the exporter drifted from spec — v=0 should
            # write the "0 0 0" triplet unconditionally.
            report.warn(
                f"{fname}: kp {k} ({KP_NAMES[k]}) has v=0 but nonzero "
                f"coords ({kx:.4f}, {ky:.4f})"
            )
    return placed


def _validate_yaml(dataset_dir: Path, report: Report) -> dict | None:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        report.error(f"missing {yaml_path}")
        return None
    try:
        y = yaml.safe_load(yaml_path.read_text())
    except Exception as e:
        report.error(f"could not parse {yaml_path}: {e}")
        return None

    for field in ("path", "train", "val", "nc", "names", "kpt_shape", "flip_idx"):
        if field not in y:
            report.error(f"data.yaml missing field: {field}")
    if y.get("nc") != 1:
        report.error(f"data.yaml nc={y.get('nc')}, expected 1")
    if y.get("kpt_shape") != EXPECTED_KPT_SHAPE:
        report.error(
            f"data.yaml kpt_shape={y.get('kpt_shape')}, "
            f"expected {EXPECTED_KPT_SHAPE}"
        )
    if y.get("flip_idx") != EXPECTED_FLIP_IDX:
        report.error(
            f"data.yaml flip_idx={y.get('flip_idx')}, "
            f"expected {EXPECTED_FLIP_IDX} (image-space L/R pairing)"
        )
    return y


def validate_dataset(dataset_dir: Path, report: Report) -> None:
    report.note(f"dataset: {dataset_dir}")

    _validate_yaml(dataset_dir, report)

    split_counts: dict[str, tuple[int, int]] = {}
    stem_locations: dict[str, list[str]] = defaultdict(list)
    kp_coverage: Counter[int] = Counter()
    total_frames = 0

    for split in ("train", "val"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            report.error(f"missing {img_dir}")
            continue
        if not lbl_dir.exists():
            report.error(f"missing {lbl_dir}")
            continue

        images = sorted(img_dir.glob("*.jpg"))
        labels = sorted(lbl_dir.glob("*.txt"))
        split_counts[split] = (len(images), len(labels))

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        for s in img_stems - lbl_stems:
            report.error(f"{split}: image {s}.jpg has no matching label")
        for s in lbl_stems - img_stems:
            report.error(f"{split}: label {s}.txt has no matching image")
        for s in img_stems:
            stem_locations[s].append(split)

        for lbl_path in labels:
            lines = [
                l for l in lbl_path.read_text().splitlines() if l.strip()
            ]
            if not lines:
                report.error(f"{split}/{lbl_path.name}: empty label file")
                continue
            if len(lines) > 1:
                report.warn(
                    f"{split}/{lbl_path.name}: {len(lines)} instances "
                    f"(expected 1 per single-subject convention)"
                )
            for line in lines:
                placed = _validate_label_line(
                    line, report, f"{split}/{lbl_path.name}"
                )
                if placed is not None:
                    total_frames += 1
                    for k in placed:
                        kp_coverage[k] += 1

    for split, (imgs, lbls) in split_counts.items():
        marker = "✓ paired" if imgs == lbls else "✗ mismatch"
        report.note(f"  {split}: {imgs} images, {lbls} labels  {marker}")

    if total_frames == 0:
        report.error("dataset contains zero valid training frames")
        return

    report.note("  v=2 coverage per keypoint slot:")
    for k, name in enumerate(KP_NAMES):
        count = kp_coverage[k]
        pct = int(100 * count / total_frames)
        marker = ""
        if pct < 50:
            marker = " ← low"
            report.warn(
                f"{name} coverage only {pct}% of frames — reviewer may be "
                f"under-placing this keypoint"
            )
        report.note(f"    {name:<8s} {count:>4d}/{total_frames}  ({pct}%){marker}")

    # Leakage: a frame stem appearing in both train and val is split drift.
    dupes = {s: sp for s, sp in stem_locations.items() if len(sp) > 1}
    if dupes:
        report.error(
            f"{len(dupes)} frame(s) appear in both train and val "
            f"(leakage)"
        )
        for s, sp in list(dupes.items())[:5]:
            report.error(f"  {s}: in {', '.join(sp)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dataset",
        help="Dataset name under data/labels/exports/ (e.g. sheep-pose-v0.1)",
    )
    g.add_argument(
        "--path",
        help="Absolute path to the dataset directory (overrides --dataset)",
    )
    args = parser.parse_args()

    if args.dataset:
        dataset_dir = DEFAULT_EXPORTS_DIR / args.dataset
    else:
        dataset_dir = Path(args.path).resolve()

    if not dataset_dir.exists():
        print(f"No such directory: {dataset_dir}", file=sys.stderr)
        return 2

    report = Report()
    validate_dataset(dataset_dir, report)
    report.render()
    return 0 if report.passed() else 1


if __name__ == "__main__":
    sys.exit(main())
