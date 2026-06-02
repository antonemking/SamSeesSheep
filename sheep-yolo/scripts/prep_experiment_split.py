"""Build a leakage-safe train/test split for the AP-10K transfer experiment.

The backend export (`/api/export/keypoints/all`) writes frames named
`<video_id>_<frame_idx>.jpg` into train/ and val/ with a *random per-video*
frame split. That's fine for normal training but leaks for an A-vs-B accuracy
comparison: adjacent frames of one clip are near-duplicates, so a random split
puts near-identical frames in both train and test and inflates the score.

This script re-partitions a frozen export into a clean leave-one-clip-out
split: every frame of the held-out `video_id` becomes the test set (with
ground-truth labels), every frame of every other clip becomes train. Both
arms of the experiment then train on the identical train set and are scored
on the identical, never-seen held-out clip.

Usage (on the pod):

    uv run python sheep-yolo/scripts/prep_experiment_split.py \
        --export-dir /workspace/SamSeesSheep/data/labels/exports/sheep-pose-v0.4 \
        --heldout IMG_1234 \
        --out /workspace/ap10k-exp/sheep-heldout
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Mirrors backend/routes/export.py _write_data_yaml — the sheep_head schema.
KPT_SHAPE = "[5, 3]"
FLIP_IDX = "[0, 2, 1, 4, 3]"


def _frame_video_id(stem: str) -> str:
    """`IMG_1234_0007` -> `IMG_1234`. The frame index is the last _NNNN chunk."""
    return stem.rsplit("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-dir", required=True, type=Path,
                    help="A dataset dir produced by the backend export "
                         "(has train/ and val/ with images/ + labels/).")
    ap.add_argument("--heldout", required=True,
                    help="video_id to hold out entirely as the test set.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--symlink", action="store_true",
                    help="Symlink images instead of copying.")
    args = ap.parse_args()

    # Gather every exported frame from BOTH original splits — we re-partition
    # from scratch, so the export's own train/val boundary is discarded.
    src_images: list[Path] = []
    for split in ("train", "val"):
        d = args.export_dir / split / "images"
        if d.is_dir():
            src_images.extend(sorted(d.glob("*.jpg")))
    if not src_images:
        raise SystemExit(f"No images found under {args.export_dir}/{{train,val}}/images")

    video_ids = sorted({_frame_video_id(p.stem) for p in src_images})
    if args.heldout not in video_ids:
        raise SystemExit(
            f"Held-out clip {args.heldout!r} not in export. "
            f"Available video_ids: {video_ids}"
        )

    for split in ("train", "test"):
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "test": 0}
    missing_label = 0
    for img in src_images:
        split = "test" if _frame_video_id(img.stem) == args.heldout else "train"
        lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            missing_label += 1
            continue
        dst_img = args.out / split / "images" / img.name
        dst_lbl = args.out / split / "labels" / lbl.name
        if args.symlink:
            if not dst_img.exists():
                dst_img.symlink_to(img.resolve())
        else:
            shutil.copyfile(img, dst_img)
        shutil.copyfile(lbl, dst_lbl)
        counts[split] += 1

    if counts["test"] == 0:
        raise SystemExit(
            f"Held-out clip {args.heldout!r} produced 0 test frames "
            f"(no labels?). Pick a clip with reviewed keypoints."
        )

    yaml_text = (
        f"# Leave-one-clip-out split for the AP-10K transfer experiment.\n"
        f"# Held out: {args.heldout}\n"
        f"path: .\n"
        f"train: train/images\n"
        f"val: test/images\n"
        f"nc: 1\n"
        f"names: ['sheep_head']\n"
        f"kpt_shape: {KPT_SHAPE}\n"
        f"flip_idx: {FLIP_IDX}\n"
    )
    (args.out / "data.yaml").write_text(yaml_text)

    print(f"[split] held-out clip: {args.heldout}")
    print(f"[split] train frames: {counts['train']}  (clips: "
          f"{[v for v in video_ids if v != args.heldout]})")
    print(f"[split] test  frames: {counts['test']}")
    if missing_label:
        print(f"[split] skipped {missing_label} frames with no label file")
    print(f"[split] data.yaml -> {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
