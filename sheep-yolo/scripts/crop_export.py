"""Build a TOP-DOWN keypoint dataset from a single-shot YOLO-pose export.

Top-down pose = detect the head, crop to it, then run a keypoint net on the
*zoomed* crop. This script makes the training data for that second stage: it
takes our existing full-frame export (one .txt per image, each line a
head bbox + 5 keypoints, all normalized to the FRAME) and emits one crop
image per instance, with the bbox + keypoints renormalized to the CROP.

Why this is the whole experiment: in single-shot, a ~234 px head in a 1080p
frame is ~78 px once the frame is letterboxed to 640 for inference. A crop of
~300 px fed at 256 keeps the head ~150 px — roughly 2x the pixels on the ears
and nose. This script is where that zoom gets baked into the labels; the
hypothesis is that it tightens ear-angle σ. If it doesn't, we've spent an
afternoon and learned to drop it.

The crop convention here (MARGIN, square=False) MUST match bench_topdown.py's
inference crop, or train/serve drift will swamp any real signal.

Source default is the v0.4 export in the laptop backup mirror; output lands in
the repo's exports dir, train/val split preserved 1:1 (a held-out val frame
stays held-out). Empty label files (hard negatives) produce no crops — the
crop stage only ever sees positive instances.

  python crop_export.py                  # v0.4 export -> sheep-pose-v0.4-crops
  python crop_export.py --src <dir> --out <dir> --margin 0.30
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import cv2

# Crop padding as a fraction of the head box on each side. Kept in lockstep
# with bench_topdown.py — change one, change both.
MARGIN = 0.30
KPT_SHAPE = (5, 3)

ROOT = Path(__file__).resolve().parents[2]          # repo root (sheep-seg)
DEFAULT_SRC = Path.home() / "Backups/sheep-seg/labels/exports/sheep-pose-v0.4-yolo26n"
DEFAULT_OUT = ROOT / "data/labels/exports/sheep-pose-v0.4-crops"

# Absolute `path:` so `yolo train` resolves train/val regardless of cwd.
# (The pod export uses `path: .` and relies on cd-ing into the dataset dir;
# for local runs an absolute path is less fragile.)
DATA_YAML = (
    "path: {root}\n"
    "train: train/images\n"
    "val: val/images\n"
    "nc: 1\n"
    "names: ['sheep_head']\n"
    "kpt_shape: [5, 3]\n"
    "flip_idx: [0, 2, 1, 4, 3]\n"
)


def parse_line(line: str):
    """One YOLO-pose label line -> (cls, bbox_norm, kpts_norm).

    bbox_norm = (cx, cy, w, h); kpts_norm = [(px, py, v) x5], all in [0,1]
    relative to the FRAME.
    """
    p = line.split()
    cls = int(float(p[0]))
    cx, cy, w, h = (float(v) for v in p[1:5])
    rest = [float(v) for v in p[5:]]
    kpts = [(rest[i], rest[i + 1], int(rest[i + 2])) for i in range(0, len(rest), 3)]
    return cls, (cx, cy, w, h), kpts


def crop_instance(img, bbox_norm, kpts_norm, margin=MARGIN):
    """Crop one head + renormalize its label to the crop.

    Returns (crop_img, new_line_str) or None if the crop is degenerate.
    """
    H, W = img.shape[:2]
    cx, cy, w, h = bbox_norm
    bw, bh = w * W, h * H
    x1, y1 = (cx * W) - bw / 2, (cy * H) - bh / 2
    x2, y2 = x1 + bw, y1 + bh

    # Expand by margin, clamp to image, snap to int pixel grid.
    ex1 = int(max(0, round(x1 - margin * bw)))
    ey1 = int(max(0, round(y1 - margin * bh)))
    ex2 = int(min(W, round(x2 + margin * bw)))
    ey2 = int(min(H, round(y2 + margin * bh)))
    cw, ch = ex2 - ex1, ey2 - ey1
    if cw < 8 or ch < 8:
        return None

    crop = img[ey1:ey2, ex1:ex2]

    # Head box within the crop, renormalized to crop dims.
    nx1, ny1 = (x1 - ex1) / cw, (y1 - ey1) / ch
    nx2, ny2 = (x2 - ex1) / cw, (y2 - ey1) / ch
    ncx = min(1.0, max(0.0, (nx1 + nx2) / 2))
    ncy = min(1.0, max(0.0, (ny1 + ny2) / 2))
    nw = min(1.0, nx2 - nx1)
    nh = min(1.0, ny2 - ny1)

    parts = ["0", f"{ncx:.6f}", f"{ncy:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
    for px, py, v in kpts_norm:
        if v <= 0:                       # absent in this frame: keep the 0,0,0 slot
            parts += ["0.000000", "0.000000", "0"]
            continue
        kpx = (px * W - ex1) / cw
        kpy = (py * H - ey1) / ch
        # A visible kpt should sit inside the padded crop; if numerics push
        # it just outside, clamp rather than drop (matches Ultralytics' oob
        # handling at train time).
        kpx = min(1.0, max(0.0, kpx))
        kpy = min(1.0, max(0.0, kpy))
        parts += [f"{kpx:.6f}", f"{kpy:.6f}", str(v)]
    return crop, " ".join(parts)


def process_split(src_split: Path, out_split: Path, margin: float):
    img_dir = src_split / "images"
    lbl_dir = src_split / "labels"
    out_img = out_split / "images"
    out_lbl = out_split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_imgs = n_inst = n_skipped = 0
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        lines = [ln for ln in lbl_path.read_text().splitlines() if ln.strip()]
        if not lines:
            continue                      # hard-negative frame: no crops
        img_path = next((img_dir / f"{lbl_path.stem}{e}"
                         for e in (".jpg", ".jpeg", ".png")
                         if (img_dir / f"{lbl_path.stem}{e}").exists()), None)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        n_imgs += 1
        for i, line in enumerate(lines):
            _, bbox, kpts = parse_line(line)
            res = crop_instance(img, bbox, kpts, margin)
            if res is None:
                n_skipped += 1
                continue
            crop, new_line = res
            stem = f"{lbl_path.stem}_i{i}"
            cv2.imwrite(str(out_img / f"{stem}.jpg"), crop)
            (out_lbl / f"{stem}.txt").write_text(new_line + "\n")
            n_inst += 1
    return n_imgs, n_inst, n_skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="source single-shot YOLO-pose export (has train/ val/)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output crops dataset dir")
    ap.add_argument("--margin", type=float, default=MARGIN,
                    help="crop padding as fraction of head box (match bench)")
    args = ap.parse_args()

    if not (args.src / "train" / "labels").exists():
        raise SystemExit(f"no train/labels under {args.src}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data.yaml").write_text(DATA_YAML.format(root=args.out.resolve()))

    print(f"src:    {args.src}")
    print(f"out:    {args.out}")
    print(f"margin: {args.margin}\n")
    total_inst = 0
    for split in ("train", "val"):
        src_split = args.src / split
        if not (src_split / "labels").exists():
            print(f"[{split}] missing, skipped")
            continue
        n_imgs, n_inst, n_skip = process_split(src_split, args.out / split, args.margin)
        total_inst += n_inst
        print(f"[{split}] {n_imgs} frames -> {n_inst} crops "
              f"({n_skip} degenerate skipped)")
    print(f"\nTotal: {total_inst} crop instances")
    print(f"Train with:  bash {Path(__file__).parent / 'train_topdown.sh'}")


if __name__ == "__main__":
    main()
