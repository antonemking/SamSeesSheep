"""Convert AP-10K (COCO-style animal pose) into a YOLO-pose dataset.

AP-10K (arXiv 2108.12617) ships COCO-format keypoint annotations: 10,015
images across 54 species, each with a shared 17-keypoint skeleton. This
script flattens every species into a single `animal` class and emits the
train/val layout Ultralytics expects, so a base `yolo26n-pose.pt` can be
pretrained on it (Stage 1 of the AP-10K transfer experiment).

WHY single-class: Stage 2 fine-tunes on our 5-keypoint `sheep_head` schema,
which discards the 17-keypoint pose head entirely (only the backbone + neck
transfer). So the class taxonomy in Stage 1 is irrelevant to what carries
over — collapsing to one class keeps the converter trivial and robust.

Input layout (the official AP-10K release, untarred):

    <ap10k-root>/
      data/                         # all images, flat
        000000000001.jpg ...
      annotations/
        ap10k-train-split1.json     # COCO keypoints
        ap10k-val-split1.json
        ap10k-test-split1.json

Output layout (--out, default /workspace/ap10k-yolo):

    images/{train,val}/*.jpg        # symlinks into <ap10k-root>/data by default
    labels/{train,val}/*.txt        # YOLO-pose: cls cx cy w h (px py v)*17
    data.yaml                       # kpt_shape [17,3] + flip_idx

Usage (on the pod):

    uv run python sheep-yolo/scripts/prep_ap10k.py \
        --ap10k-root /workspace/ap-10k \
        --out /workspace/ap10k-yolo
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# AP-10K's 17-keypoint order (1-indexed in the paper):
#   1 L_Eye 2 R_Eye 3 Nose 4 Neck 5 Root_of_tail
#   6 L_Shoulder 7 L_Elbow 8 L_F_Paw 9 R_Shoulder 10 R_Elbow 11 R_F_Paw
#   12 L_Hip 13 L_Knee 14 L_B_Paw 15 R_Hip 16 R_Knee 17 R_B_Paw
# flip_idx maps each 0-indexed slot to its left/right mirror, so Ultralytics
# can apply horizontal-flip augmentation without scrambling laterality.
KPT_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "Root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw", "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw", "R_Hip", "R_Knee", "R_B_Paw",
]
FLIP_IDX = [1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13]
N_KPT = 17


def _find_annotation(ann_dir: Path, split: str) -> Path:
    """Locate the COCO json for a split, tolerant of the -split{1,2,3} suffix."""
    candidates = sorted(ann_dir.glob(f"ap10k-{split}-split*.json"))
    if not candidates:
        candidates = sorted(ann_dir.glob(f"*{split}*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No '{split}' annotation json found in {ann_dir}. "
            f"Expected something like ap10k-{split}-split1.json."
        )
    return candidates[0]


def _convert_split(
    coco_json: Path, img_root: Path, out_dir: Path, split: str, symlink: bool,
) -> dict:
    data = json.loads(coco_json.read_text())

    images = {img["id"]: img for img in data["images"]}
    anns_by_img: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    written = skipped_no_img = skipped_no_ann = 0

    for img_id, img in images.items():
        anns = anns_by_img.get(img_id, [])
        if not anns:
            skipped_no_ann += 1
            continue

        fname = Path(img["file_name"]).name
        src = img_root / fname
        if not src.exists():
            skipped_no_img += 1
            continue

        w, h = float(img["width"]), float(img["height"])
        lines = []
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]  # COCO: top-left x,y + w,h, absolute
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nbw, nbh = bw / w, bh / h

            kp = ann.get("keypoints", [])
            if len(kp) != N_KPT * 3:
                # Malformed/absent keypoints — skip this instance, keep others.
                continue
            parts = []
            for i in range(N_KPT):
                px, py, v = kp[3 * i], kp[3 * i + 1], int(kp[3 * i + 2])
                if v == 0:
                    parts += ["0.000000", "0.000000", "0"]
                else:
                    parts += [f"{px / w:.6f}", f"{py / h:.6f}", str(v)]
            lines.append(
                "0 "
                f"{cx:.6f} {cy:.6f} {nbw:.6f} {nbh:.6f} " + " ".join(parts)
            )

        if not lines:
            skipped_no_ann += 1
            continue

        dst_img = img_out / fname
        if not dst_img.exists():
            if symlink:
                dst_img.symlink_to(src.resolve())
            else:
                import shutil
                shutil.copyfile(src, dst_img)
        (lbl_out / f"{Path(fname).stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1

    return {
        "split": split,
        "images_written": written,
        "skipped_missing_image": skipped_no_img,
        "skipped_no_annotation": skipped_no_ann,
    }


def _write_data_yaml(out_dir: Path) -> None:
    flip = ", ".join(str(i) for i in FLIP_IDX)
    yaml_text = (
        "# AP-10K flattened to a single class for backbone pretraining.\n"
        "# Stage 2 (sheep_head, 5 kpts) discards this 17-kpt head; only the\n"
        "# backbone + neck transfer. See docs/AP10K_TRANSFER.md.\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['animal']\n"
        f"kpt_shape: [{N_KPT}, 3]\n"
        f"flip_idx: [{flip}]\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ap10k-root", required=True, type=Path,
                    help="Untarred AP-10K dir containing data/ and annotations/")
    ap.add_argument("--out", type=Path, default=Path("/workspace/ap10k-yolo"))
    ap.add_argument("--copy", action="store_true",
                    help="Copy images instead of symlinking (uses more disk).")
    args = ap.parse_args()

    ann_dir = args.ap10k_root / "annotations"
    img_root = args.ap10k_root / "data"
    if not ann_dir.is_dir() or not img_root.is_dir():
        raise SystemExit(
            f"Expected {ann_dir} and {img_root} to exist. "
            f"Point --ap10k-root at the untarred AP-10K release."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for split in ("train", "val"):
        coco_json = _find_annotation(ann_dir, split)
        print(f"[ap10k] {split}: {coco_json.name}")
        summary.append(
            _convert_split(coco_json, img_root, args.out, split,
                           symlink=not args.copy)
        )
    _write_data_yaml(args.out)

    print("\n[ap10k] Conversion summary:")
    for s in summary:
        print(f"  {s}")
    print(f"\n[ap10k] data.yaml -> {args.out / 'data.yaml'}")
    print(f"[ap10k] kpt_shape=[{N_KPT}, 3]  flip_idx={FLIP_IDX}")


if __name__ == "__main__":
    main()
