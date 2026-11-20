"""Export reviewed sheep-face-36 labels to an Ultralytics YOLO-pose dataset of head crops, split by CLIP.

Replaces the local exporter's `--val 0.2` (a random split by frame, which breaks CLAUDE.md decision 6).

  python scripts/v1/export_yolo.py --batch trainset/day_3656 --batch trainset/night --batch trainset/day_3678 \
      --val-clip IMG_3678 --mask-clip Sheep_at_night --mask-slots 27-34 \
      --forbid-ids judge/manifest.json --forbid-ids judge/gt_blind20.json --out train/data/v1.0-b

A batch folder holds labels.json (heads: id, clip, head, box, kpts[36][x, y, v], outline), frames/<id>.png and
manifest.json (source clip path + sha256), which is what make_tasks.py writes.

Visibility in the label rows: v=2 visible, v=1 occluded (estimated), v=0 absent.

MASKED slots (--mask-clip + --mask-slots): the labeler skipped these slots on those clips on purpose.
--mask-if-unmarked extends that to any head, of any clip, where none of the slots is marked. The YOLO
row format has no fourth state, so the row holds the placeholder `0 0 0` there (no position loss in Ultralytics),
and the truth is written beside it in kpt_mask.json: image stem -> masked slots. A stock Ultralytics run would
still read the placeholder as "absent" in its visibility loss, so train with scripts/v1/masked_kobj.py, which drops
masked slots from that loss. data.yaml names the file under `kpt_mask`, and the split manifest counts masked and
absent apart.

There is no --allow-judge. Judge clips, forbidden frame ids and forbidden content hashes stop the export before
anything is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_split as cs  # noqa: E402

N = 36
CROP_MARGIN, CROP = 0.3, 320
# sheep-face-36 horizontal-flip pairs (image-side L/R). Must equal schema.FLIP_PAIRS; the caller can check with
# --flip-idx-json, and tests pin the list.
FLIP_PAIRS = [(1, 2), (3, 4), (5, 7), (6, 8), (9, 13), (10, 14), (11, 15), (12, 16),
              (17, 19), (18, 20), (22, 23), (27, 28), (29, 30), (31, 32), (33, 34)]
FLIP_IDX = list(range(N))
for _a, _b in FLIP_PAIRS:
    FLIP_IDX[_a], FLIP_IDX[_b] = _b, _a


def crop_window(box, margin=CROP_MARGIN):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]      # square, may exceed the frame (padded)


def label_row(head, masked, np):
    """One head -> (row values, v per slot after cropping, window). Pure geometry; no image needed."""
    win = crop_window(head["box"])
    s = CROP / (win[2] - win[0])
    M = np.float32([[s, 0, -win[0] * s], [0, s, -win[1] * s]])
    kp = np.array(head["kpts"], float)
    if kp.shape != (N, 3):
        raise cs.SplitError(f"head {head.get('id')} has kpts of shape {kp.shape}, expected ({N}, 3)")
    xy = (M @ np.concatenate([kp[:, :2], np.ones((N, 1))], 1).T).T / CROP
    v = kp[:, 2].astype(int)
    v[(xy < 0).any(1) | (xy > 1).any(1)] = 0
    v[list(masked)] = 0                       # placeholder only; kpt_mask.json says these are masked, not absent
    xy[v == 0] = 0
    pts = np.array(head["outline"]) if head.get("outline") else kp[v > 0, :2]
    if len(pts) == 0:
        return None, v, M
    bb = np.clip((M @ np.concatenate([pts, np.ones((len(pts), 1))], 1).T).T / CROP, 0, 1)
    (x0, y0), (x1, y1) = bb.min(0), bb.max(0)
    row = [0, float((x0 + x1) / 2), float((y0 + y1) / 2), float(x1 - x0), float(y1 - y0)]
    for i in range(N):
        row += [float(xy[i, 0]), float(xy[i, 1]), int(v[i])]
    return row, v, M


def load_batches(batch_dirs):
    heads, sources = [], []
    for b in batch_dirs:
        b = Path(b)
        rows = json.loads((b / "labels.json").read_text())
        for h in rows:
            h["_frames"] = str(b / "frames")
            h["_batch"] = b.name
        heads += rows
        mf = b / "manifest.json"
        m = json.loads(mf.read_text()) if mf.exists() else {}
        sources.append({"batch": b.name, "source": m.get("source", ""), "sha256": m.get("sha256", ""),
                        "labels_sha256": cs.sha256_bytes((b / "labels.json").read_bytes()), "heads": len(rows)})
    return heads, sources


def export(heads, sources, out, *, val_clips, mask_clips=(), mask_slots=(), forbidden_ids=(), forbidden_sha=(),
           mask_if_unmarked=False, flip_idx=FLIP_IDX, not_judgeable=None):
    """mask_if_unmarked: also mask mask_slots on any head, of any clip, where NONE of those slots is marked (the
    labeler skipped the whole group on that head). A head with some of them marked keeps marks and absents."""
    import cv2
    import numpy as np

    mask_slots = sorted(mask_slots)
    guard = cs.guard_training_set(heads, sources, forbidden_ids, forbidden_sha)      # before anything is written
    assign = cs.assign_by_clip(heads, val_clips)
    if mask_slots:
        cs.check_flip_closed(mask_slots, flip_idx)
    unknown = set(mask_clips) - set(assign)
    if unknown:
        raise cs.SplitError(f"--mask-clip names clips with no heads: {sorted(unknown)}")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise cs.SplitError(f"{out} is not empty; export into a fresh folder so no stale crop survives")
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    written, state_rows, kpt_mask, missing, empty = [], [], {}, [], []
    marks_dropped = Counter()
    for h in heads:
        clip = str(h["clip"])
        masked = cs.masked_slots_for(clip, mask_clips, mask_slots)
        if not masked and mask_if_unmarked and mask_slots and all(h["kpts"][i][2] == 0 for i in mask_slots):
            masked = list(mask_slots)
        marks_dropped[clip] += sum(1 for i in masked if h["kpts"][i][2] > 0)
        row, v, M = label_row(h, masked, np)
        name = f"{h['id']}_h{h['head']}"
        if row is None:
            empty.append(name)
            continue
        fr = cv2.imread(os.path.join(h["_frames"], h["id"] + ".png"))
        if fr is None:
            missing.append(name)
            continue
        img = cv2.warpAffine(fr, M, (CROP, CROP), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(114, 114, 114))
        sp = assign[clip]
        cv2.imwrite(str(out / "images" / sp / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / sp / f"{name}.txt").write_text(
            " ".join(f"{t:.6f}" if isinstance(t, float) else str(t) for t in row) + "\n")
        if masked:
            kpt_mask[name] = masked
        written.append((clip, sp, name))
        state_rows.append((clip, v.tolist(), masked))
    if missing:
        raise cs.SplitError(f"{len(missing)} frames missing (e.g. {missing[0]}); the export is incomplete")
    cs.check_no_straddle((c, sp) for c, sp, _ in written)

    (out / "kpt_mask.json").write_bytes(cs.canonical({
        "meaning": "image stem -> slots that are MASKED: not marked on purpose. No position loss, no absent signal. "
                   "The label rows hold the placeholder 0 0 0 there; train with scripts/v1/masked_kobj.py.",
        "slots": mask_slots, "clips": sorted(mask_clips), "images": kpt_mask}))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nkpt_shape: [{N}, 3]\nflip_idx: {list(flip_idx)}\n"
        f"names:\n  0: sheep_face\n"
        f"# Masked slots (not absent) are listed per image in this file. A stock run reads them as absent:\n"
        f"# train through scripts/v1/masked_kobj.py.\nkpt_mask: kpt_mask.json\n")

    per_clip_heads = Counter(c for c, _, _ in written)
    per_clip_masked = Counter(c for c, _, n in written if n in kpt_mask)
    states = cs.count_states(state_rows)
    total = Counter()
    for c in states.values():
        total.update(c)
    manifest = {
        "dataset": out.name, "split_unit": "clip", "per_frame_split": False,
        "splits": {sp: sorted(c for c, s in assign.items() if s == sp) for sp in ("train", "val")},
        "heads": {sp: sum(1 for _, s, _ in written if s == sp) for sp in ("train", "val")},
        "clips": {c: {"split": assign[c], "heads_marked": per_clip_heads[c],
                      "not_judgeable": (not_judgeable or {}).get(c),
                      "points": states.get(c, {}), "heads_masked": per_clip_masked[c],
                      "masked_slots": mask_slots if per_clip_masked[c] else [],
                      "marks_dropped_by_mask": marks_dropped[c]} for c in sorted(assign)},
        "points_total": {s: total.get(s, 0) for s in cs.STATES},
        "masking": {"slots": mask_slots, "clips": sorted(mask_clips), "images": len(kpt_mask),
                    "also_heads_with_no_mark_in_slots": bool(mask_if_unmarked),
                    "file": "kpt_mask.json", "row_placeholder": "0 0 0",
                    "note": "masked is not absent; honoured in training by scripts/v1/masked_kobj.py"},
        "evaluation_footage_guard": guard,
        "assertions": {"no_clip_in_both_splits": True, "no_judge_clip": True, "no_forbidden_frame_id": True,
                       "no_forbidden_source_hash": True, "no_test_only_source": True, "allow_judge_passed": False},
        "sources": sources, "crop": {"px": CROP, "margin": CROP_MARGIN}, "heads_without_points": empty,
    }
    (out / "split_manifest.json").write_bytes(cs.canonical(manifest))
    sha = cs.sha256_bytes((out / "split_manifest.json").read_bytes())
    (out / "split_manifest.json.sha256").write_text(f"{sha}  split_manifest.json\n")
    return manifest, sha


def _ids(paths):
    out = set()
    for p in paths:
        for row in json.loads(Path(p).read_text()):
            out.add(row["id"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", action="append", required=True, help="folder with labels.json, frames/, manifest.json")
    ap.add_argument("--val-clip", action="append", required=True, help="held-out clip (whole clip goes to val)")
    ap.add_argument("--mask-clip", action="append", default=[], help="clip whose --mask-slots are masked")
    ap.add_argument("--mask-slots", default="", help="e.g. 27-34")
    ap.add_argument("--mask-if-unmarked", action="store_true",
                    help="also mask --mask-slots on any head where none of those slots is marked")
    ap.add_argument("--forbid-ids", action="append", default=[], help="json list of rows with 'id' (judge manifest)")
    ap.add_argument("--forbid-sha", action="append", default=[], help="sha256 of footage that never trains")
    ap.add_argument("--flip-idx-json", help="json list; must equal the built-in sheep-face-36 flip_idx")
    ap.add_argument("--not-judgeable", help="json {clip: n} of heads ticked not judgeable, for the manifest")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.flip_idx_json and json.loads(Path(a.flip_idx_json).read_text()) != FLIP_IDX:
        raise cs.SplitError("flip_idx differs from the schema's; fix FLIP_PAIRS before exporting")
    if (a.mask_clip or a.mask_if_unmarked) and not a.mask_slots:
        raise cs.SplitError("--mask-clip / --mask-if-unmarked need --mask-slots")
    if a.mask_slots and not (a.mask_clip or a.mask_if_unmarked):
        raise cs.SplitError("--mask-slots needs --mask-clip or --mask-if-unmarked")
    heads, sources = load_batches(a.batch)
    nj = json.loads(Path(a.not_judgeable).read_text()) if a.not_judgeable else None
    m, sha = export(heads, sources, a.out, val_clips=a.val_clip, mask_clips=a.mask_clip,
                    mask_slots=cs.parse_slots(a.mask_slots), mask_if_unmarked=a.mask_if_unmarked,
                    forbidden_ids=_ids(a.forbid_ids),
                    forbidden_sha=a.forbid_sha, not_judgeable=nj)
    for c, d in m["clips"].items():
        print(f"{c:>16}  {d['split']:5}  heads {d['heads_marked']:3}  masked heads {d['heads_masked']:3}  {d['points']}")
    print(f"train {m['heads']['train']}  val {m['heads']['val']}  points {m['points_total']}")
    print(f"split manifest: {Path(a.out) / 'split_manifest.json'}  sha256 {sha}")


if __name__ == "__main__":
    main()
