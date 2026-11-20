"""End-to-end export on synthetic grey frames (no footage). Skips without OpenCV."""
import json

import pytest

import clip_split as cs
import export_yolo as ex

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

NIGHT, DAY, VAL = "Sheep_at_night", "IMG_3656", "IMG_3678"
EDGE = list(range(27, 35))


def make_batch(root, clip, n_frames=3, edge_placed=True):
    b = root / clip
    (b / "frames").mkdir(parents=True)
    rows = []
    for f in range(n_frames):
        fid = f"{clip}_{f:05d}"
        cv2.imwrite(str(b / "frames" / f"{fid}.png"), np.full((720, 1280, 3), 90, np.uint8))
        k = [[420 + 5 * i, 300 + 4 * i, 2] for i in range(36)]
        k[10][2] = 1                                   # one occluded point
        k[24] = [0, 0, 0]                              # one point that really is absent (deleted)
        if not edge_placed:
            for i in EDGE:
                k[i] = [0, 0, 0]                       # the labeler did not place the silhouette edge
        rows.append({"id": fid, "clip": clip, "frame": f, "head": 0, "box": [400, 280, 640, 520], "kpts": k,
                     "outline": None})
    (b / "labels.json").write_text(json.dumps(rows))
    (b / "manifest.json").write_text(json.dumps({"source": f"/clips/{clip}.MOV", "sha256": clip.lower()}))
    return str(b)


def run(tmp_path, **kw):
    batches = [make_batch(tmp_path / "in", DAY), make_batch(tmp_path / "in", NIGHT, edge_placed=False),
               make_batch(tmp_path / "in", VAL)]
    heads, sources = ex.load_batches(batches)
    args = dict(val_clips=[VAL], mask_clips=[NIGHT], mask_slots=EDGE)
    args.update(kw)
    return ex.export(heads, sources, tmp_path / "ds", **args)


def rows_of(ds, split):
    return {p.stem: p.read_text().split() for p in (ds / "labels" / split).glob("*.txt")}


def test_no_clip_appears_in_both_splits(tmp_path):
    m, _ = run(tmp_path)
    ds = tmp_path / "ds"
    clips = {sp: {stem.rsplit("_", 2)[0] for stem in rows_of(ds, sp)} for sp in ("train", "val")}
    assert clips == {"train": {DAY, NIGHT}, "val": {VAL}}
    assert not clips["train"] & clips["val"]
    assert {p.stem for p in (ds / "images" / "val").glob("*.jpg")} == set(rows_of(ds, "val"))
    assert m["splits"] == {"train": [DAY, NIGHT], "val": [VAL]} and m["heads"] == {"train": 6, "val": 3}
    assert m["per_frame_split"] is False and m["assertions"]["no_clip_in_both_splits"] is True


def test_night_edge_slots_are_masked_not_absent_and_day_keeps_its_outline(tmp_path):
    m, _ = run(tmp_path)
    ds = tmp_path / "ds"
    mask = json.loads((ds / "kpt_mask.json").read_text())
    train = rows_of(ds, "train")
    night = {s for s in train if s.startswith(NIGHT)}
    day = {s for s in train if s.startswith(DAY)}
    assert set(mask["images"]) == night and all(v == EDGE for v in mask["images"].values())   # only night heads
    for stem in day:                                              # day heads: edge slots supervised
        v = [int(train[stem][5 + 3 * i + 2]) for i in range(36)]
        assert all(v[i] == 2 for i in EDGE) and v[24] == 0 and v[10] == 1
    # counted apart: the 8 edge slots of each night head are masked; the one deleted point stays absent
    assert m["clips"][NIGHT]["points"] == {"visible": 3 * 26, "occluded": 3, "masked": 3 * 8, "absent": 3}
    assert m["clips"][DAY]["points"] == {"visible": 3 * 34, "occluded": 3, "masked": 0, "absent": 3}
    assert m["clips"][DAY]["masked_slots"] == [] and m["clips"][NIGHT]["masked_slots"] == EDGE
    assert m["clips"][DAY]["heads_masked"] == 0 and m["clips"][NIGHT]["heads_masked"] == 3
    assert "kpt_mask: kpt_mask.json" in (ds / "data.yaml").read_text()
    for s in cs.STATES:
        assert m["points_total"][s] == sum(c["points"][s] for c in m["clips"].values())
    assert sum(m["points_total"].values()) == 9 * 36


def test_a_mark_inside_a_masked_slot_is_reported_not_silently_lost(tmp_path):
    batches = [make_batch(tmp_path / "in", DAY), make_batch(tmp_path / "in", NIGHT, edge_placed=True),
               make_batch(tmp_path / "in", VAL)]
    heads, sources = ex.load_batches(batches)
    m, _ = ex.export(heads, sources, tmp_path / "ds", val_clips=[VAL], mask_clips=[NIGHT], mask_slots=EDGE)
    assert m["clips"][NIGHT]["marks_dropped_by_mask"] == 3 * 8 and m["clips"][DAY]["marks_dropped_by_mask"] == 0


def test_manifest_sha_matches_the_file(tmp_path):
    _, sha = run(tmp_path)
    ds = tmp_path / "ds"
    assert sha == cs.sha256_bytes((ds / "split_manifest.json").read_bytes())
    assert (ds / "split_manifest.json.sha256").read_text().split()[0] == sha


def test_judge_material_stops_the_export_before_anything_is_written(tmp_path):
    batches = [make_batch(tmp_path / "in", DAY), make_batch(tmp_path / "in", "IMG_4425"),
               make_batch(tmp_path / "in", VAL)]
    heads, sources = ex.load_batches(batches)
    with pytest.raises(cs.SplitError, match="judge clip"):
        ex.export(heads, sources, tmp_path / "ds", val_clips=[VAL])
    assert not (tmp_path / "ds").exists()
    with pytest.raises(cs.SplitError, match="forbidden"):
        run(tmp_path / "b", forbidden_ids={f"{DAY}_00001"})
    with pytest.raises(cs.SplitError, match="content hash"):
        run(tmp_path / "c", forbidden_sha={NIGHT.lower()})


def test_export_refuses_a_non_empty_output_folder(tmp_path):
    run(tmp_path)
    heads, sources = ex.load_batches([str(tmp_path / "in" / c) for c in (DAY, NIGHT, VAL)])
    with pytest.raises(cs.SplitError, match="not empty"):
        ex.export(heads, sources, tmp_path / "ds", val_clips=[VAL])


def test_mask_if_unmarked_masks_only_heads_with_no_mark_in_the_group(tmp_path):
    day = make_batch(tmp_path / "in", DAY, n_frames=3)
    rows = json.loads((tmp_path / "in" / DAY / "labels.json").read_text())
    for i in EDGE:
        rows[0]["kpts"][i] = [0, 0, 0]                 # head 0: whole edge group skipped
    for i in EDGE[:4]:
        rows[1]["kpts"][i] = [0, 0, 0]                 # head 1: half placed -> the rest really is absent
    (tmp_path / "in" / DAY / "labels.json").write_text(json.dumps(rows))
    heads, sources = ex.load_batches([day, make_batch(tmp_path / "in", VAL)])
    off, _ = ex.export(heads, sources, tmp_path / "off", val_clips=[VAL], mask_slots=EDGE)
    assert off["clips"][DAY]["points"]["masked"] == 0 and off["clips"][DAY]["points"]["absent"] == 3 + 8 + 4
    heads, sources = ex.load_batches([day, str(tmp_path / "in" / VAL)])
    on, _ = ex.export(heads, sources, tmp_path / "on", val_clips=[VAL], mask_slots=EDGE, mask_if_unmarked=True)
    assert on["clips"][DAY]["heads_masked"] == 1
    assert on["clips"][DAY]["points"]["masked"] == 8 and on["clips"][DAY]["points"]["absent"] == 3 + 4
    assert list(json.loads((tmp_path / "on" / "kpt_mask.json").read_text())["images"]) == [f"{DAY}_00000_h0"]
    assert on["clips"][VAL]["heads_masked"] == 0
