"""Rules of the v1.0 clip-level split and the masked state. Standard library + pytest only."""
import pytest

import clip_split as cs
import export_yolo as ex


def heads(clips=("IMG_3656", "Sheep_at_night", "IMG_3678"), per_clip=4):
    return [{"id": f"{c}_{f:05d}", "clip": c, "head": 0} for c in clips for f in range(per_clip)]


def test_split_is_by_clip_and_no_clip_is_in_both_splits():
    hs = heads()
    a = cs.assign_by_clip(hs, ["IMG_3678"])
    assert a == {"IMG_3656": "train", "IMG_3678": "val", "Sheep_at_night": "train"}
    rows = [(h["clip"], a[h["clip"]]) for h in hs]
    cs.check_no_straddle(rows)                                        # passes: every clip has one split
    by_split = {sp: {c for c, s in rows if s == sp} for sp in ("train", "val")}
    assert not by_split["train"] & by_split["val"]


def test_straddle_is_caught():
    with pytest.raises(cs.SplitError, match="both train and val"):
        cs.check_no_straddle([("A", "train"), ("B", "val"), ("A", "val")])


def test_split_refuses_to_guess():
    with pytest.raises(cs.SplitError, match="no held-out clip"):
        cs.assign_by_clip(heads(), [])
    with pytest.raises(cs.SplitError, match="no heads"):
        cs.assign_by_clip(heads(), ["IMG_9999"])
    with pytest.raises(cs.SplitError, match="nothing left"):
        cs.assign_by_clip(heads(clips=("A",)), ["A"])
    with pytest.raises(cs.SplitError, match="has no clip"):
        cs.assign_by_clip([{"id": "x_00001"}], ["A"])


@pytest.mark.parametrize("clip", ["IMG_3877", "IMG_4425", "img_3877"])
def test_judge_clip_is_refused(clip):
    with pytest.raises(cs.SplitError, match="judge clip"):
        cs.guard_training_set(heads(clips=("IMG_3656", clip)))


def test_judge_frame_id_under_another_clip_name_is_refused():
    with pytest.raises(cs.SplitError, match="judge clip"):
        cs.guard_training_set([{"id": "IMG_4425_00030", "clip": "renamed"}])


def test_forbidden_frame_id_hash_and_test_only_path_are_refused():
    with pytest.raises(cs.SplitError, match="forbidden"):
        cs.guard_training_set(heads(), forbidden_ids={"IMG_3656_00002"})
    with pytest.raises(cs.SplitError, match="content hash"):
        cs.guard_training_set(heads(), [{"source": "/x/copy.mov", "sha256": "ABC"}], forbidden_sha={"abc"})
    with pytest.raises(cs.SplitError, match="test-only-NEVER-TRAIN"):
        cs.guard_training_set(heads(), [{"source": "/Volumes/X/test-only-NEVER-TRAIN/a.mov", "sha256": "1"}])
    with pytest.raises(cs.SplitError, match="judge clip"):
        cs.guard_training_set(heads(), [{"source": "/x/IMG_3877.MOV", "sha256": "1"}])


def test_clean_set_passes_and_says_what_was_checked():
    g = cs.guard_training_set(heads(), [{"source": "/x/IMG_3656.MOV", "sha256": "1"}], {"IMG_3877_00010"}, {"ff"})
    assert g["violations"] == 0 and g["allow_judge"] is False and g["heads_checked"] == 12


def test_exporter_has_no_allow_judge_switch():
    with pytest.raises(SystemExit):
        ex.main(["--batch", "x", "--val-clip", "A", "--out", "y", "--allow-judge"])


def test_masked_is_a_state_of_its_own_never_absent():
    assert cs.state(0, masked=True) == "masked"
    assert cs.state(0, masked=False) == "absent"
    assert cs.state(2, masked=False) == "visible" and cs.state(1, masked=False) == "occluded"
    slots = cs.parse_slots("27-34")
    assert slots == list(range(27, 35))
    assert cs.masked_slots_for("Sheep_at_night", ["Sheep_at_night"], slots) == slots
    assert cs.masked_slots_for("IMG_3656", ["Sheep_at_night"], slots) == []       # day heads keep their outline
    n = cs.count_states([("night", [2] * 27 + [0] * 8 + [1], slots), ("day", [2] * 35 + [0], [])])
    assert n["night"] == {"visible": 27, "occluded": 1, "masked": 8, "absent": 0}
    assert n["day"] == {"visible": 35, "occluded": 0, "masked": 0, "absent": 1}


def test_masked_set_must_be_closed_under_the_flip():
    cs.check_flip_closed(range(27, 35), ex.FLIP_IDX)
    with pytest.raises(cs.SplitError, match="flip partner"):
        cs.check_flip_closed([27, 29, 30], ex.FLIP_IDX)


def test_flip_idx_is_the_sheep_face_36_one():
    assert ex.FLIP_IDX == [0, 2, 1, 4, 3, 7, 8, 5, 6, 13, 14, 15, 16, 9, 10, 11, 12, 19, 20, 17, 18, 21, 23, 22,
                           24, 25, 26, 28, 27, 30, 29, 32, 31, 34, 33, 35]
    assert [ex.FLIP_IDX[i] for i in ex.FLIP_IDX] == list(range(36))
