"""Stdlib tests for pose summaries, the Jev look/dont mapping, and the farmer demo.

    python experiments/sheep-triage-jev/test_triage.py

Overlay-video tests need OpenCV and are skipped without it. None of these
tests call the API.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from glance_list import (
    LOOK,
    SKIP,
    UNCHECKED,
    badge_for,
    build_cards,
    reason_for,
    render_html,
    write_glance_list,
)
from jev_client import JevError, decision_from_noul, request_body, triage_track
from pose_features import ear_angles, summarize_frames, synthetic_frames
from pose_runner import frame_from_result
from render_overlay import badges_from_triage, header_parts, load_frames, save_frames
from run_pipeline import run

HERE = Path(__file__).resolve().parent
HAS_CV2 = all(importlib.util.find_spec(m) is not None for m in ("cv2", "numpy"))


def _pt(x, y, conf=1.0):
    return [x, y, conf]


class EarAngleTests(unittest.TestCase):
    def test_horizontal_ears_are_ninety_degrees(self):
        # nose above the ear-base midpoint; tips stick straight out.
        kpts = [
            _pt(150, 80),
            _pt(120, 140),
            _pt(180, 140),
            _pt(70, 140),
            _pt(230, 140),
        ]
        left, right = ear_angles(kpts, 0.4)
        self.assertAlmostEqual(left, 90.0, places=5)
        self.assertAlmostEqual(right, 90.0, places=5)

    def test_low_confidence_tip_is_absent(self):
        kpts = [
            _pt(150, 80),
            _pt(120, 140),
            _pt(180, 140),
            _pt(70, 140, 0.1),
            _pt(230, 140),
        ]
        left, right = ear_angles(kpts, 0.4)
        self.assertIsNone(left)
        self.assertAlmostEqual(right, 90.0, places=5)


class SummaryTests(unittest.TestCase):
    def test_synthetic_tracks_separate_steady_from_jumpy(self):
        frames, fps = synthetic_frames()
        tracks, dropped = summarize_frames(frames, fps=fps, kpt_conf=0.4, min_frames=2)
        self.assertEqual(dropped, [99])
        by_id = {t["track_id"]: t for t in tracks}
        steady = by_id[1]["ear_angle_deg"]
        jumpy = by_id[7]["ear_angle_deg"]
        self.assertEqual(steady["left_std"], 0.0)
        self.assertEqual(steady["right_median"], 90.0)
        self.assertGreater(jumpy["left_std"], 20.0)
        self.assertEqual(jumpy["n_right"], 0)
        self.assertGreater(by_id[7]["motion"]["centroid_path_px"], 100.0)
        self.assertEqual(by_id[1]["motion"]["centroid_path_px"], 0.0)

    def test_state_is_numbers_not_a_secret(self):
        frames, fps = synthetic_frames()
        tracks, _ = summarize_frames(frames, fps=fps, kpt_conf=0.4, min_frames=2)
        from pose_features import state_for_jev

        state = state_for_jev(tracks[0])
        blob = json.dumps(state)
        self.assertNotIn("TYPESAFE_API_KEY", blob)
        self.assertNotIn("sk-", blob)
        self.assertNotIn("base64", blob)
        self.assertIn("track", state)
        self.assertIn("No image is attached", state["task"])


class _Array:
    """Just enough of a numpy array for frame_from_result, without numpy."""

    def __init__(self, data):
        self._data = data

    def astype(self, _typ):
        return _Array([int(v) for v in self._data])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        if isinstance(item, list):
            return _Array(item)
        return item

    @property
    def shape(self):
        if self._data and isinstance(self._data[0], list):
            return (len(self._data), len(self._data[0]))
        return (len(self._data),)

    def tolist(self):
        return self._data


class _Tensor:
    def __init__(self, data):
        self._data = data

    def cpu(self):
        return self

    def numpy(self):
        return _Array(self._data)


class _Boxes:
    def __init__(self, ids, xyxy):
        self.id = None if ids is None else _Tensor(ids)
        self.xyxy = _Tensor(xyxy)


class _Kpts:
    def __init__(self, data):
        self.data = _Tensor(data)


class _Result:
    def __init__(self, boxes, keypoints):
        self.boxes = boxes
        self.keypoints = keypoints


class PoseParseTests(unittest.TestCase):
    def test_frame_from_result(self):
        result = _Result(
            _Boxes([3], [[0, 0, 10, 20]]),
            _Kpts([[[1, 2, 0.9], [3, 4, 0.8], [5, 6, 0.7], [7, 8, 0.6], [9, 10, 0.5]]]),
        )
        frame = frame_from_result(result)
        self.assertEqual(list(frame), [3])
        self.assertEqual(frame[3]["box"], [0.0, 0.0, 10.0, 20.0])
        self.assertEqual(len(frame[3]["kpts"]), 5)
        self.assertEqual(frame[3]["kpts"][0][2], 0.9)

    def test_missing_ids(self):
        result = _Result(_Boxes(None, []), _Kpts([]))
        self.assertEqual(frame_from_result(result), {})


class JevMappingTests(unittest.TestCase):
    def test_threshold(self):
        self.assertEqual(decision_from_noul(0.5, 0.5), "look")
        self.assertEqual(decision_from_noul(0.49, 0.5), "dont")
        self.assertEqual(decision_from_noul(0.0, 0.5), "dont")

    def test_request_is_a_noul_and_omits_the_key(self):
        body = request_body({"track_id": 1})
        question = body["questions"]["look"]
        self.assertEqual(question["type"], "noul")
        self.assertNotIn("api_key", json.dumps(body))
        self.assertNotIn("pain", json.dumps(body).lower())

    def test_client_maps_response(self):
        seen = {}

        class Resp:
            def read(self):
                return json.dumps(
                    {
                        "model": "jev-1.13.0",
                        "answers": {"look": {"type": "noul", "noul": 0.82}},
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                    }
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def opener(req, timeout=0):
            seen["timeout"] = timeout
            seen["auth"] = req.get_header("Authorization")
            seen["body"] = json.loads(req.data.decode())
            return Resp()

        out = triage_track(
            {"track_id": 4},
            api_key="sk-test-secret",
            threshold=0.5,
            opener=opener,
        )
        self.assertEqual(out["decision"], "look")
        self.assertEqual(out["noul"], 0.82)
        self.assertEqual(seen["auth"], "Bearer sk-test-secret")
        self.assertNotIn("sk-test-secret", json.dumps(seen["body"]))
        self.assertNotIn("sk-test-secret", json.dumps(out))

    def test_http_error_scrubs_the_key(self):
        key = "sk-test-do-not-leak"

        def opener(req, timeout=0):
            raise urllib.error.HTTPError(
                req.full_url,
                401,
                "unauthorized",
                hdrs=None,
                fp=io.BytesIO(f"rejected {key}".encode()),
            )

        with self.assertRaises(JevError) as caught:
            triage_track({"track_id": 1}, api_key=key, opener=opener)
        self.assertNotIn(key, str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))
        self.assertIn("401", str(caught.exception))


class PipelineTests(unittest.TestCase):
    def test_synthetic_pose_only_writes_artifacts_without_a_key(self):
        env = os.environ.copy()
        env.pop("TYPESAFE_API_KEY", None)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.dict(os.environ, env, clear=True):
                code = run(["--synthetic", "--pose-only", "--out", str(out)])
            self.assertEqual(code, 0)
            tracks = json.loads((out / "tracks.json").read_text())
            triage = json.loads((out / "triage.json").read_text())
            summary = (out / "summary.txt").read_text()
            self.assertEqual(tracks["mode"], "pose-only")
            self.assertEqual(triage["reason"], "--pose-only")
            self.assertEqual(sorted(t["track_id"] for t in tracks["tracks"]), [1, 7])
            self.assertTrue(all(row["decision"] is None for row in triage["tracks"]))
            self.assertIn("decision=—", summary)
            blob = (out / "tracks.json").read_text() + (out / "triage.json").read_text()
            self.assertNotIn("TYPESAFE", blob)

    def test_missing_clip_and_weights_explain_how_to_recover(self):
        with self.assertRaises(SystemExit) as missing_clip:
            run(["--clip", "/tmp/samseessheep-no-such-clip.mov", "--pose-only"])
        self.assertIn("clip not found", str(missing_clip.exception))
        self.assertIn("Test_Clip_Morning", str(missing_clip.exception))

        with tempfile.NamedTemporaryFile(suffix=".mov") as handle:
            with self.assertRaises(SystemExit) as missing_weights:
                run(
                    [
                        "--clip",
                        handle.name,
                        "--weights",
                        "/tmp/samseessheep-no-such-weights.pt",
                        "--pose-only",
                    ]
                )
        self.assertIn("weights not found", str(missing_weights.exception))
        self.assertIn("sheep-pose-v0.7-yolo26n.pt", str(missing_weights.exception))

    def test_missing_key_stays_pose_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.dict(os.environ, {}, clear=True):
                code = run(["--synthetic", "--out", str(out)])
            self.assertEqual(code, 0)
            triage = json.loads((out / "triage.json").read_text())
            self.assertEqual(triage["mode"], "pose-only")
            self.assertIn("TYPESAFE_API_KEY", triage["reason"])

    def test_key_triggers_jev_and_is_not_stored(self):
        key = "sk-live-should-not-land-on-disk"

        def opener(req, timeout=0):
            class Resp:
                def read(self_inner):
                    track = json.loads(req.data.decode())["state"]["track"]
                    noul = 0.1 if track["track_id"] == 1 else 0.9
                    return json.dumps(
                        {"model": "jev-test", "answers": {"look": {"type": "noul", "noul": noul}}}
                    ).encode()

                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *args):
                    return False

            return Resp()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": key}, clear=True):
                with mock.patch("run_pipeline.triage_track", wraps=_wrap(opener)):
                    code = run(["--synthetic", "--out", str(out)])
            self.assertEqual(code, 0)
            triage = json.loads((out / "triage.json").read_text())
            by_id = {row["track_id"]: row["decision"] for row in triage["tracks"]}
            self.assertEqual(by_id[1], "dont")
            self.assertEqual(by_id[7], "look")
            blob = "\n".join(p.read_text() for p in out.iterdir())
            self.assertNotIn(key, blob)


def _track(tid, first, last, *, left_std=2.0, right_std=2.0, n_right=None, path=0.0, width=200.0):
    n = last - first + 1
    return {
        "track_id": tid,
        "n_frames": n,
        "frame_first": first,
        "frame_last": last,
        "duration_s": n / 30.0,
        "facing_fraction": 1.0,
        "ear_angle_deg": {
            "left_median": 90.0,
            "left_std": left_std,
            "right_median": 90.0,
            "right_std": right_std if n_right != 0 else None,
            "n_left": n,
            "n_right": n if n_right is None else n_right,
        },
        "motion": {"centroid_path_px": path, "bbox_median_w_px": width},
    }


def _docs(decisions, *, mode="jev", errors=()):
    tracks = [
        _track(1, 0, 299),
        _track(4, 60, 239, left_std=28.0),
        _track(9, 150, 449, path=1500.0),
        _track(12, 30, 329, n_right=0),
    ]
    tracks_doc = {
        "clip": "/Users/someone/sheep-triage-jev/IMG_3877.MOV",
        "fps": 30.0,
        "n_frames": 450,
        "dropped_track_ids": [77],
        "tracks": tracks,
    }
    rows = []
    for track in tracks:
        tid = track["track_id"]
        row = {"track_id": tid, "decision": decisions.get(tid), "noul": 0.123456}
        if tid in errors:
            row["error"] = "Jev HTTP 500"
        rows.append(row)
    return tracks_doc, {"mode": mode, "model": "jev-latest", "tracks": rows}


class FarmerBadgeTests(unittest.TestCase):
    def test_look_and_dont_become_look_and_skip(self):
        self.assertEqual(badge_for("look"), "Look")
        self.assertEqual(badge_for("dont"), "Skip")

    def test_anything_else_is_not_checked(self):
        for value in (None, "", "maybe", "LOOK"):
            self.assertEqual(badge_for(value), UNCHECKED)

    def test_overlay_reads_badges_from_triage(self):
        triage = {
            "tracks": [
                {"track_id": 1, "decision": "look"},
                {"track_id": 2, "decision": "dont"},
                {"track_id": 3},
            ]
        }
        self.assertEqual(badges_from_triage(triage), {1: LOOK, 2: SKIP, 3: UNCHECKED})

    def test_overlay_header_never_fakes_a_skip(self):
        self.assertEqual(header_parts({1: LOOK, 2: LOOK}), [("LOOK 2", LOOK), ("SKIP 0", SKIP)])
        self.assertEqual(header_parts({1: UNCHECKED}), [("NOT SORTED YET", UNCHECKED)])

    def test_frames_round_trip(self):
        frames, _ = synthetic_frames()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.json"
            save_frames(path, frames)
            loaded = load_frames(path)
        self.assertEqual(len(loaded), len(frames))
        self.assertEqual(sorted(loaded[0]), [1, 7, 99])
        self.assertEqual(loaded[3][7]["kpts"][0], frames[3][7]["kpts"][0])


class GlanceListTests(unittest.TestCase):
    def test_look_cards_come_first_with_time_reason_and_jump(self):
        tracks_doc, triage = _docs({1: "dont", 4: "look", 9: "look", 12: "dont"})
        cards = build_cards(tracks_doc, triage)
        self.assertEqual([c["badge"] for c in cards], [LOOK, LOOK, SKIP, SKIP])
        self.assertEqual([c["track_id"] for c in cards], [4, 9, 1, 12])
        page = render_html(tracks_doc, triage, video="annotated.mp4")
        self.assertIn("2 of 4 need a look, 2 can be skipped.", page)
        self.assertIn('<video id="replay" src="annotated.mp4"', page)
        self.assertIn('href="annotated.mp4#t=2.00" data-t="2.00">Jump to 0:02</a>', page)
        self.assertIn("In view 0:02 – 0:08", page)
        self.assertIn("Left ear kept moving — its angle varied by about 28°.", page)
        self.assertLess(page.index("Sheep #9"), page.index("Sheep #1<"))

    def test_all_look_says_everyone_needs_a_look(self):
        tracks_doc, triage = _docs({1: "look", 4: "look", 9: "look", 12: "look"})
        page = render_html(tracks_doc, triage, video="annotated.mp4")
        self.assertIn("Every sheep in this clip needs a look (4 of 4).", page)
        self.assertIn("Skip 0", page)
        self.assertNotIn('class="card skip"', page)
        self.assertEqual(page.count('class="card look"'), 4)
        self.assertIn("still flagged for a glance", page)

    def test_pose_only_run_is_not_sorted(self):
        tracks_doc, triage = _docs({}, mode="pose-only")
        page = render_html(tracks_doc, triage, video=None)
        self.assertIn("Not sorted yet", page)
        self.assertNotIn('class="card look"', page)
        self.assertNotIn('class="card skip"', page)

    def test_failed_checks_are_not_dressed_up(self):
        tracks_doc, triage = _docs({1: "dont", 4: "look"}, errors=(9, 12))
        page = render_html(tracks_doc, triage, video=None)
        self.assertIn("1 of 4 need a look, 1 can be skipped, 2 could not be checked.", page)
        self.assertEqual(page.count("The look/skip check failed for this one."), 2)

    def test_without_video_names_the_timestamp(self):
        tracks_doc, triage = _docs({4: "look"})
        page = render_html(tracks_doc, triage, video=None)
        self.assertNotIn("<video", page)
        self.assertNotIn("a class=\"jump\"", page)
        self.assertIn("At 0:02 in the clip.", page)

    def test_farmer_words_only(self):
        tracks_doc, triage = _docs({1: "dont", 4: "look"})
        page = render_html(tracks_doc, triage, video="annotated.mp4").lower()
        for word in ("noul", "jev", "0.123", "decision", "track_id", "/users/someone"):
            self.assertNotIn(word, page)
        self.assertIn("img_3877.mov", page)
        self.assertIn("1 very brief detection was ignored", page)

    def test_reasons_are_plain_english(self):
        self.assertEqual(
            reason_for(_track(4, 0, 99, right_std=30.0)),
            "Right ear kept moving — its angle varied by about 30°.",
        )
        self.assertEqual(reason_for(_track(12, 0, 99, n_right=0)), "Right ear hard to see on most frames.")
        self.assertEqual(reason_for(_track(9, 0, 299, path=1500.0)), "Walked around a lot while in view.")
        self.assertEqual(reason_for(_track(3, 0, 9)), "Only in view for a moment (0.3 s).")
        self.assertIn("steady", reason_for(_track(1, 0, 299), SKIP))

    def test_page_from_a_mocked_jev_run(self):
        key = "sk-live-glance-list"

        def opener(req, timeout=0):
            class Resp:
                def read(self_inner):
                    track = json.loads(req.data.decode())["state"]["track"]
                    noul = 0.2 if track["track_id"] == 1 else 0.8
                    return json.dumps({"answers": {"look": {"type": "noul", "noul": noul}}}).encode()

                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *args):
                    return False

            return Resp()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": key}, clear=True):
                with mock.patch("run_pipeline.triage_track", wraps=_wrap(opener)):
                    self.assertEqual(run(["--synthetic", "--out", str(out)]), 0)
            page = write_glance_list(out).read_text()
        self.assertIn("1 of 2 need a look, 1 can be skipped.", page)
        self.assertLess(page.index("Sheep #7"), page.index("Sheep #1<"))
        self.assertNotIn(key, page)
        self.assertNotIn("<video", page)


@unittest.skipUnless(HAS_CV2, "OpenCV not installed; run in the sheep-yolo env")
class OverlayTests(unittest.TestCase):
    def test_demo_run_writes_video_and_page(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(run(["--synthetic", "--pose-only", "--demo", "--out", str(out)]), 0)
            cap = cv2.VideoCapture(str(out / "annotated.mp4"))
            n = 0
            while cap.read()[0]:
                n += 1
            cap.release()
            page = (out / "glance-list.html").read_text()
            self.assertEqual(n, 8)
            self.assertEqual(len(load_frames(out / "frames.json")), 8)
            self.assertIn('<video id="replay" src="annotated.mp4"', page)
            self.assertIn("Not sorted yet", page)

    def test_badges_are_burned_from_the_decision(self):
        import cv2
        import numpy as np
        from render_overlay import BADGE_BG, draw_track

        frames, _ = synthetic_frames()

        def colored(img, badge):
            return int(np.all(img == np.array(BADGE_BG[badge], np.uint8), axis=-1).sum())

        look = np.zeros((600, 640, 3), np.uint8)
        draw_track(cv2, look, frames[0][7], 7, LOOK, scale=1.0, kpt_conf=0.4, top=0)
        skip = np.zeros((600, 640, 3), np.uint8)
        draw_track(cv2, skip, frames[0][1], 1, SKIP, scale=1.0, kpt_conf=0.4, top=0)
        self.assertGreater(colored(look, LOOK), 1000)
        self.assertEqual(colored(look, SKIP), 0)
        self.assertGreater(colored(skip, SKIP), 1000)
        self.assertEqual(colored(skip, LOOK), 0)


@unittest.skipIf(HAS_CV2, "OpenCV is installed")
class DemoWithoutOpenCVTests(unittest.TestCase):
    def test_demo_explains_the_env(self):
        with self.assertRaises(SystemExit) as caught:
            run(["--synthetic", "--pose-only", "--demo", "--out", "/tmp/samseessheep-no-demo"])
        self.assertIn("uv run --project sheep-yolo", str(caught.exception))


def _wrap(opener):
    def _call(state, *, api_key, threshold, model, timeout):
        return triage_track(
            state,
            api_key=api_key,
            threshold=threshold,
            model=model,
            timeout=timeout,
            opener=opener,
        )

    return _call


if __name__ == "__main__":
    raise SystemExit(unittest.main())
