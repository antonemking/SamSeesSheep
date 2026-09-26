"""Stdlib tests for pose summaries, the SPFES ear pack, the book rules, the Jev
look / skip / cannot mapping, the pen call, and the farmer's morning brief.

    python experiments/sheep-triage-jev/test_triage.py

Overlay-video tests need OpenCV and are skipped without it. None of these
tests call the API: Jev is a local fake, and urllib's urlopen is patched to
fail the test if anything reaches for the network.
"""

from __future__ import annotations

import html
import importlib.util
import io
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import types
import unittest
import urllib.error
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest import mock

from glance_list import (
    LOOK,
    PAGE_NAME,
    SKIP,
    UNCHECKED,
    badge_for,
    build_cards,
    detail,
    farmer_reason,
    headline,
    pen_line,
    reason_for,
    render_html,
    thumb_path,
    write_glance_list,
)
from jev_client import (
    LOOK_ID,
    PEN_CALLS,
    PEN_ID,
    PEN_QUESTIONS,
    REASON_ID,
    SPFES_EAR_PROTOCOL,
    TRACK_QUESTIONS,
    TRIAGE_ID,
    ChoiceAnswer,
    JevError,
    decide,
    fitting_reason,
    pen_request_body,
    pen_state,
    request_body,
    triage_pen,
    triage_track,
)
from pose_features import (
    MIN_EAR_FRAMES,
    EarPack,
    EarStats,
    ear_angles,
    state_for_jev,
    summarize_frames,
    summarize_track,
    synthetic_frames,
)
from pose_runner import frame_from_result
from clip_frames import TURNS, Orientation, orientation, recorded_rotation, turn, turn_frames, upright_frames
from pose_runner import track_clip
from render_overlay import badges_from_triage, crop_box, header_parts, load_frames, save_frames, thumb_frames
from run_pipeline import run
from spfes_ear import DECISIONS, JEV_REASONS, REASONS, REASONS_FOR, Verdict, book_triage

HERE = Path(__file__).resolve().parent
HAS_CV2 = all(importlib.util.find_spec(m) is not None for m in ("cv2", "numpy"))
HAS_VIDEO = HAS_CV2 and importlib.util.find_spec("av") is not None


def setUpModule() -> None:
    guard = mock.patch(
        "urllib.request.urlopen",
        side_effect=AssertionError("unit tests must not reach the network"),
    )
    guard.start()
    unittest.addModuleCleanup(guard.stop)


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
        self.assertEqual(decide("look", 0.5, 0.5), "look")
        self.assertEqual(decide("look", 0.49, 0.5), "cannot")
        self.assertEqual(decide("skip", 0.0, 0.5), "skip")
        self.assertEqual(decide("skip", 0.99, 0.5), "skip")
        self.assertEqual(decide("cannot", 0.99, 0.5), "cannot")

    def test_request_is_a_noul_and_omits_the_key(self):
        body = request_body({"track_id": 1})
        question = body["questions"]["look"]
        self.assertEqual(question["type"], "noul")
        self.assertNotIn("api_key", json.dumps(body))
        self.assertNotIn("pain", json.dumps(body).lower())

    def test_client_maps_response(self):
        seen = {}

        def opener(req, timeout=0):
            seen["timeout"] = timeout
            seen["auth"] = req.get_header("Authorization")
            seen["body"] = json.loads(req.data.decode())
            return _Resp(
                {
                    "model": "jev-1.13.0",
                    "answers": _track_answers("look", 0.82, "flutter"),
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            )

        out = triage_track(
            {"track_id": 4},
            api_key="sk-test-secret",
            threshold=0.5,
            opener=opener,
        )
        self.assertEqual(out.decision, "look")
        self.assertEqual(out.reason, "flutter")
        self.assertEqual(out.noul, 0.82)
        self.assertEqual(out.model, "jev-1.13.0")
        self.assertEqual(seen["auth"], "Bearer sk-test-secret")
        self.assertNotIn("sk-test-secret", json.dumps(seen["body"]))
        self.assertNotIn("sk-test-secret", json.dumps(out.to_json()))

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
        jev = FakeJev(lambda pack: ("skip", 0.1, "steady") if pack.track_id == 1 else ("look", 0.9, "flutter"))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": key}, clear=True):
                code = run(["--synthetic", "--out", str(out)], opener=jev)
            self.assertEqual(code, 0)
            triage = json.loads((out / "triage.json").read_text())
            by_id = {row["track_id"]: row["decision"] for row in triage["tracks"]}
            self.assertEqual(by_id[1], "skip")
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


def visible_text(page: str) -> str:
    """What a reader sees: the page without style, script, tags or attributes."""
    text = re.sub(r"<(style|script)\b.*?</\1>", " ", page, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text))


def main_view(page: str) -> str:
    """The page above the collapsed section: masthead, summary and the Look list."""
    return page.split('<details class="rest">')[0].split("<footer>")[0]


def rest_section(page: str) -> str:
    match = re.search(r'<details class="rest">.*?</details>', page, flags=re.DOTALL)
    return match.group(0) if match else ""


# Never on the farmer's page: engine and model names, raw numbers and internal
# vocabulary, file paths, and diagnosis language.
FORBIDDEN_TEXT = (
    "noul",
    "jev",
    "typesafe",
    "model",
    "json",
    "yolo",
    "spfes",
    "threshold",
    "probab",
    "confidence",
    "score",
    "decision",
    "track",
    "cannot",
    "pain",
    "diagnos",
    "welfare",
    "walk_now",
    "walk_tomorrow",
    "not_sure",
    "one_ear_missing",
    "not_facing",
    "ears_unmeasurable",
    "0.123",
    ".mp4",
    ".jpg",
    "http",
    "/",
    "\\",
)
FORBIDDEN_MARKUP = (
    "noul",
    "jev",
    "typesafe",
    "0.123",
    "track_id",
    "/users/",
    "walk_now",
    "not_sure",
    "decision",
    "http",
)


class GlanceListTests(unittest.TestCase):
    def test_look_cards_come_first_with_time_reason_and_jump(self):
        tracks_doc, triage = _docs({1: "dont", 4: "look", 9: "look", 12: "dont"})
        cards = build_cards(tracks_doc, triage)
        self.assertEqual([c["badge"] for c in cards], [LOOK, LOOK, SKIP, SKIP])
        self.assertEqual([c["track_id"] for c in cards], [4, 9, 1, 12])
        page = render_html(tracks_doc, triage, video="annotated.mp4")
        self.assertIn("Check these 2 tomorrow.", page)
        self.assertIn('<video id="replay" src="annotated.mp4"', page)
        self.assertIn(
            'href="annotated.mp4#t=2.00" data-t="2.00" data-label="Sheep #4 · from 0:02">Watch from 0:02</a>', page
        )
        self.assertIn("Seen 0:02 – 0:08", page)
        self.assertIn("Left ear kept moving — its angle varied by about 28°.", page)
        self.assertLess(page.index("Sheep #9"), page.index("Sheep #1<"))
        self.assertIn("2 looked fine", rest_section(page))

    def test_all_look_says_everyone_needs_a_look(self):
        tracks_doc, triage = _docs({1: "look", 4: "look", 9: "look", 12: "look"})
        page = render_html(tracks_doc, triage, video="annotated.mp4")
        self.assertIn("Check all 4 tomorrow.", page)
        self.assertIn("Every sheep seen was flagged (4 of 4).", page)
        self.assertEqual(rest_section(page), "")
        self.assertEqual(page.count('class="card look'), 4)
        self.assertIn("still flagged for a glance", page)

    def test_pose_only_run_is_not_sorted(self):
        tracks_doc, triage = _docs({}, mode="pose-only")
        page = render_html(tracks_doc, triage, video=None)
        self.assertIn("Not sorted yet", page)
        self.assertNotIn('class="card look', page)
        self.assertNotIn("Looked fine", rest_section(page))
        self.assertIn("4 not checked", rest_section(page))

    def test_failed_checks_are_not_dressed_up(self):
        tracks_doc, triage = _docs({1: "dont", 4: "look"}, errors=(9, 12))
        page = render_html(tracks_doc, triage, video=None)
        self.assertIn("Check this one tomorrow.", page)
        self.assertIn("2 not checked · 1 looked fine", rest_section(page))
        self.assertEqual(rest_section(page).count("The ear check failed for this one."), 2)
        self.assertNotIn("failed", main_view(page))

    def test_without_video_names_the_timestamp(self):
        tracks_doc, triage = _docs({4: "look"})
        page = render_html(tracks_doc, triage, video=None)
        self.assertNotIn("<video", page)
        self.assertNotIn("<script", page)
        self.assertNotIn('class="watch', page)
        self.assertIn("Seen 0:02 – 0:08", page)

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
        jev = FakeJev(lambda pack: ("skip", 0.2, "steady") if pack.track_id == 1 else ("look", 0.8, "flutter"))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": key}, clear=True):
                self.assertEqual(run(["--synthetic", "--out", str(out)], opener=jev), 0)
            page = write_glance_list(out).read_text()
        self.assertIn("Check this one tomorrow.", page)
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
            page = (out / PAGE_NAME).read_text()
            thumbs = sorted(p.name for p in (out / "thumbs").iterdir())
            self.assertEqual(n, 8)
            self.assertEqual(len(load_frames(out / "frames.json")), 8)
            self.assertIn('<video id="replay" src="annotated.mp4"', page)
            self.assertIn("Not sorted yet", page)
            self.assertEqual(thumbs, ["sheep-1.jpg", "sheep-7.jpg"])
            self.assertIn(f'src="{thumb_path(7)}"', page)
            self.assertEqual(cv2.imread(str(out / thumb_path(7))).shape, (240, 320, 3))

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


class _Resp:
    """The context-managed response object urllib's opener returns."""

    def __init__(self, payload: Any):
        self._raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _Resp:  # noqa: PYI034 - typing.Self needs 3.11; these tests run on older stdlib Pythons
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _choice_answer(picked: str, options: tuple[str, ...], p: float = 0.9) -> dict[str, Any]:
    rest = (1.0 - p) / (len(options) - 1)
    return {
        "type": "choice",
        "choice": picked,
        "confidence": 0.8,
        "probabilities": {o: (p if o == picked else rest) for o in options},
    }


def _track_answers(triage: str, noul: float, reason: str) -> dict[str, Any]:
    return {
        TRIAGE_ID: _choice_answer(triage, DECISIONS),
        LOOK_ID: {"type": "noul", "noul": noul},
        REASON_ID: _choice_answer(reason, JEV_REASONS),
    }


Rule = Callable[[EarPack], tuple[str, float, str]]


def book_rule(pack: EarPack) -> tuple[str, float, str]:
    verdict = book_triage(pack)
    return verdict.decision, 0.9 if verdict.decision == "look" else 0.1, verdict.reason


class FakeJev:
    """Offline stand-in for the System One endpoint. Answers each track from ``rule(pack)``."""

    def __init__(
        self,
        rule: Rule = book_rule,
        *,
        pen: str = "later",
        fail_track_ids: tuple[int, ...] = (),
        fail_pen: bool = False,
    ):
        self.rule = rule
        self.pen = pen
        self.fail_track_ids = set(fail_track_ids)
        self.fail_pen = fail_pen
        self.bodies: list[dict[str, Any]] = []

    def _fail(self, req: Any, code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(req.full_url, code, "fake", hdrs=None, fp=io.BytesIO(b"fake failure"))  # type: ignore[arg-type]

    def __call__(self, req: Any, timeout: float = 0) -> _Resp:
        body = json.loads(req.data.decode())
        self.bodies.append(body)
        if PEN_ID in body["questions"]:
            if self.fail_pen:
                raise self._fail(req, 529)
            answers = {PEN_ID: _choice_answer(self.pen, PEN_CALLS)}
        else:
            pack = EarPack.from_json(body["state"]["track"])
            if pack.track_id in self.fail_track_ids:
                raise self._fail(req, 500)
            answers = _track_answers(*self.rule(pack))
        return _Resp({"model": "jev-test", "answers": answers, "usage": {"input_tokens": 12, "output_tokens": 3}})

    @property
    def pen_calls(self) -> int:
        return sum(PEN_ID in body["questions"] for body in self.bodies)


def _head_at(
    cx: float,
    cy: float,
    left_deg: float,
    right_deg: float,
    *,
    nose_conf: float = 1.0,
    right_conf: float = 1.0,
) -> dict[str, Any]:
    """Head facing up the image. An ear at 90° sticks straight out, 0° points at the nose, 180° straight back."""
    lb, rb = [cx - 30.0, cy, 1.0], [cx + 30.0, cy, 1.0]

    def tip(base: list[float], deg: float, sign: int, conf: float) -> list[float]:
        rad = math.radians(deg)
        return [base[0] + sign * 50 * math.sin(rad), base[1] - 50 * math.cos(rad), conf]

    return {
        "box": [cx - 90, cy - 80, cx + 90, cy + 60],
        "kpts": [[cx, cy - 60, nose_conf], lb, rb, tip(lb, left_deg, -1, 1.0), tip(rb, right_deg, 1, right_conf)],
    }


def spfes_frames() -> list[dict[int, dict[str, Any]]]:
    """Three seconds, one sheep per SPFES ear case, no video. Each case is named by its book verdict.

    #3 skip/steady (and walks across the frame), #5 look/flutter, #8 look/asymmetry,
    #12 look/carriage, #15 cannot/not_facing, #21 look/one_ear_missing, #99 a one-frame flicker.
    """
    frames: list[dict[int, dict[str, Any]]] = []
    for i in range(90):
        jitter = (-2.0, 0.0, 2.0)[i % 3]
        frame: dict[int, dict[str, Any]] = {}
        frame[3] = _head_at(120 + 4 * i, 520, 92 + jitter, 88 - jitter)
        if i < 75:
            frame[5] = _head_at(180, 220, (55.0, 125.0)[(i // 3) % 2], 95 + jitter)
        if i >= 15:
            frame[8] = _head_at(480, 220, 90 + jitter, 128 + jitter)
        if i >= 30:
            frame[12] = _head_at(780, 220, 150 + jitter, 150 - jitter)
        if i < 60:
            frame[15] = _head_at(820, 520, 90, 90, nose_conf=1.0 if i % 5 == 0 else 0.0)
        if i >= 45:
            frame[21] = _head_at(1110, 520, 95 + jitter, 95, right_conf=0.0)
        if i == 10:
            frame[99] = _head_at(1110, 220, 90, 90)
        frames.append(frame)
    return frames


SPFES_EXPECTED = {
    3: Verdict("skip", "steady"),
    5: Verdict("look", "flutter"),
    8: Verdict("look", "asymmetry"),
    12: Verdict("look", "carriage"),
    15: Verdict("cannot", "not_facing"),
    21: Verdict("look", "one_ear_missing"),
}


def spfes_run_dir(root: Path) -> Path:
    """A --demo-style run folder (frames.json + tracks.json) for --from-run, from the SPFES fixture."""
    src = root / "spfes-source"
    src.mkdir()
    save_frames(src / "frames.json", spfes_frames())
    (src / "tracks.json").write_text(json.dumps({"clip": "synthetic", "weights": None, "fps": 30.0, "tracks": []}))
    return src


def _pack(
    *,
    facing: float = 1.0,
    left: tuple[float, float | None, float | None] = (1.0, 90.0, 2.0),
    right: tuple[float, float | None, float | None] = (1.0, 90.0, 2.0),
    asymmetry: float | None = 3.0,
    n_frames: int = 90,
) -> EarPack:
    return EarPack(
        track_id=1,
        n_frames=n_frames,
        duration_s=round(n_frames / 30.0, 2),
        facing_fraction=facing,
        left_ear=EarStats(*left),
        right_ear=EarStats(*right),
        asymmetry_deg=asymmetry,
    )


def _obs(frame: int, head: dict[str, Any]) -> dict[str, Any]:
    return {"frame": frame, **head}


class FeaturePackTests(unittest.TestCase):
    def _pack_of(self, observations: list[dict[str, Any]]) -> EarPack:
        summary = summarize_track(1, observations, n_source_frames=len(observations), fps=30.0, kpt_conf=0.4)
        return EarPack.from_json(summary["ear_pack"])

    def test_synthetic_tracks(self):
        tracks, _ = summarize_frames(synthetic_frames()[0], fps=30.0, kpt_conf=0.4, min_frames=2)
        steady, jumpy = (EarPack.from_json(t["ear_pack"]) for t in tracks)
        self.assertEqual(steady.left_ear, EarStats(1.0, 90.0, 0.0))
        self.assertEqual(steady.right_ear, EarStats(1.0, 90.0, 0.0))
        self.assertEqual(steady.asymmetry_deg, 0.0)
        self.assertEqual((steady.n_frames, steady.duration_s, steady.facing_fraction), (8, 0.27, 1.0))
        self.assertEqual(jumpy.left_ear, EarStats(1.0, 45.0, 45.0))
        self.assertEqual(jumpy.right_ear, EarStats(0.0, None, None))
        self.assertIsNone(jumpy.asymmetry_deg)

    def test_asymmetry_is_per_frame_not_a_difference_of_medians(self):
        pairs = [(80, 100), (90, 90), (100, 140)]
        pack = self._pack_of([_obs(i, _head_at(300, 300, left, right)) for i, (left, right) in enumerate(pairs)])
        self.assertEqual(pack.left_ear.median_deg, 90.0)
        self.assertEqual(pack.right_ear.median_deg, 100.0)
        self.assertEqual(pack.asymmetry_deg, 20.0)

    def test_measured_fraction_counts_facing_frames_only(self):
        heads = [_head_at(300, 300, 90, 90, nose_conf=1.0 if i % 2 else 0.0) for i in range(8)]
        heads[1] = _head_at(300, 300, 90, 90, right_conf=0.0)
        pack = self._pack_of([_obs(i, h) for i, h in enumerate(heads)])
        self.assertEqual(pack.facing_fraction, 0.5)
        self.assertEqual(pack.left_ear.measured_fraction, 1.0)
        self.assertEqual(pack.right_ear.measured_fraction, 0.75)

    def test_too_few_measured_frames_leave_the_numbers_out(self):
        heads = [_head_at(300, 300, 90, 120) for _ in range(MIN_EAR_FRAMES - 1)]
        pack = self._pack_of([_obs(i, h) for i, h in enumerate(heads)])
        self.assertEqual(pack.left_ear, EarStats(1.0, None, None))
        self.assertIsNone(pack.asymmetry_deg)

    def test_pack_is_ear_only(self):
        tracks, _ = summarize_frames(spfes_frames(), fps=30.0, kpt_conf=0.4, min_frames=2)
        for track in tracks:
            self.assertEqual(
                set(track["ear_pack"]),
                {"track_id", "n_frames", "duration_s", "facing_fraction", "left_ear", "right_ear", "asymmetry_deg"},
            )
            self.assertEqual(set(track["ear_pack"]["left_ear"]), {"measured_fraction", "median_deg", "std_deg"})

    def test_json_round_trip(self):
        pack = _pack(right=(0.2, None, None), asymmetry=None)
        self.assertEqual(EarPack.from_json(json.loads(json.dumps(pack.to_json()))), pack)

    def test_jev_state_is_the_pack_with_notes_and_no_motion(self):
        tracks, _ = summarize_frames(spfes_frames(), fps=30.0, kpt_conf=0.4, min_frames=2)
        state = state_for_jev(tracks[0])
        self.assertEqual(state["track"], tracks[0]["ear_pack"])
        blob = json.dumps(state).lower()
        for word in ("centroid", "motion", "coverage", "keypoint_visibility", "bbox", "speed", "path"):
            self.assertNotIn(word, blob)
        for field in ("median_deg", "std_deg", "asymmetry_deg", "facing_fraction", "measured_fraction"):
            self.assertIn(field, state["field_notes"])


class BookRuleTests(unittest.TestCase):
    def check(self, pack: EarPack, decision: str, reason: str) -> None:
        self.assertEqual(book_triage(pack), Verdict(decision, reason))  # type: ignore[arg-type]

    def test_steady_even_ears_are_skipped(self):
        self.check(_pack(), "skip", "steady")

    def test_not_facing_wins_over_every_sign(self):
        self.check(_pack(facing=0.49, left=(1.0, 160.0, 40.0)), "cannot", "not_facing")
        self.check(_pack(facing=0.5), "skip", "steady")

    def test_unmeasurable_ears_cannot_be_checked(self):
        self.check(_pack(left=(0.49, 90.0, 2.0), right=(0.2, 90.0, 2.0)), "cannot", "ears_unmeasurable")
        self.check(_pack(left=(1.0, None, None), right=(1.0, None, None), asymmetry=None), "cannot", "ears_unmeasurable")

    def test_flutter(self):
        self.check(_pack(right=(1.0, 90.0, 15.0)), "look", "flutter")
        self.check(_pack(right=(1.0, 90.0, 14.9)), "skip", "steady")

    def test_asymmetry(self):
        self.check(_pack(right=(1.0, 115.0, 2.0), asymmetry=25.0), "look", "asymmetry")
        self.check(_pack(right=(1.0, 115.0, 2.0), asymmetry=24.9), "skip", "steady")

    def test_carriage_band_is_inclusive(self):
        self.check(_pack(left=(1.0, 74.9, 2.0), right=(1.0, 80.0, 2.0)), "look", "carriage")
        self.check(_pack(left=(1.0, 140.1, 2.0), right=(1.0, 139.0, 2.0)), "look", "carriage")
        self.check(_pack(left=(1.0, 75.0, 2.0), right=(1.0, 140.0, 2.0), asymmetry=3.0), "skip", "steady")

    def test_one_ear_missing_while_facing(self):
        self.check(_pack(right=(0.1, None, None), asymmetry=None), "look", "one_ear_missing")

    def test_asymmetry_needs_both_ears_seen(self):
        self.check(_pack(right=(0.3, 150.0, 2.0), asymmetry=60.0), "look", "one_ear_missing")

    def test_first_matching_sign_names_the_reason(self):
        self.check(_pack(left=(1.0, 150.0, 20.0), asymmetry=60.0), "look", "flutter")
        self.check(_pack(left=(1.0, 150.0, 2.0), asymmetry=60.0), "look", "asymmetry")

    def test_brief_tracks_are_not_a_look(self):
        self.check(_pack(n_frames=4), "skip", "steady")

    def test_fixture_cases(self):
        tracks, dropped = summarize_frames(spfes_frames(), fps=30.0, kpt_conf=0.4, min_frames=2)
        self.assertEqual(dropped, [99])
        got = {t["track_id"]: book_triage(EarPack.from_json(t["ear_pack"])) for t in tracks}
        self.assertEqual(got, SPFES_EXPECTED)

    def test_reasons_always_fit_the_decision(self):
        for facing in (0.2, 1.0):
            for measured in (0.2, 1.0):
                for median in (60.0, 100.0, 150.0):
                    for std in (2.0, 30.0):
                        verdict = book_triage(_pack(facing=facing, left=(measured, median, std)))
                        self.assertIn(verdict.reason, REASONS_FOR[verdict.decision])


class JevQuestionTests(unittest.TestCase):
    def test_track_request_asks_three_typed_questions(self):
        body = request_body({"task": "t", "track": _pack().to_json()})
        questions = body["questions"]
        self.assertEqual({q: questions[q]["type"] for q in questions}, {"triage": "choice", "look": "noul", "reason": "choice"})
        self.assertEqual(tuple(questions[TRIAGE_ID]["criteria"]), DECISIONS)
        self.assertEqual(tuple(questions[REASON_ID]["criteria"]), JEV_REASONS)
        self.assertNotIn("not_sure", questions[REASON_ID]["criteria"])
        self.assertEqual(set(questions[LOOK_ID]["criteria"]), {"true", "false"})
        for question in questions.values():
            self.assertEqual(question["instructions"]["protocol"], SPFES_EAR_PROTOCOL)
        self.assertEqual(body["state"]["track"]["left_ear"]["median_deg"], 90.0)

    def test_protocol_names_the_ear_signs_and_what_is_not_a_sign(self):
        text = SPFES_EAR_PROTOCOL.lower()
        for phrase in ("spfes", "carriage", "asymmetry", "posture change", "cannot check", "not toward the camera"):
            self.assertIn(phrase, text)
        self.assertIn("walking, speed and a short time in view are not signs", text)
        self.assertNotIn("pain", json.dumps([TRACK_QUESTIONS, PEN_QUESTIONS]).lower())

    def test_pen_request(self):
        state = pen_state([Verdict("look", "flutter"), Verdict("look", "carriage"), Verdict("cannot", "not_facing")], failed=1)
        self.assertEqual(state["sheep_tracked"], 4)
        self.assertEqual(state["counts"], {"look": 2, "skip": 0, "cannot": 1, "check_failed": 1})
        self.assertEqual(state["look_reasons"], {"flutter": 1, "carriage": 1})
        self.assertEqual(state["cannot_reasons"], {"not_facing": 1})
        body = pen_request_body(state)
        self.assertEqual(tuple(body["questions"][PEN_ID]["criteria"]), PEN_CALLS)
        self.assertEqual(body["questions"][PEN_ID]["type"], "choice")


def _ask(answers: dict[str, Any], *, threshold: float = 0.5) -> Any:
    return triage_track(
        {"track": _pack().to_json()},
        api_key="sk-test",
        threshold=threshold,
        opener=lambda req, timeout=0: _Resp({"model": "jev-test", "answers": answers}),
    )


class JevResponseTests(unittest.TestCase):
    def test_typed_class_decides_and_noul_only_gates_a_look(self):
        cases = [
            # (typed class, noul, jev reason) -> (decision, reason, gated)
            (("look", 0.8, "flutter"), ("look", "flutter", False)),
            (("look", 0.5, "carriage"), ("look", "carriage", False)),
            (("look", 0.49, "flutter"), ("cannot", "not_sure", True)),
            (("look", 0.05, "asymmetry"), ("cannot", "not_sure", True)),
            (("skip", 0.99, "steady"), ("skip", "steady", False)),
            (("skip", 0.05, "steady"), ("skip", "steady", False)),
            (("cannot", 0.99, "not_facing"), ("cannot", "not_facing", False)),
            (("cannot", 0.02, "ears_unmeasurable"), ("cannot", "ears_unmeasurable", False)),
        ]
        for (typed, noul, reason), (decision, final_reason, gated) in cases:
            with self.subTest(typed=typed, noul=noul):
                out = _ask(_track_answers(typed, noul, reason))
                self.assertEqual((out.typed_class, out.decision, out.reason, out.gated), (typed, decision, final_reason, gated))

    def test_the_gate_moves_with_the_threshold(self):
        answers = _track_answers("look", 0.8, "flutter")
        self.assertEqual(_ask(answers, threshold=0.8).decision, "look")
        self.assertEqual(_ask(answers, threshold=0.9).decision, "cannot")
        self.assertEqual(_ask(_track_answers("skip", 0.95, "steady"), threshold=0.0).decision, "skip")

    def test_reason_is_bent_to_fit_the_decision(self):
        answer = ChoiceAnswer("steady", 0.4, {"steady": 0.5, "carriage": 0.3, "flutter": 0.1, "not_facing": 0.1})
        self.assertEqual(fitting_reason("look", answer), "carriage")
        self.assertEqual(fitting_reason("skip", answer), "steady")
        self.assertEqual(fitting_reason("cannot", answer), "not_facing")
        self.assertEqual(fitting_reason("cannot", ChoiceAnswer("steady", 1.0, {"steady": 1.0})), "not_facing")

    def test_record_keeps_both_answers_for_the_ab(self):
        out = _ask(_track_answers("cannot", 0.7, "not_facing")).to_json()
        self.assertEqual((out["decision"], out["reason"], out["noul"]), ("cannot", "not_facing", 0.7))
        self.assertEqual(out["choice"]["choice"], "cannot")
        self.assertAlmostEqual(sum(out["choice"]["probabilities"].values()), 1.0)
        self.assertEqual(set(out["reason_choice"]["probabilities"]), set(JEV_REASONS))

    def test_record_puts_typed_class_noul_and_final_call_side_by_side(self):
        out = _ask(_track_answers("look", 0.3, "carriage")).to_json()
        self.assertEqual(list(out)[:6], ["typed_class", "noul", "threshold", "decision", "gated", "reason"])
        self.assertEqual(
            (out["typed_class"], out["noul"], out["decision"], out["gated"], out["reason"]),
            ("look", 0.3, "cannot", True, "not_sure"),
        )
        self.assertEqual(out["reason_choice"]["choice"], "carriage")

    def test_malformed_answers_are_errors_not_defaults(self):
        good = _track_answers("look", 0.8, "flutter")
        broken = {
            "no triage": {k: v for k, v in good.items() if k != TRIAGE_ID},
            "no reason": {k: v for k, v in good.items() if k != REASON_ID},
            "unknown option": {**good, TRIAGE_ID: {**good[TRIAGE_ID], "choice": "maybe"}},
            "noul above one": {**good, LOOK_ID: {"type": "noul", "noul": 1.3}},
            "noul as text": {**good, LOOK_ID: {"type": "noul", "noul": "0.8"}},
            "noul not finite": {**good, LOOK_ID: {"type": "noul", "noul": float("nan")}},
            "wrong type": {**good, LOOK_ID: good[TRIAGE_ID]},
            "no confidence": {**good, REASON_ID: {k: v for k, v in good[REASON_ID].items() if k != "confidence"}},
            "no probabilities": {**good, REASON_ID: {"type": "choice", "choice": "flutter", "confidence": 1.0}},
            "gate reason from jev": {**good, REASON_ID: {**good[REASON_ID], "choice": "not_sure"}},
        }
        for name, answers in broken.items():
            with self.subTest(name), self.assertRaises(JevError):
                _ask(answers)

    def test_transport_failures_are_jev_errors(self):
        def not_json(req, timeout=0):
            return _Resp(b"<html>502</html>")

        def too_slow(req, timeout=0):
            raise TimeoutError

        def reset(req, timeout=0):
            raise ConnectionResetError("peer reset")

        for opener in (not_json, too_slow, reset):
            with self.subTest(opener.__name__), self.assertRaises(JevError):
                triage_track({}, api_key="sk-test", opener=opener)
        with self.assertRaises(JevError):
            triage_track({}, api_key="", opener=not_json)

    def test_pen_call(self):
        answers = {PEN_ID: _choice_answer("walk_tomorrow", PEN_CALLS)}
        pen = triage_pen({}, api_key="sk-test", opener=lambda req, timeout=0: _Resp({"answers": answers}))
        self.assertEqual(pen.call, "walk_tomorrow")
        self.assertEqual(pen.to_json()["call"], "walk_tomorrow")
        for stale in ("panic", "walk_now"):
            answers = {PEN_ID: {**_choice_answer("walk_tomorrow", PEN_CALLS), "choice": stale}}
            with self.subTest(stale), self.assertRaises(JevError):
                triage_pen({}, api_key="sk-test", opener=lambda req, timeout=0: _Resp({"answers": answers}))

    def test_pen_question_is_framed_for_the_next_morning(self):
        question = PEN_QUESTIONS[PEN_ID]
        self.assertEqual(tuple(question["criteria"]), ("walk_tomorrow", "later", "fine"))
        text = json.dumps(question).lower()
        self.assertIn("overnight", text)
        self.assertIn("tomorrow", text)
        self.assertNotIn("walk the pen now", text)
        self.assertIn("overnight", pen_state([Verdict("look", "flutter")])["task"])


class SpfesBadgeTests(unittest.TestCase):
    def test_new_vocabulary(self):
        self.assertEqual(badge_for("skip"), SKIP)
        self.assertEqual(badge_for("cannot"), UNCHECKED)
        triage = {"tracks": [{"track_id": 1, "decision": "cannot"}, {"track_id": 2, "decision": "skip"}]}
        self.assertEqual(badges_from_triage(triage), {1: UNCHECKED, 2: SKIP})

    def test_overlay_header_counts_not_checked_once_the_check_ran(self):
        self.assertEqual(
            header_parts({1: LOOK, 2: UNCHECKED}),
            [("LOOK 1", LOOK), ("SKIP 0", SKIP), ("NOT CHECKED 1", UNCHECKED)],
        )
        self.assertEqual(
            header_parts({1: UNCHECKED, 2: UNCHECKED}, ran=True),
            [("LOOK 0", LOOK), ("SKIP 0", SKIP), ("NOT CHECKED 2", UNCHECKED)],
        )
        self.assertEqual(header_parts({1: UNCHECKED}, ran=False), [("NOT SORTED YET", UNCHECKED)])


def _spfes_docs(rows: dict[int, dict[str, Any]], *, pen: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
    tracks, _ = summarize_frames(spfes_frames(), fps=30.0, kpt_conf=0.4, min_frames=2)
    tracks_doc = {"clip": "synthetic", "fps": 30.0, "n_frames": 90, "dropped_track_ids": [99], "tracks": tracks}
    triage = {
        "mode": "jev",
        "pen": pen,
        "tracks": [{"track_id": t["track_id"], "noul": 0.5, **rows.get(t["track_id"], {})} for t in tracks],
    }
    return tracks_doc, triage


class FarmerReasonTests(unittest.TestCase):
    def test_every_reason_reads_plainly_with_its_number(self):
        cases = {
            ("flutter", _pack(left=(1.0, 90.0, 31.0))): "Left ear kept changing position — its angle varied by about 31°.",
            ("flutter", _pack(left=(1.0, 90.0, 31.0), right=(1.0, 90.0, 18.0))): (
                "Both ears kept changing position — their angles varied by about 31°."
            ),
            ("asymmetry", _pack(asymmetry=38.2)): "Ears held unevenly — left and right differed by about 38°.",
            ("carriage", _pack(right=(1.0, 151.0, 2.0))): "Right ear held unusually far back (about 151°).",
            ("carriage", _pack(left=(1.0, 62.0, 2.0))): "Left ear held unusually far forward (about 62°).",
            ("one_ear_missing", _pack(right=(0.1, None, None))): (
                "Facing the camera, but its right ear was hard to see on most frames."
            ),
            ("steady", _pack()): "Both ears seen, even and steady.",
            ("not_facing", _pack(facing=0.1)): "Couldn't see its face well enough — it was mostly turned away.",
            ("ears_unmeasurable", _pack(left=(0.1, None, None))): "Couldn't see its ears well enough to check.",
            ("not_sure", _pack(left=(1.0, 90.0, 31.0))): "Not sure enough to call — the ear signs were weak or mixed.",
        }
        for (reason, pack), expected in cases.items():
            with self.subTest(reason):
                self.assertEqual(farmer_reason(reason, pack), expected)  # type: ignore[arg-type]

    def test_reasons_without_numbers_still_read(self):
        empty = _pack(left=(0.0, None, None), right=(0.0, None, None), asymmetry=None)
        for reason in REASONS:
            with self.subTest(reason):
                text = farmer_reason(reason, empty)
                self.assertTrue(text.endswith("."))
                self.assertNotIn("None", text)


class SpfesGlanceListTests(unittest.TestCase):
    def rows(self) -> dict[int, dict[str, Any]]:
        return {tid: v.to_json() for tid, v in SPFES_EXPECTED.items()}

    def test_cannot_is_not_checked_with_a_plain_reason(self):
        tracks_doc, triage = _spfes_docs(self.rows())
        cards = {c["track_id"]: c for c in build_cards(tracks_doc, triage)}
        self.assertEqual(cards[15]["badge"], UNCHECKED)
        self.assertEqual(cards[15]["status"], "cannot")
        self.assertEqual(cards[15]["reason"], "Couldn't see its face well enough — it was mostly turned away.")
        self.assertEqual(cards[3]["reason"], "Both ears seen, even and steady.")
        self.assertIn("kept changing position", cards[5]["reason"])
        page = render_html(tracks_doc, triage, video=None)
        self.assertIn("Check these 4 tomorrow.", page)
        self.assertIn("1 not checked · 1 looked fine", rest_section(page))
        self.assertIn("Couldn't see its face well enough — it was mostly turned away.", rest_section(page))
        self.assertNotIn("The ear check failed", page)

    def test_all_cannot_says_why(self):
        tracks_doc, triage = _spfes_docs({tid: Verdict("cannot", "not_facing").to_json() for tid in SPFES_EXPECTED})
        self.assertEqual(
            headline(build_cards(tracks_doc, triage)),
            "Couldn't check any of the 6 sheep — their faces or ears weren't clear enough.",
        )
        unsure = {tid: Verdict("cannot", "not_sure").to_json() for tid in SPFES_EXPECTED}
        self.assertEqual(
            headline(build_cards(*_spfes_docs(unsure))),
            "Couldn't check any of the 6 sheep, so there's no list for tomorrow.",
        )

    def test_pen_line(self):
        for call, text in (
            ("walk_tomorrow", "Walk the pen first thing tomorrow."),
            ("later", "No rush — your next routine walk-through is soon enough."),
            ("fine", "The pen looks fine — no special walk needed."),
        ):
            with self.subTest(call):
                tracks_doc, triage = _spfes_docs(self.rows(), pen={"call": call, "confidence": 0.9})
                self.assertEqual(pen_line(triage), (call, text))
                self.assertIn(text, main_view(render_html(tracks_doc, triage, video=None)))
        for pen in (None, {"call": None, "error": "Jev HTTP 529"}, {"call": "panic"}):
            with self.subTest(pen=pen):
                tracks_doc, triage = _spfes_docs(self.rows(), pen=pen)
                self.assertIsNone(pen_line(triage))
                self.assertNotIn('class="pen', render_html(tracks_doc, triage, video=None))

    def test_old_walk_now_runs_read_as_tomorrow(self):
        tracks_doc, triage = _spfes_docs(self.rows(), pen={"call": "walk_now", "confidence": 0.9})
        self.assertEqual(pen_line(triage), ("walk_tomorrow", "Walk the pen first thing tomorrow."))
        page = render_html(tracks_doc, triage, video=None)
        self.assertIn('<section class="summary pen-walk-tomorrow">', page)
        self.assertIsNone(re.search(r"\bnow\b", visible_text(main_view(page)), flags=re.IGNORECASE))

    def test_page_keeps_the_vocabulary_internal(self):
        tracks_doc, triage = _spfes_docs(self.rows(), pen={"call": "walk_now"})
        page = render_html(tracks_doc, triage, video="annotated.mp4").lower()
        for word in ("cannot", "noul", "jev", "spfes", "walk_now", "one_ear_missing", "not_facing", "decision"):
            self.assertNotIn(word, page)
        markup = re.sub(r"<(style|script)>.*?</\1>", "", page, flags=re.DOTALL)
        self.assertNotIn("{", markup)


def _brief_rows(decisions: dict[int, tuple[str, float | None, str]]) -> dict[int, dict[str, Any]]:
    return {tid: {"decision": d, "noul": noul, "reason": r} for tid, (d, noul, r) in decisions.items()}


def _only(decisions: dict[int, tuple[str, float | None, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The SPFES fixture cut down to just these sheep."""
    tracks_doc, triage = _spfes_docs(_brief_rows(decisions))
    tracks_doc["tracks"] = [t for t in tracks_doc["tracks"] if t["track_id"] in decisions]
    triage["tracks"] = [r for r in triage["tracks"] if r["track_id"] in decisions]
    return tracks_doc, triage


MIXED = {
    3: ("skip", 0.05, "steady"),
    5: ("look", 0.71, "flutter"),
    8: ("look", 0.93, "asymmetry"),
    12: ("look", 0.86, "carriage"),
    15: ("cannot", 0.9, "not_facing"),
    21: ("look", 0.71, "one_ear_missing"),
}


class MorningBriefTests(unittest.TestCase):
    def page(self, decisions=MIXED, **kwargs: Any) -> str:
        tracks_doc, triage = _spfes_docs(_brief_rows(decisions), pen=kwargs.pop("pen", None))
        return render_html(tracks_doc, triage, video=kwargs.pop("video", "annotated.mp4"), **kwargs)

    def test_look_cards_are_ranked_most_clearly_flagged_first(self):
        cards = build_cards(*_spfes_docs(_brief_rows(MIXED)))
        looks = [(c["track_id"], c["rank"]) for c in cards if c["badge"] == LOOK]
        # 8 (0.93), 12 (0.86), then the 0.71 tie in the order they were first seen: 5 at 0:00, 21 at 0:01.
        self.assertEqual(looks, [(8, 1), (12, 2), (5, 3), (21, 4)])
        self.assertTrue(all(c["rank"] is None for c in cards if c["badge"] != LOOK))
        checklist = re.search(r'<ol class="checklist">.*?</ol>', self.page(), flags=re.DOTALL).group(0)
        self.assertEqual(re.findall(r"<h2>Sheep #(\d+)</h2>", checklist), ["8", "12", "5", "21"])
        ranks = re.findall(r'<span class="rank" aria-hidden="true">(\d)</span>', checklist)
        self.assertEqual(ranks, ["1", "2", "3", "4"])

    def test_a_look_without_a_strength_goes_last(self):
        decisions = {**MIXED, 5: ("look", None, "flutter")}
        looks = [c["track_id"] for c in build_cards(*_spfes_docs(_brief_rows(decisions))) if c["badge"] == LOOK]
        self.assertEqual(looks, [8, 12, 21, 5])

    def test_main_view_is_the_date_the_count_and_look_cards_only(self):
        page = self.page(pen={"call": "later"}, day=date(2026, 9, 24), pen_label="North pen")
        top = main_view(page)
        self.assertLess(top.index("Thursday 24 September"), top.index("Check these 4 tomorrow."))
        self.assertLess(top.index("Check these 4 tomorrow."), top.index('<ol class="checklist">'))
        self.assertEqual(sorted(re.findall(r"Sheep #(\d+)", visible_text(top))), ["12", "21", "5", "8"])
        for word in ("looked fine", "not checked", "skip"):
            self.assertNotIn(word, visible_text(top).lower())

    def test_skip_and_not_checked_sit_in_a_collapsed_section_after_the_list(self):
        page = self.page()
        rest = rest_section(page)
        self.assertTrue(rest.startswith('<details class="rest">'), rest[:40])
        self.assertNotIn("open", rest.split(">")[0])
        self.assertLess(page.index('<ol class="checklist">'), page.index('<details class="rest">'))
        self.assertIn("1 not checked · 1 looked fine", rest)
        self.assertLess(rest.index("<h3>Not checked</h3>"), rest.index("<h3>Looked fine</h3>"))
        self.assertLess(rest.index("Sheep #15"), rest.index("Sheep #3<"))
        self.assertIn("Couldn't see its face well enough", rest)
        self.assertIn("Both ears seen, even and steady.", rest)

    def test_nothing_to_check(self):
        all_fine = {tid: ("skip", 0.05, "steady") for tid in MIXED}
        page = self.page(all_fine, pen={"call": "fine"})
        self.assertIn("Nothing needs a check tomorrow.", page)
        self.assertIn("All 6 sheep looked fine.", page)
        self.assertIn("The pen looks fine — no special walk needed.", page)
        self.assertNotIn('<ol class="checklist">', page)
        self.assertIn("6 looked fine", rest_section(page))

        one = build_cards(*_only({3: ("skip", 0.05, "steady")}))
        self.assertEqual(headline(one), "Nothing needs a check tomorrow.")
        self.assertEqual(detail(one), "The one sheep seen looked fine.")

        unsure = {
            **all_fine,
            **{tid: ("cannot", 0.4, "not_sure") for tid in (5, 8, 12, 21)},
            15: ("cannot", 0.9, "not_facing"),
        }
        mixed_empty = self.page(unsure)
        self.assertIn("Nothing flagged for tomorrow.", mixed_empty)
        self.assertIn("1 looked fine and 5 couldn't be checked — see below.", mixed_empty)
        self.assertNotIn("Nothing needs a check", mixed_empty)

    def test_no_sheep_at_all(self):
        tracks_doc = {"clip": "synthetic", "fps": 30.0, "n_frames": 90, "tracks": []}
        page = render_html(tracks_doc, {"mode": "jev", "tracks": []}, video=None)
        self.assertIn("No sheep were seen in this footage.", page)
        self.assertEqual(rest_section(page), "")
        self.assertNotIn('<ol class="checklist">', page)

    def test_one_sheep_all_look(self):
        cards = build_cards(*_only({5: ("look", 0.9, "flutter")}))
        self.assertEqual(headline(cards), "Check the one sheep tomorrow.")
        self.assertIsNone(detail(cards))
        partial = build_cards(*_spfes_docs(_brief_rows({5: ("look", 0.9, "flutter")})))
        self.assertEqual(headline(partial), "Check this one tomorrow.")

    def test_no_engine_words_numbers_paths_or_diagnosis_in_any_state(self):
        tracks_doc, triage = _docs({1: "dont", 4: "look"}, errors=(9,))
        states = {
            "mixed": self.page(pen={"call": "walk_tomorrow"}, day=date(2026, 9, 24), pen_label="North pen"),
            "old walk_now": self.page(pen={"call": "walk_now"}),
            "all look": self.page({tid: ("look", 0.9, "flutter") for tid in MIXED}),
            "all fine": self.page({tid: ("skip", 0.05, "steady") for tid in MIXED}, pen={"call": "fine"}),
            "all unsure": self.page({tid: ("cannot", 0.3, "not_sure") for tid in MIXED}, pen={"call": "later"}),
            "no video": self.page(video=None),
            "thumbs": self.page(thumbs={tid: thumb_path(tid) for tid in MIXED}),
            "failed": render_html(tracks_doc, triage, video="annotated.mp4"),
            "pose-only": render_html(*_docs({}, mode="pose-only"), video=None),
        }
        for name, page in states.items():
            with self.subTest(name):
                text = visible_text(page).lower()
                for word in FORBIDDEN_TEXT:
                    self.assertNotIn(word, text, f"{word!r} is visible")
                for word in FORBIDDEN_MARKUP:
                    self.assertNotIn(word, page.lower(), f"{word!r} is in the markup")
                self.assertNotIn("{", re.sub(r"<(style|script)>.*?</\1>", "", page, flags=re.DOTALL))
                self.assertNotIn("@import", page)
                self.assertNotIn("0.9", text)

    def test_date_and_pen_label(self):
        page = self.page(day=date(2026, 9, 24), pen_label="North pen")
        self.assertIn('<h1><time datetime="2026-09-24">Thursday 24 September</time></h1>', page)
        self.assertIn('<p class="eyebrow">North pen · Morning brief</p>', page)
        self.assertIn("<title>Thursday 24 September — North pen — morning brief</title>", page)
        self.assertIn("3 s of footage, sorted overnight", page)

        undated = self.page()
        self.assertIn("<h1><time>Morning brief</time></h1>", undated)
        self.assertIn("Your flock · Morning brief", undated)

        tracks_doc, triage = _spfes_docs(_brief_rows(MIXED))
        tracks_doc.update({"footage_date": "2026-03-01", "pen_label": "Lambing shed"})
        page = render_html(tracks_doc, triage, video=None)
        self.assertIn("Sunday 1 March", page)
        self.assertIn("Lambing shed · Morning brief", page)
        overridden = render_html(tracks_doc, triage, video=None, day=date(2026, 3, 2), pen_label="Top field")
        self.assertIn("Monday 2 March", overridden)
        self.assertIn("Top field · Morning brief", overridden)
        tracks_doc["footage_date"] = "last tuesday"
        self.assertIn("<h1><time>Morning brief</time></h1>", render_html(tracks_doc, triage, video=None))

    def test_watch_buttons_open_a_hidden_player(self):
        page = self.page()
        self.assertIn('<div class="player" id="player" hidden>', page)
        self.assertIn("<script>", page)
        self.assertIn('data-t="1.00" data-label="Sheep #12 · from 0:01">Watch from 0:01</a>', page)
        self.assertIn('data-label="Sheep #3 · from 0:00">Watch</a>', rest_section(page))
        self.assertIn('href="annotated.mp4" data-t="0" data-label="Full replay">Watch the full replay</a>', page)

    def test_thumbnails_only_when_the_replay_made_them(self):
        page = self.page(thumbs={8: thumb_path(8), 3: thumb_path(3)})
        self.assertIn('<img class="thumb" src="thumbs/sheep-8.jpg" alt="Sheep #8"', page)
        self.assertIn('<img class="mini" src="thumbs/sheep-3.jpg" alt="Sheep #3"', page)
        self.assertEqual(page.count("<img"), 2)
        self.assertEqual(page.count('class="card look no-thumb"'), 3)


def _write_run(root: Path, tracks_doc: dict[str, Any], triage: dict[str, Any], *, mtime: date | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "tracks.json").write_text(json.dumps(tracks_doc))
    (root / "triage.json").write_text(json.dumps(triage))
    if mtime is not None:
        stamp = datetime.combine(mtime, datetime.min.time()).replace(hour=12).timestamp()
        os.utime(root / "tracks.json", (stamp, stamp))
    return root


class BriefRunFolderTests(unittest.TestCase):
    def test_old_run_folders_still_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Before the SPFES check: look / dont, no reasons, no ear pack, no pen call, no date.
            tracks_doc, triage = _docs({1: "dont", 4: "look", 9: "look", 12: "dont"})
            old = _write_run(Path(tmp) / "pr2", tracks_doc, triage, mtime=date(2026, 9, 20))
            (old / "annotated.mp4").write_bytes(b"")
            page = write_glance_list(old).read_text()
            # Before the overnight framing: walk_now, and a glance list beside it.
            spfes_tracks, spfes_triage = _spfes_docs(_brief_rows(MIXED), pen={"call": "walk_now"})
            pr3 = _write_run(Path(tmp) / "pr3", spfes_tracks, spfes_triage, mtime=date(2026, 9, 25))
            (pr3 / "glance-list.html").write_text("old page")
            pr3_page = write_glance_list(pr3).read_text()
            names = sorted(p.name for p in pr3.iterdir())
            self.assertEqual(names, ["glance-list.html", PAGE_NAME, "tracks.json", "triage.json"])
        self.assertIn("Sunday 20 September", page)
        self.assertIn("Check these 2 tomorrow.", page)
        self.assertIn("Your flock · Morning brief", page)
        self.assertIn('<video id="replay" src="annotated.mp4"', page)
        self.assertIn("2 looked fine", rest_section(page))
        self.assertIn("Friday 25 September", pr3_page)
        self.assertIn("Walk the pen first thing tomorrow.", pr3_page)

    def test_rebuild_takes_pen_and_date_flags(self):
        from glance_list import main as rebuild

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _write_run(Path(tmp) / "run", *_spfes_docs(_brief_rows(MIXED)))
            thumbs = run_dir / "thumbs"
            thumbs.mkdir()
            (run_dir / thumb_path(12)).write_bytes(b"")
            with mock.patch("sys.stdout", io.StringIO()):
                self.assertEqual(rebuild([str(run_dir), "--pen", "North pen", "--date", "2026-09-24"]), 0)
            page = (run_dir / PAGE_NAME).read_text()
            with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
                rebuild([str(run_dir), "--date", "24/09/2026"])
        self.assertIn("Thursday 24 September", page)
        self.assertIn("North pen · Morning brief", page)
        self.assertEqual(page.count("<img"), 1)

    def test_pipeline_records_the_date_and_pen(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            out = Path(tmp) / "run"
            args = ["--synthetic", "--pose-only", "--out", str(out), "--pen", " North pen ", "--date", "2026-09-24"]
            self.assertEqual(run(args), 0)
            tracks_doc = json.loads((out / "tracks.json").read_text())
            summary = (out / "summary.txt").read_text()
            default = Path(tmp) / "default"
            self.assertEqual(run(["--synthetic", "--pose-only", "--out", str(default)]), 0)
            default_doc = json.loads((default / "tracks.json").read_text())
            with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
                run(["--synthetic", "--pose-only", "--out", str(out), "--date", "tomorrow"])
        self.assertEqual((tracks_doc["footage_date"], tracks_doc["pen_label"]), ("2026-09-24", "North pen"))
        self.assertEqual(tracks_doc["frame_rotation"], 0)
        self.assertIn("footage date: 2026-09-24  pen label: North pen", summary)
        self.assertEqual((default_doc["footage_date"], default_doc["pen_label"]), (date.today().isoformat(), None))

    def test_from_run_keeps_the_earlier_date_and_pen_unless_told_otherwise(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            src = spfes_run_dir(Path(tmp))
            stamp = datetime(2026, 9, 21, 12).timestamp()
            os.utime(src / "tracks.json", (stamp, stamp))
            first, again, moved = (Path(tmp) / name for name in ("first", "again", "moved"))
            self.assertEqual(run(["--from-run", str(src), "--pose-only", "--out", str(first), "--pen", "North pen"]), 0)
            first_doc = json.loads((first / "tracks.json").read_text())
            shutil.copy(src / "frames.json", first / "frames.json")  # as a --demo run would have kept it
            self.assertEqual(run(["--from-run", str(first), "--pose-only", "--out", str(again)]), 0)
            again_doc = json.loads((again / "tracks.json").read_text())
            redated = ["--from-run", str(first), "--pose-only", "--out", str(moved), "--date", "2026-09-22"]
            self.assertEqual(run(redated), 0)
            moved_doc = json.loads((moved / "tracks.json").read_text())
        self.assertEqual((first_doc["footage_date"], first_doc["pen_label"]), ("2026-09-21", "North pen"))
        self.assertEqual((again_doc["footage_date"], again_doc["pen_label"]), ("2026-09-21", "North pen"))
        self.assertEqual((moved_doc["footage_date"], moved_doc["pen_label"]), ("2026-09-22", "North pen"))


class ThumbnailPickTests(unittest.TestCase):
    def test_picks_a_frame_with_the_face_toward_the_camera(self):
        frames = spfes_frames()
        picks = thumb_frames(frames, {3, 5, 15, 21}, 0.4)
        self.assertEqual(set(picks), {3, 5, 15, 21})
        self.assertEqual(picks[15] % 5, 0)  # sheep 15 shows its face on every fifth frame only
        self.assertEqual(picks[3], 44)  # all frames equal: the one nearest the middle of 0..89
        self.assertNotIn(99, thumb_frames(frames, {3}, 0.4))

    def test_crop_is_four_by_three_and_stays_in_frame(self):
        boxes = (
            ([100, 100, 280, 240], 1.0),
            ([0, 0, 60, 40], 1.0),
            ([1180, 560, 1240, 620], 1.0),
            ([0, 0, 2000, 900], 0.5),
        )
        for box, scale in boxes:
            with self.subTest(box=box):
                left, top, right, bottom = crop_box(box, scale, 1240, 620)
                self.assertGreaterEqual(left, 0)
                self.assertGreaterEqual(top, 0)
                self.assertLessEqual(right, 1240)
                self.assertLessEqual(bottom, 620)
                self.assertAlmostEqual((right - left) / (bottom - top), 4 / 3, delta=0.02)
        left, top, right, bottom = crop_box([100, 100, 280, 240], 1.0, 1240, 620)
        self.assertLessEqual(left, 100)
        self.assertGreaterEqual(right, 280)


class SpfesPipelineTests(unittest.TestCase):
    def run_from_fixture(self, tmp: str, *extra: str, jev: FakeJev | None = None, key: str | None = "sk-fixture") -> Path:
        out = Path(tmp) / "out"
        env = {"TYPESAFE_API_KEY": key} if key else {}
        with mock.patch.dict(os.environ, env, clear=True):
            self.code = run(["--from-run", str(spfes_run_dir(Path(tmp))), "--out", str(out), *extra], opener=jev)
        return out

    def test_from_run_asks_jev_per_track_then_the_pen(self):
        jev = FakeJev(pen="walk_tomorrow")
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_from_fixture(tmp, jev=jev)
            triage = json.loads((out / "triage.json").read_text())
            tracks = json.loads((out / "tracks.json").read_text())
            summary = (out / "summary.txt").read_text()
        self.assertEqual(self.code, 0)
        self.assertEqual(len(jev.bodies), len(SPFES_EXPECTED) + 1)
        self.assertEqual(jev.pen_calls, 1)
        self.assertEqual(
            {r["track_id"]: Verdict(r["decision"], r["reason"]) for r in triage["tracks"]}, SPFES_EXPECTED
        )
        self.assertEqual(triage["pen"]["call"], "walk_tomorrow")
        self.assertEqual(triage["question_set"], "spfes-ear-v2")
        self.assertNotIn("book_rules", triage)
        self.assertTrue(tracks["from_run"].endswith("spfes-source"))
        self.assertIn("typed: look 4  skip 1  cannot 1", summary)
        self.assertIn("final: look 4  skip 1  cannot 1  failed 0  (gate turned 0 typed look into cannot)", summary)
        self.assertIn("pen: walk_tomorrow", summary)
        pen_body = jev.bodies[-1]
        self.assertEqual(pen_body["state"]["counts"], {"look": 4, "skip": 1, "cannot": 1, "check_failed": 0})

    def test_book_baseline_sits_beside_jev(self):
        always_look = FakeJev(lambda pack: ("look", 0.99, "flutter"))
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_from_fixture(tmp, "--triage", "book", jev=always_look)
            triage = json.loads((out / "triage.json").read_text())
            summary = (out / "summary.txt").read_text()
            page = write_glance_list(out).read_text()
        self.assertEqual(self.code, 0)
        self.assertEqual(triage["book_rules"]["flutter_deg"], 15.0)
        for row in triage["tracks"]:
            self.assertEqual(row["decision"], "look")
            self.assertEqual(Verdict(row["book"]["decision"], row["book"]["reason"]), SPFES_EXPECTED[row["track_id"]])
        self.assertIn("book:  look 4  skip 1  cannot 1", summary)
        self.assertIn("book vs final: same call on 4 of 6", summary)
        self.assertIn("Check all 6 tomorrow.", page)
        self.assertIn("Every sheep seen was flagged (6 of 6).", page)

    def test_gate_changes_only_unsure_looks_and_the_ab_uses_the_final_call(self):
        # Book calls: 3 skip, 5/8/12/21 look, 15 cannot. Jev: sure looks on 5 and 8, unsure looks on 12
        # and 21, a skip with a high noul on 3, and a typed cannot on 15.
        answers = {
            3: ("skip", 0.97, "steady"),
            5: ("look", 0.9, "flutter"),
            8: ("look", 0.8, "asymmetry"),
            12: ("look", 0.3, "carriage"),
            21: ("look", 0.2, "one_ear_missing"),
            15: ("cannot", 0.9, "not_facing"),
        }
        jev = FakeJev(lambda pack: answers[pack.track_id])
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_from_fixture(tmp, "--triage", "book", jev=jev)
            triage = json.loads((out / "triage.json").read_text())
            tracks_doc = json.loads((out / "tracks.json").read_text())
            summary = (out / "summary.txt").read_text()
            page = write_glance_list(out).read_text()
        self.assertEqual(self.code, 0)
        badges = {c["track_id"]: c["badge"] for c in build_cards(tracks_doc, triage)}
        self.assertEqual(badges, {3: SKIP, 5: LOOK, 8: LOOK, 12: UNCHECKED, 21: UNCHECKED, 15: UNCHECKED})
        rows = {r["track_id"]: r for r in triage["tracks"]}
        side_by_side = {tid: (r["typed_class"], r["noul"], r["decision"], r["gated"], r["reason"]) for tid, r in rows.items()}
        self.assertEqual(
            side_by_side,
            {
                3: ("skip", 0.97, "skip", False, "steady"),
                5: ("look", 0.9, "look", False, "flutter"),
                8: ("look", 0.8, "look", False, "asymmetry"),
                12: ("look", 0.3, "cannot", True, "not_sure"),
                21: ("look", 0.2, "cannot", True, "not_sure"),
                15: ("cannot", 0.9, "cannot", False, "not_facing"),
            },
        )
        self.assertIn("gate turned 2 typed look into cannot", summary)
        self.assertIn("typed: look 4  skip 1  cannot 1", summary)
        self.assertIn("final: look 2  skip 1  cannot 3", summary)
        self.assertIn("book vs final: same call on 4 of 6", summary)
        self.assertIn("typed=look  noul=0.3  decision=cannot/not_sure  book=look/carriage", summary)
        self.assertEqual(jev.bodies[-1]["state"]["cannot_reasons"], {"not_facing": 1, "not_sure": 2})
        self.assertIn("Check these 2 tomorrow.", page)
        self.assertIn("3 not checked · 1 looked fine", rest_section(page))
        unsure = "Not sure enough to call — the ear signs were weak or mixed."
        self.assertEqual(rest_section(page).count(unsure), 2)

    def test_book_without_a_key_still_writes_the_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_from_fixture(tmp, "--triage", "book", key=None)
            triage = json.loads((out / "triage.json").read_text())
            page = write_glance_list(out).read_text()
        self.assertEqual(self.code, 0)
        self.assertEqual(triage["mode"], "pose-only")
        self.assertIsNone(triage["pen"])
        self.assertTrue(all(row["decision"] is None and "book" in row for row in triage["tracks"]))
        self.assertIn("Not sorted yet", page)

    def test_failures_are_recorded_and_the_pen_still_runs(self):
        jev = FakeJev(fail_track_ids=(8,))
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_from_fixture(tmp, jev=jev)
            triage = json.loads((out / "triage.json").read_text())
            page = write_glance_list(out).read_text()
        self.assertEqual(self.code, 3)
        row = next(r for r in triage["tracks"] if r["track_id"] == 8)
        self.assertIsNone(row["decision"])
        self.assertIn("HTTP 500", row["error"])
        self.assertEqual(jev.bodies[-1]["state"]["counts"]["check_failed"], 1)
        self.assertEqual(rest_section(page).count("The ear check failed for this one."), 1)

    def test_pen_failure_and_no_pen(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_from_fixture(tmp, jev=FakeJev(fail_pen=True))
            triage = json.loads((out / "triage.json").read_text())
            page = write_glance_list(out).read_text()
        self.assertEqual(self.code, 3)
        self.assertIn("529", triage["pen"]["error"])
        self.assertNotIn('class="pen', page)

        jev = FakeJev()
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_from_fixture(tmp, "--no-pen", jev=jev)
            self.assertIsNone(json.loads((out / "triage.json").read_text())["pen"])
        self.assertEqual((self.code, jev.pen_calls), (0, 0))

    def test_from_run_needs_frames(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as caught:
            run(["--from-run", tmp, "--pose-only", "--out", str(Path(tmp) / "out")])
        self.assertIn("--demo", str(caught.exception))


@unittest.skipUnless(HAS_CV2, "OpenCV not installed; run in the sheep-yolo env")
class SpfesOverlayTests(unittest.TestCase):
    def test_demo_from_the_fixture(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "sk-fixture"}, clear=True):
                code = run(
                    [
                        "--from-run", str(spfes_run_dir(Path(tmp))), "--demo", "--out", str(out),
                        "--pen", "North pen", "--date", "2026-09-24",
                    ],
                    opener=FakeJev(pen="walk_tomorrow"),
                )
            cap = cv2.VideoCapture(str(out / "annotated.mp4"))
            n = 0
            while cap.read()[0]:
                n += 1
            cap.release()
            page = (out / PAGE_NAME).read_text()
            thumbs = sorted(p.name for p in (out / "thumbs").iterdir())
        self.assertEqual(code, 0)
        self.assertEqual(n, 90)
        self.assertIn("Thursday 24 September", page)
        self.assertIn("North pen · Morning brief", page)
        self.assertIn("Walk the pen first thing tomorrow.", page)
        self.assertIn("<h3>Not checked</h3>", rest_section(page))
        self.assertEqual(thumbs, sorted(f"sheep-{tid}.jpg" for tid in SPFES_EXPECTED))
        for tid in SPFES_EXPECTED:
            self.assertEqual(page.count(f'src="{thumb_path(tid)}"'), 1)

    def test_cannot_is_burned_grey(self):
        import cv2
        import numpy as np

        from render_overlay import BADGE_BG, draw_track

        img = np.zeros((600, 640, 3), np.uint8)
        draw_track(cv2, img, _head_at(300, 300, 90, 90), 15, UNCHECKED, scale=1.0, kpt_conf=0.4, top=0)
        grey = int(np.all(img == np.array(BADGE_BG[UNCHECKED], np.uint8), axis=-1).sum())
        self.assertGreater(grey, 1000)
        for badge in (LOOK, SKIP):
            self.assertEqual(int(np.all(img == np.array(BADGE_BG[badge], np.uint8), axis=-1).sum()), 0)


class TurnCoordinatesTests(unittest.TestCase):
    """A 96 × 64 picture: a box and a keypoint near its top-left corner, turned clockwise."""

    FRAME = [{1: {"box": [10.0, 5.0, 30.0, 20.0], "kpts": [[12.0, 6.0, 0.9]]}}]

    def test_each_quarter_turn(self):
        cases = {
            90: ([44.0, 10.0, 59.0, 30.0], [58.0, 12.0, 0.9]),
            180: ([66.0, 44.0, 86.0, 59.0], [84.0, 58.0, 0.9]),
            270: ([5.0, 66.0, 20.0, 86.0], [6.0, 84.0, 0.9]),
        }
        for degrees, (box, kpt) in cases.items():
            with self.subTest(degrees):
                turned = turn_frames(self.FRAME, degrees, 96, 64)[0][1]
                self.assertEqual(turned["box"], box)
                self.assertEqual(turned["kpts"], [kpt])

    def test_no_turn_is_the_same_frames_and_turns_undo(self):
        self.assertIs(turn_frames(self.FRAME, 0, 96, 64), self.FRAME)
        self.assertIs(turn_frames(self.FRAME, 360, 96, 64), self.FRAME)
        there = turn_frames(self.FRAME, 90, 96, 64)
        self.assertEqual(turn_frames(there, 270, 64, 96), self.FRAME)
        self.assertEqual(turn_frames(self.FRAME, -90, 96, 64), turn_frames(self.FRAME, 270, 96, 64))

    def test_recorded_rotation(self):
        self.assertEqual(recorded_rotation({"frame_rotation": 180}), 180)
        for doc in ({}, {"frame_rotation": None}, {"frame_rotation": "180"}, {"frame_rotation": True}):
            with self.subTest(doc=doc):
                self.assertIsNone(recorded_rotation(doc))


# The upright test picture: a grey sheep with a white strip of ears along its top.
PIC_W, PIC_H = 96, 64
SHEEP = (30, 16, 70, 48)
# Where that sheep sits in the pixels a clip stores when it must be turned this far to show upright.
STORED_SHEEP = {0: SHEEP, 90: (16, 26, 48, 66), 180: (26, 16, 66, 48), 270: (16, 30, 48, 70)}


def _sheep_picture() -> Any:
    import numpy as np

    img = np.zeros((PIC_H, PIC_W, 3), np.uint8)
    x1, y1, x2, y2 = SHEEP
    img[y1:y2, x1:x2] = 120
    img[y1 : y1 + 8, x1:x2] = 255
    return img


def _ears_up(img: Any) -> bool:
    """True when the brightest rows of the sheep are at its top, as in the upright picture."""
    import numpy as np

    grey = img.max(axis=2) if img.ndim == 3 else img
    ys, xs = np.nonzero(grey > 60)
    sheep = grey[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(float)
    quarter = max(1, sheep.shape[0] // 4)
    return float(sheep[:quarter].mean()) > float(sheep[-quarter:].mean()) + 40


def _mp4_boxes(data: bytes, start: int, end: int) -> Any:
    i = start
    while i + 8 <= end:
        size, kind = struct.unpack(">I4s", data[i : i + 8])
        header = 8
        if size == 1:
            size, header = struct.unpack(">Q", data[i + 8 : i + 16])[0], 16
        elif size == 0:
            size = end - i
        yield kind, i + header, i + size
        i += size


def _set_track_rotation(path: Path, degrees_cw: int) -> None:
    """Write a clockwise quarter turn into the video track header's display matrix, as an iPhone does."""
    data = bytearray(path.read_bytes())
    a, b = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[degrees_cw]
    matrix = struct.pack(">9i", a << 16, b << 16, 0, -b << 16, a << 16, 0, 0, 0, 1 << 30)
    for kind, body, end in _mp4_boxes(data, 0, len(data)):
        if kind != b"moov":
            continue
        for kind2, body2, end2 in _mp4_boxes(data, body, end):
            if kind2 != b"trak":
                continue
            for kind3, body3, _ in _mp4_boxes(data, body2, end2):
                if kind3 == b"tkhd":
                    at = body3 + 4 + (32 if data[body3] == 1 else 20) + 16
                    data[at : at + 36] = matrix
    path.write_bytes(bytes(data))


def _turned_clip(path: Path, degrees_cw: int, n_frames: int = 6) -> Path:
    """A synthetic clip stored turned, whose container says to turn it ``degrees_cw`` to show it upright."""
    from fractions import Fraction

    import av

    stored = turn(_sheep_picture(), -degrees_cw % 360)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=Fraction(10))
        stream.width, stream.height, stream.pix_fmt = stored.shape[1], stored.shape[0], "yuv420p"
        stream.options = {"crf": "10"}
        for i in range(n_frames):
            frame = av.VideoFrame.from_ndarray(stored, format="bgr24")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    if degrees_cw:
        _set_track_rotation(path, degrees_cw)
    return path


class _FakeTensor:
    def __init__(self, data: Any):
        self._data = data

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> Any:
        return self._data


class FakeYOLO:
    """Stands in for ultralytics.YOLO: 'detects' the bright block in whatever frame it is handed."""

    instances: list[FakeYOLO] = []

    def __init__(self, weights: str):
        self.seen: list[tuple[Any, dict[str, Any]]] = []
        FakeYOLO.instances.append(self)

    def track(self, source: Any, **kwargs: Any) -> list[Any]:
        import numpy as np

        self.seen.append((source.copy(), kwargs))
        ys, xs = np.nonzero(source.max(axis=2) > 60)
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        kpts = np.zeros((1, 5, 3))  # nothing confident, so the replay draws only the box
        boxes = types.SimpleNamespace(id=_FakeTensor(np.array([1])), xyxy=_FakeTensor(np.array([box])))
        return [types.SimpleNamespace(boxes=boxes, keypoints=types.SimpleNamespace(data=_FakeTensor(kpts)))]


@unittest.skipUnless(HAS_VIDEO, "needs OpenCV and PyAV; run in the sheep-yolo env")
class ClipOrientationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        FakeYOLO.instances.clear()
        fake = types.ModuleType("ultralytics")
        fake.YOLO = FakeYOLO  # type: ignore[attr-defined]
        real = sys.modules.get("ultralytics")
        sys.modules["ultralytics"] = fake
        if real is None:
            self.addCleanup(sys.modules.pop, "ultralytics", None)
        else:
            self.addCleanup(sys.modules.__setitem__, "ultralytics", real)

    def clip(self, degrees_cw: int) -> Path:
        return _turned_clip(self.tmp / f"turned-{degrees_cw}.mp4", degrees_cw)

    def test_the_fixture_is_stored_turned(self):
        import av

        for degrees in (90, 180, 270):
            with self.subTest(degrees), av.open(str(self.clip(degrees))) as container:
                stored = next(container.decode(video=0)).to_ndarray(format="bgr24")
            self.assertFalse(_ears_up(stored) and stored.shape[:2] == (PIC_H, PIC_W))

    def test_reads_the_rotation_the_container_asks_for(self):
        for degrees in TURNS:
            with self.subTest(degrees):
                stored = (PIC_W, PIC_H) if degrees in (0, 180) else (PIC_H, PIC_W)
                self.assertEqual(orientation(self.clip(degrees)), Orientation(degrees, *stored))

    def test_frames_come_out_upright_and_upright_clips_are_left_alone(self):
        import av
        import numpy as np

        for degrees in TURNS:
            with self.subTest(degrees):
                frames = list(upright_frames(self.clip(degrees)))
                self.assertEqual(len(frames), 6)
                for img in frames:
                    self.assertEqual(img.shape, (PIC_H, PIC_W, 3))
                    self.assertTrue(_ears_up(img))
        with av.open(str(self.clip(0))) as container:
            stored = [f.to_ndarray(format="bgr24") for f in container.decode(video=0)]
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(upright_frames(self.clip(0)), stored)))
        self.assertEqual(len(list(upright_frames(self.clip(180), max_frames=2))), 2)

    def test_image_and_coordinate_turns_agree(self):
        import numpy as np

        img = np.zeros((PIC_H, PIC_W), np.uint8)
        img[5, 12] = 255  # the pixel whose centre is (12.5, 5.5)
        for degrees in TURNS:
            with self.subTest(degrees):
                turned = turn(img, degrees)
                y, x = (int(v) for v in np.argwhere(turned == 255)[0])
                point = [{1: {"box": [0, 0, 1, 1], "kpts": [[12.5, 5.5, 1.0]]}}]
                self.assertEqual(turn_frames(point, degrees, PIC_W, PIC_H)[0][1]["kpts"][0][:2], [x + 0.5, y + 0.5])

    def test_tracking_is_handed_upright_frames(self):
        for degrees in (0, 90, 180, 270):
            with self.subTest(degrees):
                frames = track_clip(
                    Path("fake.pt"), self.clip(degrees), conf=0.25, max_frames=4, tracker="bytetrack.yaml", imgsz=None
                )
                model = FakeYOLO.instances[-1]
                self.assertEqual(len(model.seen), 4)
                for img, kwargs in model.seen:
                    self.assertEqual(img.shape, (PIC_H, PIC_W, 3))
                    self.assertTrue(_ears_up(img))
                    self.assertTrue(kwargs["persist"])
                box = frames[0][1]["box"]
                for got, want in zip(box, SHEEP):
                    self.assertAlmostEqual(got, want, delta=2)

    def run_demo(self, *argv: str) -> Path:
        out = self.tmp / f"out-{len(list(self.tmp.iterdir()))}"
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(run([*argv, "--pose-only", "--demo", "--out", str(out)]), 0)
        return out

    def replay_first_frame(self, out: Path) -> Any:
        import av

        with av.open(str(out / "annotated.mp4")) as container:
            frame = next(container.decode(video=0))
        self.assertEqual(frame.rotation, 0)
        return frame.to_ndarray(format="bgr24")

    def assert_upright_outputs(self, out: Path) -> None:
        import cv2

        thumb = cv2.imread(str(out / thumb_path(1)))
        self.assertTrue(_ears_up(thumb), "thumbnail is not upright")
        replay = self.replay_first_frame(out)
        self.assertEqual(replay.shape, (PIC_H, PIC_W, 3))
        x1, y1, x2, y2 = SHEEP
        ears = replay[y1 + 2 : y1 + 6, x1 + 8 : x2 - 8].mean()
        body = replay[y2 - 10 : y2 - 4, x1 + 8 : x2 - 8].mean()
        self.assertGreater(ears, body + 40, "replay is not upright")

    def test_a_demo_run_from_a_turned_clip_is_upright_end_to_end(self):
        weights = self.tmp / "fake.pt"
        weights.write_bytes(b"")
        for degrees in (0, 90, 180, 270):
            with self.subTest(degrees):
                out = self.run_demo("--clip", str(self.clip(degrees)), "--weights", str(weights))
                self.assertEqual(json.loads((out / "tracks.json").read_text())["frame_rotation"], degrees)
                self.assert_upright_outputs(out)

    def old_run(self, degrees: int) -> tuple[Path, Path]:
        """A run folder from before frame_rotation: frames.json in the clip's stored coordinates."""
        import av
        import numpy as np

        clip = self.clip(degrees)
        box = STORED_SHEEP[degrees]
        with av.open(str(clip)) as container:
            stored = next(container.decode(video=0)).to_ndarray(format="bgr24")
        ys, xs = np.nonzero(stored.max(axis=2) > 60)
        for got, want in zip((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), box):
            self.assertAlmostEqual(float(got), want, delta=2)
        run_dir = self.tmp / f"old-{degrees}"
        run_dir.mkdir()
        save_frames(run_dir / "frames.json", [{1: {"box": list(box), "kpts": [[0.0, 0.0, 0.0]] * 5}}] * 6)
        old = {"clip": str(clip), "weights": None, "fps": 10.0, "tracks": []}
        (run_dir / "tracks.json").write_text(json.dumps(old))
        return run_dir, clip

    def test_old_runs_tracked_on_stored_frames_are_turned_to_match(self):
        # How the upside-down thumbnails happened: OpenCV on that machine left the clip as stored.
        for degrees in (90, 180, 270):
            with self.subTest(degrees), mock.patch("clip_frames.opencv_rotation", return_value=0):
                run_dir, _ = self.old_run(degrees)
                out = self.run_demo("--from-run", str(run_dir))
                self.assertEqual(json.loads((out / "tracks.json").read_text())["frame_rotation"], degrees)
                box = load_frames(out / "frames.json")[0][1]["box"]
                self.assertEqual(box, [float(v) for v in SHEEP])
                self.assert_upright_outputs(out)

    def test_old_runs_opencv_already_turned_are_left_alone(self):
        with mock.patch("clip_frames.opencv_rotation", return_value=180):
            run_dir, clip = self.old_run(180)
            upright = [{1: {"box": list(SHEEP), "kpts": [[0.0, 0.0, 0.0]] * 5}}] * 6
            save_frames(run_dir / "frames.json", upright)
            out = self.run_demo("--from-run", str(run_dir))
        self.assertEqual(load_frames(out / "frames.json")[0][1]["box"], [float(v) for v in SHEEP])
        self.assert_upright_outputs(out)

    def test_opencv_probe_matches_what_opencv_does_here(self):
        import cv2

        from clip_frames import opencv_rotation

        clip = self.clip(180)
        cap = cv2.VideoCapture(str(clip))
        ok, img = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.assertEqual(opencv_rotation(clip), 0 if not _ears_up(img) else 180)
        self.assertEqual(opencv_rotation(self.clip(0)), 0)

    def test_rebuilding_an_old_replay_turns_it_too(self):
        from render_overlay import main as rebuild

        with mock.patch("clip_frames.opencv_rotation", return_value=0):
            run_dir, _ = self.old_run(180)
            (run_dir / "triage.json").write_text(json.dumps({"mode": "pose-only", "tracks": [{"track_id": 1}]}))
            with mock.patch("sys.stdout", io.StringIO()):
                self.assertEqual(rebuild([str(run_dir)]), 0)
        self.assert_upright_outputs(run_dir)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
