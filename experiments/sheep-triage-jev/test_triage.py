"""Stdlib tests for pose summaries and the Jev look/dont mapping.

    python experiments/sheep-triage-jev/test_triage.py
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from jev_client import JevError, decision_from_noul, request_body, triage_track
from pose_features import ear_angles, summarize_frames, synthetic_frames
from pose_runner import frame_from_result
from run_pipeline import run

HERE = Path(__file__).resolve().parent


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
