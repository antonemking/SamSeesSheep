#!/usr/bin/env python3
"""Sheep ear triage: YOLO-pose tracks → SPFES ear pack per track → TypeSafe Jev.

Pose-only (no API key):

    python experiments/sheep-triage-jev/run_pipeline.py --synthetic --pose-only

    uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py \\
        --clip test-clips/Test_Clip_Morning.mov --pose-only

Full triage (TYPESAFE_API_KEY must be set in the environment, never committed):

    uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py \\
        --clip test-clips/Test_Clip_Morning.mov

Each track gets look / skip / cannot and a reason from Jev; one more call
turns the pen's counts into walk_tomorrow / later / fine. --triage book also
writes the fixed ear-rule baseline beside Jev in triage.json. --from-run
re-asks an earlier --demo run folder without tracking the clip again.

Meant to run overnight on the day's footage. Add --demo to also write
annotated.mp4 (Look / Skip / Not checked replay), one thumbnail per sheep, and
morning-brief.html into the same --out folder. --pen names the pen on the
brief; --date is the day the footage was taken (default: the clip file's date).

Jev runs on this machine or in the cloud. It does not run on a Pi.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from clip_frames import orientation, recorded_rotation, to_upright
from glance_list import iso_day, write_glance_list
from jev_client import (
    DECISION_RULE,
    ENDPOINT,
    QUESTION_SET,
    JevError,
    Opener,
    pen_state,
    triage_pen,
    triage_track,
)
from pose_features import (
    KPT_NAMES,
    EarPack,
    EarStats,
    state_for_jev,
    summarize_frames,
    synthetic_frames,
)
from render_overlay import FRAMES_NAME, load_frames, render, save_frames
from spfes_ear import BOOK_RULES, DECISIONS, Verdict, book_triage

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = REPO / "sheep-yolo" / "weights" / "sheep-pose-v0.7-yolo26n.pt"
ENV_KEY = "TYPESAFE_API_KEY"


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _ear_text(ear: EarStats) -> str:
    seen = f"{ear.measured_fraction:.0%}"
    if ear.median_deg is None:
        return f"— ({seen})"
    return f"{ear.median_deg:.0f}°±{_fmt(ear.std_deg)} ({seen})"


def _verdict_text(decision: str | None, reason: str | None) -> str:
    if not decision:
        return "—"
    return f"{decision}/{reason}" if reason else decision


def _counts(verdicts: list[tuple[str | None, str | None]]) -> str:
    c = Counter(decision for decision, _ in verdicts)
    return "  ".join(f"{d} {c[d]}" for d in DECISIONS)


def _summary_text(meta: dict[str, Any], tracks: list[dict[str, Any]], triage: dict[str, Any]) -> str:
    rows = {row["track_id"]: row for row in triage["tracks"]}
    lines = [
        f"clip: {meta.get('clip')}",
        f"footage date: {meta.get('footage_date')}  pen label: {meta.get('pen_label') or '—'}",
        (
            f"mode: {triage['mode']}  model: {triage.get('model')}  threshold: {triage.get('threshold')}  "
            f"questions: {QUESTION_SET}"
        ),
        f"weights: {meta.get('weights')}",
        f"frames: {meta.get('n_frames')}  fps: {meta.get('fps')}",
        f"tracks: {len(tracks)}  dropped_short: {meta.get('dropped_track_ids')}",
        "",
    ]
    typed: list[tuple[str | None, str | None]] = []
    final: list[tuple[str | None, str | None]] = []
    book: list[tuple[str | None, str | None]] = []
    for track in tracks:
        pack = EarPack.from_json(track["ear_pack"])
        row = rows.get(track["track_id"], {})
        typed.append((row.get("typed_class"), None))
        final.append((row.get("decision"), row.get("reason")))
        line = (
            f"track {pack.track_id}  frames={pack.n_frames}  facing={pack.facing_fraction:.2f}  "
            f"L={_ear_text(pack.left_ear)}  R={_ear_text(pack.right_ear)}  |L−R|={_fmt(pack.asymmetry_deg)}  "
            f"typed={_fmt(row.get('typed_class'))}  noul={_fmt(row.get('noul'))}  "
            f"decision={_verdict_text(row.get('decision'), row.get('reason'))}"
        )
        if "book" in row:
            book.append((row["book"]["decision"], row["book"]["reason"]))
            line += f"  book={_verdict_text(row['book']['decision'], row['book']['reason'])}"
        lines.append(line)
    lines.append("")
    if triage["mode"] == "jev":
        gated = sum(bool(row.get("gated")) for row in triage["tracks"])
        lines.append(f"typed: {_counts(typed)}")
        lines.append(f"final: {_counts(final)}  failed {triage['failures']}  (gate turned {gated} typed look into cannot)")
    if book:
        lines.append(f"book:  {_counts(book)}")
        if triage["mode"] == "jev":
            compared = [(f, b) for f, b in zip(final, book) if f[0] is not None]
            agree = sum(f[0] == b[0] for f, b in compared)
            lines.append(f"book vs final: same call on {agree} of {len(compared)}")
    pen = triage.get("pen")
    if isinstance(pen, dict):
        lines.append(f"pen: {pen.get('call') or 'FAILED ' + str(pen.get('error'))}")
    return "\n".join(lines) + "\n"


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", type=Path, help="Source video. Not the rendered assets/*.mp4 showcase.")
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="Built-in two-track fixture. No weights, no video, no API key required for --pose-only.",
    )
    src.add_argument(
        "--from-run",
        type=Path,
        metavar="RUN_DIR",
        help="Re-triage an earlier --demo run folder from its frames.json, without tracking again.",
    )
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument(
        "--pose-only",
        action="store_true",
        help="Write pose summaries and skip Jev, even if TYPESAFE_API_KEY is set.",
    )
    p.add_argument(
        "--triage",
        choices=("jev", "book"),
        default="jev",
        help="jev (default): Jev decides. book: also write the fixed ear-rule baseline beside Jev "
        "in triage.json for A/B. Jev still drives the demo.",
    )
    p.add_argument(
        "--no-pen", action="store_true", help="Skip the pen-level walk tomorrow / later / fine call."
    )
    p.add_argument("--pen", default=None, help='Pen or flock name for the morning brief, e.g. "North pen".')
    p.add_argument(
        "--date",
        type=iso_day,
        default=None,
        help="Day the footage was taken, YYYY-MM-DD. Default: the clip file's date "
        "(--from-run: the earlier run's; --synthetic: today).",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence.")
    p.add_argument("--kpt-conf", type=float, default=0.4, help="Keypoint confidence.")
    p.add_argument("--min-frames", type=int, default=2)
    p.add_argument("--fps", type=float, default=None, help="Override clip fps.")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence gate on a typed look: noul below this turns it into cannot (Not checked).",
    )
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--demo",
        action="store_true",
        help="Also write frames.json, annotated.mp4 (Look / Skip / Not checked replay), thumbs/ and "
        "morning-brief.html. Needs OpenCV.",
    )
    p.add_argument("--demo-width", type=int, default=1280, help="Max width of annotated.mp4.")
    return p.parse_args(argv)


def _load_from_run(args: argparse.Namespace) -> tuple[list[dict[int, dict]], float, dict[str, Any]]:
    run_dir = args.from_run.expanduser().resolve()
    frames_path, tracks_path = run_dir / FRAMES_NAME, run_dir / "tracks.json"
    if not frames_path.is_file() or not tracks_path.is_file():
        raise SystemExit(
            f"{run_dir} needs {FRAMES_NAME} and tracks.json.\n"
            "Only runs made with --demo keep per-frame poses. Run the clip once with --demo, then use --from-run."
        )
    old = json.loads(tracks_path.read_text())
    fps = float(args.fps if args.fps is not None else old["fps"])
    clip = old.get("clip")
    if args.demo and clip != "synthetic" and not (clip and Path(clip).is_file()):
        raise SystemExit(
            f"clip recorded in {tracks_path} is missing: {clip}\n"
            "The replay needs the source video. Move it back, or drop --demo and rebuild the replay later with\n"
            "  render_overlay.py <out> --clip <video>"
        )
    source = Path(clip) if clip and clip != "synthetic" and Path(clip).is_file() else None
    day = old.get("footage_date") or _file_day(source or tracks_path)
    frames = load_frames(frames_path)
    rotation = 0 if clip == "synthetic" else recorded_rotation(old)
    needs = ("av", "numpy") if rotation is not None else ("av", "numpy", "cv2")
    if source is not None and all(importlib.util.find_spec(name) is not None for name in needs):
        frames, rotation = to_upright(frames, source, rotation)
    meta = {
        "clip": clip,
        "weights": old.get("weights"),
        "fps": fps,
        "from_run": str(run_dir),
        "footage_date": day,
        "pen_label": old.get("pen_label"),
    }
    if rotation is not None:
        meta["frame_rotation"] = rotation
    return frames, fps, meta


def _file_day(path: Path) -> str:
    return date.fromtimestamp(path.stat().st_mtime).isoformat()


def _load_frames(args: argparse.Namespace) -> tuple[list[dict[int, dict]], float, dict[str, Any]]:
    if args.synthetic:
        frames, default_fps = synthetic_frames()
        fps = float(args.fps or default_fps)
        meta = {"clip": "synthetic", "weights": None, "fps": fps, "footage_date": date.today().isoformat()}
        return frames, fps, {**meta, "pen_label": None, "frame_rotation": 0}
    if args.from_run is not None:
        return _load_from_run(args)

    clip = args.clip.expanduser().resolve()
    if not clip.is_file():
        raise SystemExit(
            f"clip not found: {clip}\n"
            "Source clips are local and gitignored. Point --clip at one, for example\n"
            "  test-clips/Test_Clip_Morning.mov\n"
            "  test-clips/IMG_3651.MOV\n"
            "assets/synced-lanes-6ewes-pro-Test_Clip_Morning.mp4 is a rendered\n"
            "showcase, not a source clip. With no clip, use --synthetic --pose-only."
        )
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(
            f"weights not found: {weights}\n"
            "sheep-yolo/weights/*.pt is gitignored. Download v0.7 (about 6 MB) with:\n"
            "  mkdir -p sheep-yolo/weights\n"
            "  curl -L -o sheep-yolo/weights/sheep-pose-v0.7-yolo26n.pt \\\n"
            "    https://huggingface.co/antking1/sheep-pose-yolo26n/resolve/main/sheep-pose-v0.7-yolo26n.pt\n"
            "Or pass --weights. Hugging Face: antking1/sheep-pose-yolo26n."
        )
    from pose_runner import read_fps, track_clip

    fps = args.fps if args.fps is not None else read_fps(clip)
    rotation = orientation(clip).rotation
    turned = f", turned {rotation}° as the clip asks" if rotation else ""
    print(f"tracking {clip.name} with {weights.name} (fps={fps:.2f}{turned})")
    frames = track_clip(
        weights,
        clip,
        conf=args.conf,
        max_frames=args.max_frames,
        tracker=args.tracker,
        imgsz=args.imgsz,
    )
    meta = {"clip": str(clip), "weights": str(weights), "fps": fps, "footage_date": _file_day(clip)}
    return frames, fps, {**meta, "pen_label": None, "frame_rotation": rotation}


def _out_stem(args: argparse.Namespace) -> str:
    if args.synthetic:
        return "synthetic"
    if args.from_run is not None:
        return f"{args.from_run.expanduser().resolve().name}-rerun"
    return str(args.clip.stem)


def _jev_tracks(
    tracks: list[dict[str, Any]], args: argparse.Namespace, api_key: str, opener: Opener | None
) -> tuple[list[dict[str, Any]], int]:
    print(f"calling Jev ({args.model}) for {len(tracks)} track(s)")
    rows: list[dict[str, Any]] = []
    failures = 0
    for summary in tracks:
        record: dict[str, Any] = {"track_id": summary["track_id"]}
        try:
            result = triage_track(
                state_for_jev(summary),
                api_key=api_key,
                threshold=args.threshold,
                model=args.model,
                timeout=args.timeout,
                opener=opener,
            )
        except JevError as exc:
            failures += 1
            record.update({"decision": None, "reason": None, "noul": None, "error": str(exc)})
            print(f"  track {summary['track_id']}: FAILED {exc}")
        else:
            record.update(result.to_json())
            print(
                f"  track {summary['track_id']}: typed {result.typed_class}, noul={result.noul:.3f} "
                f"→ {result.decision} ({result.reason})"
            )
        rows.append(record)
    return rows, failures


def _jev_pen(
    rows: list[dict[str, Any]], failures: int, args: argparse.Namespace, api_key: str, opener: Opener | None
) -> dict[str, Any] | None:
    verdicts = [Verdict(row["decision"], row["reason"]) for row in rows if row.get("decision")]
    if not verdicts:
        return None
    try:
        pen = triage_pen(
            pen_state(verdicts, failed=failures),
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            opener=opener,
        )
    except JevError as exc:
        print(f"  pen: FAILED {exc}")
        return {"call": None, "error": str(exc)}
    print(f"  pen: {pen.call}")
    return pen.to_json()


def run(argv: list[str] | None = None, *, opener: Opener | None = None) -> int:
    """Run the pipeline. ``opener`` replaces urllib's for every Jev call (tests use it)."""
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.demo:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            raise SystemExit(
                "--demo needs OpenCV and NumPy. Run it in the sheep-yolo env:\n"
                "  uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py ... --demo"
            )
    frames, fps, meta = _load_frames(args)
    if args.date is not None:
        meta["footage_date"] = args.date.isoformat()
    if args.pen and args.pen.strip():
        meta["pen_label"] = args.pen.strip()
    tracks, dropped = summarize_frames(
        frames,
        fps=fps,
        kpt_conf=args.kpt_conf,
        min_frames=args.min_frames,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (HERE / "runs" / f"{stamp}-{_out_stem(args)}")
    out.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get(ENV_KEY, "").strip()
    if args.pose_only:
        mode, reason = "pose-only", "--pose-only"
    elif not api_key:
        mode, reason = "pose-only", f"{ENV_KEY} is not set"
    else:
        mode, reason = "jev", None

    meta.update(
        {
            "mode": mode,
            "n_frames": len(frames),
            "kpt_conf": args.kpt_conf,
            "det_conf": args.conf,
            "min_frames": args.min_frames,
            "dropped_track_ids": dropped,
            "keypoint_order": list(KPT_NAMES),
        }
    )
    track_rows = [{**summary, "jev_state": state_for_jev(summary)} for summary in tracks]
    _dump(out / "tracks.json", {**meta, "tracks": track_rows})

    triage: dict[str, Any]
    failures = 0
    pen: dict[str, Any] | None = None
    if mode == "pose-only":
        rows = [{"track_id": s["track_id"], "decision": None, "reason": None, "noul": None} for s in tracks]
        triage = {"mode": "pose-only", "reason": reason, "model": None, "threshold": args.threshold}
        print(f"pose-only ({reason}). Jev was not called.")
    else:
        rows, failures = _jev_tracks(tracks, args, api_key, opener)
        if not args.no_pen:
            pen = _jev_pen(rows, failures, args, api_key, opener)
        triage = {
            "mode": "jev",
            "reason": None,
            "endpoint": ENDPOINT,
            "model": args.model,
            "threshold": args.threshold,
            "decision_rule": DECISION_RULE,
            "failures": failures,
        }
    triage["question_set"] = QUESTION_SET
    triage["pen"] = pen
    if args.triage == "book":
        triage["book_rules"] = BOOK_RULES
        for row, summary in zip(rows, tracks):
            row["book"] = book_triage(EarPack.from_json(summary["ear_pack"])).to_json()
    triage["tracks"] = rows

    _dump(out / "triage.json", triage)
    text = _summary_text(meta, tracks, triage)
    (out / "summary.txt").write_text(text)
    sys.stdout.write(text)
    if args.demo:
        save_frames(out / FRAMES_NAME, frames)
        clip = str(meta["clip"])
        render(
            out,
            frames,
            triage,
            clip=None if clip == "synthetic" else Path(clip),
            fps=fps,
            kpt_conf=args.kpt_conf,
            max_width=args.demo_width,
        )
        print(f"wrote {write_glance_list(out)}")
    print(f"wrote {out}")
    if api_key:
        for path in out.iterdir():
            if path.is_file() and path.suffix in {".json", ".txt", ".html"} and api_key in path.read_text():
                raise SystemExit(f"refusing to leave the API key in run artifacts ({path.name})")
    pen_failed = pen is not None and pen.get("error") is not None
    return 3 if failures or pen_failed else 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
