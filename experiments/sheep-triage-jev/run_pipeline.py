#!/usr/bin/env python3
"""Sheep-track triage experiment: YOLO-pose tracks, then optional TypeSafe Jev.

Pose-only (no API key):

    python experiments/sheep-triage-jev/run_pipeline.py --synthetic --pose-only

    uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py \\
        --clip test-clips/Test_Clip_Morning.mov --pose-only

Full triage (TYPESAFE_API_KEY must be set in the environment, never committed):

    uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py \\
        --clip test-clips/Test_Clip_Morning.mov

Add --demo to also write annotated.mp4 (Look / Skip replay) and
glance-list.html into the same --out folder.

Jev runs on this machine or in the cloud. It does not run on a Pi.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from jev_client import JevError, triage_track
from pose_features import state_for_jev, summarize_frames, synthetic_frames

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = REPO / "sheep-yolo" / "weights" / "sheep-pose-v0.7-yolo26n.pt"
ENV_KEY = "TYPESAFE_API_KEY"


def _dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _summary_text(meta: dict, tracks: list[dict], triage_rows: list[dict]) -> str:
    by_id = {row["track_id"]: row for row in triage_rows}
    lines = [
        f"clip: {meta.get('clip')}",
        f"mode: {meta.get('mode')}",
        f"weights: {meta.get('weights')}",
        f"frames: {meta.get('n_frames')}  fps: {meta.get('fps')}",
        f"tracks: {len(tracks)}  dropped_short: {meta.get('dropped_track_ids')}",
        "",
    ]
    for track in tracks:
        ear = track["ear_angle_deg"]
        row = by_id.get(track["track_id"], {})
        decision = row.get("decision")
        noul = row.get("noul")
        lines.append(
            "track {tid}  frames={n}  coverage={cov}  "
            "L={l}° σ={ls}  R={r}° σ={rs}  path={path}px  "
            "decision={decision}  noul={noul}".format(
                tid=track["track_id"],
                n=track["n_frames"],
                cov=_fmt(track["coverage"]),
                l=_fmt(ear["left_median"]),
                ls=_fmt(ear["left_std"]),
                r=_fmt(ear["right_median"]),
                rs=_fmt(ear["right_std"]),
                path=_fmt(track["motion"]["centroid_path_px"]),
                decision=decision if decision else "—",
                noul=_fmt(noul) if noul is not None else "—",
            )
        )
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
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument(
        "--pose-only",
        action="store_true",
        help="Write pose summaries and skip Jev, even if TYPESAFE_API_KEY is set.",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence.")
    p.add_argument("--kpt-conf", type=float, default=0.4, help="Keypoint confidence.")
    p.add_argument("--min-frames", type=int, default=2)
    p.add_argument("--fps", type=float, default=None, help="Override clip fps.")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--threshold", type=float, default=0.5, help="noul >= threshold → look.")
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--demo",
        action="store_true",
        help="Also write frames.json, annotated.mp4 (Look / Skip replay) and glance-list.html. Needs OpenCV.",
    )
    p.add_argument("--demo-width", type=int, default=1280, help="Max width of annotated.mp4.")
    return p.parse_args(argv)


def _load_frames(args: argparse.Namespace) -> tuple[list, float, dict]:
    if args.synthetic:
        frames, fps = synthetic_frames()
        meta = {
            "clip": "synthetic",
            "weights": None,
            "fps": args.fps or fps,
        }
        return frames, meta["fps"], meta

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
    print(f"tracking {clip.name} with {weights.name} (fps={fps:.2f})")
    frames = track_clip(
        weights,
        clip,
        conf=args.conf,
        max_frames=args.max_frames,
        tracker=args.tracker,
        imgsz=args.imgsz,
    )
    meta = {"clip": str(clip), "weights": str(weights), "fps": fps}
    return frames, fps, meta


def run(argv: list[str] | None = None) -> int:
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
    tracks, dropped = summarize_frames(
        frames,
        fps=fps,
        kpt_conf=args.kpt_conf,
        min_frames=args.min_frames,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = "synthetic" if args.synthetic else args.clip.stem
    out = args.out or (HERE / "runs" / f"{stamp}-{stem}")
    out.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get(ENV_KEY, "").strip()
    if args.pose_only:
        mode = "pose-only"
        reason = "--pose-only"
    elif not api_key:
        mode = "pose-only"
        reason = f"{ENV_KEY} is not set"
    else:
        mode = "jev"
        reason = None

    meta.update(
        {
            "mode": mode,
            "n_frames": len(frames),
            "kpt_conf": args.kpt_conf,
            "det_conf": args.conf,
            "min_frames": args.min_frames,
            "dropped_track_ids": dropped,
            "keypoint_order": [
                "nose",
                "L_ear_base",
                "R_ear_base",
                "L_ear_tip",
                "R_ear_tip",
            ],
        }
    )
    track_rows = []
    for summary in tracks:
        track_rows.append({**summary, "jev_state": state_for_jev(summary)})
    _dump(out / "tracks.json", {**meta, "tracks": track_rows})

    triage_rows: list[dict] = []
    failures = 0
    if mode == "pose-only":
        for summary in tracks:
            triage_rows.append(
                {"track_id": summary["track_id"], "decision": None, "noul": None}
            )
        triage = {
            "mode": "pose-only",
            "reason": reason,
            "model": None,
            "threshold": args.threshold,
            "tracks": triage_rows,
        }
        print(f"pose-only ({reason}). Jev was not called.")
    else:
        print(f"calling Jev ({args.model}) for {len(tracks)} track(s)")
        for summary in tracks:
            record = {"track_id": summary["track_id"]}
            try:
                decision = triage_track(
                    state_for_jev(summary),
                    api_key=api_key,
                    threshold=args.threshold,
                    model=args.model,
                    timeout=args.timeout,
                )
            except JevError as exc:
                failures += 1
                record["error"] = str(exc)
                record["decision"] = None
                record["noul"] = None
                print(f"  track {summary['track_id']}: FAILED {exc}")
            else:
                record.update(decision)
                print(
                    f"  track {summary['track_id']}: {decision['decision']} "
                    f"(noul={decision['noul']:.3f})"
                )
            triage_rows.append(record)
        triage = {
            "mode": "jev",
            "reason": None,
            "endpoint": "https://api.typesafe.ai/v1/systemone",
            "model": args.model,
            "threshold": args.threshold,
            "decision_rule": "look if noul >= threshold else dont",
            "failures": failures,
            "tracks": triage_rows,
        }

    _dump(out / "triage.json", triage)
    text = _summary_text(meta, tracks, triage_rows)
    (out / "summary.txt").write_text(text)
    sys.stdout.write(text)
    if args.demo:
        from glance_list import write_glance_list
        from render_overlay import FRAMES_NAME, render, save_frames

        save_frames(out / FRAMES_NAME, frames)
        render(
            out,
            frames,
            triage,
            clip=None if args.synthetic else Path(meta["clip"]),
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
    return 3 if failures else 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
