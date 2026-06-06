"""Roboflow `supervision` spike (LOR-123) — does the supervision toolkit buy us
anything over the hand-rolled OpenCV annotation in `render_ekg.py`?

This is a *throwaway probe*, not a production path. It runs v0.7 YOLO-pose over a
short bounded sample of one test clip and renders an annotated MP4 using
supervision's video IO + keypoint annotators, plus a `PolygonZone` evaluation of
the existing ROI. The ear-angle geometry is reused verbatim from `render_ekg.py`
so numbers stay comparable.

supervision is NOT a declared dependency of this repo (see docs/supervision-spike.md
for the rationale). Run it one-off with uv:

    cd sheep-yolo
    uv run --with supervision scripts/render_supervision_spike.py

CAVEAT: `sv.ByteTrack` is deprecated in supervision 0.28.0 (tracking is moving to a
separate `trackers` package). This spike deliberately does NOT track — it runs
per-frame detection only. The production EKG renderer keeps using Ultralytics'
built-in ByteTrack via `model.track(...)`. Do not migrate tracking here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Reuse the ear-angle math + skeleton from the production EKG renderer rather
# than re-deriving it — keeps the spike honest and the numbers comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_ekg import SKEL, ears  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Same ROI the EKG renderer uses to pick out the calm ewe (x1, y1, x2, y2).
ROI = (1220, 80, 1560, 390)
MAGENTA = sv.Color(r=255, g=0, b=255)
CYAN = sv.Color(r=0, g=200, b=255)
KPT_TH = 0.4
# supervision's EdgeAnnotator uses 1-based keypoint indices; our SKEL is
# zero-based because the rest of SamSeesSheep indexes the five YOLO-pose points
# directly as nose/l-base/r-base/l-tip/r-tip.
SV_SKEL = [(a + 1, b + 1) for a, b in SKEL]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", default=str(ROOT.parent / "test-clips" / "IMG_3651.MOV"),
                   help="source video")
    p.add_argument("--weights", default="weights/sheep-pose-v0.7-yolo26n.pt",
                   help="YOLO-pose weights (relative to sheep-yolo/ or absolute)")
    p.add_argument("--output", default="artifacts/supervision-spike-IMG_3651.mp4",
                   help="annotated MP4 output (gitignored artifacts/ dir)")
    p.add_argument("--max-frames", type=int, default=90,
                   help="bounded sample length so the spike verifies quickly")
    p.add_argument("--start-frame", type=int, default=0,
                   help="skip ahead before sampling; the calm ewe enters the ROI "
                        "around frame ~380, so --start-frame 380 populates the zone")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--raw-sv-keypoint-annotators", action="store_true",
                   help="draw raw supervision keypoint annotators without the "
                        "SamSeesSheep confidence filter; useful only for API "
                        "inspection, not showcase output")
    return p.parse_args()


def resolve(path: str) -> Path:
    """Allow weights/output to be given relative to sheep-yolo/ or absolutely."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def roi_polygon() -> np.ndarray:
    x1, y1, x2, y2 = ROI
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def ear_text(kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> tuple[str, tuple[int, int]] | None:
    """Re-pack supervision KeyPoints back into the [x, y, conf] rows ears() wants,
    return a label + anchor (the nose) or None if nose is missing."""
    k = np.concatenate([kpts_xy, kpts_conf[:, None]], axis=1)  # (5, 3)
    if k[0, 2] <= KPT_TH:
        return None
    lft, rgt = ears(k)
    parts = []
    if not np.isnan(lft):
        parts.append(f"L{lft:.0f}")
    if not np.isnan(rgt):
        parts.append(f"R{rgt:.0f}")
    if not parts:
        return None
    return " ".join(parts), (int(k[0, 0]), int(k[0, 1]))


def edge_safe_keypoints(keypoints: sv.KeyPoints) -> sv.KeyPoints:
    """Return a copy with low-confidence points zeroed for EdgeAnnotator.

    supervision's keypoint annotators are generic drawing helpers: they do not
    apply our YOLO-pose confidence threshold. EdgeAnnotator skips zeroed points,
    so this keeps low-confidence ears/noses from producing noisy skeleton lines.
    """
    if keypoints.confidence is None or len(keypoints) == 0:
        return keypoints
    xy = keypoints.xy.copy()
    xy[keypoints.confidence <= KPT_TH] = 0
    return sv.KeyPoints(
        xy=xy,
        class_id=keypoints.class_id,
        confidence=keypoints.confidence,
        data=keypoints.data,
    )


def draw_confident_vertices(scene: np.ndarray, keypoints: sv.KeyPoints) -> None:
    """Draw only keypoints above SamSeesSheep's confidence threshold.

    VertexAnnotator currently draws every xy coordinate it receives, regardless
    of confidence. That made the first spike output look worse than the original
    EKG renderer. Keep showcase visuals confidence-aware.
    """
    if len(keypoints) == 0:
        return
    conf = (keypoints.confidence if keypoints.confidence is not None
            else np.ones(keypoints.xy.shape[:2]))
    for pts, scores in zip(keypoints.xy, conf):
        for (x, y), score in zip(pts, scores):
            if score <= KPT_TH:
                continue
            cv2.circle(scene, (int(x), int(y)), 5, (255, 0, 255), -1)


def main() -> None:
    args = parse_args()
    clip = resolve(args.clip)
    weights = resolve(args.weights)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    print(f"supervision {sv.__version__}")
    model = YOLO(str(weights))

    # --- supervision video IO ---
    video_info = sv.VideoInfo.from_video_path(str(clip))
    print(f"video: {video_info.width}x{video_info.height} @ {video_info.fps}fps, "
          f"{video_info.total_frames} frames; sampling first {args.max_frames}")
    frames = sv.get_video_frames_generator(source_path=str(clip), start=args.start_frame)

    # --- supervision annotators ---
    vertex_annotator = sv.VertexAnnotator(color=MAGENTA, radius=5)
    edge_annotator = sv.EdgeAnnotator(color=MAGENTA, thickness=2, edges=SV_SKEL)
    box_annotator = sv.BoxAnnotator(thickness=1)

    # --- supervision zone (ROI) evaluation ---
    zone = sv.PolygonZone(polygon=roi_polygon())
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=CYAN, thickness=2)

    n_det = 0
    n_in_zone = 0
    frames_with_zone_hit = 0
    written = 0

    with sv.VideoSink(target_path=str(output), video_info=video_info) as sink:
        for i, frame in enumerate(frames):
            if i >= args.max_frames:
                break
            result = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

            detections = sv.Detections.from_ultralytics(result)
            keypoints = sv.KeyPoints.from_ultralytics(result)

            in_zone = zone.trigger(detections=detections)  # bool per detection
            n_det += len(detections)
            n_in_zone += int(in_zone.sum())
            if in_zone.any():
                frames_with_zone_hit += 1

            annotated = frame.copy()
            annotated = box_annotator.annotate(annotated, detections)
            if args.raw_sv_keypoint_annotators:
                # API inspection mode only: draws all keypoint xy values exactly
                # as supervision receives them, without SamSeesSheep filtering.
                annotated = edge_annotator.annotate(annotated, keypoints)
                annotated = vertex_annotator.annotate(annotated, keypoints)
            else:
                annotated = edge_annotator.annotate(annotated, edge_safe_keypoints(keypoints))
                draw_confident_vertices(annotated, keypoints)
            annotated = zone_annotator.annotate(annotated)

            # Overlay reused ear-angle readout for instances inside the ROI.
            if keypoints.xy is not None and len(keypoints.xy):
                conf = (keypoints.confidence if keypoints.confidence is not None
                        else np.ones(keypoints.xy.shape[:2]))
                for d in range(len(keypoints.xy)):
                    if d < len(in_zone) and not in_zone[d]:
                        continue
                    label = ear_text(keypoints.xy[d], conf[d])
                    if label is None:
                        continue
                    text, anchor = label
                    cv2.putText(annotated, text, (anchor[0] + 8, anchor[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

            sink.write_frame(annotated)
            written += 1

    print(f"wrote {output}  ({written} frames)")
    print(f"detections: {n_det} total, {n_in_zone} inside ROI; "
          f"{frames_with_zone_hit}/{written} frames had a sheep in the ROI")


if __name__ == "__main__":
    main()
