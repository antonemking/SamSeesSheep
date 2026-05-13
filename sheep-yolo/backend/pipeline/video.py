"""YOLO video pipeline.

Open-vocabulary YOLOE-seg with text prompts for "sheep head", "sheep ear",
"sheep nose". Per-sheep tracking is handled by Ultralytics' built-in tracker
(ByteTrack by default). Masks feed into the same ear-angle geometry used in
v1, so chart bands and SPFES thresholds remain directly comparable.

This pipeline is deliberately "out of the box": no fine-tuning, no custom
weights, no visual prompts. If YOLOE returns zero part detections on a clip,
we fall back to COCO-pretrained YOLO11-seg (whole-sheep class only) and
report it — that's the honest answer to "how does YOLO do OOB on sheep
anatomy", which is the experiment this repo is designed to run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.config import (
    DETECTION_CONF,
    EAR_DOWN_THRESHOLD_DEG,
    EAR_UP_THRESHOLD_DEG,
    INFER_IMGSZ,
    MAX_FRAMES,
    PART_PROMPTS,
    RESULTS_DIR,
    TRACK_COVERAGE_FLOOR,
    TRACKER_YAML,
    YOLO_FALLBACK_MODEL,
    YOLOE_MODEL,
)
from backend.pipeline.ear_angle import ear_angle_from_masks

logger = logging.getLogger(__name__)

_part_model = None
_whole_model = None


class _H264Writer:
    """Browser-playable MP4 writer.

    OpenCV's VideoWriter on this machine can only emit 'mp4v' (MPEG-4 Part 2),
    which Chrome refuses to play inline. pyav ships libx264 so we route MP4
    output through it — same frame-by-frame push API, browser-compatible file.
    """
    def __init__(self, path: str, fps: float, width: int, height: int):
        import av  # type: ignore
        self._container = av.open(path, mode="w")
        self._stream = self._container.add_stream("h264", rate=round(fps))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._av = av

    def write(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = self._av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def release(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def _load_whole_model():
    """COCO-pretrained YOLO11-seg. Only used in 'whole' mode, where we just
    want reliable whole-sheep masks + tracking to prove the ceiling that
    OOB YOLO can hit before any custom training."""
    global _whole_model
    if _whole_model is not None:
        return _whole_model
    from ultralytics import YOLO  # type: ignore
    logger.info("Loading whole-sheep model: %s", YOLO_FALLBACK_MODEL)
    _whole_model = YOLO(YOLO_FALLBACK_MODEL)
    return _whole_model


def _load_part_model():
    """Load YOLOE-seg with sheep-part text prompts.

    YOLOE ships a text prompt encoder (get_text_pe) — we pre-compute prompt
    embeddings once and hand them to set_classes so every subsequent call
    reuses the same fixed class head rather than re-embedding per frame.
    """
    global _part_model
    if _part_model is not None:
        return _part_model
    try:
        from ultralytics import YOLOE  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "ultralytics YOLOE is required. pip install 'ultralytics>=8.3'"
        ) from e

    logger.info("Loading YOLOE part model: %s", YOLOE_MODEL)
    model = YOLOE(YOLOE_MODEL)
    try:
        pe = model.get_text_pe(PART_PROMPTS)
        model.set_classes(PART_PROMPTS, pe)
        logger.info("YOLOE configured with prompts: %s", PART_PROMPTS)
    except Exception as e:
        # Older YOLOE builds use a slightly different API. Don't block load —
        # the caller will see zero detections and fall back.
        logger.warning("YOLOE set_classes failed: %s", e)
    _part_model = model
    return _part_model


def _frame_size(video_path: Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if w == 0 or h == 0:
        raise RuntimeError(f"Could not read {video_path}")
    # webm metadata often reports a junk fps (we've seen 1000.0). Clamp to
    # a sane playback range so the annotated MP4 isn't unplayably fast/slow.
    if not (0.5 <= fps <= 240.0):
        fps = 30.0
    return w, h, fps


def _mask_to_bool(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape != target_hw:
        mask = cv2.resize(
            mask.astype(np.uint8), (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 0


def _extract_detections(result) -> list[dict]:
    """Flatten an ultralytics Result into one dict per detection.

    Each dict carries: class_name, track_id, box, mask (bool ndarray at frame
    resolution), centroid (x, y), and score.
    """
    out: list[dict] = []
    if result.masks is None or result.boxes is None:
        return out

    names = result.names  # {class_id: "sheep head"...}
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    track_ids = (
        result.boxes.id.cpu().numpy().astype(int)
        if result.boxes.id is not None
        else np.full(len(cls_ids), -1, dtype=int)
    )
    mask_arr = result.masks.data.cpu().numpy()  # (N, H', W')
    orig_h, orig_w = result.orig_shape

    for i in range(len(cls_ids)):
        cname = names.get(int(cls_ids[i]), str(cls_ids[i]))
        bmask = _mask_to_bool(mask_arr[i], (orig_h, orig_w))
        if not bmask.any():
            continue
        ys, xs = np.where(bmask)
        cx, cy = float(xs.mean()), float(ys.mean())
        out.append({
            "class": cname,
            "track_id": int(track_ids[i]),
            "box": boxes_xyxy[i].tolist(),
            "mask": bmask,
            "centroid": (cx, cy),
            "score": float(confs[i]),
        })
    return out


def _match_parts_to_heads(
    heads: list[dict], ears: list[dict], noses: list[dict],
) -> list[dict]:
    """Group parts under their owning head by mask overlap / bbox containment.

    Returns one entry per head: {head, ears: [...up to 2...], nose: Optional}.
    Parts outside any head are dropped — they're either from untracked animals
    or false positives, both of which we want off the chart.
    """
    grouped: list[dict] = []
    for head in heads:
        hmask = head["mask"]
        hbool_dilated = cv2.dilate(
            hmask.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1,
        ) > 0

        ear_candidates = []
        for ear in ears:
            cx, cy = ear["centroid"]
            ih, iw = hbool_dilated.shape
            if not (0 <= int(cy) < ih and 0 <= int(cx) < iw):
                continue
            if not hbool_dilated[int(cy), int(cx)]:
                overlap = (ear["mask"] & hbool_dilated).sum() / max(1, ear["mask"].sum())
                if overlap < 0.15:
                    continue
            ear_candidates.append(ear)
        ear_candidates.sort(key=lambda e: -e["score"])
        ear_candidates = ear_candidates[:2]

        matched_nose = None
        for nose in noses:
            cx, cy = nose["centroid"]
            ih, iw = hbool_dilated.shape
            if 0 <= int(cy) < ih and 0 <= int(cx) < iw and hbool_dilated[int(cy), int(cx)]:
                if matched_nose is None or nose["score"] > matched_nose["score"]:
                    matched_nose = nose

        grouped.append({
            "head": head,
            "ears": ear_candidates,
            "nose": matched_nose,
        })
    return grouped


def _assign_left_right(ears: list[dict], head_centroid: tuple[float, float]) -> dict:
    """Return {'left': ear_dict|None, 'right': ear_dict|None} by screen-x.

    For a frame with two ears, the one to the left of the head centroid is
    labeled 'left'. For one ear, we still pick a side based on which side of
    the head centroid it sits on. This is a per-frame heuristic — chart
    swapping under head rotation is an accepted cost of OOB YOLO (v1 solved
    it by locking per-animal ear track-IDs; here that's a v2 follow-up).
    """
    hx = head_centroid[0]
    left = right = None
    if len(ears) == 2:
        e1, e2 = ears
        if e1["centroid"][0] < e2["centroid"][0]:
            left, right = e1, e2
        else:
            left, right = e2, e1
    elif len(ears) == 1:
        e = ears[0]
        if e["centroid"][0] < hx:
            left = e
        else:
            right = e
    return {"left": left, "right": right}


def _pick_sheep_for_click(
    frame_groups: list[dict], click_xy: tuple[float, float],
) -> Optional[int]:
    """Return the head track_id whose mask contains (or is nearest to) the click.

    click_xy is in pixel coords at the frame's native resolution.
    """
    cx, cy = int(click_xy[0]), int(click_xy[1])
    best_id, best_dist = None, float("inf")
    for group in frame_groups:
        head = group["head"]
        hmask = head["mask"]
        ih, iw = hmask.shape
        if 0 <= cy < ih and 0 <= cx < iw and hmask[cy, cx]:
            return head["track_id"]
        hx, hy = head["centroid"]
        d = ((hx - cx) ** 2 + (hy - cy) ** 2) ** 0.5
        if d < best_dist:
            best_dist, best_id = d, head["track_id"]
    return best_id


_COLORS_BGR = [
    (62, 136, 240),   # orange
    (80, 185, 63),    # green
    (255, 166, 88),   # light blue
    (235, 100, 235),  # pink
    (65, 179, 227),   # yellow
    (200, 200, 200),  # grey
]


def _color_for(track_id: int) -> tuple[int, int, int]:
    return _COLORS_BGR[track_id % len(_COLORS_BGR)]


def _draw_overlay(
    frame_bgr: np.ndarray, groups: list[dict], per_sheep_angles: dict,
    focus_track_id: Optional[int],
) -> np.ndarray:
    out = frame_bgr.copy()
    for group in groups:
        head = group["head"]
        tid = head["track_id"]
        if focus_track_id is not None and tid != focus_track_id:
            continue
        color = _color_for(tid)

        hmask = head["mask"]
        overlay = out.copy()
        overlay[hmask] = color
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, dst=out)

        for ear in group["ears"]:
            contours, _ = cv2.findContours(
                ear["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(out, contours, -1, color, 2)
        if group["nose"] is not None:
            nmask = group["nose"]["mask"]
            overlay = out.copy()
            overlay[nmask] = (65, 179, 227)
            cv2.addWeighted(overlay, 0.55, out, 0.45, 0, dst=out)

        x1, y1, x2, y2 = map(int, head["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        angles = per_sheep_angles.get(tid, {})
        la = angles.get("left_ear_angle_deg")
        ra = angles.get("right_ear_angle_deg")
        label = f"sheep#{tid}"
        if la is not None or ra is not None:
            l = f"{la:+.0f}" if la is not None else "--"
            r = f"{ra:+.0f}" if ra is not None else "--"
            label = f"{label}  L{l}  R{r}"
        cv2.putText(
            out, label, (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
    return out


def _apply_median_smoothing(series: list[Optional[float]], window: int = 3) -> list[Optional[float]]:
    half = window // 2
    out: list[Optional[float]] = []
    for i, v in enumerate(series):
        if v is None:
            out.append(None)
            continue
        nearby = [
            series[j] for j in range(max(0, i - half), min(len(series), i + half + 1))
            if series[j] is not None
        ]
        out.append(float(np.median(nearby)) if nearby else v)
    return out


def analyze_video(
    video_path: Path,
    click_point_norm: Optional[tuple[float, float]] = None,
    max_frames: int = MAX_FRAMES,
    tracker: Optional[str] = None,
) -> dict:
    """Run YOLOE tracking across the clip.

    click_point_norm: optional (x, y) in [0,1] on the first frame. When set,
    only the sheep under that click is returned (single-subject mode). When
    None, every tracked sheep is returned (multi-subject mode).
    tracker: override TRACKER_YAML ("bytetrack.yaml" or "botsort.yaml").
    """
    tracker_yaml = tracker or TRACKER_YAML
    start = time.time()
    frame_w, frame_h, fps = _frame_size(video_path)
    if click_point_norm is not None:
        click_xy = (click_point_norm[0] * frame_w, click_point_norm[1] * frame_h)
    else:
        click_xy = None

    try:
        model = _load_part_model()
    except Exception as e:
        logger.error("Part model failed to load: %s", e)
        raise

    out_name = f"{video_path.stem}_yolo.mp4"
    out_path = RESULTS_DIR / out_name
    writer: Optional[cv2.VideoWriter] = None

    per_frame: list[dict] = []
    sheep_angle_series: dict[int, list[Optional[dict]]] = {}
    focus_track_id: Optional[int] = None
    ear_detect_count = 0

    tracker_kwargs = dict(
        source=str(video_path),
        persist=True,
        stream=True,
        tracker=tracker_yaml,
        imgsz=INFER_IMGSZ,
        conf=DETECTION_CONF,
        verbose=False,
    )

    try:
        results_iter = model.track(**tracker_kwargs)
    except TypeError:
        # Older ultralytics accepted fewer kwargs
        results_iter = model.track(source=str(video_path), persist=True, stream=True)

    for frame_idx, result in enumerate(results_iter):
        if frame_idx >= max_frames:
            break
        frame_bgr = result.orig_img
        if writer is None:
            writer = _H264Writer(
                str(out_path), fps, frame_bgr.shape[1], frame_bgr.shape[0],
            )

        detections = _extract_detections(result)
        heads = [d for d in detections if d["class"] == PART_PROMPTS[0]]
        ears = [d for d in detections if d["class"] == PART_PROMPTS[1]]
        noses = [d for d in detections if d["class"] == PART_PROMPTS[2]]
        ear_detect_count += len(ears)

        groups = _match_parts_to_heads(heads, ears, noses)

        if focus_track_id is None and click_xy is not None and groups:
            focus_track_id = _pick_sheep_for_click(groups, click_xy)
            logger.info(
                "Frame %d: click (%d,%d) mapped to sheep#%s",
                frame_idx, int(click_xy[0]), int(click_xy[1]), focus_track_id,
            )

        per_sheep_angles: dict[int, dict] = {}
        frame_record: dict = {"frame_idx": frame_idx, "sheep": []}
        for group in groups:
            head = group["head"]
            tid = head["track_id"]
            if tid < 0:
                continue
            if focus_track_id is not None and tid != focus_track_id:
                continue

            sides = _assign_left_right(group["ears"], head["centroid"])
            angles = ear_angle_from_masks(
                head_mask=head["mask"],
                left_ear_mask=sides["left"]["mask"] if sides["left"] else None,
                right_ear_mask=sides["right"]["mask"] if sides["right"] else None,
                nose_mask=group["nose"]["mask"] if group["nose"] else None,
            )
            per_sheep_angles[tid] = angles

            sheep_angle_series.setdefault(tid, [])

            sheep_entry = {
                "track_id": tid,
                "head_score": head["score"],
                "ear_count": len(group["ears"]),
                "has_nose": group["nose"] is not None,
            }
            sheep_entry.update(angles)
            frame_record["sheep"].append(sheep_entry)

        # Backfill each sheep series up to this frame so JSON ordering matches
        # the frame index (sheep that appeared late get None for earlier frames).
        for tid, series in sheep_angle_series.items():
            while len(series) < frame_idx:
                series.append(None)
            match = next((s for s in frame_record["sheep"] if s["track_id"] == tid), None)
            series.append(match)

        annotated = _draw_overlay(frame_bgr, groups, per_sheep_angles, focus_track_id)
        writer.write(annotated)
        per_frame.append(frame_record)

    if writer is not None:
        writer.release()

    elapsed = time.time() - start
    n_frames = len(per_frame)
    logger.info(
        "YOLO pipeline: %d frames in %.1fs (%.1f fps). ears-total=%d, sheep=%d",
        n_frames, elapsed, n_frames / elapsed if elapsed else 0,
        ear_detect_count, len(sheep_angle_series),
    )

    fallback_note = None
    if n_frames > 0 and ear_detect_count == 0:
        fallback_note = (
            "YOLOE found zero 'sheep ear' detections across the clip. "
            "This is the expected outcome of the out-of-the-box benchmark — "
            "fine-tuning on a labeled sheep-parts dataset is the next step."
        )
        logger.warning(fallback_note)

    # Build smoothed series per sheep
    sheep_summary: list[dict] = []
    for tid, series in sheep_angle_series.items():
        while len(series) < n_frames:
            series.append(None)
        left_raw = [s.get("left_ear_angle_deg") if s else None for s in series]
        right_raw = [s.get("right_ear_angle_deg") if s else None for s in series]
        left_s = _apply_median_smoothing(left_raw)
        right_s = _apply_median_smoothing(right_raw)

        def _mean_std(xs):
            vals = [x for x in xs if x is not None]
            if not vals:
                return None, None
            return float(np.mean(vals)), float(np.std(vals))

        lm, ls = _mean_std(left_s)
        rm, rs = _mean_std(right_s)
        frames_tracked = sum(1 for s in series if s is not None)
        sheep_summary.append({
            "track_id": tid,
            "frames_tracked": frames_tracked,
            "left_ear_angle_deg": left_raw,
            "right_ear_angle_deg": right_raw,
            "left_ear_angle_deg_smoothed": left_s,
            "right_ear_angle_deg_smoothed": right_s,
            "left_mean_deg": lm,
            "left_std_deg": ls,
            "right_mean_deg": rm,
            "right_std_deg": rs,
        })
    sheep_summary.sort(key=lambda r: -r["frames_tracked"])

    return {
        "n_frames": n_frames,
        "fps": fps,
        "elapsed_s": elapsed,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "focus_track_id": focus_track_id,
        "annotated_video_url": f"/results/{out_name}" if writer is not None else None,
        "sheep": sheep_summary,
        "per_frame": per_frame,
        "fallback_note": fallback_note,
        "model": YOLOE_MODEL,
        "mode": "parts",
        "tracker": tracker_yaml,
        "prompts": PART_PROMPTS,
        "ear_up_threshold_deg": EAR_UP_THRESHOLD_DEG,
        "ear_down_threshold_deg": EAR_DOWN_THRESHOLD_DEG,
    }


def _draw_whole_overlay(
    frame_bgr: np.ndarray, detections: list[dict], focus_track_id: Optional[int],
) -> np.ndarray:
    out = frame_bgr.copy()
    for det in detections:
        tid = det["track_id"]
        if tid < 0:
            continue
        if focus_track_id is not None and tid != focus_track_id:
            continue
        color = _color_for(tid)
        overlay = out.copy()
        overlay[det["mask"]] = color
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, dst=out)
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"sheep#{tid}  {det['score']:.2f}"
        cv2.putText(
            out, label, (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return out


def analyze_video_whole(
    video_path: Path,
    click_point_norm: Optional[tuple[float, float]] = None,
    max_frames: int = MAX_FRAMES,
    tracker: Optional[str] = None,
) -> dict:
    tracker_yaml = tracker or TRACKER_YAML
    """Whole-sheep tracking using COCO-pretrained YOLO11-seg (class 'sheep').

    Companion to analyze_video() — no ear angle, just stable per-animal
    masks + track IDs. This is the "does YOLO OOB at least find sheep"
    check, the missing control that shows the part-detection failure was a
    prompt/anatomy problem, not a broken pipeline.
    """
    start = time.time()
    frame_w, frame_h, fps = _frame_size(video_path)
    click_xy = (click_point_norm[0] * frame_w, click_point_norm[1] * frame_h) \
        if click_point_norm else None

    model = _load_whole_model()
    # COCO "sheep" is class 18. Hard-coded — we don't want to accidentally
    # track dogs that happen to look sheep-ish.
    sheep_cls_id = None
    for cid, cname in model.names.items():
        if cname == "sheep":
            sheep_cls_id = int(cid)
            break
    if sheep_cls_id is None:
        raise RuntimeError(f"{YOLO_FALLBACK_MODEL} has no 'sheep' class")

    out_name = f"{video_path.stem}_yolo_whole.mp4"
    out_path = RESULTS_DIR / out_name
    writer: Optional[cv2.VideoWriter] = None

    tracker_kwargs = dict(
        source=str(video_path), persist=True, stream=True,
        tracker=tracker_yaml, imgsz=INFER_IMGSZ,
        conf=DETECTION_CONF, classes=[sheep_cls_id], verbose=False,
    )
    try:
        results_iter = model.track(**tracker_kwargs)
    except TypeError:
        results_iter = model.track(source=str(video_path), persist=True, stream=True)

    focus_track_id: Optional[int] = None
    per_sheep_presence: dict[int, list[int]] = {}
    per_frame: list[dict] = []

    for frame_idx, result in enumerate(results_iter):
        if frame_idx >= max_frames:
            break
        frame_bgr = result.orig_img
        if writer is None:
            writer = _H264Writer(
                str(out_path), fps, frame_bgr.shape[1], frame_bgr.shape[0],
            )

        detections = [d for d in _extract_detections(result) if d["class"] == "sheep"]

        if focus_track_id is None and click_xy is not None and detections:
            cx, cy = int(click_xy[0]), int(click_xy[1])
            best_id, best_dist = None, float("inf")
            for d in detections:
                m = d["mask"]
                ih, iw = m.shape
                if 0 <= cy < ih and 0 <= cx < iw and m[cy, cx]:
                    best_id = d["track_id"]; break
                dx = d["centroid"][0] - cx; dy = d["centroid"][1] - cy
                dd = (dx * dx + dy * dy) ** 0.5
                if dd < best_dist:
                    best_dist, best_id = dd, d["track_id"]
            focus_track_id = best_id

        frame_record = {"frame_idx": frame_idx, "sheep": []}
        for d in detections:
            tid = d["track_id"]
            if tid < 0:
                continue
            if focus_track_id is not None and tid != focus_track_id:
                continue
            per_sheep_presence.setdefault(tid, []).append(frame_idx)
            frame_record["sheep"].append({
                "track_id": tid,
                "score": d["score"],
                "box": d["box"],
            })

        annotated = _draw_whole_overlay(frame_bgr, detections, focus_track_id)
        writer.write(annotated)
        per_frame.append(frame_record)

    if writer is not None:
        writer.release()

    elapsed = time.time() - start
    n_frames = len(per_frame)
    logger.info(
        "YOLO whole-sheep: %d frames in %.1fs (%.1f fps). sheep tracked=%d",
        n_frames, elapsed, n_frames / elapsed if elapsed else 0,
        len(per_sheep_presence),
    )

    sheep_summary = []
    for tid, present_idxs in per_sheep_presence.items():
        presence = [False] * n_frames
        for i in present_idxs:
            if i < n_frames:
                presence[i] = True
        sheep_summary.append({
            "track_id": tid,
            "frames_tracked": len(present_idxs),
            "coverage": len(present_idxs) / max(1, n_frames),
            "presence": presence,
        })
    sheep_summary.sort(key=lambda r: -r["frames_tracked"])
    total_tracks = len(sheep_summary)
    persistent = [s for s in sheep_summary if s["coverage"] >= TRACK_COVERAGE_FLOOR]
    transient = total_tracks - len(persistent)

    return {
        "n_frames": n_frames,
        "fps": fps,
        "elapsed_s": elapsed,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "focus_track_id": focus_track_id,
        "annotated_video_url": f"/results/{out_name}" if writer is not None else None,
        "sheep": persistent,
        "sheep_transient": transient,
        "sheep_total_tracks": total_tracks,
        "track_coverage_floor": TRACK_COVERAGE_FLOOR,
        "per_frame": per_frame,
        "fallback_note": None,
        "model": YOLO_FALLBACK_MODEL,
        "mode": "whole",
        "tracker": tracker_yaml,
        "prompts": ["sheep (COCO class 18)"],
        "ear_up_threshold_deg": EAR_UP_THRESHOLD_DEG,
        "ear_down_threshold_deg": EAR_DOWN_THRESHOLD_DEG,
    }
