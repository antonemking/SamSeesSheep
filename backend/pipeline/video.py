"""Video segmentation + tracking with SAM 3 Video.

Processes a video as a sequence of frames. Uses SAM 3's text-prompted
video tracker to propagate masks across frames — much faster than
running SAM 3 per-frame independently.

Output: per-frame ear angles over time.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-loaded model cache
_video_model = None
_video_processor = None
_device = None


def _free_sam3_image_model():
    """Unload the SAM 3 image model to free VRAM for the video model."""
    import gc
    import torch
    from backend.pipeline import segment as _seg
    if _seg._sam3_model is not None:
        logger.info("Unloading SAM 3 image model to free VRAM...")
        _seg._sam3_model.cpu()
        _seg._sam3_model = None
        _seg._sam3_processor = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_video_model():
    """Load SAM 3 Video model from HuggingFace.

    Uses fp32 — Turing GPUs (1660 Ti) don't accelerate fp16. The
    PYTORCH_CUDA_ALLOC_CONF setting reduces memory fragmentation so
    we can fit the model + per-frame inference state on 6GB.
    """
    global _video_model, _video_processor, _device
    try:
        import os
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        # Free image model first — they don't both fit on a 6GB GPU
        _free_sam3_image_model()

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading SAM 3 Video on %s...", _device)
        _video_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
        _video_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(_device)
        _video_model.eval()
        logger.info(
            "SAM 3 Video loaded. VRAM allocated: %.2f GB",
            torch.cuda.memory_allocated() / 1024 ** 3,
        )
    except Exception as e:
        logger.error("Failed to load SAM 3 Video: %s", e)
        _video_model = None


def _extract_frames(
    video_path: Path, max_frames: int = 30, target_fps: float = 2.0
) -> list[Image.Image]:
    """Extract evenly-spaced frames from a video as PIL Images.

    Falls back to even sampling when fps metadata is unreliable
    (common for webm files where pyav can return time-base values).
    """
    import imageio.v3 as iio

    frames = iio.imread(str(video_path), plugin="pyav")
    total = len(frames)
    if total == 0:
        return []

    # Try to read fps; webm often reports junk values (e.g. 1000)
    actual_fps = 30.0
    try:
        meta = iio.immeta(str(video_path), plugin="pyav")
        meta_fps = float(meta.get("fps", 30.0))
        if 0.5 <= meta_fps <= 240:
            actual_fps = meta_fps
    except Exception:
        pass

    # Sample target_fps frames per second of video, clamped to max_frames.
    # If we don't have many frames, just take them all (up to max_frames).
    if total <= max_frames:
        target_count = total
    else:
        duration_s = total / actual_fps
        target_count = max(min(max_frames, total),
                           min(max_frames, int(duration_s * target_fps)))
        target_count = max(target_count, min(max_frames, 8))  # at least 8 frames

    idxs = np.linspace(0, total - 1, target_count).astype(int)
    # Downscale to keep VRAM usage bounded (SAM 3 internal res is small anyway)
    MAX_DIM = 768
    pil_frames = []
    for i in idxs:
        img = Image.fromarray(frames[i])
        w, h = img.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        pil_frames.append(img)
    logger.info(
        "Video: %d total frames, fps_used=%.1f, sampling %d frames at %s",
        total, actual_fps, len(pil_frames), pil_frames[0].size,
    )
    return pil_frames


def _pick_primary_object(
    obj_id_to_score: dict, obj_id_to_mask: dict, frame_h: int, frame_w: int
) -> Optional[int]:
    """Pick the object ID that looks most like the target head/body.

    Prefers high score AND size in a reasonable range (2-60% of frame).
    """
    best_id = None
    best_metric = 0.0
    for oid, score in obj_id_to_score.items():
        mask_logits = obj_id_to_mask.get(oid)
        if mask_logits is None:
            continue
        area_frac = float((mask_logits.squeeze(0) > 0).sum().item()) / (frame_h * frame_w)
        if area_frac < 0.005 or area_frac > 0.75:
            continue
        metric = score * (1.0 if 0.02 <= area_frac <= 0.50 else 0.5)
        if metric > best_metric:
            best_metric = metric
            best_id = oid
    return best_id


def _mask_from_logits(
    mask_logits, threshold: float = 0.0,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert SAM 3 mask logits tensor to a uint8 binary mask.

    Resizes to target_size (H, W) if provided, since SAM 3 Video outputs
    masks at its internal resolution, not the input frame resolution.
    """
    mask = (mask_logits.squeeze(0) > threshold).cpu().numpy().astype(np.uint8) * 255
    if target_size is not None and mask.shape != target_size:
        h, w = target_size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def analyze_video(
    video_path: Path,
    subject: str = "sheep",
    max_frames: int = 30,
    click_point: Optional[tuple[float, float]] = None,
) -> dict:
    """Run SAM 3 Video tracking across a clip.

    Args:
        click_point: optional (x, y) normalized 0-1 click on frame 0. When
            provided, we pick the tracked object whose frame-0 mask contains
            (or is nearest to) this point, locking tracking to that subject.

    Returns a dict with per-frame ear angles and summary stats.
    """
    global _video_model

    if _video_model is None:
        _load_video_model()
    if _video_model is None:
        raise RuntimeError("SAM 3 Video model failed to load")

    import torch
    from backend.pipeline.ear_angle import (
        _compute_anatomical_midline,
        _compute_ear_direction,
        _normalize_angle,
        _classify_ear_position,
    )

    start = time.time()
    pil_frames = _extract_frames(video_path, max_frames=max_frames)
    if not pil_frames:
        return {"error": "no frames extracted", "per_frame": []}

    frame_w, frame_h = pil_frames[0].size

    # Three separate sessions (one per prompt). Single-session multi-prompt
    # hits OOM on 6GB GPUs because each additional prompt adds tracking state.
    head_key = f"{subject} head"
    ear_key = f"{subject} ear"
    nose_key = f"{subject} nose"
    prompt_list = [head_key, ear_key, nose_key]

    results_per_prompt: dict[str, list[dict]] = {}
    for prompt_text in prompt_list:
        t0 = time.time()
        session = _video_processor.init_video_session(
            video=pil_frames, inference_device=_device
        )
        _video_processor.add_text_prompt(session, prompt_text)

        per_frame_outputs = []
        for i, output in enumerate(
            _video_model.propagate_in_video_iterator(
                session, show_progress_bar=False
            )
        ):
            per_frame_outputs.append({
                "obj_id_to_mask": {k: v.cpu() for k, v in output.obj_id_to_mask.items()},
                "obj_id_to_score": dict(output.obj_id_to_score),
                "object_ids": list(output.object_ids),
            })
        results_per_prompt[prompt_text] = per_frame_outputs

        logger.info(
            "Tracked '%s' across %d frames in %.1fs",
            prompt_text, len(pil_frames), time.time() - t0,
        )
        # Free tracking state from this prompt before the next session
        session.reset_inference_session()
        del session
        torch.cuda.empty_cache()

    per_frame = []

    # Pick a stable primary ID per prompt.
    # If a click_point was provided, pick the object whose frame-0 mask
    # contains or is nearest to that point — locks tracking to that subject.
    def click_primary(prompt_results: list[dict]) -> Optional[int]:
        if not click_point or not prompt_results:
            return None
        cx_norm, cy_norm = click_point
        target_x, target_y = cx_norm * frame_w, cy_norm * frame_h
        frame0 = prompt_results[0]
        best_id, best_dist = None, float("inf")
        for oid, mask_t in frame0["obj_id_to_mask"].items():
            m = _mask_from_logits(mask_t, target_size=(frame_h, frame_w))
            if m.sum() < 20:
                continue
            # If click is inside the mask, distance = 0 (best match)
            if m[int(target_y), int(target_x)] > 127:
                return oid
            # Else use centroid distance as fallback
            coords = np.where(m > 127)
            cy, cx = coords[0].mean(), coords[1].mean()
            d = ((cy - target_y) ** 2 + (cx - target_x) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_id = oid
        return best_id

    def stable_primary(prompt_results: list[dict]) -> Optional[int]:
        votes = {}
        for fr in prompt_results[:min(5, len(prompt_results))]:
            pid = _pick_primary_object(
                fr["obj_id_to_score"], fr["obj_id_to_mask"], frame_h, frame_w
            )
            if pid is not None:
                votes[pid] = votes.get(pid, 0) + 1
        if not votes:
            return None
        return max(votes, key=votes.get)

    # For head: click selection is critical (determines which animal we track)
    # Log candidates to help debug which animal got selected
    if click_point is not None:
        frame0_heads = results_per_prompt[head_key][0]
        logger.info("Click at normalized (%.2f, %.2f) = pixel (%d, %d)",
                    click_point[0], click_point[1],
                    int(click_point[0] * frame_w), int(click_point[1] * frame_h))
        for oid, mask_t in frame0_heads["obj_id_to_mask"].items():
            m = _mask_from_logits(mask_t, target_size=(frame_h, frame_w))
            if m.sum() < 20:
                continue
            coords = np.where(m > 127)
            cy, cx = coords[0].mean(), coords[1].mean()
            area_frac = m.sum() / (frame_h * frame_w * 255)
            score = frame0_heads["obj_id_to_score"].get(oid, 0)
            tx = int(click_point[0] * frame_w)
            ty = int(click_point[1] * frame_h)
            inside = m[ty, tx] > 127 if 0 <= ty < frame_h and 0 <= tx < frame_w else False
            logger.info(
                "  head obj_id=%d: score=%.2f, centroid=(%.0f,%.0f), area=%.2f%%, click_inside=%s",
                oid, score, cx, cy, area_frac*100, inside,
            )

    # Note: avoid `a or b` because a valid object ID can be 0 (falsy in Python)
    primary_head = click_primary(results_per_prompt[head_key])
    if primary_head is None:
        primary_head = stable_primary(results_per_prompt[head_key])
    primary_nose = click_primary(results_per_prompt[nose_key])
    if primary_nose is None:
        primary_nose = stable_primary(results_per_prompt[nose_key])
    logger.info("Selected primary_head=%s, primary_nose=%s", primary_head, primary_nose)

    for i in range(len(pil_frames)):
        frame_data = {"frame_idx": i}
        frame_w, frame_h = pil_frames[i].size

        head_out = results_per_prompt[head_key][i]
        ear_out = results_per_prompt[ear_key][i]
        nose_out = results_per_prompt[nose_key][i]

        # Head mask
        head_mask = None
        if primary_head is not None and primary_head in head_out["obj_id_to_mask"]:
            head_mask = _mask_from_logits(
                head_out["obj_id_to_mask"][primary_head],
                target_size=(frame_h, frame_w),
            )
            if head_mask.sum() == 0:
                head_mask = None
            else:
                frame_data["head_confidence"] = head_out["obj_id_to_score"].get(primary_head)

        if head_mask is None:
            per_frame.append(frame_data)
            continue

        # Head bbox
        coords = np.where(head_mask > 127)
        head_bbox = (
            coords[0].min(), coords[1].min(),
            coords[0].max(), coords[1].max(),
        )

        # Ears: only keep those attached to the tracked head.
        # Require ear centroid to sit inside the head mask, OR the ear mask
        # to overlap the head mask by > 15%. This rejects ears belonging to
        # other animals that happen to fall inside the head's bounding box.
        head_bool = head_mask > 127
        head_area = head_bool.sum()
        dilated_head = cv2.dilate(
            head_mask, np.ones((15, 15), np.uint8), iterations=1,
        ) > 127
        ear_candidates = []
        for oid, mask_t in ear_out["obj_id_to_mask"].items():
            m = _mask_from_logits(mask_t, target_size=(frame_h, frame_w))
            if m.sum() < 50:
                continue
            ear_bool = m > 127
            coords = np.where(ear_bool)
            cy, cx = coords[0].mean(), coords[1].mean()
            # Accept if centroid falls inside the (dilated) head mask
            # OR if at least 15% of the ear mask overlaps the head
            centroid_inside = dilated_head[int(cy), int(cx)]
            overlap = (ear_bool & dilated_head).sum()
            overlap_frac = overlap / ear_bool.sum()
            if not centroid_inside and overlap_frac < 0.15:
                continue
            ear_candidates.append(
                (m, ear_out["obj_id_to_score"][oid], cx, oid)
            )
        ear_candidates.sort(key=lambda e: -e[1])  # by score desc
        ear_candidates = ear_candidates[:2]
        ear_candidates.sort(key=lambda e: e[2])   # then by x asc

        left_ear_mask = ear_candidates[0][0] if len(ear_candidates) >= 1 else None
        right_ear_mask = ear_candidates[1][0] if len(ear_candidates) >= 2 else None

        if left_ear_mask is not None:
            frame_data["left_ear_confidence"] = ear_candidates[0][1]
        if right_ear_mask is not None:
            frame_data["right_ear_confidence"] = ear_candidates[1][1]

        # Nose mask
        nose_mask = None
        if primary_nose is not None and primary_nose in nose_out["obj_id_to_mask"]:
            nm = _mask_from_logits(
                nose_out["obj_id_to_mask"][primary_nose],
                target_size=(frame_h, frame_w),
            )
            mc = np.where(nm > 127)
            if len(mc[0]) > 20:
                cy, cx = mc[0].mean(), mc[1].mean()
                if (head_bbox[0] <= cy <= head_bbox[2] and
                        head_bbox[1] <= cx <= head_bbox[3]):
                    nose_mask = nm
                    frame_data["nose_confidence"] = nose_out["obj_id_to_score"].get(primary_nose)

        # Compute ear angles (anatomical midline if we have nose + ears)
        # Convert masks to boolean for the ear_angle helpers
        lb = (left_ear_mask > 127) if left_ear_mask is not None else None
        rb = (right_ear_mask > 127) if right_ear_mask is not None else None
        nb = (nose_mask > 127) if nose_mask is not None else None

        anatomical = _compute_anatomical_midline(nb, lb, rb)
        if anatomical is not None:
            head_up_angle, head_center = anatomical
        elif lb is not None or rb is not None:
            # Fallback: assume head is upright (head-up = image-up = 90°),
            # use ear midpoint as head center
            head_up_angle = 90.0
            pts = []
            for m in (lb, rb):
                if m is None:
                    continue
                coords = np.where(m)
                if len(coords[0]):
                    pts.append((coords[1].mean(), coords[0].mean()))
            if not pts:
                per_frame.append(frame_data)
                continue
            cx_mean = sum(p[0] for p in pts) / len(pts)
            cy_mean = sum(p[1] for p in pts) / len(pts)
            head_center = (cx_mean, cy_mean)
            frame_data["midline_fallback"] = "vertical"
        else:
            per_frame.append(frame_data)
            continue

        head_horizontal = head_up_angle - 90.0
        frame_data["head_midline_angle_deg"] = head_up_angle

        for side, ear_bool in (("left", lb), ("right", rb)):
            if ear_bool is None:
                continue
            direction = _compute_ear_direction(ear_bool, head_center)
            if direction is None:
                continue
            dx, dy = direction
            world_angle = float(np.degrees(np.arctan2(-dy, dx)))
            rel = _normalize_angle(world_angle - head_horizontal)
            if side == "left":
                rel = _normalize_angle(-rel + 180) if rel > 0 else _normalize_angle(-rel - 180)
            frame_data[f"{side}_ear_angle_deg"] = rel
            frame_data[f"{side}_ear_position"] = _classify_ear_position(rel).value

        per_frame.append(frame_data)

    # Build an annotated GIF showing masks tracking across frames
    gif_url = _build_annotated_gif(
        pil_frames, per_frame, results_per_prompt,
        primary_head, primary_nose, prompt_list, video_path,
    )

    elapsed = time.time() - start
    logger.info("Video analysis: %d frames in %.1fs", len(pil_frames), elapsed)

    left_angles = [f["left_ear_angle_deg"] for f in per_frame if "left_ear_angle_deg" in f]
    right_angles = [f["right_ear_angle_deg"] for f in per_frame if "right_ear_angle_deg" in f]

    return {
        "n_frames": len(pil_frames),
        "elapsed_s": elapsed,
        "per_frame": per_frame,
        "annotated_gif_url": gif_url,
        "summary": {
            "frames_with_measurement": sum(
                1 for f in per_frame
                if "left_ear_angle_deg" in f or "right_ear_angle_deg" in f
            ),
            "left_mean_deg": float(np.mean(left_angles)) if left_angles else None,
            "left_std_deg": float(np.std(left_angles)) if left_angles else None,
            "right_mean_deg": float(np.mean(right_angles)) if right_angles else None,
            "right_std_deg": float(np.std(right_angles)) if right_angles else None,
        },
    }


def _build_annotated_gif(
    pil_frames: list[Image.Image],
    per_frame: list[dict],
    results_per_prompt: dict,
    primary_head: Optional[int],
    primary_nose: Optional[int],
    prompt_list: list[str],
    video_path: Path,
) -> Optional[str]:
    """Composite SAM 3 masks onto each frame and save as a looping GIF.

    Head = green fill, ears = orange contours, nose = yellow dot.
    """
    try:
        import imageio.v3 as iio
    except Exception:
        return None

    from backend.config import RESULTS_DIR
    head_key, ear_key, nose_key = prompt_list

    annotated = []
    for i, pil in enumerate(pil_frames):
        frame = np.array(pil).copy()  # RGB
        h, w = frame.shape[:2]

        # Head: green translucent fill
        head_mask_bool = None
        dilated_head = None
        head_out = results_per_prompt[head_key][i]
        if primary_head is not None and primary_head in head_out["obj_id_to_mask"]:
            m = _mask_from_logits(
                head_out["obj_id_to_mask"][primary_head], target_size=(h, w),
            )
            if m.sum() > 0:
                head_mask_bool = m > 127
                frame[head_mask_bool] = (
                    frame[head_mask_bool].astype(np.float32) * 0.55 +
                    np.array([63, 185, 80], dtype=np.float32) * 0.45
                ).astype(np.uint8)
                dilated_head = cv2.dilate(
                    m, np.ones((15, 15), np.uint8), iterations=1,
                ) > 127

        # Ears: thick contour in orange, filtered to the tracked head
        ear_out = results_per_prompt[ear_key][i]
        for oid, mask_t in ear_out["obj_id_to_mask"].items():
            m = _mask_from_logits(mask_t, target_size=(h, w))
            if m.sum() < 50:
                continue
            ear_bool = m > 127
            # Skip ears not attached to the tracked head (e.g., other animal's ears)
            if dilated_head is not None:
                coords = np.where(ear_bool)
                cy, cx = int(coords[0].mean()), int(coords[1].mean())
                centroid_inside = dilated_head[cy, cx]
                overlap_frac = (ear_bool & dilated_head).sum() / max(1, ear_bool.sum())
                if not centroid_inside and overlap_frac < 0.15:
                    continue
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (240, 136, 62), 3)

        # Nose: yellow filled region (only the tracked nose)
        nose_out = results_per_prompt[nose_key][i]
        if primary_nose is not None and primary_nose in nose_out["obj_id_to_mask"]:
            m = _mask_from_logits(
                nose_out["obj_id_to_mask"][primary_nose], target_size=(h, w),
            )
            if m.sum() >= 20:
                bool_m = m > 127
                frame[bool_m] = (
                    frame[bool_m].astype(np.float32) * 0.4 +
                    np.array([227, 179, 65], dtype=np.float32) * 0.6
                ).astype(np.uint8)

        # Overlay ear angle readings in corner
        fd = per_frame[i]
        la = fd.get("left_ear_angle_deg")
        ra = fd.get("right_ear_angle_deg")
        text = f"f{i}  L: {la:+.1f}  R: {ra:+.1f}" if la is not None and ra is not None else f"f{i}"
        cv2.putText(
            frame, text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

        annotated.append(frame)

    # Write as GIF
    gif_name = f"{video_path.stem}_annotated.gif"
    gif_path = RESULTS_DIR / gif_name
    try:
        iio.imwrite(
            str(gif_path), annotated, loop=0, duration=250, plugin="pillow"
        )
        logger.info(
            "Annotated GIF: %d frames, %.0f KB",
            len(annotated), gif_path.stat().st_size / 1024,
        )
    except Exception as e:
        logger.warning("GIF export failed: %s", e)
        return None
    return f"/results/{gif_name}"
