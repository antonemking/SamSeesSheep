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


def unload_video_model():
    """Drop the SAM 3 Video model from GPU and clear the allocator.

    Call between failed analyze_video attempts so the next retry starts
    with a clean allocator state rather than fragmented residue.
    """
    global _video_model, _video_processor
    import gc
    import torch
    if _video_model is not None:
        logger.info("Unloading SAM 3 Video model to reclaim VRAM...")
        try:
            _video_model.cpu()
        except Exception:
            pass
        _video_model = None
        _video_processor = None
        gc.collect()
        if torch.cuda.is_available():
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
    # 384 because clips with multiple sheep and busy scenes push a single
    # SAM 3 Video session past 4.8 GB at 512 px, leaving no headroom for
    # ear and nose sessions on a 6 GB GPU. We're in labeling mode now —
    # the reviewer drags dots anyway, so SAM's auto-placement precision
    # at 384 vs 512 is marginal. YOLO training runs at imgsz=640 with the
    # HUMAN keypoint labels, not SAM-derived ones, so mask crispness
    # doesn't transfer to the trained model. Revert to 512 if moving to
    # A100 or if OOMs stop being a thing on 6 GB.
    MAX_DIM = 384
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


def _apply_median_smoothing(
    per_frame: list[dict], key: str, window: int = 3
) -> None:
    """Add a smoothed version of `key` to each frame dict as `<key>_smoothed`.

    Uses a centered median filter over `window` frames. Frames missing the
    raw key keep no smoothed value. Window edges use what's available.
    """
    if window < 2:
        return
    values = [f.get(key) for f in per_frame]
    half = window // 2
    for i, raw in enumerate(values):
        if raw is None:
            continue
        nearby = [
            values[j] for j in range(max(0, i - half), min(len(values), i + half + 1))
            if values[j] is not None
        ]
        if nearby:
            per_frame[i][f"{key}_smoothed"] = float(np.median(nearby))


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
        _compute_head_midline_pca,
        _compute_ear_direction,
        _extract_ear_keypoints,
        _mask_centroid,
        _normalize_angle,
        _classify_ear_position,
    )

    start = time.time()
    pil_frames = _extract_frames(video_path, max_frames=max_frames)
    if not pil_frames:
        return {"error": "no frames extracted", "per_frame": []}

    frame_w, frame_h = pil_frames[0].size

    # Three separate sessions (head, ear, nose). Single-session multi-prompt
    # hits OOM on 6GB GPUs because each additional prompt adds tracking state.
    # The nose session anchors the dorsal midline via nose→ear-midpoint — an
    # unambiguous reference that doesn't wobble with head-mask shape noise
    # the way a head-PCA midline does. Worth the extra pass for chart stability.
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
                "removed_obj_ids": set(output.removed_obj_ids or []),
            })
        results_per_prompt[prompt_text] = per_frame_outputs

        logger.info(
            "Tracked '%s' across %d frames in %.1fs · peak VRAM %.2f GB",
            prompt_text, len(pil_frames), time.time() - t0,
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        )
        # Free tracking state from this prompt before the next session.
        # empty_cache alone isn't enough — Python refs to mask tensors
        # survive until gc runs, which can keep ~500MB alive on a 6GB GPU.
        import gc as _gc
        session.reset_inference_session()
        del session
        _gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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
    logger.info(
        "Selected primary_head=%s, primary_nose=%s", primary_head, primary_nose,
    )

    # Lock to specific ear object IDs.
    # Scan frames in order to find up to 2 ears that are clearly attached
    # to our tracked head. Only those 2 IDs are followed across the rest
    # of the video — the dilated-head attachment filter rejects "flash"
    # detections from other animals. Scans ALL frames (not just the first
    # 5), because profile-view starts can hide the far ear for ~10 frames
    # before the sheep turns and exposes it.
    locked_ear_ids: set = set()
    if primary_head is not None:
        for i in range(len(pil_frames)):
            fw, fh = pil_frames[i].size
            head_t = results_per_prompt[head_key][i]["obj_id_to_mask"].get(primary_head)
            if head_t is None:
                continue
            hm = _mask_from_logits(head_t, target_size=(fh, fw))
            if hm.sum() == 0:
                continue
            dilated = cv2.dilate(hm, np.ones((15, 15), np.uint8), iterations=1) > 127
            cands = []
            for oid, mt in results_per_prompt[ear_key][i]["obj_id_to_mask"].items():
                if oid in locked_ear_ids:
                    continue
                m = _mask_from_logits(mt, target_size=(fh, fw))
                if m.sum() < 50:
                    continue
                eb = m > 127
                cs = np.where(eb)
                cy, cx = int(cs[0].mean()), int(cs[1].mean())
                inside = dilated[cy, cx]
                overlap = (eb & dilated).sum() / max(1, eb.sum())
                if inside or overlap >= 0.15:
                    score = results_per_prompt[ear_key][i]["obj_id_to_score"].get(oid, 0)
                    cands.append((oid, score))
            # Add highest-scoring ear candidates until we have 2
            cands.sort(key=lambda x: -x[1])
            for oid, _ in cands:
                if len(locked_ear_ids) >= 2:
                    break
                locked_ear_ids.add(oid)
            if len(locked_ear_ids) >= 2:
                break
    logger.info("Locked ear IDs: %s", locked_ear_ids)

    # Assign each locked ear obj_id permanently to "left" or "right" so every
    # frame's blue/orange traces follow the same physical ear. Without this,
    # per-frame x-sorting was relabeling ears whenever they crossed in screen
    # space (sheep turns head → labels flip → chart swaps mid-clip).
    # Assignment uses screen-x at the first frame where both ears are visible.
    locked_left_oid: Optional[int] = None
    locked_right_oid: Optional[int] = None
    if len(locked_ear_ids) == 2:
        for i in range(len(pil_frames)):
            fw, fh = pil_frames[i].size
            centroids: dict[int, float] = {}
            for oid in locked_ear_ids:
                mt = results_per_prompt[ear_key][i]["obj_id_to_mask"].get(oid)
                if mt is None:
                    continue
                m = _mask_from_logits(mt, target_size=(fh, fw))
                if m.sum() < 50:
                    continue
                cs = np.where(m > 127)
                centroids[oid] = float(cs[1].mean())
            if len(centroids) == 2:
                by_x = sorted(centroids.keys(), key=lambda o: centroids[o])
                locked_left_oid, locked_right_oid = by_x[0], by_x[1]
                logger.info(
                    "Ear sides assigned at frame %d: left_oid=%s right_oid=%s",
                    i, locked_left_oid, locked_right_oid,
                )
                break
    elif len(locked_ear_ids) == 1:
        locked_left_oid = next(iter(locked_ear_ids))

    # Tracks the previous frame's head-up axis so we can flip PCA sign
    # ambiguities temporally. Ears-based disambiguation fails when ears
    # straddle the long axis; temporal continuity catches those.
    prev_head_up_angle: Optional[float] = None

    for i in range(len(pil_frames)):
        frame_data = {"frame_idx": i}
        frame_w, frame_h = pil_frames[i].size

        head_out = results_per_prompt[head_key][i]
        ear_out = results_per_prompt[ear_key][i]
        nose_out = results_per_prompt[nose_key][i]

        # Head mask. If primary_head was removed/suppressed or its mask is
        # empty on this frame, we mark the frame as a tracking gap.
        head_mask = None
        removed = head_out.get("removed_obj_ids") or set()
        if (primary_head is not None
                and primary_head in head_out["obj_id_to_mask"]
                and primary_head not in removed):
            head_mask = _mask_from_logits(
                head_out["obj_id_to_mask"][primary_head],
                target_size=(frame_h, frame_w),
            )
            if head_mask.sum() == 0:
                head_mask = None
            else:
                frame_data["head_confidence"] = head_out["obj_id_to_score"].get(primary_head)

        if head_mask is None:
            frame_data["tracking_gap"] = True
            per_frame.append(frame_data)
            continue

        frame_data["tracking_gap"] = False

        # Head bbox (y_min, x_min, y_max, x_max in pixel coords)
        coords = np.where(head_mask > 127)
        head_bbox = (
            coords[0].min(), coords[1].min(),
            coords[0].max(), coords[1].max(),
        )
        # Emit as {x, y, w, h} top-left + size in pixel coords so the
        # review UI can render it directly and the export endpoint can
        # normalize to YOLO-pose center-xywh at write time.
        frame_data["head_bbox"] = {
            "x": int(head_bbox[1]),
            "y": int(head_bbox[0]),
            "w": int(head_bbox[3] - head_bbox[1]),
            "h": int(head_bbox[2] - head_bbox[0]),
        }

        # Ears: look up each locked obj_id directly, assigning to the same
        # left/right trace every frame. The dilated-head attachment filter
        # rejects the rare case where a locked ear drifts away from the
        # tracked head.
        dilated_head = cv2.dilate(
            head_mask, np.ones((15, 15), np.uint8), iterations=1,
        ) > 127
        left_ear_mask = None
        right_ear_mask = None
        for oid, mask_t in ear_out["obj_id_to_mask"].items():
            if oid not in locked_ear_ids:
                continue
            m = _mask_from_logits(mask_t, target_size=(frame_h, frame_w))
            if m.sum() < 50:
                continue
            ear_bool = m > 127
            coords = np.where(ear_bool)
            cy, cx = int(coords[0].mean()), int(coords[1].mean())
            centroid_inside = dilated_head[cy, cx]
            overlap_frac = (ear_bool & dilated_head).sum() / ear_bool.sum()
            if not centroid_inside and overlap_frac < 0.15:
                continue
            score = ear_out["obj_id_to_score"].get(oid, 0)
            if oid == locked_left_oid:
                left_ear_mask = m
                frame_data["left_ear_confidence"] = score
            elif oid == locked_right_oid:
                right_ear_mask = m
                frame_data["right_ear_confidence"] = score

        lb = (left_ear_mask > 127) if left_ear_mask is not None else None
        rb = (right_ear_mask > 127) if right_ear_mask is not None else None
        hb = head_mask > 127

        # Nose mask for anatomical midline — preferred over head-PCA.
        nose_mask_bool = None
        if primary_nose is not None and primary_nose in nose_out["obj_id_to_mask"]:
            nm = _mask_from_logits(
                nose_out["obj_id_to_mask"][primary_nose],
                target_size=(frame_h, frame_w),
            )
            mc = np.where(nm > 127)
            if len(mc[0]) > 20:
                cy, cx = mc[0].mean(), mc[1].mean()
                if (head_bbox[0] <= cy <= head_bbox[2]
                        and head_bbox[1] <= cx <= head_bbox[3]):
                    nose_mask_bool = nm > 127
                    frame_data["nose_confidence"] = nose_out["obj_id_to_score"].get(primary_nose)

        if lb is None and rb is None:
            per_frame.append(frame_data)
            continue

        # Prefer nose-anchored midline (unambiguous, stable). Fall back to
        # head-PCA midline when nose isn't detected on this frame.
        midline_source = None
        midline = _compute_anatomical_midline(nose_mask_bool, lb, rb)
        if midline is not None:
            midline_source = "nose"
        else:
            midline = _compute_head_midline_pca(hb, lb, rb)
            midline_source = "head_pca" if midline is not None else None
        if midline is None:
            per_frame.append(frame_data)
            continue
        head_up_angle, head_center = midline
        frame_data["midline_source"] = midline_source

        # Sign-flip guard only applies to the PCA fallback — anatomical is
        # already unambiguous. Prevents the wobble-then-flip we used to see
        # when ears straddled the head's long axis.
        if midline_source == "head_pca" and prev_head_up_angle is not None:
            diff = ((head_up_angle - prev_head_up_angle + 180) % 360) - 180
            if abs(diff) > 90:
                head_up_angle = (head_up_angle + 180) % 360
                if head_up_angle > 180:
                    head_up_angle -= 360
                frame_data["midline_flipped"] = True
        prev_head_up_angle = head_up_angle

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

        # Candidate keypoints for the YOLO-pose labeling pipeline.
        # Schema: [nose, L-base, R-base, L-tip, R-tip], matching the
        # flip_idx: [0, 2, 1, 4, 3] that training expects. Each slot is
        # {"x": px, "y": py, "v": visibility} where visibility is:
        #   0 = not derivable (mask missing)
        #   1 = auto-derived from SAM 3 masks (awaiting human review)
        #   2 = human-reviewed (set later by the labeling UI)
        #
        # IMPORTANT — image-space L/R convention.
        # YOLO-pose with flip_idx=[0, 2, 1, 4, 3] expects "left" to mean
        # screen-left in the current image, not the animal's anatomical
        # left. So per frame, we assign the leftmost ear (smallest x) to
        # slots 1/3 and the rightmost ear to slots 2/4. This is DIFFERENT
        # from the chart's left/right, which stays pinned to a specific
        # obj_id across the clip so each trace follows the same physical
        # ear. The two concerns diverge intentionally.
        head_centroid_xy = _mask_centroid(head_mask > 127)
        kps = [{"x": 0.0, "y": 0.0, "v": 0} for _ in range(5)]
        if nose_mask_bool is not None:
            nc = _mask_centroid(nose_mask_bool)
            if nc is not None:
                kps[0] = {"x": nc[0], "y": nc[1], "v": 1}
        if head_centroid_xy is not None:
            # Assign each detected ear mask to the L or R slot based on its
            # centroid's x-position relative to the head centroid's x.
            # With both ears visible, smaller-x = L, larger-x = R. With only
            # one ear (profile view), compare to head_center_x: ear on the
            # left half of the head → L slot, right half → R slot.
            head_cx = head_centroid_xy[0]
            detected = []
            for mask in (lb, rb):
                if mask is not None:
                    cx = float(np.where(mask)[1].mean())
                    detected.append((cx, mask))
            screen_left_mask = None
            screen_right_mask = None
            if len(detected) == 2:
                detected.sort(key=lambda t: t[0])
                screen_left_mask = detected[0][1]
                screen_right_mask = detected[1][1]
            elif len(detected) == 1:
                cx, mask = detected[0]
                if cx < head_cx:
                    screen_left_mask = mask
                else:
                    screen_right_mask = mask
            if screen_left_mask is not None:
                kp = _extract_ear_keypoints(screen_left_mask, head_centroid_xy)
                if kp is not None:
                    (bx, by), (tx, ty) = kp
                    kps[1] = {"x": bx, "y": by, "v": 1}  # L-base (screen-left)
                    kps[3] = {"x": tx, "y": ty, "v": 1}  # L-tip  (screen-left)
            if screen_right_mask is not None:
                kp = _extract_ear_keypoints(screen_right_mask, head_centroid_xy)
                if kp is not None:
                    (bx, by), (tx, ty) = kp
                    kps[2] = {"x": bx, "y": by, "v": 1}  # R-base (screen-right)
                    kps[4] = {"x": tx, "y": ty, "v": 1}  # R-tip  (screen-right)
        frame_data["candidate_keypoints"] = kps
        frame_data["frame_width"] = frame_w
        frame_data["frame_height"] = frame_h

        per_frame.append(frame_data)

    # Build an annotated GIF showing masks tracking across frames
    gif_url = _build_annotated_gif(
        pil_frames, per_frame, results_per_prompt,
        primary_head, primary_nose, locked_ear_ids, prompt_list, video_path,
    )

    # Persist raw sampled frames + a review.json for the keypoint labeler.
    # Frames are always re-written (idempotent — same sampling → same frames).
    # review.json is seeded ONLY if it doesn't exist, so re-running /analyze
    # on a video that's already been labeled won't wipe human review state.
    _persist_label_artifacts(pil_frames, per_frame, video_path)

    elapsed = time.time() - start
    logger.info("Video analysis: %d frames in %.1fs", len(pil_frames), elapsed)

    # Keypoint coverage log — helps the labeler see how much auto-annotation
    # is usable vs. will need manual placement.
    kp_names = ["nose", "L_base", "R_base", "L_tip", "R_tip"]
    kp_coverage = [0] * 5
    for f in per_frame:
        for k, kp in enumerate(f.get("candidate_keypoints") or []):
            if kp.get("v", 0) > 0:
                kp_coverage[k] += 1
    logger.info(
        "Keypoint auto-coverage over %d frames: %s",
        len(per_frame),
        ", ".join(f"{n}={c}" for n, c in zip(kp_names, kp_coverage)),
    )

    # Smooth the per-frame angles with a 3-frame rolling median.
    # Median filter wins over averaging: single outlier frames (mask
    # jitter, brief mis-detection) get rejected, real trends survive.
    _apply_median_smoothing(per_frame, "left_ear_angle_deg", window=3)
    _apply_median_smoothing(per_frame, "right_ear_angle_deg", window=3)

    left_angles = [f["left_ear_angle_deg_smoothed"] for f in per_frame if "left_ear_angle_deg_smoothed" in f]
    right_angles = [f["right_ear_angle_deg_smoothed"] for f in per_frame if "right_ear_angle_deg_smoothed" in f]

    gaps = sum(1 for f in per_frame if f.get("tracking_gap"))
    result = {
        "n_frames": len(pil_frames),
        "elapsed_s": elapsed,
        "per_frame": per_frame,
        "annotated_gif_url": gif_url,
        "summary": {
            "frames_with_measurement": sum(
                1 for f in per_frame
                if "left_ear_angle_deg" in f or "right_ear_angle_deg" in f
            ),
            "frames_tracked": len(per_frame) - gaps,
            "frames_gap": gaps,
            "left_mean_deg": float(np.mean(left_angles)) if left_angles else None,
            "left_std_deg": float(np.std(left_angles)) if left_angles else None,
            "right_mean_deg": float(np.mean(right_angles)) if right_angles else None,
            "right_std_deg": float(np.std(right_angles)) if right_angles else None,
        },
    }

    # Drop references to the accumulated per-prompt mask tensors BEFORE the
    # final cache flush so empty_cache can actually reclaim what they held.
    # Without this, running several analyses back-to-back fragments the
    # allocator until even a 5-frame retry OOMs — precisely the failure
    # mode reported 2026-04-18. results_per_prompt was keeping ~1-2GB alive
    # across successful calls because Python's gc hadn't fired between them.
    import gc
    del results_per_prompt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(
            "Post-analysis VRAM: %.2f GB allocated, %.2f GB reserved",
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )

    return result


def _persist_label_artifacts(
    pil_frames: list[Image.Image],
    per_frame: list[dict],
    video_path: Path,
) -> None:
    """Save raw sampled frames + seed a review.json for the keypoint labeler.

    Layout produced:
      data/labels/{video_id}/frames/frame0000.jpg ... frameNNNN.jpg
      data/labels/{video_id}/review.json

    review.json is seeded only if missing. Re-running /analyze on an already-
    labeled video leaves the review state intact; only the frame files get
    rewritten (deterministic — same sampling produces the same frames).
    """
    import json
    from backend.config import LABELS_DIR

    video_id = video_path.stem
    label_dir = LABELS_DIR / video_id
    frames_dir = label_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, pil in enumerate(pil_frames):
        frame_path = frames_dir / f"frame{i:04d}.jpg"
        pil.convert("RGB").save(frame_path, "JPEG", quality=92)

    review_path = label_dir / "review.json"
    if review_path.exists():
        logger.info(
            "Label dir %s already has review.json — leaving human state intact",
            label_dir.name,
        )
        return

    # Seed review.json with candidate state. Each keypoint carries two flags:
    #   auto_v: what SAM 3 derived (0 missing, 1 derived) — immutable record
    #   v:      current state in the review workflow (0, 1, or 2=reviewed)
    # The exporter only trusts v=2; see LOG.md in sheep-yolo/sheep-seg-conversation
    # for the YOLO-pose v-flag mapping.
    frames_out = []
    for i, f in enumerate(per_frame):
        kps = f.get("candidate_keypoints") or []
        review_kps = []
        for kp in kps:
            v = int(kp.get("v", 0))
            review_kps.append({
                "x": float(kp.get("x", 0.0)),
                "y": float(kp.get("y", 0.0)),
                "v": v,
                "auto_v": v,
            })
        frames_out.append({
            "frame_idx": i,
            "frame_path": f"frames/frame{i:04d}.jpg",
            "tracking_gap": bool(f.get("tracking_gap", True)),
            "head_bbox": f.get("head_bbox"),
            "frame_width": f.get("frame_width"),
            "frame_height": f.get("frame_height"),
            "keypoints": review_kps,
        })

    review_payload = {
        "video_id": video_id,
        "schema_version": 1,
        "kpt_names": ["nose", "left_ear_base", "right_ear_base",
                      "left_ear_tip", "right_ear_tip"],
        "flip_idx": [0, 2, 1, 4, 3],
        "n_frames": len(per_frame),
        "frames": frames_out,
    }
    review_path.write_text(json.dumps(review_payload, indent=2))
    logger.info(
        "Seeded %s with %d frames of candidate keypoints",
        review_path, len(per_frame),
    )


def _build_annotated_gif(
    pil_frames: list[Image.Image],
    per_frame: list[dict],
    results_per_prompt: dict,
    primary_head: Optional[int],
    primary_nose: Optional[int],
    locked_ear_ids: set,
    prompt_list: list[str],
    video_path: Path,
) -> Optional[str]:
    """Composite SAM 3 masks onto each frame and save as a looping GIF.

    Head = green fill, ears = orange contours, nose = yellow fill.
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

        # Ears: thick contour in orange, only the locked ear IDs
        ear_out = results_per_prompt[ear_key][i]
        for oid, mask_t in ear_out["obj_id_to_mask"].items():
            if locked_ear_ids and oid not in locked_ear_ids:
                continue
            m = _mask_from_logits(mask_t, target_size=(h, w))
            if m.sum() < 50:
                continue
            ear_bool = m > 127
            # Also require the ear to still be attached to the tracked head
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
