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


def _full_pipeline_enabled() -> bool:
    """True on hardware with enough VRAM for the full 3-session pipeline.

    Priority:
    1. SHEEPSEG_FULL_PIPELINE=1 or =0 (explicit override).
    2. Auto-detect: CUDA device with >=15 GB total VRAM (RTX 4090 / A100 /
       anything bigger than a 6 GB card). 1660 Ti on a 6 GB laptop → survival.
    """
    import os
    env = os.environ.get("SHEEPSEG_FULL_PIPELINE")
    if env is not None:
        return env == "1"
    try:
        import torch
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return total_gb >= 15.0
    except Exception:
        pass
    return False


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
    except Exception:
        # logger.exception includes the full traceback — critical for
        # diagnosing load failures (HF auth, CUDA mismatch, OOM, etc.)
        # that the wrapper RuntimeError in analyze_video doesn't preserve.
        logger.exception("Failed to load SAM 3 Video")
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
    # MAX_DIM adapts to hardware:
    # - Full pipeline (24 GB+): 512 for crisper masks. SAM's auto-derived
    #   ear base/tip lands closer to anatomy, reviewer drags less.
    # - Survival (6 GB): 384 because at 512 a single SAM 3 Video session
    #   peaks at ~4.8 GB, leaving no room for the ear session to run
    #   afterward. Mask edge precision drops slightly but reviewers drag
    #   anyway and YOLO training uses HUMAN labels, not SAM-derived ones.
    MAX_DIM = 512 if _full_pipeline_enabled() else 384
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


def _build_head_instance(
    head_oid: int,
    head_out_i: dict,
    ear_out_i: dict,
    nose_out_i: Optional[dict],
    ear_left_oid: Optional[int],
    ear_right_oid: Optional[int],
    fw: int, fh: int,
) -> Optional[dict]:
    """Build one instance dict for a single head obj_id in a single frame.

    Returns None if the head has no mask or the mask is empty. Used for
    secondary heads in multi-subject mode; the primary head's instance is
    assembled inline in the main per-frame loop (which also computes the
    chart's ear-angle fields from the same head).

    Output keys: obj_id, head_bbox, candidate_keypoints, head_confidence.
    No midline / angle fields — those live frame-level, on the primary.
    """
    from backend.pipeline.ear_angle import (
        _extract_ear_keypoints,
        _mask_centroid,
    )

    head_t = head_out_i["obj_id_to_mask"].get(head_oid)
    if head_t is None:
        return None
    head_mask = _mask_from_logits(head_t, target_size=(fh, fw))
    if head_mask.sum() == 0:
        return None
    coords = np.where(head_mask > 127)
    head_bbox = {
        "x": int(coords[1].min()),
        "y": int(coords[0].min()),
        "w": int(coords[1].max() - coords[1].min()),
        "h": int(coords[0].max() - coords[0].min()),
    }
    dilated_head = cv2.dilate(
        head_mask, np.ones((15, 15), np.uint8), iterations=1,
    ) > 127

    left_mask, right_mask = None, None
    for oid, mt in ear_out_i["obj_id_to_mask"].items():
        if oid not in {ear_left_oid, ear_right_oid}:
            continue
        m = _mask_from_logits(mt, target_size=(fh, fw))
        if m.sum() < 50:
            continue
        eb = m > 127
        cs = np.where(eb)
        cy, cx = int(cs[0].mean()), int(cs[1].mean())
        if (not dilated_head[cy, cx]
                and (eb & dilated_head).sum() / max(1, eb.sum()) < 0.15):
            continue
        if oid == ear_left_oid:
            left_mask = m
        elif oid == ear_right_oid:
            right_mask = m
    lb = (left_mask > 127) if left_mask is not None else None
    rb = (right_mask > 127) if right_mask is not None else None

    # Nose attached to this head (centroid inside this head's bbox).
    # The primary gets explicit nose-tracking via primary_nose; secondaries
    # settle for "any nose obj whose centroid falls inside my bbox."
    nose_bool = None
    if nose_out_i is not None:
        ymin, xmin = coords[0].min(), coords[1].min()
        ymax, xmax = coords[0].max(), coords[1].max()
        for nose_oid, mt in nose_out_i["obj_id_to_mask"].items():
            nm = _mask_from_logits(mt, target_size=(fh, fw))
            mc = np.where(nm > 127)
            if len(mc[0]) <= 20:
                continue
            ncy, ncx = mc[0].mean(), mc[1].mean()
            if ymin <= ncy <= ymax and xmin <= ncx <= xmax:
                nose_bool = nm > 127
                break

    # Candidate keypoints (same image-space L/R convention as primary).
    head_centroid_xy = _mask_centroid(head_mask > 127)
    kps = [{"x": 0.0, "y": 0.0, "v": 0} for _ in range(5)]
    if nose_bool is not None:
        nc = _mask_centroid(nose_bool)
        if nc is not None:
            kps[0] = {"x": nc[0], "y": nc[1], "v": 1}
    if head_centroid_xy is not None:
        head_cx = head_centroid_xy[0]
        detected = []
        for mask in (lb, rb):
            if mask is not None:
                cx = float(np.where(mask)[1].mean())
                detected.append((cx, mask))
        sl_mask, sr_mask = None, None
        if len(detected) == 2:
            detected.sort(key=lambda t: t[0])
            sl_mask = detected[0][1]
            sr_mask = detected[1][1]
        elif len(detected) == 1:
            cx, mask = detected[0]
            if cx < head_cx:
                sl_mask = mask
            else:
                sr_mask = mask
        if sl_mask is not None:
            kp = _extract_ear_keypoints(sl_mask, head_centroid_xy)
            if kp is not None:
                (bx, by), (tx, ty) = kp
                kps[1] = {"x": bx, "y": by, "v": 1}
                kps[3] = {"x": tx, "y": ty, "v": 1}
        if sr_mask is not None:
            kp = _extract_ear_keypoints(sr_mask, head_centroid_xy)
            if kp is not None:
                (bx, by), (tx, ty) = kp
                kps[2] = {"x": bx, "y": by, "v": 1}
                kps[4] = {"x": tx, "y": ty, "v": 1}
    return {
        "obj_id": int(head_oid),
        "head_bbox": head_bbox,
        "candidate_keypoints": kps,
        "head_confidence": float(
            head_out_i["obj_id_to_score"].get(head_oid, 0) or 0
        ),
    }


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
        raise RuntimeError(
            "SAM 3 Video model failed to load. See the uvicorn log's "
            "preceding traceback (tag 'Failed to load SAM 3 Video') for "
            "the underlying cause."
        )

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

    # Prompt list adapts to hardware:
    # - Full pipeline (24 GB+): head + ear + nose — all three SAM 3 Video
    #   sessions. Nose gives anatomical midline and auto-seeds nose keypoint.
    # - Survival (6 GB): head + ear only. Nose is dropped because three
    #   4.8GB-peak sessions can't fit back-to-back on a 6GB GPU. Nose
    #   keypoints become manual placement in the labeler.
    head_key = f"{subject} head"
    ear_key = f"{subject} ear"
    nose_key = f"{subject} nose"
    full_pipeline = _full_pipeline_enabled()
    prompt_list = [head_key, ear_key, nose_key] if full_pipeline else [head_key, ear_key]
    logger.info(
        "Pipeline mode: %s (%d sessions)",
        "full (head+ear+nose)" if full_pipeline else "survival (head+ear)",
        len(prompt_list),
    )

    results_per_prompt: dict[str, list[dict]] = {}
    for prompt_idx, prompt_text in enumerate(prompt_list):
        # If the model was unloaded after the prior session (see below),
        # reload it for this one. Adds ~15s per session but guarantees a
        # clean allocator — empirically the only way to fit two back-to-back
        # sessions on 6GB for busier clips.
        if _video_model is None:
            _load_video_model()
            if _video_model is None:
                raise RuntimeError("SAM 3 Video model failed to reload between sessions")

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
        # Free tracking state from this prompt.
        import gc as _gc
        session.reset_inference_session()
        del session
        _gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Force-unload the model if there's another session to run AND we're
        # in survival mode. empty_cache alone doesn't defrag enough on 6GB to
        # fit a second 4.8GB-peak session back-to-back. Reload costs ~15s.
        # On 24GB+ (full pipeline) there's plenty of headroom; no unload needed.
        if not full_pipeline and prompt_idx < len(prompt_list) - 1:
            unload_video_model()

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
    # Nose session is disabled for 6GB reliability — primary_nose stays None
    # and every frame's nose_mask_bool stays None (head-PCA midline fallback).
    primary_nose = None
    if nose_key in results_per_prompt:
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

    # === MULTI-SUBJECT: collect secondary tracked heads (schema v2) ===
    #
    # Past this point, `primary_head` is the "observability" head — its ear
    # angles drive the frame-level chart and the GIF's title overlay. For
    # training data, we emit `instances[]` per frame containing the primary
    # plus every other head that appears in at least MIN_HEAD_FRAMES frames.
    # This prevents the single-subject false-negative pathology: previously,
    # unlabeled sheep in a labeled frame taught the model "don't detect here."
    # With multi-subject labels, every visible sheep is a positive example.
    min_head_frames = max(2, len(pil_frames) // 10)
    head_frame_counts: dict[int, int] = {}
    for fr in results_per_prompt[head_key]:
        for oid, mask_t in fr["obj_id_to_mask"].items():
            m = _mask_from_logits(mask_t, target_size=(frame_h, frame_w))
            if m.sum() >= 50:
                head_frame_counts[oid] = head_frame_counts.get(oid, 0) + 1
    tracked_head_oids = {
        oid for oid, c in head_frame_counts.items() if c >= min_head_frames
    }
    if primary_head is not None:
        tracked_head_oids.add(primary_head)
    secondary_head_oids = sorted(
        tracked_head_oids - ({primary_head} if primary_head is not None else set())
    )
    logger.info(
        "Multi-head: primary=%s, secondary=%s (min_frames=%d, counts=%s)",
        primary_head, secondary_head_oids, min_head_frames,
        dict(sorted(head_frame_counts.items(), key=lambda kv: -kv[1])[:10]),
    )

    # Per-head ear locking for secondaries — mirrors the primary logic above
    # but scoped per head. `claimed_ear_oids` prevents two heads from
    # "stealing" the same ear obj_id (happens when heads are close together).
    claimed_ear_oids: set = set(locked_ear_ids)
    secondary_head_to_ears: dict[int, set] = {}
    for head_oid in secondary_head_oids:
        ears_for_head: set = set()
        for i in range(len(pil_frames)):
            fw, fh = pil_frames[i].size
            head_t = results_per_prompt[head_key][i]["obj_id_to_mask"].get(head_oid)
            if head_t is None:
                continue
            hm = _mask_from_logits(head_t, target_size=(fh, fw))
            if hm.sum() == 0:
                continue
            dilated = cv2.dilate(
                hm, np.ones((15, 15), np.uint8), iterations=1,
            ) > 127
            cands = []
            for ear_oid, mt in results_per_prompt[ear_key][i]["obj_id_to_mask"].items():
                if ear_oid in claimed_ear_oids or ear_oid in ears_for_head:
                    continue
                m = _mask_from_logits(mt, target_size=(fh, fw))
                if m.sum() < 50:
                    continue
                eb = m > 127
                cs = np.where(eb)
                cy, cx = int(cs[0].mean()), int(cs[1].mean())
                if (not dilated[cy, cx]
                        and (eb & dilated).sum() / max(1, eb.sum()) < 0.15):
                    continue
                score = results_per_prompt[ear_key][i]["obj_id_to_score"].get(ear_oid, 0)
                cands.append((ear_oid, score))
            cands.sort(key=lambda x: -x[1])
            for ear_oid, _ in cands:
                if len(ears_for_head) >= 2:
                    break
                ears_for_head.add(ear_oid)
            if len(ears_for_head) >= 2:
                break
        secondary_head_to_ears[head_oid] = ears_for_head
        claimed_ear_oids |= ears_for_head

    # Per-head ear side assignment (screen-x) for secondaries.
    secondary_head_to_ear_sides: dict[int, tuple] = {}
    for head_oid in secondary_head_oids:
        ears = secondary_head_to_ears.get(head_oid, set())
        l_oid, r_oid = None, None
        if len(ears) == 1:
            l_oid = next(iter(ears))
        elif len(ears) == 2:
            for i in range(len(pil_frames)):
                fw, fh = pil_frames[i].size
                centroids = {}
                for ear_oid in ears:
                    mt = results_per_prompt[ear_key][i]["obj_id_to_mask"].get(ear_oid)
                    if mt is None:
                        continue
                    m = _mask_from_logits(mt, target_size=(fh, fw))
                    if m.sum() < 50:
                        continue
                    cs = np.where(m > 127)
                    centroids[ear_oid] = float(cs[1].mean())
                if len(centroids) == 2:
                    by_x = sorted(centroids.keys(), key=lambda o: centroids[o])
                    l_oid, r_oid = by_x[0], by_x[1]
                    break
        secondary_head_to_ear_sides[head_oid] = (l_oid, r_oid)
    logger.info(
        "Secondary ear locks: %s",
        {h: {"left": ls[0], "right": ls[1]} for h, ls in secondary_head_to_ear_sides.items()},
    )

    # Tracks the previous frame's head-up axis so we can flip PCA sign
    # ambiguities temporally. Ears-based disambiguation fails when ears
    # straddle the long axis; temporal continuity catches those.
    prev_head_up_angle: Optional[float] = None

    for i in range(len(pil_frames)):
        frame_data = {"frame_idx": i}
        frame_w, frame_h = pil_frames[i].size
        frame_data["frame_width"] = frame_w
        frame_data["frame_height"] = frame_h

        head_out = results_per_prompt[head_key][i]
        ear_out = results_per_prompt[ear_key][i]
        nose_out = (
            results_per_prompt[nose_key][i]
            if nose_key in results_per_prompt
            else None
        )

        # === PRIMARY head processing ===
        # Populates frame-level observability fields (ear angles, midline,
        # head_confidence) used by the dashboard chart and the annotated
        # GIF's title overlay. Also produces the primary's instance dict
        # (or None if the primary head is missing in this frame).
        primary_instance: Optional[dict] = None
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

        if head_mask is not None:
            # Head bbox (y_min, x_min, y_max, x_max in pixel coords)
            coords = np.where(head_mask > 127)
            head_bbox_tuple = (
                coords[0].min(), coords[1].min(),
                coords[0].max(), coords[1].max(),
            )
            primary_bbox = {
                "x": int(head_bbox_tuple[1]),
                "y": int(head_bbox_tuple[0]),
                "w": int(head_bbox_tuple[3] - head_bbox_tuple[1]),
                "h": int(head_bbox_tuple[2] - head_bbox_tuple[0]),
            }

            # Ears: look up each locked obj_id directly, assigning to the
            # same left/right trace every frame. Dilated-head attachment
            # filter rejects a locked ear drifting away from the tracked head.
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
                cs = np.where(ear_bool)
                cy, cx = int(cs[0].mean()), int(cs[1].mean())
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
            if (primary_nose is not None
                    and nose_out is not None
                    and primary_nose in nose_out["obj_id_to_mask"]):
                nm = _mask_from_logits(
                    nose_out["obj_id_to_mask"][primary_nose],
                    target_size=(frame_h, frame_w),
                )
                mc = np.where(nm > 127)
                if len(mc[0]) > 20:
                    cy, cx = mc[0].mean(), mc[1].mean()
                    if (head_bbox_tuple[0] <= cy <= head_bbox_tuple[2]
                            and head_bbox_tuple[1] <= cx <= head_bbox_tuple[3]):
                        nose_mask_bool = nm > 127
                        frame_data["nose_confidence"] = nose_out["obj_id_to_score"].get(primary_nose)

            # Midline + ear angles (frame-level observability). Skip when no
            # ears detected; primary instance is still built from head+bbox+
            # whatever ear keypoints we can compute.
            if lb is not None or rb is not None:
                midline_source = None
                midline = _compute_anatomical_midline(nose_mask_bool, lb, rb)
                if midline is not None:
                    midline_source = "nose"
                else:
                    midline = _compute_head_midline_pca(hb, lb, rb)
                    midline_source = "head_pca" if midline is not None else None

                if midline is not None:
                    head_up_angle, head_center = midline
                    frame_data["midline_source"] = midline_source
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
                            rel = (
                                _normalize_angle(-rel + 180) if rel > 0
                                else _normalize_angle(-rel - 180)
                            )
                        frame_data[f"{side}_ear_angle_deg"] = rel
                        frame_data[f"{side}_ear_position"] = _classify_ear_position(rel).value

            # Candidate keypoints for the primary (image-space L/R convention).
            # Schema: [nose, L-base, R-base, L-tip, R-tip]. Visibility:
            #   0 = not derivable, 1 = auto-derived from SAM, 2 = human-reviewed.
            head_centroid_xy = _mask_centroid(head_mask > 127)
            kps = [{"x": 0.0, "y": 0.0, "v": 0} for _ in range(5)]
            if nose_mask_bool is not None:
                nc = _mask_centroid(nose_mask_bool)
                if nc is not None:
                    kps[0] = {"x": nc[0], "y": nc[1], "v": 1}
            if head_centroid_xy is not None:
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
                        kps[1] = {"x": bx, "y": by, "v": 1}
                        kps[3] = {"x": tx, "y": ty, "v": 1}
                if screen_right_mask is not None:
                    kp = _extract_ear_keypoints(screen_right_mask, head_centroid_xy)
                    if kp is not None:
                        (bx, by), (tx, ty) = kp
                        kps[2] = {"x": bx, "y": by, "v": 1}
                        kps[4] = {"x": tx, "y": ty, "v": 1}

            primary_instance = {
                "obj_id": int(primary_head),
                "head_bbox": primary_bbox,
                "candidate_keypoints": kps,
                "head_confidence": float(
                    head_out["obj_id_to_score"].get(primary_head, 0) or 0
                ),
            }

        # === SECONDARY heads ===
        # Each secondary head becomes an additional entry in instances[].
        # No chart-driving fields; just {obj_id, head_bbox, keypoints,
        # head_confidence}.
        instances: list[dict] = []
        if primary_instance is not None:
            instances.append(primary_instance)
        for head_oid in secondary_head_oids:
            el, er = secondary_head_to_ear_sides.get(head_oid, (None, None))
            inst = _build_head_instance(
                head_oid, head_out, ear_out, nose_out, el, er, frame_w, frame_h,
            )
            if inst is not None:
                instances.append(inst)
        frame_data["instances"] = instances
        frame_data["tracking_gap"] = (len(instances) == 0)

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
    # is usable vs. will need manual placement. In multi-subject mode this
    # counts slot-hits across every instance in every frame; the denominator
    # is (n_instances × n_frames) rather than just n_frames.
    kp_names = ["nose", "L_base", "R_base", "L_tip", "R_tip"]
    kp_coverage = [0] * 5
    total_instance_frames = 0
    for f in per_frame:
        for inst in f.get("instances") or []:
            total_instance_frames += 1
            for k, kp in enumerate(inst.get("candidate_keypoints") or []):
                if kp.get("v", 0) > 0:
                    kp_coverage[k] += 1
    logger.info(
        "Keypoint auto-coverage over %d instance-frames: %s",
        total_instance_frames,
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

    # Seed review.json with multi-subject candidate state (schema v2).
    #
    # Per-frame structure:
    #   {frame_idx, frame_path, frame_width, frame_height, tracking_gap,
    #    instances: [{obj_id, head_bbox, keypoints: [5]}, ...]}
    #
    # Each keypoint carries two flags:
    #   auto_v: what SAM 3 derived (0 missing, 1 derived) — immutable record
    #   v:      current state in the review workflow (0, 1, or 2=reviewed)
    # The exporter only trusts v=2; see LOG.md in sheep-yolo/sheep-seg-conversation
    # for the YOLO-pose v-flag mapping.
    frames_out = []
    for i, f in enumerate(per_frame):
        instances_out = []
        for inst in f.get("instances") or []:
            kps = inst.get("candidate_keypoints") or []
            review_kps = []
            for kp in kps:
                v = int(kp.get("v", 0))
                review_kps.append({
                    "x": float(kp.get("x", 0.0)),
                    "y": float(kp.get("y", 0.0)),
                    "v": v,
                    "auto_v": v,
                })
            instances_out.append({
                "obj_id": int(inst.get("obj_id")),
                "head_bbox": inst.get("head_bbox"),
                "head_confidence": inst.get("head_confidence"),
                "keypoints": review_kps,
            })
        frames_out.append({
            "frame_idx": i,
            "frame_path": f"frames/frame{i:04d}.jpg",
            "tracking_gap": bool(f.get("tracking_gap", True)),
            "frame_width": f.get("frame_width"),
            "frame_height": f.get("frame_height"),
            "instances": instances_out,
        })

    # Collect all obj_ids that appear anywhere in the clip; the UI uses this
    # to assign stable colors per sheep without rescanning every frame.
    all_obj_ids = sorted({
        int(inst["obj_id"])
        for f in frames_out
        for inst in f["instances"]
    })

    review_payload = {
        "video_id": video_id,
        "schema_version": 2,
        "kpt_names": ["nose", "left_ear_base", "right_ear_base",
                      "left_ear_tip", "right_ear_tip"],
        "flip_idx": [0, 2, 1, 4, 3],
        "n_frames": len(per_frame),
        "obj_ids": all_obj_ids,
        "frames": frames_out,
    }
    review_path.write_text(json.dumps(review_payload, indent=2))
    n_instance_rows = sum(len(f["instances"]) for f in frames_out)
    logger.info(
        "Seeded %s with %d frames, %d instance-rows across %d obj_ids",
        review_path, len(per_frame), n_instance_rows, len(all_obj_ids),
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
    head_key, ear_key = prompt_list[0], prompt_list[1]
    # prompt_list may be [head, ear] (2-session VRAM mode) or [head, ear, nose].
    nose_key = prompt_list[2] if len(prompt_list) >= 3 else None

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

        # Nose: yellow filled region (only the tracked nose). Skipped when
        # the nose session wasn't run (2-session VRAM mode).
        nose_out = (
            results_per_prompt[nose_key][i]
            if nose_key is not None and nose_key in results_per_prompt
            else None
        )
        if (primary_nose is not None
                and nose_out is not None
                and primary_nose in nose_out["obj_id_to_mask"]):
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
