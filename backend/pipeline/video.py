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


def _load_video_model():
    """Load SAM 3 Video model from HuggingFace."""
    global _video_model, _video_processor, _device
    try:
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading SAM 3 Video on %s...", _device)
        _video_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
        _video_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(_device)
        _video_model.eval()
        logger.info("SAM 3 Video model loaded.")
    except Exception as e:
        logger.error("Failed to load SAM 3 Video: %s", e)
        _video_model = None


def _extract_frames(
    video_path: Path, max_frames: int = 30, target_fps: float = 2.0
) -> list[Image.Image]:
    """Extract evenly-spaced frames from a video as PIL Images."""
    import imageio.v3 as iio

    frames = iio.imread(str(video_path), plugin="pyav")
    total = len(frames)
    if total == 0:
        return []

    # Get the actual fps from metadata if possible
    try:
        meta = iio.immeta(str(video_path), plugin="pyav")
        actual_fps = float(meta.get("fps", 30.0))
    except Exception:
        actual_fps = 30.0

    # Sample at target_fps or evenly if that would exceed max_frames
    duration_s = total / actual_fps
    target_count = min(max_frames, max(2, int(duration_s * target_fps)))
    idxs = np.linspace(0, total - 1, target_count).astype(int)

    pil_frames = [Image.fromarray(frames[i]) for i in idxs]
    logger.info(
        "Video: %d frames @ %.1f fps (%.1fs), sampling %d frames",
        total, actual_fps, duration_s, len(pil_frames),
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


def _mask_from_logits(mask_logits, threshold: float = 0.0) -> np.ndarray:
    """Convert SAM 3 mask logits tensor to a uint8 binary mask."""
    mask = (mask_logits.squeeze(0) > threshold).cpu().numpy().astype(np.uint8) * 255
    return mask


def analyze_video(
    video_path: Path,
    subject: str = "sheep",
    max_frames: int = 30,
) -> dict:
    """Run SAM 3 Video tracking across a clip.

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

    # One session with all three prompts — propagation runs once
    head_key = f"{subject} head"
    ear_key = f"{subject} ear"
    nose_key = f"{subject} nose"
    prompt_list = [head_key, ear_key, nose_key]

    t0 = time.time()
    session = _video_processor.init_video_session(
        video=pil_frames, inference_device=_device
    )
    _video_processor.add_text_prompt(session, prompt_list)
    prompt_text_to_id = {v: k for k, v in session.prompts.items()}

    # Collect per-frame outputs, split by prompt via obj_id_to_prompt_id
    results_per_prompt: dict[str, list[dict]] = {p: [] for p in prompt_list}
    for i, output in enumerate(
        _video_model.propagate_in_video_iterator(session, show_progress_bar=False)
    ):
        # Each prompt gets its own per-frame view of the outputs
        per_prompt = {p: {"obj_id_to_mask": {}, "obj_id_to_score": {},
                         "object_ids": []} for p in prompt_list}
        for obj_id, mask_t in output.obj_id_to_mask.items():
            pid = session.obj_id_to_prompt_id.get(obj_id)
            if pid is None or pid not in session.prompts:
                continue
            prompt_text = session.prompts[pid]
            if prompt_text not in per_prompt:
                continue
            per_prompt[prompt_text]["obj_id_to_mask"][obj_id] = mask_t.cpu()
            per_prompt[prompt_text]["obj_id_to_score"][obj_id] = output.obj_id_to_score.get(obj_id, 0.0)
            per_prompt[prompt_text]["object_ids"].append(obj_id)
        for p in prompt_list:
            results_per_prompt[p].append(per_prompt[p])

    logger.info(
        "Tracked %d prompts across %d frames in %.1fs",
        len(prompt_list), len(pil_frames), time.time() - t0,
    )
    del session
    torch.cuda.empty_cache()

    per_frame = []

    # Pick a stable primary ID per prompt from the first few frames with detections
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

    primary_head = stable_primary(results_per_prompt[head_key])
    primary_nose = stable_primary(results_per_prompt[nose_key])

    for i in range(len(pil_frames)):
        frame_data = {"frame_idx": i}

        head_out = results_per_prompt[head_key][i]
        ear_out = results_per_prompt[ear_key][i]
        nose_out = results_per_prompt[nose_key][i]

        # Head mask
        head_mask = None
        if primary_head is not None and primary_head in head_out["obj_id_to_mask"]:
            head_mask = _mask_from_logits(head_out["obj_id_to_mask"][primary_head])
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

        # Ears: pick the 2 highest-confidence ones inside head bbox, sort by x
        ear_candidates = []
        for oid, mask_t in ear_out["obj_id_to_mask"].items():
            m = _mask_from_logits(mask_t)
            if m.sum() < 50:
                continue
            mc = np.where(m > 127)
            cy, cx = mc[0].mean(), mc[1].mean()
            if (head_bbox[0] - 50 <= cy <= head_bbox[2] + 50 and
                    head_bbox[1] - 50 <= cx <= head_bbox[3] + 50):
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
            nm = _mask_from_logits(nose_out["obj_id_to_mask"][primary_nose])
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
        if anatomical is not None and (lb is not None or rb is not None):
            head_up_angle, head_center = anatomical
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

    elapsed = time.time() - start
    logger.info("Video analysis: %d frames in %.1fs", len(pil_frames), elapsed)

    # Summary stats
    left_angles = [f["left_ear_angle_deg"] for f in per_frame if "left_ear_angle_deg" in f]
    right_angles = [f["right_ear_angle_deg"] for f in per_frame if "right_ear_angle_deg" in f]

    return {
        "n_frames": len(pil_frames),
        "elapsed_s": elapsed,
        "per_frame": per_frame,
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
