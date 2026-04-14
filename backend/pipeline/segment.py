"""SAM segmentation pipeline for sheep facial features.

Uses Meta's SAM via HuggingFace transformers (auto-downloads weights).
Produces masks for the whole animal, then uses point prompts for ear regions.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from backend.config import MIN_MASK_AREA_FRACTION
from backend.models import SegmentationResult

logger = logging.getLogger(__name__)

# Global model cache
_model = None
_processor = None
_device = None

# SAM 3 (text-prompted) model cache
_sam3_model = None
_sam3_processor = None


def _free_sam3_video_model():
    """Unload the SAM 3 video model to free VRAM for the image model."""
    from backend.pipeline import video as _vid
    if _vid._video_model is not None:
        logger.info("Unloading SAM 3 video model to free VRAM...")
        _vid._video_model.cpu()
        _vid._video_model = None
        _vid._video_processor = None
        torch.cuda.empty_cache()


def _load_sam3_model():
    """Load SAM 3 model from HuggingFace. Downloads ~3 GB on first run."""
    global _sam3_model, _sam3_processor, _device
    try:
        from transformers import Sam3Model, Sam3Processor

        # Free video model first — they don't both fit on a 6GB GPU
        _free_sam3_video_model()

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading SAM 3 on %s...", _device)
        _sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
        _sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(_device)
        _sam3_model.eval()
        logger.info("SAM 3 model loaded.")
    except Exception as e:
        logger.error("Failed to load SAM 3 model: %s", e)
        _sam3_model = None


def _load_model():
    """Load SAM 2.1 model from HuggingFace. Downloads ~350MB on first run."""
    global _model, _processor, _device
    try:
        from transformers import Sam2Model, Sam2Processor

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "facebook/sam2.1-hiera-small"
        logger.info("Loading SAM 2.1 hiera-small on %s...", _device)
        _processor = Sam2Processor.from_pretrained(model_id)
        _model = Sam2Model.from_pretrained(model_id).to(_device)
        _model.eval()
        logger.info("SAM 2.1 model loaded.")
    except Exception as e:
        logger.error("Failed to load SAM 2.1 model: %s", e)
        _model = None


def _mask_to_base64_png(mask: np.ndarray) -> str:
    """Convert a binary mask (H, W) to base64-encoded PNG."""
    if mask.max() > 1:
        mask_uint8 = mask.astype(np.uint8)
    else:
        mask_uint8 = (mask.astype(np.float32) * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _predict_mask(
    image_pil: Image.Image,
    points: list[list[int]],
    labels: list[int] | None = None,
    target_area: tuple[float, float] | None = None,
) -> Optional[np.ndarray]:
    """Run SAM 2.1 with point prompts. Returns the best mask or None.

    Args:
        points: list of [x, y] coordinates
        labels: 1=positive (include), 0=negative (exclude). Defaults to all positive.
        target_area: (min_frac, max_frac) of image area. If set, prefer masks
                     in this size range over highest score. Use for heads.
    """
    if _model is None or _processor is None:
        return None

    if labels is None:
        labels = [1] * len(points)

    # SAM 2 expects 4-level nesting: [image, object, point, xy]
    input_points = [[points]]
    input_labels = [[labels]]
    inputs = _processor(
        image_pil,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)

    masks = _processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
    )

    scores = outputs.iou_scores.cpu().numpy()[0][0]
    mask_array = masks[0][0].numpy()
    h, w = mask_array.shape[1], mask_array.shape[2]
    total_pixels = h * w

    if target_area is not None:
        # Pick the best-scoring mask that falls within the target size range
        min_frac, max_frac = target_area
        best_idx = None
        best_score = -1
        for i in range(len(scores)):
            frac = mask_array[i].sum() / total_pixels
            if min_frac <= frac <= max_frac and scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        # Fallback: pick the largest mask if none matched
        if best_idx is None:
            areas = [mask_array[i].sum() for i in range(len(scores))]
            best_idx = int(np.argmax(areas))
    else:
        best_idx = scores.argmax()

    best_mask = mask_array[best_idx]
    return best_mask.astype(np.uint8) * 255


def _create_extraction(image: np.ndarray, head_mask: np.ndarray) -> str:
    """Extract the face from the image, remove background, render in grey material.

    Produces the SAM 3D Body / Blender clay-render look:
    - Black background
    - Face extracted and converted to grey/silver material
    - Subtle edge lighting for depth
    """
    h, w = image.shape[:2]
    mask_bool = head_mask > 127

    # Convert to greyscale and boost to silver/clay look
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize and shift to mid-grey range (clay material look)
    grey = cv2.normalize(grey, None, 60, 210, cv2.NORM_MINMAX)

    # Apply slight gaussian blur for the smooth clay/scan feel
    grey = cv2.GaussianBlur(grey, (3, 3), 0)

    # Edge detection for rim lighting effect
    edges = cv2.Canny(grey, 50, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)

    # Build output: black background, grey face, bright edges
    output = np.zeros((h, w, 3), dtype=np.uint8)

    # Apply grey material where mask is
    for c in range(3):
        output[:, :, c] = np.where(mask_bool, grey, 0)

    # Add subtle edge highlights (white rim light)
    edge_mask = (edges > 30) & mask_bool
    output[edge_mask] = [220, 230, 240]

    # Crop to face bounding box with padding
    coords = np.where(mask_bool)
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        pad = int(max(y_max - y_min, x_max - x_min) * 0.15)
        y_min = max(0, y_min - pad)
        y_max = min(h, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(w, x_max + pad)

        # Make it square
        crop_h = y_max - y_min
        crop_w = x_max - x_min
        size = max(crop_h, crop_w)
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
        y1 = max(0, cy - size // 2)
        y2 = min(h, y1 + size)
        x1 = max(0, cx - size // 2)
        x2 = min(w, x1 + size)

        output = output[y1:y2, x1:x2]

    # Resize to standard output
    output = cv2.resize(output, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    _, buf = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _create_ear_overlay(image: np.ndarray, masks: dict[str, np.ndarray]) -> str:
    """Create a view with colored SPFES facial regions highlighted.

    Supports ears, eyes, and nose — all key SPFES action unit regions.
    """
    overlay = image.copy()
    # BGR colors for OpenCV
    colors = {
        "left_ear": (0, 220, 0),       # green
        "right_ear": (0, 120, 255),     # orange
        "left_eye": (255, 180, 0),      # cyan
        "right_eye": (255, 180, 0),     # cyan
        "nose": (0, 220, 220),          # yellow
    }
    labels = {
        "left_ear": "Ear",
        "right_ear": "Ear",
        "left_eye": "Eye",
        "right_eye": "Eye",
        "nose": "Nose",
    }

    draw_order = ["left_ear", "right_ear", "left_eye", "right_eye", "nose"]

    for name in draw_order:
        mask = masks.get(name)
        if mask is None:
            continue
        mask_bool = mask > 127
        if not mask_bool.any():
            continue
        color = colors.get(name, (200, 200, 200))

        # Color overlay
        overlay[mask_bool] = (
            overlay[mask_bool].astype(np.float32) * 0.4 +
            np.array(color, dtype=np.float32) * 0.6
        ).astype(np.uint8)

        # Contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 3)

        # Label at centroid
        coords = np.where(mask_bool)
        cy, cx = int(coords[0].mean()), int(coords[1].mean())
        label = labels.get(name, name)
        cv2.putText(overlay, label, (cx - 12, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def segment_sheep(
    image: np.ndarray, photo_id: str, demo_mode: bool = False
) -> SegmentationResult:
    """Run SAM segmentation on a sheep photo.

    Strategy:
    1. Prompt SAM at image center to get the main animal mask (head/body)
    2. From the head bounding box, prompt upper-left and upper-right for ears
    """
    global _model

    if demo_mode:
        return _generate_demo_masks(image, photo_id)

    # Lazy-load model
    if _model is None:
        _load_model()
    if _model is None:
        logger.warning("SAM unavailable, using demo masks for %s", photo_id)
        return _generate_demo_masks(image, photo_id)

    start = time.time()
    h, w = image.shape[:2]

    # Convert BGR to RGB PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    result_masks = {}
    raw_masks = {}
    confidence = {}

    # Step 1: Find the animal face.
    # Strategy: sample a grid of points across the image, run SAM on each,
    # and pick the mask that looks most like a face (medium-sized, compact).
    grid_points = [
        [w * x // 4, h * y // 4]
        for y in range(1, 4)
        for x in range(1, 4)
    ]

    best_mask = None
    best_score = -1

    for pt in grid_points:
        mask = _predict_mask(image_pil, [pt])
        if mask is None:
            continue
        area = (mask > 127).sum()
        area_frac = area / (h * w)
        # We want a face-sized mask: between 2% and 40% of the image
        if area_frac < 0.02 or area_frac > 0.40:
            continue
        # Score by compactness (area / bounding_box_area) — faces are compact
        coords = np.where(mask > 127)
        bbox_area = (coords[0].max() - coords[0].min()) * (coords[1].max() - coords[1].min())
        if bbox_area == 0:
            continue
        compactness = area / bbox_area
        # Prefer medium-sized, compact masks
        score = compactness * (1.0 - abs(area_frac - 0.10))
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is not None:
        head_mask = best_mask
        result_masks["head"] = _mask_to_base64_png(head_mask)
        raw_masks["head"] = head_mask
        confidence["head"] = 0.9

        # Find head bounding box
        coords = np.where(head_mask > 127)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        head_h = y_max - y_min
        head_w = x_max - x_min

        kern_size = max(3, int(min(head_w, head_h) * 0.12))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
        eroded = cv2.erode(head_mask, kernel, iterations=1)
        protrusions = cv2.subtract(head_mask, eroded)
        protrusions[int(y_min + head_h * 0.45):, :] = 0

        mid_x = x_min + head_w // 2
        left_ear_mask = protrusions.copy(); left_ear_mask[:, mid_x:] = 0
        right_ear_mask = protrusions.copy(); right_ear_mask[:, :mid_x] = 0

        if (left_ear_mask > 127).sum() > 50:
            result_masks["left_ear"] = _mask_to_base64_png(left_ear_mask)
            raw_masks["left_ear"] = left_ear_mask
            confidence["left_ear"] = 0.8
        if (right_ear_mask > 127).sum() > 50:
            result_masks["right_ear"] = _mask_to_base64_png(right_ear_mask)
            raw_masks["right_ear"] = right_ear_mask
            confidence["right_ear"] = 0.8

    # Create visualizations
    if "head" in raw_masks:
        result_masks["_extraction"] = _create_extraction(image, raw_masks["head"])
    if raw_masks:
        result_masks["_ear_overlay"] = _create_ear_overlay(image, raw_masks)

    elapsed_ms = (time.time() - start) * 1000
    logger.info("Segmented %s in %.0fms", photo_id, elapsed_ms)

    return SegmentationResult(
        photo_id=photo_id,
        head_mask_found="head" in result_masks,
        left_ear_mask_found="left_ear" in result_masks,
        right_ear_mask_found="right_ear" in result_masks,
        masks=result_masks,
        confidence_scores=confidence,
        segmentation_time_ms=elapsed_ms,
    )


def segment_sheep_at_point(
    image: np.ndarray, photo_id: str, px: int, py: int
) -> SegmentationResult:
    """Run SAM with a user-clicked point on the animal's face.

    Much more precise than the grid search — the user clicks directly
    on the face, and SAM segments from that point.
    """
    global _model

    if _model is None:
        _load_model()
    if _model is None:
        return _generate_demo_masks(image, photo_id)

    start = time.time()
    h, w = image.shape[:2]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    result_masks = {}
    raw_masks = {}
    confidence = {}

    # Step 1: Segment at the clicked point — this is the face
    head_mask = _predict_mask(image_pil, [[px, py]], target_area=(0.02, 0.40))

    if head_mask is not None and (head_mask > 127).sum() > h * w * MIN_MASK_AREA_FRACTION:
        result_masks["head"] = _mask_to_base64_png(head_mask)
        raw_masks["head"] = head_mask
        confidence["head"] = 0.95

        # Find head bounding box for ear prompts
        coords = np.where(head_mask > 127)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        head_h = y_max - y_min
        head_w = x_max - x_min

        # Ear isolation via morphological protrusion detection:
        # Erode the head mask, subtract from original → protrusions (ears).
        # Split at the head midline → left ear, right ear.
        kern_size = max(3, int(min(head_w, head_h) * 0.12))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kern_size, kern_size)
        )
        eroded = cv2.erode(head_mask, kernel, iterations=1)
        protrusions = cv2.subtract(head_mask, eroded)
        # Only keep upper 45% of head bbox (ears are at the top)
        protrusions[int(y_min + head_h * 0.45):, :] = 0

        mid_x = x_min + head_w // 2

        left_ear_mask = protrusions.copy()
        left_ear_mask[:, mid_x:] = 0

        right_ear_mask = protrusions.copy()
        right_ear_mask[:, :mid_x] = 0

        if (left_ear_mask > 127).sum() > 50:
            result_masks["left_ear"] = _mask_to_base64_png(left_ear_mask)
            raw_masks["left_ear"] = left_ear_mask
            confidence["left_ear"] = 0.8

        if (right_ear_mask > 127).sum() > 50:
            result_masks["right_ear"] = _mask_to_base64_png(right_ear_mask)
            raw_masks["right_ear"] = right_ear_mask
            confidence["right_ear"] = 0.8

    # Visualizations
    if "head" in raw_masks:
        result_masks["_extraction"] = _create_extraction(image, raw_masks["head"])
    if raw_masks:
        result_masks["_ear_overlay"] = _create_ear_overlay(image, raw_masks)

    elapsed_ms = (time.time() - start) * 1000
    logger.info("Click-segmented %s at (%d,%d) in %.0fms", photo_id, px, py, elapsed_ms)

    return SegmentationResult(
        photo_id=photo_id,
        head_mask_found="head" in result_masks,
        left_ear_mask_found="left_ear" in result_masks,
        right_ear_mask_found="right_ear" in result_masks,
        masks=result_masks,
        confidence_scores=confidence,
        segmentation_time_ms=elapsed_ms,
    )


def segment_sheep_multipoint(
    image: np.ndarray,
    photo_id: str,
    face_pt: tuple[int, int],
    left_ear_pt: tuple[int, int] | None = None,
    right_ear_pt: tuple[int, int] | None = None,
    left_eye_pt: tuple[int, int] | None = None,
    right_eye_pt: tuple[int, int] | None = None,
    nose_pt: tuple[int, int] | None = None,
) -> SegmentationResult:
    """Segment SPFES facial action unit regions with user-clicked landmarks.

    Uses SAM's positive/negative prompt capability:
    - Head mask: all points as positive prompts
    - Each feature: feature point positive, face center negative
      → SAM returns only that region, not the whole face

    SPFES regions (McLennan & Mahmoud 2019):
    - Ears: position/orientation
    - Eyes: orbital tightening
    - Nose: nostril dilation, bridge tension
    """
    global _model

    if _model is None:
        _load_model()
    if _model is None:
        return _generate_demo_masks(image, photo_id)

    start = time.time()
    h, w = image.shape[:2]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    result_masks = {}
    raw_masks = {}
    confidence = {}

    # Head mask: use all clicked points as positive prompts
    all_points = [list(face_pt)]
    feature_points = {
        "left_ear": left_ear_pt,
        "right_ear": right_ear_pt,
        "left_eye": left_eye_pt,
        "right_eye": right_eye_pt,
        "nose": nose_pt,
    }
    for pt in feature_points.values():
        if pt:
            all_points.append(list(pt))

    head_mask = _predict_mask(
        image_pil, all_points, labels=[1] * len(all_points),
        target_area=(0.02, 0.40),
    )

    if head_mask is not None and (head_mask > 127).sum() > h * w * MIN_MASK_AREA_FRACTION:
        result_masks["head"] = _mask_to_base64_png(head_mask)
        raw_masks["head"] = head_mask
        confidence["head"] = 0.95

        # Head bounding box for sizing constraints
        coords = np.where(head_mask > 127)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        head_size = min(y_max - y_min, x_max - x_min)

        for name, pt in feature_points.items():
            if pt is None:
                continue

            is_ear = "ear" in name

            if is_ear:
                # Ears are protrusions — pos/neg works well
                feature_mask = _predict_mask(
                    image_pil,
                    [list(pt), list(face_pt)],
                    labels=[1, 0],
                )
            else:
                # Eyes/nose are inside the face — use positive-only SAM
                # then clip to a small radius to prevent runaway masks
                feature_mask = _predict_mask(image_pil, [list(pt)])

                if feature_mask is not None:
                    # Radius constraint: eyes ~12%, nose ~15% of head size
                    radius_frac = 0.12 if "eye" in name else 0.15
                    radius = max(15, int(head_size * radius_frac))

                    constraint = np.zeros_like(feature_mask)
                    cv2.circle(constraint, (pt[0], pt[1]), radius, 255, -1)

                    # Intersect SAM mask with radius constraint and head mask
                    feature_mask = cv2.bitwise_and(feature_mask, constraint)
                    feature_mask = cv2.bitwise_and(feature_mask, head_mask)

            if feature_mask is not None and (feature_mask > 127).sum() > 50:
                result_masks[name] = _mask_to_base64_png(feature_mask)
                raw_masks[name] = feature_mask
                confidence[name] = 0.9

    # Visualizations
    if "head" in raw_masks:
        result_masks["_extraction"] = _create_extraction(image, raw_masks["head"])
    if raw_masks:
        result_masks["_ear_overlay"] = _create_ear_overlay(image, raw_masks)

    elapsed_ms = (time.time() - start) * 1000
    n_features = sum(1 for v in feature_points.values() if v is not None)
    logger.info(
        "SPFES segmented %s (%d features) in %.0fms",
        photo_id, n_features, elapsed_ms,
    )

    return SegmentationResult(
        photo_id=photo_id,
        head_mask_found="head" in result_masks,
        left_ear_mask_found="left_ear" in result_masks,
        right_ear_mask_found="right_ear" in result_masks,
        masks=result_masks,
        confidence_scores=confidence,
        segmentation_time_ms=elapsed_ms,
    )


def _generate_demo_masks(image: np.ndarray, photo_id: str) -> SegmentationResult:
    """Generate synthetic demo masks for testing without SAM."""
    h, w = image.shape[:2]

    head_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(head_mask, (w // 2, h // 2), (w // 4, h // 3), 0, 0, 360, 255, -1)

    left_ear_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(left_ear_mask, (w // 2 - w // 5, h // 2 - h // 4), (w // 12, h // 6), -30, 0, 360, 255, -1)

    right_ear_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(right_ear_mask, (w // 2 + w // 5, h // 2 - h // 4), (w // 12, h // 6), 30, 0, 360, 255, -1)

    raw_masks = {"head": head_mask, "left_ear": left_ear_mask, "right_ear": right_ear_mask}

    return SegmentationResult(
        photo_id=photo_id,
        head_mask_found=True,
        left_ear_mask_found=True,
        right_ear_mask_found=True,
        masks={
            "head": _mask_to_base64_png(head_mask),
            "left_ear": _mask_to_base64_png(left_ear_mask),
            "right_ear": _mask_to_base64_png(right_ear_mask),
            "_extraction": _create_extraction(image, head_mask),
            "_ear_overlay": _create_ear_overlay(image, raw_masks),
        },
        confidence_scores={"head": 0.95, "left_ear": 0.82, "right_ear": 0.80},
        segmentation_time_ms=0.0,
    )


def _build_face_mask(
    head_mask: np.ndarray,
    raw_masks: dict[str, np.ndarray],
    head_bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Build a tight face mask from landmark positions.

    Uses ears, eyes, and nose bounding boxes to compute a face-only bbox,
    then intersects it with the head mask. This excludes neck/wool from
    the downstream face extraction and depth mesh.
    """
    landmarks = [raw_masks.get(k) for k in
                 ("left_ear", "right_ear", "left_eye", "right_eye", "nose")]
    landmarks = [m for m in landmarks if m is not None]
    if len(landmarks) < 3:
        # Not enough landmarks to construct a reliable face bbox
        return None

    # Union of landmark bboxes
    ys_min, xs_min, ys_max, xs_max = [], [], [], []
    for m in landmarks:
        coords = np.where(m > 127)
        if len(coords[0]) == 0:
            continue
        ys_min.append(coords[0].min())
        xs_min.append(coords[1].min())
        ys_max.append(coords[0].max())
        xs_max.append(coords[1].max())

    if not ys_min:
        return None

    # Face bbox: tight around landmarks + ~10% padding
    y1, x1 = min(ys_min), min(xs_min)
    y2, x2 = max(ys_max), max(xs_max)
    pad_y = int((y2 - y1) * 0.10)
    pad_x = int((x2 - x1) * 0.10)
    h, w = head_mask.shape[:2]
    y1 = max(head_bbox[0], y1 - pad_y)
    y2 = min(head_bbox[2], y2 + pad_y)
    x1 = max(head_bbox[1], x1 - pad_x)
    x2 = min(head_bbox[3], x2 + pad_x)

    # Intersect head mask with face bbox
    face_mask = np.zeros_like(head_mask)
    face_mask[y1:y2, x1:x2] = head_mask[y1:y2, x1:x2]
    return face_mask


def _sam3_segment_text(image_pil: Image.Image, text: str) -> list[tuple[np.ndarray, float]]:
    """Run SAM 3 with a text prompt. Returns list of (mask, score) tuples
    sorted by score descending.
    """
    if _sam3_model is None or _sam3_processor is None:
        return []

    w, h = image_pil.size
    inputs = _sam3_processor(
        images=image_pil, text=text, return_tensors="pt"
    ).to(_device)

    with torch.no_grad():
        outputs = _sam3_model(**inputs)

    results = _sam3_processor.post_process_instance_segmentation(
        outputs, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)]
    )[0]

    out = []
    for mask, score in zip(results["masks"], results["scores"]):
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        out.append((mask_uint8, float(score)))

    out.sort(key=lambda x: -x[1])
    return out


def segment_sheep_sam3(
    image: np.ndarray, photo_id: str, subject: str = "sheep"
) -> SegmentationResult:
    """Segment sheep face using SAM 3 with text prompts — no clicks needed.

    Uses open-vocabulary segmentation to auto-detect head, ears, eyes.
    When multiple animals are in frame, picks the largest head instance
    and filters features to those inside its bounding box.

    Args:
        subject: "sheep" or "goat" — guides the text prompts
    """
    global _sam3_model

    if _sam3_model is None:
        _load_sam3_model()
    if _sam3_model is None:
        logger.warning("SAM 3 unavailable, falling back to grid search")
        return segment_sheep(image, photo_id)

    start = time.time()
    h, w = image.shape[:2]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    result_masks = {}
    raw_masks = {}
    confidence = {}

    # 1. Find the head — pick the largest high-confidence instance
    head_candidates = _sam3_segment_text(image_pil, f"{subject} head")
    if not head_candidates:
        head_candidates = _sam3_segment_text(image_pil, "head")

    head_mask = None
    head_bbox = None
    for mask, score in head_candidates:
        area_frac = (mask > 127).sum() / (h * w)
        # Head should be between 2% and 60% of image
        if 0.02 <= area_frac <= 0.60:
            head_mask = mask
            coords = np.where(mask > 127)
            head_bbox = (coords[0].min(), coords[1].min(),
                         coords[0].max(), coords[1].max())
            confidence["head"] = score
            break

    if head_mask is None:
        logger.warning("SAM 3 could not find head")
        return SegmentationResult(
            photo_id=photo_id,
            head_mask_found=False,
            left_ear_mask_found=False,
            right_ear_mask_found=False,
            masks={},
            confidence_scores={},
            segmentation_time_ms=(time.time() - start) * 1000,
        )

    result_masks["head"] = _mask_to_base64_png(head_mask)
    raw_masks["head"] = head_mask

    # 2. Find ears — filter to those inside the head bbox
    ear_candidates = _sam3_segment_text(image_pil, f"{subject} ear")
    if not ear_candidates:
        ear_candidates = _sam3_segment_text(image_pil, "ear")

    head_center_x = (head_bbox[1] + head_bbox[3]) // 2
    valid_ears = []
    for mask, score in ear_candidates:
        coords = np.where(mask > 127)
        if len(coords[0]) < 50:
            continue
        cy, cx = int(coords[0].mean()), int(coords[1].mean())
        # Ear centroid must be inside or near the head bbox
        in_head = (head_bbox[0] - 50 <= cy <= head_bbox[2] + 50 and
                   head_bbox[1] - 50 <= cx <= head_bbox[3] + 50)
        if in_head:
            valid_ears.append((mask, score, cx))
        if len(valid_ears) >= 2:
            break

    # Sort by x-position (left = viewer's left)
    valid_ears.sort(key=lambda e: e[2])
    if len(valid_ears) >= 1:
        result_masks["left_ear"] = _mask_to_base64_png(valid_ears[0][0])
        raw_masks["left_ear"] = valid_ears[0][0]
        confidence["left_ear"] = valid_ears[0][1]
    if len(valid_ears) >= 2:
        result_masks["right_ear"] = _mask_to_base64_png(valid_ears[1][0])
        raw_masks["right_ear"] = valid_ears[1][0]
        confidence["right_ear"] = valid_ears[1][1]

    # 3. Eyes — same approach, filter to inside head bbox
    eye_candidates = _sam3_segment_text(image_pil, f"{subject} eye")
    if not eye_candidates:
        eye_candidates = _sam3_segment_text(image_pil, "eye")

    valid_eyes = []
    for mask, score in eye_candidates:
        coords = np.where(mask > 127)
        if len(coords[0]) < 20:
            continue
        cy, cx = int(coords[0].mean()), int(coords[1].mean())
        # Eyes must be inside the head bbox
        if (head_bbox[0] <= cy <= head_bbox[2] and
                head_bbox[1] <= cx <= head_bbox[3]):
            valid_eyes.append((mask, score, cx))
        if len(valid_eyes) >= 2:
            break

    valid_eyes.sort(key=lambda e: e[2])
    if len(valid_eyes) >= 1:
        result_masks["left_eye"] = _mask_to_base64_png(valid_eyes[0][0])
        raw_masks["left_eye"] = valid_eyes[0][0]
        confidence["left_eye"] = valid_eyes[0][1]
    if len(valid_eyes) >= 2:
        result_masks["right_eye"] = _mask_to_base64_png(valid_eyes[1][0])
        raw_masks["right_eye"] = valid_eyes[1][0]
        confidence["right_eye"] = valid_eyes[1][1]

    # 4. Nose — single instance expected
    nose_candidates = _sam3_segment_text(image_pil, f"{subject} nose")
    if not nose_candidates:
        nose_candidates = _sam3_segment_text(image_pil, "nose")

    for mask, score in nose_candidates:
        coords = np.where(mask > 127)
        if len(coords[0]) < 20:
            continue
        cy, cx = int(coords[0].mean()), int(coords[1].mean())
        if (head_bbox[0] <= cy <= head_bbox[2] and
                head_bbox[1] <= cx <= head_bbox[3]):
            result_masks["nose"] = _mask_to_base64_png(mask)
            raw_masks["nose"] = mask
            confidence["nose"] = score
            break

    # 5. Build a tight face bbox from landmarks (ears, eyes, nose).
    # The head mask often includes the neck; this restricts to just
    # the face region for a cleaner extraction and depth mesh.
    face_mask = _build_face_mask(head_mask, raw_masks, head_bbox)
    if face_mask is not None:
        result_masks["face"] = _mask_to_base64_png(face_mask)
        raw_masks["face"] = face_mask

    # Visualizations — use tight face mask for extraction when available
    extraction_mask = face_mask if face_mask is not None else head_mask
    result_masks["_extraction"] = _create_extraction(image, extraction_mask)
    result_masks["_ear_overlay"] = _create_ear_overlay(image, raw_masks)

    elapsed_ms = (time.time() - start) * 1000
    has_nose = "nose" in result_masks
    logger.info(
        "SAM 3 segmented %s (head + %d ears + %d eyes + nose=%s) in %.0fms",
        photo_id, len(valid_ears), len(valid_eyes), has_nose, elapsed_ms,
    )

    return SegmentationResult(
        photo_id=photo_id,
        head_mask_found="head" in result_masks,
        left_ear_mask_found="left_ear" in result_masks,
        right_ear_mask_found="right_ear" in result_masks,
        masks=result_masks,
        confidence_scores=confidence,
        segmentation_time_ms=elapsed_ms,
    )
