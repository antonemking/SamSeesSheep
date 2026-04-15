"""Local VLM orchestrator using Gemma 4 E4B.

Same contract as the Claude orchestrator — takes an image, returns
structured scene analysis + SAM 3 prompts + welfare observations. The
only difference: runs entirely on the local GPU, no API key, no
network call.

Uses 4-bit quantization (bitsandbytes) so the ~15GB fp32 model fits
in roughly 2GB of VRAM, leaving room for SAM 3 alongside it on a 6GB
card.
"""

from __future__ import annotations

import io
import json
import logging
import re
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

LOCAL_MODEL_ID = "google/gemma-4-E4B-it"

_model = None
_processor = None


SYSTEM_PROMPT = """You are a vision orchestrator for a sheep / goat welfare
monitoring tool. Your job: look at one frame and decide what to ask the
SAM 3 segmentation model to find.

Output strict JSON with this schema:
{
  "scene_summary": "1-2 sentence plain English description of what you see",
  "subjects": [
    {
      "label": "short label e.g. 'ewe', 'lamb', 'goat'",
      "position": "rough position in frame e.g. 'left foreground'",
      "color": "fleece/coat color",
      "facing": "toward camera | away | left profile | right profile"
    }
  ],
  "segmentation_prompts": [
    "list of 3-6 SAM 3 text prompts to run, e.g. 'sheep head', 'sheep ear', 'sheep nose'"
  ],
  "welfare_observations": [
    "list of plain-English observations a viewer should know - ear position, posture, alertness. Be specific. NEVER use the word 'pain' or 'welfare' as outcomes - this is geometric observation only."
  ],
  "confidence_notes": "anything the user should distrust - occlusion, lighting, multiple animals confused, etc"
}

Rules:
- Be terse. No marketing language.
- If multiple animals: list each as a separate subject.
- The `segmentation_prompts` should be common nouns SAM 3 can match - "sheep head" works, "the ewe in the foreground" does not.
- Output ONLY the JSON, no preamble or markdown fences."""


def _free_sam3_models() -> None:
    """Unload SAM 3 image + video models to make VRAM room for Gemma."""
    import gc
    import torch
    from backend.pipeline import segment as _seg
    from backend.pipeline import video as _vid

    if _seg._sam3_model is not None:
        logger.info("Unloading SAM 3 image model...")
        _seg._sam3_model.cpu()
        _seg._sam3_model = None
        _seg._sam3_processor = None
    if _vid._video_model is not None:
        logger.info("Unloading SAM 3 video model...")
        _vid._video_model.cpu()
        _vid._video_model = None
        _vid._video_processor = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _load_local_vlm():
    """Load Gemma 4 E4B-it at 4-bit quantization."""
    global _model, _processor
    try:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        _free_sam3_models()

        logger.info("Loading %s at 4-bit...", LOCAL_MODEL_ID)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        _processor = AutoProcessor.from_pretrained(LOCAL_MODEL_ID)
        _model = AutoModelForImageTextToText.from_pretrained(
            LOCAL_MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
        )
        _model.eval()
        logger.info(
            "Gemma 4 loaded. VRAM allocated: %.2f GB",
            torch.cuda.memory_allocated() / 1024 ** 3,
        )
    except Exception as e:
        logger.error("Failed to load local VLM: %s", e)
        _model = None


def _resize_for_vlm(img: Image.Image) -> Image.Image:
    """Cap size to keep inference fast + memory bounded."""
    if max(img.size) > 768:
        scale = 768 / max(img.size)
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)),
            Image.LANCZOS,
        )
    return img.convert("RGB")


def orchestrate_scene_local(image_path_or_pil) -> Optional[dict]:
    """Run the local Gemma 4 orchestrator on an image.

    Returns the same dict structure as the Claude orchestrator, or None
    on failure.
    """
    global _model, _processor

    if _model is None:
        _load_local_vlm()
    if _model is None:
        return None

    try:
        import torch

        img = (
            image_path_or_pil
            if isinstance(image_path_or_pil, Image.Image)
            else Image.open(image_path_or_pil)
        )
        img = _resize_for_vlm(img)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {
                        "type": "text",
                        "text": "Analyze this frame and respond with the JSON schema.",
                    },
                ],
            },
        ]

        inputs = _processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(_model.device)

        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
            )

        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = _processor.decode(generated, skip_special_tokens=True).strip()

        # Strip markdown fences if the model added them despite instructions
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Try to extract JSON object from the text
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    logger.error(
                        "Gemma returned non-JSON: %s | text: %r", e, text[:400]
                    )
                    return None
            else:
                logger.error(
                    "Gemma returned non-JSON: %s | text: %r", e, text[:400]
                )
                return None

        data["model_used"] = LOCAL_MODEL_ID
        data["raw_text"] = text
        return data

    except Exception as e:
        logger.exception("Local orchestrator failed: %s", e)
        return None
