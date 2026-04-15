"""VLM orchestrator for SAM 3 — Claude vision describes the scene
and generates the segmentation prompts.

The pattern: the VLM looks at frame 0, identifies subjects, decides
which segmentation prompts SAM 3 should run. SAM 3 then executes the
precise vision tasks. This is the "VLM as brain, segmenter as hands"
architecture for local AI on the farm.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from typing import Optional

from PIL import Image

from backend.config import CLAUDE_MODEL

logger = logging.getLogger(__name__)


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
    "list of 3-6 SAM 3 text prompts to run, e.g. 'sheep head', 'sheep ear', 'sheep nose', 'lamb', 'ewe'"
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


def _image_to_base64(image_path_or_pil) -> tuple[str, str]:
    """Read an image and return (base64_string, media_type)."""
    if isinstance(image_path_or_pil, Image.Image):
        img = image_path_or_pil
    else:
        img = Image.open(image_path_or_pil)
    img = img.convert("RGB")
    # Cap size for the API call (and to save tokens)
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def orchestrate_scene(image_path_or_pil) -> Optional[dict]:
    """Send an image to Claude vision and get structured scene analysis.

    Returns a dict with scene_summary, subjects, segmentation_prompts,
    welfare_observations, confidence_notes. None if the API key is missing
    or the call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, orchestrator unavailable")
        return None

    try:
        import anthropic

        img_b64, media_type = _image_to_base64(image_path_or_pil)
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyze this frame and respond with the JSON schema.",
                    },
                ],
            }],
        )

        text = response.content[0].text.strip()
        # Strip markdown fences if Claude added them despite instructions
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Claude returned non-JSON: %s | text: %r", e, text[:300])
            return None

        data["model_used"] = CLAUDE_MODEL
        data["raw_text"] = text
        return data

    except Exception as e:
        logger.error("Claude orchestrator call failed: %s", e)
        return None
