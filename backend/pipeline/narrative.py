"""Claude API narrative generation for welfare analysis.

Generates a plain-English summary of the EUP% analysis,
referencing published veterinary literature (SPFES, Reefmann, Boissy, Sandem).
Honest about limitations per VALIDATION.md.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from backend.config import CLAUDE_MODEL
from backend.models import EUPResult, NarrativeResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a scientific communicator writing about animal welfare
computer vision research. You write for LinkedIn — affirmative, specific, honest
about limitations, and referencing published veterinary literature.

You are summarizing results from a feasibility study called sheep-seg: a public,
honest experiment testing whether a foundation segmentation model (SAM) can extract
ear-position welfare signals from phone photos of sheep on a small Delaware homestead.

Key rules:
- NEVER use the word "pain" as something this system detects. It detects ear position.
- NEVER use the word "welfare" as a measured outcome. Welfare is multi-dimensional.
- NEVER use emoji-style emotional labels ("happy", "sad", "stressed").
- Always reference the published literature by author name.
- Always state the dataset size and limitations.
- Use the metric name: EUP% (Ear-Up Percentage).
- Trust deltas, not absolutes — only within-animal comparisons are valid.
- If the results are negative or inconclusive, say so directly.
"""

NARRATIVE_PROMPT = """Based on the following analysis results, write a 2-3 paragraph
plain-English summary suitable for a LinkedIn post. Include:

1. What was measured and how (SAM segmentation → ear angle → EUP%)
2. The key finding (EUP% value and what it means in context of the published literature)
3. Honest limitations and what comes next

Analysis Results:
- Total photos analyzed: {total_photos}
- Photos with measurable ears: {measurable_photos}
- Segmentation success rate: {success_rate}%
- EUP%: {eup_percent}
- Ears up: {ears_up}, Ears neutral: {ears_neutral}, Ears down: {ears_down}

Published thresholds reference:
- McLennan et al. (2019): SPFES validated ear position as one of five facial action
  units for sheep pain assessment
- Reefmann et al. (2009): Forward/up ear position associated with positive valence;
  backward/low position with negative valence
- Boissy et al. (2011): Ear posture as indicator of emotional valence in sheep

Dataset context:
- 5 sheep on a single Delaware homestead
- Phone photos, handheld, variable lighting
- Single non-veterinary annotator
- This is a feasibility study, not a validated clinical tool

{additional_context}

Write the summary now. Be specific with numbers. Be honest about limitations."""


def generate_narrative(
    eup_result: EUPResult,
    additional_context: str = "",
) -> NarrativeResult:
    """Generate a Claude API narrative from EUP% results."""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set. Returning placeholder narrative.")
        return _placeholder_narrative(eup_result)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        success_rate = 0
        if eup_result.total_photos > 0:
            success_rate = round(
                (eup_result.measurable_photos / eup_result.total_photos) * 100, 1
            )

        prompt = NARRATIVE_PROMPT.format(
            total_photos=eup_result.total_photos,
            measurable_photos=eup_result.measurable_photos,
            success_rate=success_rate,
            eup_percent=eup_result.eup_percent if eup_result.eup_percent is not None else "N/A",
            ears_up=eup_result.ears_up_count,
            ears_neutral=eup_result.ears_neutral_count,
            ears_down=eup_result.ears_down_count,
            additional_context=additional_context,
        )

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        summary_text = response.content[0].text

        return NarrativeResult(
            summary=summary_text,
            methodology_note=(
                "Ear position extracted from SAM segmentation masks using 2D geometric "
                "analysis. Ear-Up Percentage (EUP%) computed as the fraction of photos "
                "where at least one ear is classified as 'up/alert' based on angle "
                "relative to the dorsal head axis. Thresholds derived from McLennan et al. "
                "(2019) and Reefmann et al. (2009)."
            ),
            limitations=(
                "N=5 sheep, single homestead, single annotator (non-veterinary), "
                "variable lighting, handheld phone camera. Within-animal deltas only. "
                "No clinical validation. See VALIDATION.md for full claim boundaries."
            ),
            references=[
                "McLennan et al. (2019) - Sheep Pain Facial Expression Scale (SPFES)",
                "Reefmann et al. (2009) - Ear posture taxonomy in sheep",
                "Boissy et al. (2011) - Ear posture and emotional valence",
                "Sandem et al. (2002) - Eye white percentage as welfare indicator",
            ],
            model_used=CLAUDE_MODEL,
            generated_at=datetime.now(),
        )

    except Exception as e:
        logger.error("Claude API call failed: %s", e)
        return _placeholder_narrative(eup_result)


def _placeholder_narrative(eup_result: EUPResult) -> NarrativeResult:
    """Generate a placeholder narrative when Claude API is unavailable."""
    eup_str = f"{eup_result.eup_percent}%" if eup_result.eup_percent is not None else "N/A"

    return NarrativeResult(
        summary=(
            f"Analysis of {eup_result.total_photos} sheep photos using SAM segmentation "
            f"found measurable ear positions in {eup_result.measurable_photos} images "
            f"({round(eup_result.measurable_photos / max(1, eup_result.total_photos) * 100)}% "
            f"segmentation success rate). The computed Ear-Up Percentage (EUP%) is {eup_str}. "
            f"\n\n"
            f"This is a feasibility study on 5 sheep from a single Delaware homestead. "
            f"EUP% thresholds are derived from the published SPFES literature (McLennan "
            f"et al., 2019) and ear posture studies (Reefmann et al., 2009). Only "
            f"within-animal deltas are valid claims at this dataset size. "
            f"\n\n"
            f"[Set ANTHROPIC_API_KEY for a detailed Claude-generated narrative.]"
        ),
        methodology_note="Placeholder — Claude API key not configured.",
        limitations="See VALIDATION.md",
        references=[
            "McLennan et al. (2019) - SPFES",
            "Reefmann et al. (2009) - Ear posture",
        ],
        model_used="placeholder",
        generated_at=datetime.now(),
    )
