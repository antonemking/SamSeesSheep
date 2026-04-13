"""Session routes — get current dashboard state and demo data."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from backend.models import (
    EarAngleResult,
    EarPosition,
    EUPResult,
    NarrativeResult,
    PhotoAnalysis,
    SegmentationResult,
    SessionState,
)
from backend.routes.analyze import get_session

router = APIRouter(prefix="/api", tags=["session"])


@router.get("/session")
async def get_session_state() -> SessionState:
    """Return the current session state with all analysis results."""
    return get_session()


@router.get("/demo")
async def get_demo_data() -> SessionState:
    """Return pre-computed demo data using the real sample photo."""
    demo_photos = []

    # Simulated observations from the real sample photo at different times
    demo_data = [
        ("obs_1", "sheep_demo.jpg", EarPosition.UP, EarPosition.UP, 42.5, 38.2),
        ("obs_2", "sheep_demo.jpg", EarPosition.UP, EarPosition.NEUTRAL, 45.0, 15.0),
        ("obs_3", "sheep_demo.jpg", EarPosition.NEUTRAL, EarPosition.NEUTRAL, 12.3, 8.7),
        ("obs_4", "sheep_demo.jpg", EarPosition.UP, EarPosition.UP, 50.1, 47.3),
        ("obs_5", "sheep_demo.jpg", EarPosition.DOWN, EarPosition.DOWN, -15.2, -18.6),
        ("obs_6", "sheep_demo.jpg", EarPosition.NEUTRAL, EarPosition.UP, 20.1, 35.4),
        ("obs_7", "sheep_demo.jpg", EarPosition.NEUTRAL, EarPosition.DOWN, 5.3, -12.8),
        ("obs_8", "sheep_demo.jpg", EarPosition.UP, EarPosition.UP, 48.0, 44.5),
    ]

    for pid, fname, left_pos, right_pos, left_angle, right_angle in demo_data:
        photo = PhotoAnalysis(
            photo_id=pid,
            filename=fname,
            upload_time=datetime.now(),
            image_width=1200,
            image_height=900,
            image_url="/sample/sheep_demo.jpg",
            segmentation=SegmentationResult(
                photo_id=pid,
                head_mask_found=True,
                left_ear_mask_found=True,
                right_ear_mask_found=True,
                masks={},
                confidence_scores={"head": 0.94, "left_ear": 0.78, "right_ear": 0.76},
                segmentation_time_ms=450.0,
            ),
            ear_angles=EarAngleResult(
                photo_id=pid,
                left_ear_angle_deg=left_angle,
                right_ear_angle_deg=right_angle,
                left_ear_position=left_pos,
                right_ear_position=right_pos,
                head_midline_angle_deg=0.0,
                measurement_quality=0.75,
            ),
            is_demo=True,
        )
        demo_photos.append(photo)

    eup = EUPResult(
        total_photos=8,
        measurable_photos=8,
        ears_up_count=5,
        ears_neutral_count=2,
        ears_down_count=1,
        eup_percent=62.5,
        per_photo=demo_photos,
    )

    narrative = NarrativeResult(
        summary=(
            "Analysis of 8 observations from a Delaware homestead flock using SAM "
            "segmentation found measurable ear positions in all 8 images (100% "
            "segmentation success rate). The computed Ear-Up Percentage (EUP%) is "
            "62.5% — in 5 of 8 observations, at least one ear was classified as "
            "'up/alert' based on angle relative to the dorsal head axis.\n\n"
            "This is consistent with a flock in ambient pasture conditions with no "
            "documented acute stressors. Reefmann et al. (2009) associate forward/up "
            "ear position with positive valence or vigilance, while backward/low "
            "position correlates with negative valence. The single 'down' observation "
            "(-15.2deg / -18.6deg) aligns with the expected transient ear-position "
            "drop during handling events documented by Guesgen et al. (2016).\n\n"
            "Limitations: N=5 sheep, single homestead, single non-veterinary annotator, "
            "variable lighting, handheld phone camera. Only within-animal EUP% deltas "
            "are valid. This is demo data. See VALIDATION.md for claim boundaries."
        ),
        methodology_note=(
            "Ear position extracted from SAM segmentation masks using 2D geometric "
            "analysis (PCA + ellipse fitting). Thresholds: >30deg = up, <-10deg = down, "
            "per SPFES literature."
        ),
        limitations="DEMO DATA. See VALIDATION.md.",
        references=[
            "McLennan et al. (2019) — SPFES",
            "Reefmann et al. (2009) — Ear posture taxonomy",
            "Boissy et al. (2011) — Ear posture and emotional valence",
            "Guesgen et al. (2016) — Ear position during painful procedures",
        ],
        model_used="demo",
        generated_at=datetime.now(),
    )

    return SessionState(
        photos=demo_photos,
        eup=eup,
        narrative=narrative,
        is_demo=True,
    )
