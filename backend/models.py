"""Pydantic models for the sheep-seg pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EarPosition(str, Enum):
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    UNMEASURABLE = "unmeasurable"


class SegmentationResult(BaseModel):
    """Result of SAM segmentation on a single photo."""

    photo_id: str
    head_mask_found: bool = False
    left_ear_mask_found: bool = False
    right_ear_mask_found: bool = False
    masks: dict[str, str] = Field(
        default_factory=dict,
        description="Region name -> base64 PNG of the binary mask",
    )
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    segmentation_time_ms: float = 0.0


class EarAngleResult(BaseModel):
    """Ear angle extraction for a single photo."""

    photo_id: str
    left_ear_angle_deg: Optional[float] = None
    right_ear_angle_deg: Optional[float] = None
    left_ear_position: EarPosition = EarPosition.UNMEASURABLE
    right_ear_position: EarPosition = EarPosition.UNMEASURABLE
    head_midline_angle_deg: Optional[float] = None
    measurement_quality: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0=unusable, 1=high quality"
    )


class PhotoAnalysis(BaseModel):
    """Complete analysis of a single photo."""

    photo_id: str
    filename: str
    upload_time: datetime = Field(default_factory=datetime.now)
    image_width: int = 0
    image_height: int = 0
    image_url: Optional[str] = None
    segmentation: Optional[SegmentationResult] = None
    ear_angles: Optional[EarAngleResult] = None
    is_demo: bool = False


class EUPResult(BaseModel):
    """EUP% computation across a set of photos."""

    total_photos: int = 0
    measurable_photos: int = 0
    ears_up_count: int = 0
    ears_neutral_count: int = 0
    ears_down_count: int = 0
    eup_percent: Optional[float] = None
    per_photo: list[PhotoAnalysis] = Field(default_factory=list)


class NarrativeResult(BaseModel):
    """Claude API-generated welfare narrative."""

    summary: str = ""
    methodology_note: str = ""
    limitations: str = ""
    references: list[str] = Field(default_factory=list)
    model_used: str = ""
    generated_at: datetime = Field(default_factory=datetime.now)


class SessionState(BaseModel):
    """Complete dashboard state."""

    photos: list[PhotoAnalysis] = Field(default_factory=list)
    eup: Optional[EUPResult] = None
    narrative: Optional[NarrativeResult] = None
    is_demo: bool = False
