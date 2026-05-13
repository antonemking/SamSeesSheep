"""Pydantic/enum types shared across the pipeline."""

from __future__ import annotations

from enum import Enum


class EarPosition(str, Enum):
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    UNMEASURABLE = "unmeasurable"
