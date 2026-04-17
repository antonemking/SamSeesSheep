"""Pydantic models shared by the ear-angle pipeline."""

from __future__ import annotations

from enum import Enum


class EarPosition(str, Enum):
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    UNMEASURABLE = "unmeasurable"
