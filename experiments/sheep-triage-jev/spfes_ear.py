"""The SPFES ear slice: triage vocabulary and the deterministic book rules.

McLennan et al. (2016), the Sheep Pain Facial Expression Scale (SPFES), scores
five facial areas. The v0.7 pose model sees only the nose and the ears, so this
experiment uses the ear items alone: unusual carriage, left/right asymmetry and
frequent posture change. It is a proxy for a slice of one scale, not a pain
score.

Jev makes the soft call. ``book_triage`` is a fixed-threshold baseline over
the same ``EarPack`` so the two can be compared run by run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

from pose_features import EarPack, EarStats

Decision = Literal["look", "skip", "cannot"]
Reason = Literal[
    "flutter",
    "asymmetry",
    "carriage",
    "one_ear_missing",
    "steady",
    "not_facing",
    "ears_unmeasurable",
]
DECISIONS: tuple[Decision, ...] = get_args(Decision)
REASONS: tuple[Reason, ...] = get_args(Reason)
REASONS_FOR: Mapping[Decision, tuple[Reason, ...]] = {
    "look": ("flutter", "asymmetry", "carriage", "one_ear_missing"),
    "skip": ("steady",),
    "cannot": ("not_facing", "ears_unmeasurable"),
}

FLUTTER_DEG = 15.0  # ear-angle std across frames
ASYMMETRY_DEG = 25.0  # median |left - right|
CARRIAGE_LOW_DEG = 75.0  # median below this: ear held far forward
CARRIAGE_HIGH_DEG = 140.0  # median above this: ear held far back
FACING_MIN = 0.5  # share of frames with the face toward the camera
MEASURED_MIN = 0.5  # share of facing frames with that ear measured

BOOK_RULES: dict[str, float] = {
    "flutter_deg": FLUTTER_DEG,
    "asymmetry_deg": ASYMMETRY_DEG,
    "carriage_low_deg": CARRIAGE_LOW_DEG,
    "carriage_high_deg": CARRIAGE_HIGH_DEG,
    "facing_min": FACING_MIN,
    "measured_min": MEASURED_MIN,
}


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: Reason

    def to_json(self) -> dict[str, str]:
        return {"decision": self.decision, "reason": self.reason}


def ear_seen(ear: EarStats) -> bool:
    """Measured on enough facing frames to judge its carriage."""
    return ear.median_deg is not None and ear.std_deg is not None and ear.measured_fraction >= MEASURED_MIN


def unusual_carriage(median_deg: float) -> bool:
    return not CARRIAGE_LOW_DEG <= median_deg <= CARRIAGE_HIGH_DEG


def book_triage(pack: EarPack) -> Verdict:
    """Fixed ear rules, first match wins. Speed and time in view play no part."""
    if pack.facing_fraction < FACING_MIN:
        return Verdict("cannot", "not_facing")
    seen = [ear for ear in (pack.left_ear, pack.right_ear) if ear_seen(ear)]
    if not seen:
        return Verdict("cannot", "ears_unmeasurable")
    if any(ear.std_deg is not None and ear.std_deg >= FLUTTER_DEG for ear in seen):
        return Verdict("look", "flutter")
    if len(seen) == 2 and pack.asymmetry_deg is not None and pack.asymmetry_deg >= ASYMMETRY_DEG:
        return Verdict("look", "asymmetry")
    if any(ear.median_deg is not None and unusual_carriage(ear.median_deg) for ear in seen):
        return Verdict("look", "carriage")
    if len(seen) == 1:
        return Verdict("look", "one_ear_missing")
    return Verdict("skip", "steady")
