"""EUP% (Ear-Up Percentage) computation.

EUP% is defined as:
    (count of photos where at least one ear is classified as "up") /
    (count of photos where at least one ear is measurable) * 100

Per VALIDATION.md: this is a within-animal delta metric.
We trust deltas, not absolutes.
"""

from __future__ import annotations

from backend.models import EarPosition, EUPResult, PhotoAnalysis


def compute_eup(photos: list[PhotoAnalysis]) -> EUPResult:
    """Compute EUP% across a set of analyzed photos.

    Only photos with at least one measurable ear are included in the denominator.
    """
    measurable = 0
    ears_up = 0
    ears_neutral = 0
    ears_down = 0

    for photo in photos:
        if photo.ear_angles is None:
            continue

        left = photo.ear_angles.left_ear_position
        right = photo.ear_angles.right_ear_position

        # A photo is measurable if at least one ear has a real classification
        positions = [
            p for p in [left, right] if p != EarPosition.UNMEASURABLE
        ]
        if not positions:
            continue

        measurable += 1

        # Classify the photo by the "most extreme" ear position
        # If either ear is UP, count as UP (alert/vigilant)
        # If either ear is DOWN, count as DOWN (negative valence)
        # Otherwise NEUTRAL
        if EarPosition.UP in positions:
            ears_up += 1
        elif EarPosition.DOWN in positions:
            ears_down += 1
        else:
            ears_neutral += 1

    eup_pct = None
    if measurable > 0:
        eup_pct = round((ears_up / measurable) * 100, 1)

    return EUPResult(
        total_photos=len(photos),
        measurable_photos=measurable,
        ears_up_count=ears_up,
        ears_neutral_count=ears_neutral,
        ears_down_count=ears_down,
        eup_percent=eup_pct,
        per_photo=photos,
    )
