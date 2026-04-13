"""CLI entry point for sheep-seg batch processing.

Usage:
    python -m src.cli [data/sample/]    Process all photos in a directory
    python -m src.cli --demo            Run with synthetic demo data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import PhotoAnalysis
from backend.pipeline.ear_angle import extract_ear_angles
from backend.pipeline.eup import compute_eup
from backend.pipeline.narrative import generate_narrative
from backend.pipeline.segment import segment_sheep


def process_directory(photo_dir: Path, demo_mode: bool = False) -> None:
    """Process all photos in a directory through the full pipeline."""
    image_extensions = {".jpg", ".jpeg", ".png", ".heic", ".bmp"}
    photo_files = sorted(
        f for f in photo_dir.iterdir()
        if f.suffix.lower() in image_extensions
    )

    if not photo_files:
        print(f"No image files found in {photo_dir}")
        return

    print(f"Processing {len(photo_files)} photos from {photo_dir}")
    print("=" * 60)

    analyses = []

    for i, photo_path in enumerate(photo_files, 1):
        print(f"\n[{i}/{len(photo_files)}] {photo_path.name}")

        image = cv2.imread(str(photo_path))
        if image is None:
            print(f"  SKIP: Could not read image")
            continue

        h, w = image.shape[:2]
        photo_id = photo_path.stem

        # Segment
        print(f"  Segmenting... ", end="", flush=True)
        seg = segment_sheep(image, photo_id, demo_mode=demo_mode)
        print(f"head={'Y' if seg.head_mask_found else 'N'} "
              f"L_ear={'Y' if seg.left_ear_mask_found else 'N'} "
              f"R_ear={'Y' if seg.right_ear_mask_found else 'N'} "
              f"({seg.segmentation_time_ms:.0f}ms)")

        # Extract ear angles
        ears = extract_ear_angles(seg)
        if ears.left_ear_angle_deg is not None:
            print(f"  Left ear:  {ears.left_ear_angle_deg:+.1f}deg ({ears.left_ear_position.value})")
        if ears.right_ear_angle_deg is not None:
            print(f"  Right ear: {ears.right_ear_angle_deg:+.1f}deg ({ears.right_ear_position.value})")

        analysis = PhotoAnalysis(
            photo_id=photo_id,
            filename=photo_path.name,
            image_width=w,
            image_height=h,
            segmentation=seg,
            ear_angles=ears,
        )
        analyses.append(analysis)

    # Compute EUP%
    print("\n" + "=" * 60)
    eup = compute_eup(analyses)
    print(f"\nRESULTS:")
    print(f"  Total photos:     {eup.total_photos}")
    print(f"  Measurable:       {eup.measurable_photos}")
    print(f"  Ears up:          {eup.ears_up_count}")
    print(f"  Ears neutral:     {eup.ears_neutral_count}")
    print(f"  Ears down:        {eup.ears_down_count}")
    print(f"  EUP%:             {eup.eup_percent}%")

    if eup.total_photos > 0:
        success_rate = round((eup.measurable_photos / eup.total_photos) * 100, 1)
        print(f"  Segmentation rate: {success_rate}%")

    # Save results
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "latest_results.json"

    results_data = {
        "eup_percent": eup.eup_percent,
        "total_photos": eup.total_photos,
        "measurable_photos": eup.measurable_photos,
        "ears_up": eup.ears_up_count,
        "ears_neutral": eup.ears_neutral_count,
        "ears_down": eup.ears_down_count,
        "photos": [a.model_dump(mode="json") for a in analyses],
    }
    results_file.write_text(json.dumps(results_data, indent=2, default=str))
    print(f"\nResults saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="sheep-seg CLI — batch photo analysis")
    parser.add_argument("photo_dir", nargs="?", default="data/sample",
                        help="Directory containing sheep photos")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic demo masks (no SAM model needed)")
    args = parser.parse_args()

    photo_dir = Path(args.photo_dir)
    if not photo_dir.exists():
        print(f"Directory not found: {photo_dir}")
        sys.exit(1)

    process_directory(photo_dir, demo_mode=args.demo)


if __name__ == "__main__":
    main()
