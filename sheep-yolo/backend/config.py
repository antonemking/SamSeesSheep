"""Configuration for sheep-yolo."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

for d in (UPLOAD_DIR, RESULTS_DIR, WEIGHTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# YOLOE (Ultralytics open-vocabulary seg) is the primary model — text-prompted
# part detection without retraining. Override via env var to swap in a custom
# checkpoint once one exists.
YOLOE_MODEL = os.getenv("YOLOE_MODEL", "yoloe-v8l-seg.pt")

# Fallback: plain COCO-pretrained YOLO11-seg. Only detects the whole "sheep"
# class. Kicks in when YOLOE returns zero part detections — lets the UI still
# show tracked bounding boxes so the user can see "YOLO OOB didn't find parts".
YOLO_FALLBACK_MODEL = os.getenv("YOLO_FALLBACK_MODEL", "yolo26l-seg.pt")

# Text prompts for the part detector. Plural forms help YOLOE generalize to
# two-ear detections per sheep.
PART_PROMPTS = ["sheep head", "sheep ear", "sheep nose"]

# Ultralytics tracker config. "bytetrack.yaml" ships with ultralytics and is
# the default. "botsort.yaml" can be set via env if ReID-style tracking helps.
TRACKER_YAML = os.getenv("TRACKER_YAML", "bytetrack.yaml")

# Inference frame cap. Longer than v1's 30 because YOLO is ~1000x faster than
# SAM 3 Video at inference — the point of the rewrite is to unlock long clips.
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "600"))

# Downscale large video frames before inference. 960px is a standard YOLO
# training size and keeps per-frame latency predictable on a 6GB GPU.
INFER_IMGSZ = int(os.getenv("INFER_IMGSZ", "960"))

# Confidence floor for detections. 0.25 filters out YOLO26's noisier
# false positives (e.g. human limbs misread as sheep) without dropping
# real sheep, which detect at 0.85+ on farm footage. Lower to 0.15 if
# running the YOLOE parts pipeline, where head detections peak ~0.30.
DETECTION_CONF = float(os.getenv("DETECTION_CONF", "0.25"))

# SPFES-derived ear-angle thresholds. Angle is measured relative to the dorsal
# head axis (nose → ear-midpoint). Values copied verbatim from sheep-seg v1 so
# the chart bands match across both tools and stay traceable to the same lit:
# McLennan & Mahmoud 2019 (SPFES), Reefmann 2009, Boissy 2011.
EAR_UP_THRESHOLD_DEG = 30.0
EAR_DOWN_THRESHOLD_DEG = -10.0

# Rejects tiny spurious mask blobs before we try to extract geometry from them.
MIN_MASK_AREA_FRACTION = 0.001

# Chart filter: drop tracks seen for less than this fraction of frames.
# ByteTrack re-issues a new ID every time a sheep briefly exits frame or gets
# occluded, so a video of 2 sheep easily produces 10+ raw track IDs. 5% is
# a noise floor that keeps real re-acquisitions (occluded-then-returned) on
# the chart while dropping single-frame flickers.
TRACK_COVERAGE_FLOOR = float(os.getenv("TRACK_COVERAGE_FLOOR", "0.15"))

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
