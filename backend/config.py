"""Configuration for sheep-seg."""

from pathlib import Path

# Register HEIC support globally
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
SAMPLE_DIR = DATA_DIR / "sample"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# SAM model config
SAM_MODEL_TYPE = "sam2.1_hiera_large"
SAM_CHECKPOINT = WEIGHTS_DIR / "sam2.1_hiera_large.pt"

# Ear angle thresholds (derived from SPFES literature)
# McLennan et al. 2019, Reefmann et al. 2009
# Ear angle relative to dorsal head axis:
#   > 30 degrees above horizontal = "up/alert" (positive valence or vigilance)
#   -10 to 30 degrees = "neutral"
#   < -10 degrees = "down/back" (negative valence, pain, or submission)
EAR_UP_THRESHOLD_DEG = 30.0
EAR_DOWN_THRESHOLD_DEG = -10.0

# Minimum contour area (fraction of image area) to consider a mask valid
MIN_MASK_AREA_FRACTION = 0.001

# Claude API
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Server
HOST = "0.0.0.0"
PORT = 8000
