#!/usr/bin/env bash
# Run on the RunPod pod to bring the sheep-seg server up.
#
# Handles the pain points from the first setup day:
#   - Ensures labels dir is symlinked to the durable Network Volume
#   - Ensures uv is installed and on PATH
#   - Ensures venv exists and is synced
#   - Sets PYTORCH_CUDA_ALLOC_CONF + SHEEPSEG_FULL_PIPELINE
#   - Starts uvicorn in the foreground so the shell shows logs
#
# Idempotent — safe to re-run.

set -e

cd "$(dirname "$0")/.."

# Load pod-side env (LABELS_VOLUME, etc.) if present
if [ -f .env.pod ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.pod
  set +a
fi

# 0. Durable labels dir on the Network Volume.
#
# The dataset is the one piece of state that represents unrecoverable human
# work. Container disk survives Stop/Resume but NOT Terminate or spot
# preemption. A Network Volume (attached via RunPod UI at pod-deploy time)
# survives both. We symlink data/labels → $LABELS_VOLUME so backend/config.py
# keeps using the same LABELS_DIR path.
#
# Set LABELS_VOLUME_SKIP=1 to bypass (only for first-time pods where you
# haven't attached a volume yet — labels will be on ephemeral container disk).
LABELS_VOLUME="${LABELS_VOLUME:-/mnt/labels}"

if [ "${LABELS_VOLUME_SKIP:-0}" = "1" ]; then
  echo "[startup] WARNING: LABELS_VOLUME_SKIP=1 — labels will live on ephemeral container disk."
  echo "[startup] Terminate or spot preemption will DESTROY labeling work. Attach a Network Volume ASAP."
elif [ -L data/labels ] && [ -d data/labels ]; then
  resolved=$(readlink -f data/labels)
  echo "[startup] data/labels → $resolved (durable volume OK)"
elif [ ! -d "$LABELS_VOLUME" ]; then
  cat >&2 <<EOF
[startup] FATAL: LABELS_VOLUME=$LABELS_VOLUME does not exist.

The labels dir needs a durable RunPod Network Volume mounted at that path,
otherwise a Terminate or spot preemption will wipe all labeling work.

Fix (in the RunPod console):
  1. Create a Network Volume (if you don't have one).
  2. Edit the pod → attach the volume with mount path = $LABELS_VOLUME.
  3. Stop + Resume the pod so the mount takes effect.
  4. Re-run this script.

Or, if you intentionally want labels on ephemeral container disk (first-time
pod setup, no volume yet), re-run with:
  LABELS_VOLUME_SKIP=1 bash scripts/start_pod_server.sh
EOF
  exit 1
else
  # Volume is mounted and data/labels isn't a symlink yet. Set it up.
  if [ -d data/labels ] && [ -n "$(ls -A data/labels 2>/dev/null)" ]; then
    echo "[startup] Migrating existing data/labels → $LABELS_VOLUME ..."
    rsync -a data/labels/ "$LABELS_VOLUME/"
    rm -rf data/labels
  elif [ -d data/labels ]; then
    rmdir data/labels
  fi
  mkdir -p "$LABELS_VOLUME"
  mkdir -p data
  ln -sfn "$LABELS_VOLUME" data/labels
  echo "[startup] data/labels → $LABELS_VOLUME (symlink created)"
fi

# 1. Install rsync if missing — needed for backup_dataset.sh on the laptop
# to pull labels off the pod. RunPod's pytorch base image doesn't ship it.
if ! command -v rsync >/dev/null 2>&1; then
  echo "[startup] Installing rsync..."
  apt-get update -qq && apt-get install -y rsync
fi

# 2. Install uv if missing (fresh pods sometimes lose it after redeploy)
if [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "[startup] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Make sure uv is on PATH for this shell
if [ -f "$HOME/.local/bin/env" ]; then
  source "$HOME/.local/bin/env"
fi

# 2. Sync deps (idempotent; fast if already up to date)
echo "[startup] uv sync..."
uv sync

# 3. Confirm CUDA is visible and torch is sane
uv run python -c "import torch; print('[startup] torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 4. Confirm HuggingFace auth (needed to download SAM 3)
if ! uv run hf auth whoami >/dev/null 2>&1; then
  echo ""
  echo "[startup] WARNING: not logged in to HuggingFace."
  echo "[startup] Run:   uv run hf auth login"
  echo "[startup] Then re-run this script."
  exit 1
fi

# 5. Start uvicorn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SHEEPSEG_FULL_PIPELINE=1

echo ""
echo "[startup] Starting uvicorn on 0.0.0.0:8000..."
echo "[startup] Open the pod's HTTP Service URL (port 8000) in your laptop browser."
echo ""

exec uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
