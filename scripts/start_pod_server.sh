#!/usr/bin/env bash
# Run on the Vast.ai instance to bring the sheep-seg server up.
#
# Handles the pain points from the first setup day:
#   - Installs rsync (most pytorch base images don't ship it)
#   - Symlinks data/labels → durable Vast Volume
#   - Persists HF cache + auth on the volume (survives instance destroy)
#   - Ensures uv is installed and on PATH
#   - Ensures venv exists and is synced
#   - Sets PYTORCH_CUDA_ALLOC_CONF + SHEEPSEG_FULL_PIPELINE
#   - Starts uvicorn in the foreground so the shell shows logs
#
# Idempotent — safe to re-run.

set -e

cd "$(dirname "$0")/.."

# Load pod-side env (LABELS_VOLUME, HF_HOME, etc.) if present
if [ -f .env.pod ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.pod
  set +a
fi

# 0a. Install rsync if missing — needed for backup_dataset.sh on the laptop
# to pull labels off the pod, and for the migration steps below. Most pytorch
# base images (incl. Vast's) don't ship it.
if ! command -v rsync >/dev/null 2>&1; then
  echo "[startup] Installing rsync..."
  apt-get update -qq && apt-get install -y rsync
fi

# 0b. Durable labels dir on the Vast Volume.
#
# The dataset is the one piece of state that represents unrecoverable human
# work. Container disk is destroyed when the instance is destroyed. A Vast
# Volume (created in Console → Volumes, attached at instance-rent time)
# survives instance destroy, so we symlink data/labels → $LABELS_VOLUME and
# backend/config.py keeps using the same LABELS_DIR path.
#
# CAVEAT vs RunPod: a Vast Volume is tied to ONE physical machine and only
# reattaches to instances on that same host. If that GPU is unavailable when
# you come back, the labels are safe but unreachable until it frees up. So the
# laptop mirror (scripts/backup_dataset.sh) is the real durability guarantee —
# back up before you destroy an instance.
#
# Set LABELS_VOLUME_SKIP=1 to bypass (only for first-time instances where you
# haven't attached a volume yet — labels will be on ephemeral container disk).
LABELS_VOLUME="${LABELS_VOLUME:-/workspace/labels}"

if [ "${LABELS_VOLUME_SKIP:-0}" = "1" ]; then
  echo "[startup] WARNING: LABELS_VOLUME_SKIP=1 — labels will live on ephemeral container disk."
  echo "[startup] Destroying this instance will WIPE labeling work. Attach a Vast Volume ASAP,"
  echo "[startup] and run scripts/backup_dataset.sh from the laptop before you destroy it."
elif [ -L data/labels ] && [ -d data/labels ]; then
  resolved=$(readlink -f data/labels)
  echo "[startup] data/labels → $resolved (durable volume OK)"
elif [ ! -d "$LABELS_VOLUME" ]; then
  cat >&2 <<EOF
[startup] FATAL: LABELS_VOLUME=$LABELS_VOLUME does not exist.

The labels dir needs a durable Vast Volume mounted at that path, otherwise
destroying this instance will wipe all labeling work.

Fix (in the Vast.ai console):
  1. Console → Volumes → create a Volume (if you don't have one) on the host
     you intend to rent GPUs from.
  2. When renting the instance, attach the Volume with mount path = $LABELS_VOLUME.
  3. Re-run this script.

Reminder: a Vast Volume is pinned to one physical machine. Keep the laptop
mirror current with scripts/backup_dataset.sh as the cross-host safety net.

Or, if you intentionally want labels on ephemeral container disk (first-time
instance, no volume yet), re-run with:
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

# 0c. HuggingFace cache on the volume.
#
# By default, HF libraries write auth token and downloaded models to
# ~/.cache/huggingface/, which is on container disk — wiped on instance destroy.
# Putting HF_HOME on the volume means: across pod redeploys, the ~3 GB
# SAM 3 download sticks around AND the auth token survives, so
# `hf auth login` is a one-time cost per volume, not per pod.
#
# The variable is exported so child processes (uv run, uvicorn) inherit it.
HF_HOME="${HF_HOME:-/workspace/.hf-cache}"
export HF_HOME
mkdir -p "$HF_HOME"

# Migrate any NEW container-disk HF state onto the volume every run.
# Rationale: users sometimes `hf auth login` in an interactive shell
# (without HF_HOME set), which writes ~/.cache/huggingface/token to
# container disk. On next boot, without this sync, start_pod_server.sh
# wouldn't see the token in HF_HOME and would warn "not logged in" even
# though the user *did* log in. rsync is idempotent: nothing to do if
# the files are already in sync.
if [ -d "$HOME/.cache/huggingface" ] && [ -n "$(ls -A "$HOME/.cache/huggingface" 2>/dev/null)" ]; then
  rsync -a "$HOME/.cache/huggingface/" "$HF_HOME/"
fi
echo "[startup] HF_HOME=$HF_HOME (on volume)"

# 1. Install uv if missing (fresh pods sometimes lose it after redeploy)
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
echo "[startup] Open http://<public-ip>:<external-port> (Vast IP Port Info → internal 8000) in your laptop browser."
echo ""

exec uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
