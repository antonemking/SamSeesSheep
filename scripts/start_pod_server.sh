#!/usr/bin/env bash
# Run on the RunPod pod to bring the sheep-seg server up.
#
# Handles the pain points from the first setup day:
#   - Ensures uv is installed and on PATH
#   - Ensures venv exists and is synced
#   - Sets PYTORCH_CUDA_ALLOC_CONF + SHEEPSEG_FULL_PIPELINE
#   - Starts uvicorn in the foreground so the shell shows logs
#
# Idempotent — safe to re-run.

set -e

cd "$(dirname "$0")/.."

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
echo "[startup] Open the pod's HTTP Service URL (port 8000) in your laptop browser."
echo ""

exec uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
