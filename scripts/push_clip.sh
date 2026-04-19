#!/usr/bin/env bash
# Laptop-side: scp a video clip to the pod, register it with the server,
# and print the dashboard URL you open in your browser to start labeling.
#
# Skips the HTTP/2 upload limit in RunPod's proxy by uploading over SSH.
#
# Usage (with .env.pod populated):
#   ./scripts/push_clip.sh /path/to/clip.mov
#
# Usage (override connection):
#   ./scripts/push_clip.sh /path/to/clip.mov 38.65.239.23 27921

set -e

cd "$(dirname "$0")/.."

# Load connection config if present
if [ -f .env.pod ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.pod
  set +a
fi

VIDEO="$1"
POD_IP="${2:-$POD_IP}"
POD_SSH_PORT="${3:-$POD_SSH_PORT}"
POD_HTTP_URL="${POD_HTTP_URL:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [ -z "$VIDEO" ]; then
  cat >&2 <<EOF
Usage: $0 <video-path> [pod-ip] [pod-ssh-port]

Example:
  $0 ~/Downloads/IMG_1234.MOV

Needs pod connection info. Either set in .env.pod or pass as arg 2/3.
EOF
  exit 1
fi

if [ ! -f "$VIDEO" ]; then
  echo "File not found: $VIDEO" >&2
  exit 1
fi

if [ -z "$POD_IP" ] || [ -z "$POD_SSH_PORT" ]; then
  echo "Missing POD_IP / POD_SSH_PORT. See .env.pod.example." >&2
  exit 1
fi

# Use the clip's basename as the remote name so sequential pushes don't
# collide on the pod's /tmp.
REMOTE_NAME="$(basename "$VIDEO")"
REMOTE_PATH="/tmp/${REMOTE_NAME}"

echo "[push] Uploading $VIDEO → root@$POD_IP:$REMOTE_PATH"
scp -q -P "$POD_SSH_PORT" -i "$SSH_KEY" "$VIDEO" "root@$POD_IP:$REMOTE_PATH"

echo "[push] Registering with server..."
RESPONSE=$(ssh -p "$POD_SSH_PORT" -i "$SSH_KEY" "root@$POD_IP" \
  "curl -s -X POST 'http://localhost:8000/api/import_video?path=${REMOTE_PATH}'")

echo "[push] Server response:"
echo "$RESPONSE"

# Extract video_id for the dashboard URL. Falls back to sed if python isn't
# on the laptop's PATH.
VIDEO_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('video_id',''))" 2>/dev/null \
  || echo "$RESPONSE" | sed -E 's/.*"video_id":"([^"]+)".*/\1/')

echo ""
echo "[push] Video ID: $VIDEO_ID"
if [ -n "$POD_HTTP_URL" ]; then
  echo "[push] Open in browser:"
  echo "         ${POD_HTTP_URL%/}/?video_id=$VIDEO_ID"
else
  echo "[push] Open in browser (fill in your pod's HTTP proxy URL):"
  echo "         https://<pod-id>-8000.proxy.runpod.net/?video_id=$VIDEO_ID"
fi
