#!/usr/bin/env bash
# Laptop-side: upload one or more .mov clips to the pod, register each with
# the labeling server, and print the dashboard URL for each clip.
#
# Uploads over SSH rather than HTTP, so it doesn't matter whether the host's
# port mapping (RunPod proxy before, direct Vast port now) chokes on large
# multipart uploads.
#
# Usage (with .env.pod populated):
#   ./scripts/push_clip.sh ~/Downloads/clip.mov
#   ./scripts/push_clip.sh ~/Downloads/clip1.mov ~/Downloads/clip2.mov
#   ./scripts/push_clip.sh ~/Downloads/sheep-*.mov
#
# Dedupe by content hash:
#   Every successful push is logged to .pushed_clips.tsv (gitignored, in the
#   project root), keyed by sha256 + pod IP. Re-pushing the same file to the
#   same pod is a no-op — the script prints the existing video_id and dashboard
#   URL instead of re-running SAM on the pod (~3 min per clip wasted).
#
#   The new labeling UI lets you select multiple subjects within one
#   video_id, so "one video_id per clip" is the intended workflow. Once
#   you've tapped everything useful out of a clip, move on to the next.
#
# Escape hatches:
#   PUSH_FORCE=1 ./scripts/push_clip.sh clip.mov       # re-push even if logged
#   rm .pushed_clips.tsv                               # wipe all dedupe state
#
# Overrides (use env vars, not positional args):
#   POD_IP=... POD_SSH_PORT=... ./scripts/push_clip.sh clip.mov

set -e

cd "$(dirname "$0")/.."

# Load connection config if present
if [ -f .env.pod ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.pod
  set +a
fi

POD_IP="${POD_IP:-}"
POD_SSH_PORT="${POD_SSH_PORT:-}"
POD_HTTP_URL="${POD_HTTP_URL:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
LOG_FILE="${PUSH_LOG_FILE:-.pushed_clips.tsv}"

if [ "$#" -lt 1 ]; then
  cat >&2 <<EOF
Usage: $0 <video-path> [<video-path> ...]

Examples:
  $0 ~/Downloads/IMG_1234.MOV
  $0 ~/Downloads/sheep-*.mov

Needs pod connection info in .env.pod (POD_IP, POD_SSH_PORT).
EOF
  exit 1
fi

if [ -z "$POD_IP" ] || [ -z "$POD_SSH_PORT" ]; then
  echo "Missing POD_IP / POD_SSH_PORT. See .env.pod.example." >&2
  exit 1
fi

# Ensure log file exists with header
if [ ! -f "$LOG_FILE" ]; then
  printf "# sha256\tpod_ip\tvideo_id\tbasename\tpushed_at\n" > "$LOG_FILE"
fi

pushed=0
skipped=0
failed=0

for VIDEO in "$@"; do
  echo ""
  echo "=== $VIDEO ==="

  if [ ! -f "$VIDEO" ]; then
    echo "[push] SKIP: file not found" >&2
    failed=$((failed + 1))
    continue
  fi

  HASH=$(sha256sum "$VIDEO" | awk '{print $1}')
  REMOTE_NAME="$(basename "$VIDEO")"
  REMOTE_PATH="/tmp/${REMOTE_NAME}"

  # Dedupe: same content + same pod = already pushed
  if [ "${PUSH_FORCE:-0}" != "1" ]; then
    EXISTING=$(awk -F'\t' -v h="$HASH" -v p="$POD_IP" '$1==h && $2==p {print; exit}' "$LOG_FILE" 2>/dev/null || true)
    if [ -n "$EXISTING" ]; then
      EXISTING_VID=$(echo "$EXISTING" | awk -F'\t' '{print $3}')
      EXISTING_AT=$(echo "$EXISTING" | awk -F'\t' '{print $5}')
      echo "[push] Already pushed ($EXISTING_AT) — video_id: $EXISTING_VID"
      if [ -n "$POD_HTTP_URL" ]; then
        echo "[push] ${POD_HTTP_URL%/}/?video_id=$EXISTING_VID"
      fi
      echo "[push] (Use PUSH_FORCE=1 to re-push.)"
      skipped=$((skipped + 1))
      continue
    fi
  fi

  echo "[push] Uploading → root@$POD_IP:$REMOTE_PATH"
  scp -q -P "$POD_SSH_PORT" -i "$SSH_KEY" "$VIDEO" "root@$POD_IP:$REMOTE_PATH"

  echo "[push] Registering with server..."
  RESPONSE=$(ssh -p "$POD_SSH_PORT" -i "$SSH_KEY" "root@$POD_IP" \
    "curl -s -X POST 'http://localhost:8000/api/import_video?path=${REMOTE_PATH}'")

  VIDEO_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('video_id',''))" 2>/dev/null \
    || echo "$RESPONSE" | sed -E 's/.*"video_id":"([^"]+)".*/\1/')

  if [ -z "$VIDEO_ID" ]; then
    echo "[push] FAIL: no video_id in server response:" >&2
    echo "$RESPONSE" >&2
    failed=$((failed + 1))
    continue
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$HASH" "$POD_IP" "$VIDEO_ID" "$REMOTE_NAME" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$LOG_FILE"

  echo "[push] video_id: $VIDEO_ID"
  if [ -n "$POD_HTTP_URL" ]; then
    echo "[push] ${POD_HTTP_URL%/}/?video_id=$VIDEO_ID"
  else
    echo "[push] set POD_HTTP_URL in .env.pod (http://<public-ip>:<external-port>) to print a clickable link · video_id=$VIDEO_ID"
  fi
  pushed=$((pushed + 1))
done

echo ""
echo "=== summary ==="
echo "  pushed:  $pushed"
echo "  skipped: $skipped (already on this pod)"
if [ "$failed" -gt 0 ]; then
  echo "  failed:  $failed"
  exit 1
fi
