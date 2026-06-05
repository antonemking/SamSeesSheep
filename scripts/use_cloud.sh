#!/usr/bin/env bash
# Flip the active cloud by re-pointing .env.pod and sheep-yolo/sheep-yolo.env
# at their .runpod or .vast variant. Lets you keep both configs side by side
# and switch in one command without re-typing pod details.
#
# Usage:
#   ./scripts/use_cloud.sh runpod
#   ./scripts/use_cloud.sh vast
#   ./scripts/use_cloud.sh            # print current active cloud

set -e

cd "$(dirname "$0")/.."

current_target() {
  local link="$1"
  if [ -L "$link" ]; then
    readlink "$link"
  elif [ -f "$link" ]; then
    echo "(plain file, not a symlink)"
  else
    echo "(missing)"
  fi
}

show_status() {
  local pod_target pod_ip
  pod_target=$(current_target .env.pod)
  pod_ip=$(grep -E '^POD_IP=' .env.pod 2>/dev/null | head -1 | cut -d= -f2)
  echo "  .env.pod                  → $pod_target"
  echo "  sheep-yolo/sheep-yolo.env → $(current_target sheep-yolo/sheep-yolo.env)"
  if [ -n "$pod_ip" ]; then
    echo "  → active POD_IP: $pod_ip"
  fi
}

if [ "$#" -eq 0 ]; then
  echo "Current cloud:"
  show_status
  exit 0
fi

CLOUD="$1"
case "$CLOUD" in
  runpod|vast) ;;
  *)
    echo "Usage: $0 {runpod|vast}" >&2
    exit 1
    ;;
esac

POD_VARIANT=".env.pod.${CLOUD}"
YOLO_VARIANT="sheep-yolo.env.${CLOUD}"

if [ ! -f "$POD_VARIANT" ]; then
  echo "Missing $POD_VARIANT. Create it (see $POD_VARIANT.example or copy from the other cloud)." >&2
  exit 1
fi
if [ ! -f "sheep-yolo/$YOLO_VARIANT" ]; then
  echo "Missing sheep-yolo/$YOLO_VARIANT." >&2
  exit 1
fi

ln -sfn "$POD_VARIANT" .env.pod
ln -sfn "$YOLO_VARIANT" sheep-yolo/sheep-yolo.env

echo "Switched to: $CLOUD"
show_status
