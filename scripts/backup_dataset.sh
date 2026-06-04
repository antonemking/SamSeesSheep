#!/usr/bin/env bash
# Laptop-side: rsync the pod's entire data/labels/ tree to a local backup dir.
#
# Why this exists: the pod's labels live on a Vast Volume, which survives
# instance destroy — but ONLY reattaches on the same physical machine, and is
# still a single point of failure (host unavailable, accidental volume delete,
# billing lapse). On Vast this off-pod mirror is the PRIMARY durability layer,
# not just a backup: run it before you destroy an instance.
#
# Covers the WHOLE labels tree, not just data/labels/exports/. That means
# in-progress per-video JSON state (segmentations, keypoint reviews, etc.)
# gets backed up too, not only finalized YOLO exports.
#
# Usage:
#   ./scripts/backup_dataset.sh                        # uses .env.pod defaults
#   ./scripts/backup_dataset.sh 38.65.239.23 27921     # override connection
#   BACKUP_DIR=/path/to/dst ./scripts/backup_dataset.sh
#
# Config (via .env.pod in project root — same as push_clip.sh / pod_ssh.sh):
#   POD_IP=38.65.239.23
#   POD_SSH_PORT=27921
#   SSH_KEY=~/.ssh/id_ed25519                          # defaults to this
#   BACKUP_DIR=~/Backups/sheep-seg/labels              # defaults to this
#
# Not automated — run manually. Weekly is usually plenty. If you want cron,
# add something like this to your user crontab (laptop must be awake):
#
#   0 18 * * 0 cd ~/dev/lorewood-advisors/sheep-seg && ./scripts/backup_dataset.sh >> ~/Backups/sheep-seg/backup.log 2>&1
#
# The --delete-after flag means the local mirror tracks the pod's state: a
# file removed on the pod will be removed from the local mirror on the next
# run. If you want append-only / versioned backups, wrap this in a per-run
# timestamped dir yourself; deliberate defaults here favor a clean mirror
# over archive-style history.

set -e

cd "$(dirname "$0")/.."

# Load connection config if present
if [ -f .env.pod ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.pod
  set +a
fi

POD_IP="${1:-$POD_IP}"
POD_SSH_PORT="${2:-$POD_SSH_PORT}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/Backups/sheep-seg/labels}"

if [ -z "$POD_IP" ] || [ -z "$POD_SSH_PORT" ]; then
  cat >&2 <<EOF
Missing POD_IP / POD_SSH_PORT.

Either put them in .env.pod (see .env.pod.example):
  POD_IP=38.65.239.23
  POD_SSH_PORT=27921

Or pass on the command line:
  $0 <pod-ip> <pod-ssh-port>
EOF
  exit 1
fi

mkdir -p "$BACKUP_DIR"

REMOTE_LABELS="/workspace/SamSeesSheep/data/labels/"

echo "[backup] Mirroring root@${POD_IP}:${REMOTE_LABELS}"
echo "[backup]        → ${BACKUP_DIR}/"
echo ""

rsync -avz --delete-after --human-readable \
  -e "ssh -p $POD_SSH_PORT -i $SSH_KEY" \
  "root@${POD_IP}:${REMOTE_LABELS}" \
  "${BACKUP_DIR%/}/"

echo ""
echo "[backup] Done."
echo "[backup] Local mirror size:  $(du -sh "${BACKUP_DIR%/}" 2>/dev/null | cut -f1)"
echo "[backup] Files in mirror:    $(find "${BACKUP_DIR%/}" -type f 2>/dev/null | wc -l)"
