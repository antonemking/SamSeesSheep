#!/usr/bin/env bash
# One-shot: bring a fresh Vast.ai instance up to where start_pod_server.sh can
# take over. Idempotent — safe to re-run after a transient SSH failure or to
# refresh the labels mirror.
#
# Does:
#   1. SSH-probes the instance to confirm reachability.
#   2. Clones (or git-pulls) SamSeesSheep into /workspace/SamSeesSheep.
#   3. Ensures /workspace/labels exists (the LABELS_VOLUME default for Vast).
#   4. rsync's labels from ~/Backups/sheep-seg/labels/ → pod /workspace/labels/.
#   5. Writes the pod-side .env.pod with LABELS_VOLUME + HF_HOME.
#   6. Prints the remaining manual steps (HF auth, start the server).
#
# Usage (reads pod target from .env.pod.vast by default):
#   ./scripts/bootstrap_vast.sh
#
# Or pass on CLI (after Vast restart gives a new host/port):
#   ./scripts/bootstrap_vast.sh ssh5.vast.ai 12345
#
# Overrides:
#   REPO_URL=...       (default: laptop's git remote origin URL)
#   LABELS_VOLUME=...  (default: /workspace/labels)
#   LOCAL_LABELS=...   (default: ~/Backups/sheep-seg/labels)
#   SEED_SKIP=1        skip the labels rsync (use when re-running after labels already seeded)

set -e

cd "$(dirname "$0")/.."

# Read Vast config (POD_IP/POD_SSH_PORT/SSH_KEY) regardless of which cloud is
# currently active. We deliberately source .env.pod.vast directly rather than
# whichever cloud is active, so bootstrapping a Vast box doesn't require
# flipping the switcher first.
if [ -f .env.pod.vast ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.pod.vast
  set +a
fi

POD_IP="${1:-$POD_IP}"
POD_SSH_PORT="${2:-$POD_SSH_PORT}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REPO_URL="${REPO_URL:-$(git config --get remote.origin.url 2>/dev/null || echo)}"
LABELS_VOLUME="${LABELS_VOLUME:-/workspace/labels}"
LOCAL_LABELS="${LOCAL_LABELS:-$HOME/Backups/sheep-seg/labels}"
REPO_DIR="/workspace/SamSeesSheep"

if [ -z "$POD_IP" ] || [ -z "$POD_SSH_PORT" ]; then
  cat >&2 <<EOF
Missing Vast SSH info.

Fill in .env.pod.vast with POD_IP / POD_SSH_PORT from the Vast console's
"SSH command" panel, or pass on the CLI:

  $0 <vast-host> <vast-ssh-port>
EOF
  exit 1
fi

if [ -z "$REPO_URL" ]; then
  echo "Couldn't determine git remote URL. Set REPO_URL=https://github.com/<user>/SamSeesSheep.git" >&2
  exit 1
fi

if [ ! -d "$LOCAL_LABELS" ]; then
  echo "Local labels mirror not found at $LOCAL_LABELS." >&2
  echo "Either run ./scripts/backup_dataset.sh against your old pod first, or set LOCAL_LABELS=..." >&2
  exit 1
fi

SSH_OPTS=(-p "$POD_SSH_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)

echo "[boot] Vast target:   root@${POD_IP}:${POD_SSH_PORT}"
echo "[boot] Repo URL:      ${REPO_URL}"
echo "[boot] Repo dir:      ${REPO_DIR}"
echo "[boot] LABELS_VOLUME: ${LABELS_VOLUME}"
echo "[boot] Local labels:  ${LOCAL_LABELS}"
echo ""

# 1. Probe SSH
echo "[boot] Probing SSH..."
ssh "${SSH_OPTS[@]}" "root@$POD_IP" "echo ok && uname -a" || {
  echo "[boot] SSH probe failed. Check POD_IP/POD_SSH_PORT and that your pubkey is on the Vast account." >&2
  exit 1
}

# 2. Clone or pull the repo
echo ""
echo "[boot] Cloning / updating ${REPO_DIR}..."
ssh "${SSH_OPTS[@]}" "root@$POD_IP" bash <<REMOTE
set -e
mkdir -p /workspace
if [ -d "$REPO_DIR/.git" ]; then
  cd "$REPO_DIR"
  git fetch --all --prune
  git pull --ff-only || echo "[boot-remote] (skipping pull — local pod branch may be ahead; check manually)"
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
REMOTE

# 3. Ensure LABELS_VOLUME exists on the pod
echo ""
echo "[boot] Ensuring ${LABELS_VOLUME} exists on pod..."
ssh "${SSH_OPTS[@]}" "root@$POD_IP" "mkdir -p '$LABELS_VOLUME' && mkdir -p /workspace/.hf-cache"

# 4. Seed labels from local backup mirror
if [ "${SEED_SKIP:-0}" = "1" ]; then
  echo ""
  echo "[boot] SEED_SKIP=1 — skipping labels rsync."
else
  echo ""
  echo "[boot] Seeding ${LABELS_VOLUME}/ from ${LOCAL_LABELS}/ ..."
  echo "[boot] (this can take a few minutes the first time; idempotent on re-run)"
  # No --delete here: we don't want to wipe pod-side state if the local mirror
  # is older than what's on the pod (e.g. if you've been labeling on Vast
  # already and forgot to back up before re-running this).
  rsync -avz --human-readable \
    -e "ssh -p $POD_SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=accept-new" \
    "${LOCAL_LABELS%/}/" \
    "root@${POD_IP}:${LABELS_VOLUME%/}/"
fi

# 5. Write pod-side .env.pod
echo ""
echo "[boot] Writing pod-side ${REPO_DIR}/.env.pod ..."
ssh "${SSH_OPTS[@]}" "root@$POD_IP" "cat > ${REPO_DIR}/.env.pod" <<EOF
# Pod-side env. Read by scripts/start_pod_server.sh on this Vast instance.
LABELS_VOLUME=${LABELS_VOLUME}
HF_HOME=/workspace/.hf-cache
EOF

# 6. Next steps
cat <<EOF

[boot] Done.

Remaining manual steps (one-time per fresh Vast instance):

  1. Point laptop scripts at Vast:
       ./scripts/use_cloud.sh vast

  2. SSH in and log into HuggingFace (needed for SAM 3 download):
       ./scripts/pod_ssh.sh
       cd ${REPO_DIR}
       uv run hf auth login    # paste your HF token
       exit

  3. Start the labeling server (detached, so the SSH session can close):
       ssh -p $POD_SSH_PORT -i $SSH_KEY root@$POD_IP \\
         'cd ${REPO_DIR} && nohup bash scripts/start_pod_server.sh > /workspace/server.log 2>&1 < /dev/null & disown'

  4. In a separate terminal, tunnel port 8000 to your laptop:
       ssh -L 8000:localhost:8000 -p $POD_SSH_PORT -i $SSH_KEY root@$POD_IP

  5. Open http://localhost:8000 in the browser. Push a clip:
       ./scripts/push_clip.sh ~/Downloads/some-clip.MOV
EOF
