#!/usr/bin/env bash
# Laptop-side: open an SSH session to the RunPod pod.
#
# Two usages:
#
#   1. Source .env.pod (pod IP / SSH port stored there):
#        ./scripts/pod_ssh.sh
#
#   2. Override on the CLI (useful after pod restart gives new IP/port):
#        ./scripts/pod_ssh.sh 38.65.239.23 27921

set -e

cd "$(dirname "$0")/.."

# Load connection config if present (ignores line-by-line errors)
if [ -f .env.pod ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.pod
  set +a
fi

# CLI args override env vars
POD_IP="${1:-$POD_IP}"
POD_SSH_PORT="${2:-$POD_SSH_PORT}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [ -z "$POD_IP" ] || [ -z "$POD_SSH_PORT" ]; then
  cat >&2 <<EOF
Missing pod connection info.

Either put them in .env.pod (see .env.pod.example):
  POD_IP=38.65.239.23
  POD_SSH_PORT=27921

Or pass on the command line:
  $0 <pod-ip> <pod-ssh-port>

Get the current values from RunPod console → your pod → Connect → SSH over exposed TCP.
EOF
  exit 1
fi

exec ssh -p "$POD_SSH_PORT" -i "$SSH_KEY" "root@$POD_IP"
