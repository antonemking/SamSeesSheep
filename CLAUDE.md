# CLAUDE.md

Guidance for agents working in this repo. The human-facing details for the
training/labeling flywheel live in `scripts/README.md` — read it once. This
file is for things you actually trip over.

## Mental model

The training loop spans three places:

| Where | What it owns |
|---|---|
| **Laptop** (this repo) | Orchestration scripts, the `sheep-yolo/` subdir (inference + σ benchmark), the off-pod backup mirror. |
| **Pod** (RunPod 4090 / L40S / H100) | Labeling backend (SAM 3 + FastAPI UI), the dataset on a Network Volume, `yolo train` runs. |
| **Network Volume** (mounted on the pod at `/workspace`) | Durable storage for labels, model cache, train runs. Survives Stop/Resume **and** Terminate. |

Pod-side repo lives at `/workspace/SamSeesSheep` (not `sheep-seg`). Labels at
`/workspace/SamSeesSheep/data/labels/` are a symlink to `/workspace/labels`.

## Two env files (easy to confuse)

| File | Used by | Purpose |
|---|---|---|
| `<repo>/.env.pod` | `push_clip.sh`, `pod_ssh.sh`, `backup_dataset.sh` | Pod IP/SSH port/HTTP URL for the labeling-side scripts. |
| `~/.sheep-yolo.env` | `train_on_pod.sh`, `fetch_dataset.sh`, `sync_weights_from_pod.sh` | Pod IP/SSH port + `DATASET=` for the training-side scripts. |

When the user redeploys/restarts the pod the SSH port changes — **both files**
need updating before scripts work again. The user usually edits `.env.pod`
themselves; ask before editing.

`.env.pod` on the pod is a different file (lives at
`/workspace/SamSeesSheep/.env.pod`) and sets pod-side things like
`LABELS_VOLUME=/workspace/labels` and `HF_HOME=/workspace/.hf-cache`.

## The daily flow

```
push_clip.sh <video.MOV>     →  upload + register, get video_id + dashboard URL
   ↓
(user labels in browser, SAM 3 propagation runs on the pod GPU)
   ↓
backup_dataset.sh             →  rsync labels to ~/Backups/sheep-seg/labels/
   ↓
train_on_pod.sh               →  batch-export + validate + yolo train (synchronous)
   ↓
sync_weights_from_pod.sh      →  pull best.pt → sheep-yolo/weights/<dataset>.pt
```

Only weights cross the wire. The dataset stays on the pod (off-pod copy is the
laptop's `~/Backups` mirror, not the working dataset).

## Booting the labeling server from scratch

The labeling server is **not** auto-started on pod boot. After a pod
Stop/Resume:

1. Update `.env.pod` (new SSH port).
2. SSH in, `cd /workspace/SamSeesSheep`, `git pull` if needed.
3. Run `bash scripts/start_pod_server.sh`. The script symlinks `data/labels`
   to the volume, installs `rsync`, `uv sync`s, checks HF auth, then
   `exec`s uvicorn on port 8000.
4. The script refuses to start if `/workspace/labels` (or whatever
   `LABELS_VOLUME` is) doesn't exist — that's the durability guardrail.

To run it detached over SSH:
```
ssh ...pod... 'cd /workspace/SamSeesSheep && nohup bash scripts/start_pod_server.sh > /workspace/server.log 2>&1 < /dev/null & disown'
```
Then tail `/workspace/server.log` for the `Uvicorn running` line before pushing
clips. First boot of a new pod takes ~2–3 min (dep install + SAM 3 model
download if HF cache isn't already on the volume).

## Gotchas you'll hit

- **`pkill -f "uvicorn backend.main"` kills your own SSH session.** `pkill -f`
  matches against the full command line, and your SSH command contains the
  literal pattern. Use `fuser -k 8000/tcp` or grab the PID via `ss -tlnp` first.
- **"Unexpected token '<', `<!DOCTYPE`... is not valid JSON" in the UI.** The
  RunPod HTTP proxy in front of port 8000 has a ~60–100 s request timeout. SAM
  3 propagation on the full head+ear+nose pipeline runs ~90 s and often
  exceeds it. The backend succeeds and writes `review.json` either way — the
  user can just refresh the page. Not a real error.
- **Pod branch drift.** `/workspace/SamSeesSheep` on the pod can be on a
  feature branch (e.g. `simplify`) while local is on `main`. Check
  `git rev-parse --abbrev-ref HEAD` on the pod before assuming code parity.
- **SAM 3 isn't serialized.** A single global `_video_model` singleton in
  `backend/pipeline/video.py:26` with no locks. Two concurrent analyze calls
  on the same pod will likely OOM even a 4090. Pure label edits (clicking
  keypoints, save) are fine in parallel — those just touch JSON files.
- **Push dedupe.** `push_clip.sh` keys on `sha256 + pod_ip` in
  `.pushed_clips.tsv`. Re-pushing the same file to the same pod is a no-op
  (saves ~3 min of SAM re-import). `PUSH_FORCE=1` to override.

## Label-count one-liner

To see reviewed-instance counts across the local backup:

```python
import json
from pathlib import Path
b = Path.home() / "Backups/sheep-seg/labels"
for vd in sorted(p for p in b.iterdir() if p.is_dir() and p.name != "exports"):
    rj = vd / "review.json"
    if not rj.exists(): continue
    d = json.loads(rj.read_text())
    inst = reviewed = 0
    for f in d.get("frames", []):
        for ins in f.get("instances", []):
            inst += 1
            if any(k.get("v") == 2 for k in ins.get("keypoints", [])):
                reviewed += 1
    print(f"{vd.name}: {reviewed}/{inst} reviewed")
```

Keypoint `v` semantics: **2 = human-reviewed, 1 = auto/pending review,
0 = absent from frame.** Per the user's working assumption, an instance with
*any* `v=2` keypoint counts as reviewed; missing keypoints (`v=0`) on a
reviewed instance usually mean the part isn't in frame, not that work is
incomplete.

## When in doubt

- Read `scripts/README.md` first — the durability story, dataset versioning,
  and the "what these scripts deliberately do NOT do" section live there.
- For pod-side specifics (volume mount, HF cache, SAM 3 install), read
  `docs/CLOUD.md`.
- The user has explicit feedback memory about scope — don't refactor or add
  abstractions during a labeling/training session. Stay minimal.
