# SETUP — SamSeesSheep Reproducibility Guide

Complete setup instructions for the labeling pipeline, cloud GPU training, and local edge inference. Assumes familiarity with the command line and Python tooling.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11** | 3.12 is the upper bound per `pyproject.toml` (`>=3.11,<3.13`). 3.11 recommended for best dependency compatibility. |
| **CUDA GPU** | 24 GB for full 3-session SAM 3 pipeline. 6 GB runs a reduced 2-session pipeline (head + ear only, nose keypoints fall back). Inference-only (sheep-yolo) works on 6 GB. |
| **Hugging Face token** | Required to download `facebook/sam3` (~3 GB). Request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3). |
| **uv** | Python package manager. Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`. |
| **git** | To clone the repo. |

Optional for cloud training:

| Requirement | Notes |
|---|---|
| **RunPod or Vast.ai account** | Cloud GPU pod for YOLO-pose training (local 6 GB cannot fit `batch=8 imgsz=640`). |
| **SSH key** | For pod access. |

---

## Local labeling setup

This runs SAM 3 labeling on your local GPU. For a 6 GB GPU, SAM 3 runs the reduced pipeline (2 sessions: head + ear). For a 24 GB GPU, the full 3-session pipeline (head + ear + nose) runs.

```bash
# Clone
git clone https://github.com/antonemking/SamSeesSheep.git
cd SamSeesSheep

# Install dependencies
uv sync

# Authenticate with Hugging Face (required once)
uv run hf auth login

# Start the labeling server
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser. Drop a sheep video (phone capture, 15–30 s, .MOV). Wait 1–3 minutes for SAM 3 to find every sheep via text prompts — no clicking required. Review each frame in the labeler, dragging keypoints onto every detected instance.

**Keyboard shortcuts:**
- `A` — accept SAM's auto-placed keypoint candidates
- `Enter` — save current frame and advance
- `S` — skip frame

**Data directory:** Reviewed annotations land in `data/labels/{video_id}/`. Each video gets `frames/` (extracted JPGs) and `review.json` (keypoint annotations in schema v2).

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | (from `hf auth login`) | Hugging Face access token |
| `HF_HOME` | `~/.cache/huggingface` | Model cache location |

### First-run behavior

First startup downloads SAM 3 (~3 GB) from Hugging Face to `HF_HOME`. Subsequent starts use the cached model. If the download fails, verify `uv run hf auth login` completed successfully and that your account has been granted access to `facebook/sam3`.

---

## Cloud GPU training

Training YOLO-pose requires a cloud GPU pod — local 6 GB VRAM cannot fit `batch=8 imgsz=640` for training. The workflow:

1. Deploy a pod on RunPod or Vast.ai
2. Clone the repo and start the labeling server on the pod
3. Push video clips from your laptop
4. Label in the pod's browser UI
5. Export the dataset and train on the pod GPU
6. Sync `best.pt` back to your laptop

### RunPod deployment

#### Create a Network Volume (durable labels storage)

In the RunPod console **before** deploying the pod:

1. Storage → **New Network Volume**. Size 10 GB.
2. Pick the same datacenter your pod will run in.
3. When creating/editing the pod, attach the volume at mount path `/mnt/labels`.

The Network Volume survives Stop/Resume, Terminate, and spot preemption. Container disk does not.

#### First pod boot

```bash
# On the pod (after first deploy)
cd /workspace
git clone https://github.com/antonemking/SamSeesSheep.git
cd SamSeesSheep
cp .env.pod.example .env.pod    # pod-side: LABELS_VOLUME, HF_HOME
# uv run hf auth login           # if the start script asks for it
bash scripts/start_pod_server.sh
```

`start_pod_server.sh` does:
1. Symlinks `data/labels` → `$LABELS_VOLUME` (refuses to boot if mount missing)
2. Installs `rsync` and runs `uv sync`
3. Checks HF auth
4. Starts uvicorn on port 8000

#### Laptop config

```bash
cp .env.pod.example .env.pod
# Edit .env.pod: fill in POD_IP, POD_SSH_PORT, POD_HTTP_URL
# from RunPod console → pod → Connect panel
```

For training-side scripts, also configure `~/.sheep-yolo.env` with the same `POD_IP` and `POD_SSH_PORT`.

#### Daily resume

```bash
# 1. Resume pod from RunPod console (~60 s)
# 2. Grab new SSH IP/port and HTTP URL from Connect panel
# 3. Update .env.pod and ~/.sheep-yolo.env on laptop

# 4. SSH in
./scripts/pod_ssh.sh

# 5. Start server (inside SSH)
cd /workspace/SamSeesSheep
git pull
bash scripts/start_pod_server.sh

# 6. Open POD_HTTP_URL in browser
```

#### Push video clips

```bash
# From laptop (second terminal)
./scripts/push_clip.sh ~/Downloads/clip.mov
```

Re-pushing the same clip is a no-op (deduped by sha256 + pod IP in `.pushed_clips.tsv`). Override: `PUSH_FORCE=1 ./scripts/push_clip.sh ...`.

#### Back up labels

```bash
# From laptop — mirrors pod data/labels/ to ~/Backups/sheep-seg/labels/
./scripts/backup_dataset.sh
```

Run weekly or after heavy labeling sessions.

#### Train

```bash
# From laptop, with pod server running
DATASET=sheep-pose-v0.4-yolo26n \
MODEL=yolo26n-pose.pt \
EPOCHS=100 \
IMGSZ=640 \
BATCH=8 \
./scripts/train_on_pod.sh
```

What happens:
1. POSTs `/api/export/keypoints/all?dataset=<DATASET>` to the pod
2. Validates the exported dataset
3. Runs `yolo train` on the pod GPU (~6 min on RTX 4090)
4. Writes `best.pt` to `/workspace/runs/pose/<DATASET>.run/weights/`

Use unique `DATASET` names for A/B runs so weights don't overwrite each other.

#### Sync weights back

```bash
./scripts/sync_weights_from_pod.sh sheep-pose-v0.4-yolo26n
```

Pulls `best.pt` (and `last.pt` if available) into `sheep-yolo/weights/`.

### Vast.ai deployment

Vast.ai instances use `scripts/bootstrap_vast.sh` for bring-up. After that, the workflow is identical — place the script's env vars in a `.env.pod.vast` file and symlink `.env.pod → .env.pod.vast`.

---

## Local inference setup

Inference runs locally on a 6 GB GPU using the trained YOLO-pose weights.

```bash
cd sheep-yolo
export YOLOE_MODEL=weights/sheep-pose-v0.4-yolo26n.pt
python -m backend.main
```

The sheep-yolo UI provides inference visualization and access to the σ benchmark.

### Run the σ benchmark

```bash
cd sheep-yolo
python scripts/bench_held_out.py IMG_3651
```

This measures per-keypoint residual σ and ear-angle σ on a held-out clip. The benchmark caches per-model predictions (pickle) so re-render iterations skip inference.

---

## Common troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `hf` command not found | PATH lost it | `source $HOME/.local/bin/env` then `uv run hf auth login` |
| SAM 3 download fails | HF auth expired or access not granted | `uv run hf auth login`; verify access to `facebook/sam3` |
| Analyze returns 500 | HF auth expired or CUDA missing | Check uvicorn logs for traceback; re-auth if needed |
| Upload fails with HTTP2 error | Browser upload drops large files | Use `scripts/push_clip.sh` instead of browser drop zone |
| Export returns 400 "No exportable frames" | No `v=2` keypoints reviewed yet | Review at least one frame at `/label/{video_id}` |
| `start_pod_server.sh` refuses to boot | Network Volume not mounted | Attach volume at `/mnt/labels` in RunPod console, Stop+Resume |
| Training crashes with empty val/images | Only one frame exported | Normal at small scale; the guarantee promotes one frame to val |
| `No best.pt found` after training | Training didn't finish or DATASET mismatch | Check pod logs; verify DATASET name consistency |
| `Connection refused` on SSH | Pod IP/port changed after restart | Update `.env.pod` from RunPod Connect panel |
| Pod resumed with new IP but scripts stale | Both env files need updating | Update `.env.pod` AND `~/.sheep-yolo.env` |
| SAM 3 OOM on concurrent calls | Global singleton, no locking | Don't trigger two `/api/analyze` calls simultaneously |
| GPU out of memory during yolo train | Batch size too large for pod GPU | Reduce BATCH (try 4 or 2) |
| Keypoint export skips frames | No reviewed instances, missing bbox, or missing source images | Check skipped counts in export response; review frames |

### Verification checklist

Confirm a working setup:

1. ✓ Labeling server starts: `uv run uvicorn backend.main:app` prints "Uvicorn running"
2. ✓ SAM 3 runs: drop a clip, wait for "SAM processing complete" in logs
3. ✓ Browser UI loads: `http://localhost:8000` shows the dashboard
4. ✓ Keypoints draggable: click a frame, drag a dot
5. ✓ Export works: `curl -X POST "http://localhost:8000/api/export/keypoints/all?dataset=test"` returns `"ok": true`
6. ✓ Inference runs: `cd sheep-yolo && python -m backend.main` loads model
7. ✓ Benchmark runs: `python scripts/bench_held_out.py IMG_3651` produces artifacts
