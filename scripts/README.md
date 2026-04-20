# Scripts — training + pod orchestration

This directory owns the **training flywheel**: labeling (sheep-seg on the pod) →
training (sheep-seg on the pod's 4090) → weights file sync to sheep-yolo (local,
for inference). It is the deliberate consolidation of what used to be scattered
across two repos.

## Why these scripts live in sheep-seg (not sheep-yolo)

Earlier drafts put the sync scripts in sheep-yolo. We pulled them into sheep-seg
because:

1. **The dataset lives here.** The YOLO-pose dataset is produced by
   `/api/export/keypoints` in this repo. Scripts that trigger exports or shell
   into the pod that runs the labeling server belong next to the labeling
   server.
2. **Training compute lives on the pod, which is a sheep-seg deployment.** The
   pod clones sheep-seg (SAM 3 + labeling UI + `ultralytics` dep). It does not
   clone sheep-yolo. Scripts that SSH into the pod to kick off `yolo train` are
   sheep-seg concerns.
3. **sheep-yolo's only responsibility is inference + σ benchmarks.** It
   consumes a `best.pt` produced elsewhere. Anything upstream of "weights file
   exists locally" is somebody else's problem from sheep-yolo's perspective.
4. **Single source of truth.** One repo owns labeling *and* training — the two
   steps that produce the artifact sheep-yolo consumes. Avoids two repos
   racing to update the same sync conventions.

## Repo split at a glance

| Location | Repo | Responsibility |
|---|---|---|
| Pod (RunPod 4090) | sheep-seg | SAM 3 pipeline, labeling UI, dataset export, `yolo train` |
| Laptop | sheep-seg | Scripts that SSH the pod: trigger training, pull weights |
| Laptop | sheep-yolo | Inference pipeline, σ benchmark, demo UI — consumes `best.pt` |

Nothing ever runs training *or* labeling locally. The 6 GB GTX 1660 Ti on the
laptop is for inference and visualization only. The 4090 on the pod does every
compute-heavy thing.

## The scripts

| Script | Runs on | Purpose |
|---|---|---|
| `pod_ssh.sh` | laptop | Generic SSH into the pod (handy for manual poking). |
| `start_pod_server.sh` | **pod** | Boots the labeling server on the pod after a pod resume. |
| `push_clip.sh` | laptop | Upload a fresh video clip to the pod's `data/uploads/`. |
| `train_on_pod.sh` | laptop | Trigger full training run on the 4090. Synchronous — logs stream to your terminal until training finishes (~10 min at current dataset scale). |
| `sync_weights_from_pod.sh` | laptop | After training, pull `best.pt` + `last.pt` into `~/dev/lorewood-advisors/sheep-yolo/weights/`. |
| `fetch_dataset.sh` | laptop | **Optional.** Pulls the dataset images to the laptop for eyeballing labels. Not part of the training loop — dataset never leaves the pod in normal use. |
| `validate_dataset.py` | pod or laptop | Dataset health checks — paired image/label counts, 20-column label format, `[0,1]` coord ranges, per-slot v=2 coverage, train/val leakage. Run as a pre-flight inside `train_on_pod.sh`; can also be run standalone any time. Exit 1 on any critical error so bad data never silently feeds into training. |

## End-of-day loop

You (labeler):

```
# Finish reviewing frames in the labeling UI, then let the YOLO side know.
# "Dataset ready, N frames on v0.X."
```

Me / YOLO side, from laptop (`~/dev/lorewood-advisors/sheep-seg`):

```
# 1. Kick off training on the pod's 4090. Synchronous: stays attached.
./scripts/train_on_pod.sh

# 2. When that finishes, pull the trained weights into sheep-yolo.
./scripts/sync_weights_from_pod.sh
```

Then from sheep-yolo:

```
export YOLOE_MODEL=~/dev/lorewood-advisors/sheep-yolo/weights/sheep-pose-v0.X.pt
# run σ-on-motionless-sheep benchmark, post results to sheep-seg-conversation/LOG.md
```

Only the ~10 MB weights file crosses the network. The hundreds of MB of
labeled frames stay on the pod.

## Configuration

All laptop-side scripts load `~/.sheep-yolo.env`:

```
POD_IP=38.65.239.23
POD_SSH_PORT=27921
```

Update this file whenever the pod restarts (RunPod assigns a new IP/port on
resume). All scripts that need SSH info read from this one place — no editing
individual scripts on pod restart.

Pod-side `start_pod_server.sh` uses a different file, `.env.pod`, for things
like `HF_HOME` and CUDA config. Only `.env.pod.example` is committed to the
repo (as a template); `.env.pod` itself is gitignored — copy the example to
`.env.pod` and fill it in locally on the pod. Keep the two env files
separate: `.env.pod` is portable across pods (lives on the pod's disk);
`~/.sheep-yolo.env` is laptop-local and tracks the pod instance you're
pointing at today.

## What happens after a pod restart (RunPod → Stop then Resume)

1. RunPod console → your pod → Resume. New public IP and SSH port.
2. Update `~/.sheep-yolo.env` on the laptop with the new values.
3. `./scripts/pod_ssh.sh` → you're in. `cd /workspace/SamSeesSheep && git pull`.
4. `bash scripts/start_pod_server.sh` → labeling server back up.
5. Resume labeling.

Disk on the pod is persistent across Stop→Resume, so all prior labels, cached
model weights, and run artifacts survive. Only the network identity changes.

## Dataset versioning

`DATASET` is the single knob. Default is `sheep-pose-v0.1`. When you bump:

- Exporter (`/api/export/keypoints`) writes to
  `data/labels/exports/sheep-pose-v0.X/`.
- `train_on_pod.sh` runs the `uv run yolo train` against that folder and
  writes weights to `/workspace/runs/pose/sheep-pose-v0.X.run/`.
- `sync_weights_from_pod.sh` writes the weights as
  `sheep-yolo/weights/sheep-pose-v0.X.pt`.

The dataset tag flows through every downstream filename, so you can trivially
A/B-compare v0.1 vs v0.2 σ benchmarks: both weight files live side-by-side in
the sheep-yolo weights dir, and `YOLOE_MODEL=...` picks which one loads.

## What these scripts deliberately do NOT do

- **Train locally.** The laptop's 6 GB VRAM can't fit `batch=8 imgsz=640`.
  Every training path runs on the pod's 4090. The laptop is for inference and
  benchmarking only.
- **Sync the dataset to the laptop as part of the training loop.** That was
  the original plan and we explicitly undid it. The dataset lives and dies on
  the pod. `fetch_dataset.sh` exists for the narrow case where you want to
  scroll JPGs locally, not as part of any normal path.
- **Auto-trigger inference or σ benchmarks.** After weights land locally, the
  benchmark run is a human decision (which motionless-sheep clip? which
  baseline to compare against?). Scripts don't automate that.

## Minimal troubleshooting

| Symptom | Likely cause |
|---|---|
| `Missing pod SSH info` | `~/.sheep-yolo.env` doesn't exist or has stale IP/port after a pod resume. |
| `No best.pt found on pod` | `train_on_pod.sh` hasn't completed yet, or `DATASET` in your env doesn't match what was trained. |
| Export step fails with 404 | Pod's labeling server isn't running. `start_pod_server.sh` on the pod. |
| SSH times out | Pod is stopped, or RunPod assigned a new IP. Check the console, update `~/.sheep-yolo.env`. |

## What sheep-yolo does with the weights

Once `sheep-yolo/weights/sheep-pose-v0.X.pt` exists locally:

```bash
cd ~/dev/lorewood-advisors/sheep-yolo
source .venv/bin/activate
export YOLOE_MODEL=weights/sheep-pose-v0.X.pt
python -m backend.main
# UI at http://localhost:8001 — upload a video, parts mode now runs the
# trained sheep-specific keypoints model instead of YOLOE open-vocab.
```

The inference code in sheep-yolo doesn't know or care that the weights came
from the pod — it just loads a `.pt` file. That's the whole point of the
split.
