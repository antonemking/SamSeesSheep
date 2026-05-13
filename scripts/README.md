# Scripts — training + pod orchestration

This directory owns the **training flywheel**: labeling (sheep-seg backend on the pod) →
training (sheep-seg on the pod's GPU) → weights file sync into the `sheep-yolo/`
subdir for local inference. As of 2026-05-12 sheep-seg and sheep-yolo live in
the same repo; this script set lives at the root because every step it touches
crosses the subdir boundary.

## Why these scripts live at the repo root (not inside sheep-yolo/)

1. **The dataset lives here.** The YOLO-pose dataset is produced by
   `/api/export/keypoints` (sheep-seg backend). Scripts that trigger exports
   or shell into the pod that runs the labeling server belong next to the
   labeling server.
2. **Training compute lives on the pod, which is a sheep-seg deployment.** The
   pod clones the repo to use its SAM 3 + labeling UI + `ultralytics`
   dependencies. The inference-only `sheep-yolo/` subdir is never installed on
   the pod. Scripts that SSH into the pod to kick off `yolo train` are
   sheep-seg-side concerns.
3. **The `sheep-yolo/` subdir's responsibility is inference + σ benchmarks.** It
   consumes a `best.pt` produced elsewhere. Anything upstream of "weights file
   exists locally" is somebody else's problem from sheep-yolo's perspective.
4. **Cross-cutting orchestration lives at the root.** These scripts coordinate
   the pod, the labeling backend, and the inference subdir — none of those
   alone owns the workflow.

## Code layout at a glance

| Location | Path | Responsibility |
|---|---|---|
| Pod (RunPod cloud GPU — 4090 / L40S / H100) | repo root | SAM 3 pipeline, labeling UI, dataset export, `yolo train` |
| Laptop | `scripts/` (repo root) | Scripts that SSH the pod: trigger training, pull weights |
| Laptop | `sheep-yolo/` subdir | Inference pipeline, σ benchmark, demo UI — consumes `best.pt` |

Nothing ever runs training *or* labeling locally. The 6 GB GTX 1660 Ti on the
laptop is for inference and visualization only. The cloud GPU on the pod does
every compute-heavy thing.

## The scripts

| Script | Runs on | Purpose |
|---|---|---|
| `pod_ssh.sh` | laptop | Generic SSH into the pod (handy for manual poking). |
| `start_pod_server.sh` | **pod** | Boots the labeling server on the pod after a pod resume. |
| `push_clip.sh` | laptop | Upload a fresh video clip to the pod's `data/uploads/`. |
| `train_on_pod.sh` | laptop | Trigger full training run on the pod's GPU. Synchronous — logs stream to your terminal until training finishes (~10 min at current dataset scale, faster on H100). |
| `sync_weights_from_pod.sh` | laptop | After training, pull `best.pt` + `last.pt` into `sheep-yolo/weights/` (in-repo). |
| `fetch_dataset.sh` | laptop | **Optional.** Pulls the dataset images to the laptop for eyeballing labels. Not part of the training loop — dataset never leaves the pod in normal use. |
| `backup_dataset.sh` | laptop | **Durability layer 2.** rsync mirror of the pod's `data/labels/` tree to `~/Backups/sheep-seg/labels/`. Covers in-progress JSON state, not just YOLO exports. Run manually (weekly is plenty); cron example in the script's docstring. |
| `validate_dataset.py` | pod or laptop | Dataset health checks — paired image/label counts, 20-column label format, `[0,1]` coord ranges, per-slot v=2 coverage, train/val leakage. Run as a pre-flight inside `train_on_pod.sh`; can also be run standalone any time. Exit 1 on any critical error so bad data never silently feeds into training. |

## End-of-day loop

You (labeler):

```
# Finish reviewing frames in the labeling UI, then let the YOLO side know.
# "Dataset ready, N frames on v0.X."
```

Me / YOLO side, from laptop (`~/dev/lorewood-advisors/sheep-seg`):

```
# 1. Kick off training on the pod's GPU. Synchronous: stays attached.
./scripts/train_on_pod.sh

# 2. When that finishes, pull the trained weights into sheep-yolo.
./scripts/sync_weights_from_pod.sh
```

Then from the sheep-yolo subdir:

```
cd sheep-yolo
export YOLOE_MODEL=weights/sheep-pose-v0.X.pt
# run σ-on-motionless-sheep benchmark, post results to docs/v0.X-benchmark.md
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

## Durability — how the dataset stays alive

The dataset is the one piece of state that represents unrecoverable human
work. Two layers protect it:

1. **RunPod Network Volume (on the pod).** `data/labels/` is a symlink to a
   volume mounted at `LABELS_VOLUME` (default `/mnt/labels`). Network Volumes
   survive Stop/Resume *and* Terminate and spot preemption — the container
   disk doesn't. `start_pod_server.sh` sets up the symlink on first run and
   refuses to boot if the expected mount path isn't there (so labels can't
   silently land on ephemeral disk). Volume is attached via the RunPod UI at
   pod-deploy time — see `docs/CLOUD.md` for the setup steps.
2. **Laptop rsync mirror (off the pod).** `backup_dataset.sh` mirrors the
   pod's full `data/labels/` tree to `~/Backups/sheep-seg/labels/` via rsync
   `--delete-after`. Covers the RunPod outage / volume-delete / billing-lapse
   scenarios the volume alone can't. Run weekly by hand; docstring has a
   cron snippet if you want it automated.

If you terminate a pod without a volume attached (or with
`LABELS_VOLUME_SKIP=1`), assume labels are gone. The whole point of these
scripts is so you never have to think about that outcome.

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
  Every training path runs on the pod's cloud GPU. The laptop is for inference
  and benchmarking only.
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
cd sheep-yolo
source .venv/bin/activate
export YOLOE_MODEL=weights/sheep-pose-v0.X.pt
python -m backend.main
# UI at http://localhost:8001 — upload a video, parts mode now runs the
# trained sheep-specific keypoints model instead of YOLOE open-vocab.
```

The inference code in sheep-yolo doesn't know or care that the weights came
from the pod — it just loads a `.pt` file. That's the whole point of the
split.
