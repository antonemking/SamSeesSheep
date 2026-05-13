# AGENTS

Operational notes for AI/code agents working in this repo. Keep this file
short, command-oriented, and in sync with `scripts/README.md`.

## Project Shape

This repo owns the sheep keypoint labeling and training flywheel:

1. Run SAM 3 + labeling UI on a RunPod GPU pod.
2. Review sheep-head keypoints in the browser.
3. Export reviewed `v=2` keypoints to YOLO-pose format.
4. Train YOLO-pose on the pod GPU.
5. Sync `best.pt` into `sheep-yolo/weights/` for local inference and sigma
   benchmarks.

The laptop is for orchestration, backups, and local inference. Do not plan
normal YOLO training on the laptop GPU.

## Important Paths

- Repo root on laptop: `/home/toneking/dev/lorewood-advisors/sheep-seg`
- Repo root on pod: `/workspace/SamSeesSheep`
- Pod labels source of truth: `/workspace/SamSeesSheep/data/labels/`
- Local labels backup: `~/Backups/sheep-seg/labels/`
- Trained pod runs: `/workspace/runs/pose/<DATASET>.run/weights/best.pt`
- Local synced weights: `sheep-yolo/weights/<DATASET>.pt`

`data/labels/` on the pod should be backed by the RunPod Network Volume
mounted at `LABELS_VOLUME` (default `/mnt/labels`). Treat labels as the most
valuable project state.

## Config Files

There are currently two connection config conventions:

- `.env.pod` in the repo root is read by:
  - `scripts/pod_ssh.sh`
  - `scripts/push_clip.sh`
  - `scripts/backup_dataset.sh`
- `~/.sheep-yolo.env` is read by:
  - `scripts/train_on_pod.sh`
  - `scripts/sync_weights_from_pod.sh`
  - `scripts/fetch_dataset.sh`

Both should contain current `POD_IP`, `POD_SSH_PORT`, and optionally `SSH_KEY`.
After a RunPod Stop/Resume, update these values from the RunPod console before
running laptop-side scripts.

Never commit secrets. `.env.pod` is gitignored; `.env.pod.example` is the
template.

## Start Or Resume The Pod Server

From laptop:

```bash
./scripts/pod_ssh.sh
```

On the pod:

```bash
cd /workspace/SamSeesSheep
git pull
bash scripts/start_pod_server.sh
```

The labeling/training export server must be running before `push_clip.sh`,
`fetch_dataset.sh`, or `train_on_pod.sh` can register/export data. If export
fails with HTTP `000`, `404`, or an empty response, first check whether the pod
server is running.

## Push Video Clips

From laptop:

```bash
./scripts/push_clip.sh /path/to/clip.mov
./scripts/push_clip.sh /path/to/clip1.mov /path/to/clip2.mov
```

The script uploads over SSH, registers the clip with the pod server, dedupes by
content hash in `.pushed_clips.tsv`, and prints the labeling URL. Use
`PUSH_FORCE=1` only when re-processing the same clip is intentional.

## Back Up Labels

From laptop:

```bash
./scripts/backup_dataset.sh
```

This mirrors the pod's full `data/labels/` tree to
`~/Backups/sheep-seg/labels/` using `rsync --delete-after`. It includes
in-progress `review.json` files and frames, not only exported YOLO datasets.

This is a mirror, not append-only history. Files deleted on the pod are deleted
from the mirror on the next backup.

## Count Reviewed Data

Useful quick checks against the local backup:

```bash
find "$HOME/Backups/sheep-seg/labels" -path '*/frames/*' -type f | wc -l
find "$HOME/Backups/sheep-seg/labels" -name review.json -type f | sort
```

Reviewed instance count and full/partial split:

```bash
jq -s 'reduce .[] as $r ({reviewed_instances:0,full_instances:0,partial_instances:0};
  .reviewed_instances += ([$r.frames[]? | (.instances // [])[]? |
    select((.keypoints // []) | any(.v==2))] | length) |
  .full_instances += ([$r.frames[]? | (.instances // [])[]? |
    select(((.keypoints // []) | length == 5) and ((.keypoints // []) | all(.v==2)))] | length) |
  .partial_instances += ([$r.frames[]? | (.instances // [])[]? |
    select(((.keypoints // []) | any(.v==2)) and
    ((((.keypoints // []) | length) != 5) or (((.keypoints // []) | any(.v!=2)))))] | length)
)' "$HOME"/Backups/sheep-seg/labels/*/review.json
```

The exporter trains only on `v=2` keypoints. `v=1` auto/pseudo labels do not
pollute normal training unless the export is explicitly called with
`pseudo=true`; do not use pseudo labels for sigma benchmarks.

## Train YOLO-Pose On The Pod

From laptop, with the pod server already running:

```bash
DATASET=sheep-pose-v0.3-yolo26n \
MODEL=yolo26n-pose.pt \
EPOCHS=100 \
IMGSZ=640 \
BATCH=8 \
./scripts/train_on_pod.sh
```

What happens:

1. POSTs to the pod server:
   `/api/export/keypoints/all?dataset=<DATASET>`
2. Validates the exported dataset on the pod.
3. Runs `uv run --project /workspace/SamSeesSheep yolo train ...`
4. Writes weights to:
   `/workspace/runs/pose/<DATASET>.run/weights/best.pt`

Use unique `DATASET` names for A/B runs so weights do not overwrite each other,
for example:

```bash
DATASET=sheep-pose-v0.3-yolo26n MODEL=yolo26n-pose.pt ./scripts/train_on_pod.sh
DATASET=sheep-pose-v0.3-yolo11n MODEL=yolo11n-pose.pt ./scripts/train_on_pod.sh
```

## Sync Weights Back

From laptop:

```bash
./scripts/sync_weights_from_pod.sh sheep-pose-v0.3-yolo26n
```

This pulls `best.pt` and usually `last.pt` into `sheep-yolo/weights/` as:

```text
sheep-yolo/weights/sheep-pose-v0.3-yolo26n.pt
sheep-yolo/weights/sheep-pose-v0.3-yolo26n.last.pt
```

## Run Local Inference / Benchmark

From laptop:

```bash
cd sheep-yolo
export YOLOE_MODEL=weights/sheep-pose-v0.3-yolo26n.pt
python -m backend.main
```

Then use the sheep-yolo UI/benchmark flow. The benchmark question is stability
of nose/ear keypoints and downstream ear-angle jitter, not just headline mAP.

## Optional Dataset Fetch

Use only when local inspection of exported JPG/TXT pairs is needed:

```bash
./scripts/fetch_dataset.sh <pod-ip> <pod-ssh-port> sheep-pose-v0.3-yolo26n
```

This is not part of the normal training loop. Training data should stay on the
pod.

## Common Failure Modes

- Missing pod SSH info: update `.env.pod` and/or `~/.sheep-yolo.env`.
- Export fails before training: start `scripts/start_pod_server.sh` on the pod.
- `No best.pt found`: training did not finish or `DATASET` names differ.
- Very small validation metrics: check actual reviewed `v=2` instance count and
  per-keypoint coverage before reading too much into mAP.
- Pod resumed with new IP/port: update both config files if both script
  families will be used.

