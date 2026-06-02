# AP-10K transfer experiment

Does pretraining the YOLO-pose **backbone** on AP-10K (animal-body pose,
[arXiv 2108.12617](https://arxiv.org/abs/2108.12617)) beat the stock
COCO-human-pose backbone for our 5-keypoint `sheep_head` task?

## The two arms

| | Stage 1 | Stage 2 (fine-tune) |
|---|---|---|
| **A — baseline** | — | `yolo26n-pose.pt` → sheep train |
| **B — experiment** | `yolo26n-pose.pt` → **AP-10K** | AP-10K weights → sheep train |

Both arms fine-tune on the **identical** train set (every labeled clip except
the held-out one) and are scored on the **identical** held-out clip with
ground-truth keypoints. Arm A is exactly what `train_on_pod.sh` does today, so
it's a fair "is this worth it?" baseline.

## The one thing to understand: it's a *backbone* transfer

AP-10K has **17 whole-body keypoints** (eyes, nose, neck, limbs, tail). Our
schema has **5 head/ear keypoints** (`nose, L/R-ear-base, L/R-ear-tip`). The
pose head's output dimension is tied to the keypoint count, so the 17-kpt head
**cannot** carry into a 5-kpt fine-tune.

When Stage 2 loads the AP-10K `.pt`, Ultralytics intersects state dicts by
shape: the **backbone + neck transfer, the pose head is reinitialized** and
trained fresh on sheep. So this experiment measures whether an
animal-body-pose-pretrained *feature extractor* helps — not whether AP-10K's
keypoints map to ours (they don't). That's the intended, correct comparison;
just don't read the result as "AP-10K keypoints transferred."

Why it might help: the base `yolo26n-pose.pt` backbone is tuned on COCO
*humans*. AP-10K gives it real four-legged-animal texture/shape/pose priors
(including sheep and the rest of Bovidae) before it ever sees our clips.

Why it might not: our task is close-up head/ear crops, a narrow domain; the
backbone may already be good enough after Stage-2 fine-tuning, leaving little
headroom. A null result is a perfectly valid (and cheap-to-get) answer.

## Why a whole clip is held out (not random frames)

Adjacent frames of one clip are near-duplicates. A random frame split leaks
near-identical frames into both train and test and inflates the score. We hold
out one entire `video_id` (leave-one-clip-out) so the test clip is genuinely
unseen. Pick a held-out clip with a healthy number of reviewed frames so the
metric isn't noisy.

## Running it

Prereqs on the pod: the labeling server's export endpoint reachable on
`localhost:8000`, and the AP-10K release untarred (default `/workspace/ap-10k`,
containing `data/` + `annotations/`). Get it from the
[AP-10K repo](https://github.com/AlexTheBad/AP-10K).

From your laptop (uses `~/.sheep-yolo.env` for pod SSH, like `train_on_pod.sh`):

```bash
HELDOUT=IMG_1234 ./scripts/run_ap10k_transfer_experiment.sh
# optional knobs: DATASET=sheep-pose-v0.4 AP10K_EPOCHS=50 FT_EPOCHS=100 ...
```

Or directly on the pod:

```bash
cd /workspace/SamSeesSheep
HELDOUT=IMG_1234 bash scripts/ap10k_transfer_experiment.sh
```

List candidate held-out clips with `ls /workspace/SamSeesSheep/data/labels`.

## What it produces

```
/workspace/runs/ap10k-transfer/
  ap10k-pretrain/weights/best.pt   # Stage-1 AP-10K backbone (17-kpt, reused across reruns)
  baseline/weights/best.pt         # Arm A
  ap10k-ft/weights/best.pt         # Arm B
  comparison.json                  # side-by-side metrics
```

`comparison.json` (also printed as a table) reports, for each arm, pose
mAP@50, pose mAP@50-95 (OKS), box mAP, precision, recall — plus the % delta of
B vs A. **Positive delta on pose mAP@50-95 ⇒ AP-10K pretraining helped on the
held-out clip.** The launcher rsyncs `comparison.json` back to
`sheep-yolo/artifacts/`.

## Cost / reruns

Stage 1 (full AP-10K, ~10K images, 50 epochs) is the expensive part — roughly
1–2 h on a 4090, longer the bigger the card's queue. It runs **once**: the
driver skips it if `ap10k-pretrain/weights/best.pt` already exists, and skips
the AP-10K→YOLO conversion if `/workspace/ap10k-yolo/data.yaml` exists. Delete
those to force a rebuild. Stage 2 (both fine-tunes) is the usual ~10 min each.

## Pieces

| File | Role |
|---|---|
| `sheep-yolo/scripts/prep_ap10k.py` | AP-10K COCO → YOLO-pose (17 kpts, single `animal` class) |
| `sheep-yolo/scripts/prep_experiment_split.py` | Leave-one-clip-out train/test split from a frozen export |
| `sheep-yolo/scripts/eval_transfer_experiment.py` | `yolo val` both arms on the held-out clip → comparison.json |
| `scripts/ap10k_transfer_experiment.sh` | Pod-side driver (all 5 stages) |
| `scripts/run_ap10k_transfer_experiment.sh` | Local SSH launcher |
