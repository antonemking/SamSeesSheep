---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: keypoint-detection
tags:
  - pose-estimation
  - keypoint-detection
  - animal-pose
  - sheep
  - yolo
  - yolo-pose
  - precision-livestock-farming
  - agriculture
---

# SamSeesSheep — sheep-head keypoint detector (YOLO26n-pose)

A lightweight (2.5 M parameter, ~6 MB) YOLO-pose model that places **five
anatomical keypoints** on every sheep head it detects in ambient pasture video:
nose, left/right ear base, left/right ear tip. From those keypoints the
[project pipeline](https://github.com/antonemking/SamSeesSheep) derives a
per-animal **ear-angle** signal.

> **What this is — and is not.** This is a *measurement instrument*: it localizes
> ear keypoints with quantified stability. It is **not** a pain detector and
> **not** a welfare scorer. Ear-angle thresholds drawn from the literature are
> reference bands, not diagnostic cutoffs. See
> [`VALIDATION.md`](https://github.com/antonemking/SamSeesSheep/blob/main/VALIDATION.md)
> for the full anti-overclaim contract.

## Keypoint schema

`kpt_shape: [5, 3]` — each keypoint is `(x, y, visibility)`. Left/right are from
the camera's perspective.

| Index | Keypoint | | Index | Keypoint |
|---|---|---|---|---|
| 0 | Nose tip | | 3 | Left ear tip |
| 1 | Left ear base | | 4 | Right ear tip |
| 2 | Right ear base | | | |

Horizontal-flip augmentation uses `flip_idx: [0, 2, 1, 4, 3]` (swaps left/right).

## Model versions

Each version is the same architecture and recipe (YOLO26n-pose, 100 epochs,
batch 8, imgsz 640); the only variable is labeled-data scale. The progression is
published as a training-scale study, **not** as monotonic improvement past v0.4.

| File | Reviewed instances | Train videos | Held-out ear-angle σ_avg (HO-1) |
|---|---|---|---|
| `sheep-pose-v0.2-yolo26n.pt` | 98  | 3  | 6.39° |
| `sheep-pose-v0.3-yolo26n.pt` | 313 | 6  | 4.52° |
| `sheep-pose-v0.4-yolo26n.pt` | 405 | 8  | 4.07° |
| `sheep-pose-v0.5-yolo26n.pt` | 428 | 9  | 3.98° |
| `sheep-pose-v0.6-yolo26n.pt` | 471 | 10 | 4.10° (per-keypoint regression) |
| **`sheep-pose-v0.7-yolo26n.pt`** | **523** | **11** | **4.08°** |

**Use `v0.7`** (latest, most data) unless you are reproducing a specific paper
figure. On a second held-out clip (HO-2, morning light), v0.7 reaches
σ_avg = **2.84°** — the lower number reflects an easier clip (larger apparent
head, higher contrast), not a model improvement. σ here is the residual standard
deviation of ear angle on a near-stationary sheep, i.e. the measurement noise
floor; the difference between adjacent versions is within bootstrap CIs.

Stock `yolo26n.pt` (COCO-pretrained) produces **zero keypoints** on these clips —
the keypoint head must be trained against the target animals.

## Usage

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

path = hf_hub_download("antking1/sheep-pose-yolo26n",
                       "sheep-pose-v0.7-yolo26n.pt")
model = YOLO(path)

results = model("sheep.jpg")           # detect heads + 5 keypoints each
kpts = results[0].keypoints.xy         # (n_sheep, 5, 2) pixel coords
```

## Scope and limitations

Trained on a single Katahdin hair-sheep flock (5 ewes) on one homestead in
Middletown, Delaware, USA, with one smartphone camera, reviewed by one
non-veterinarian operator. **No** cross-breed, cross-flock, or cross-geography
generalization is claimed. Known failure modes: dark fleece / low contrast,
heavy motion blur, profile views with ear occlusion, crowded frames, wool-covered
ears. See the repository for the full benchmark protocol and the
`verify_paper_claims.py` harness (44/44 checks).

## Links

- **Code + benchmarks:** https://github.com/antonemking/SamSeesSheep
- **Paper:** *SamSeesSheep: A Measurement Pipeline for Sheep Ear-Angle Extraction
  from Ambient Pasture Video Using Foundation-Model Annotation* (see repository)
- **Prediction caches** (to regenerate every benchmark JSON): GitHub release asset

## License

Released under **AGPL-3.0** to match Ultralytics YOLO, with which these weights
were trained. (The author's own pipeline code in the GitHub repository is MIT;
the trained weights inherit Ultralytics' AGPL-3.0.)

## Citation

```bibtex
@misc{king2026samseessheep,
  author       = {Antone King},
  title        = {SamSeesSheep: A Measurement Pipeline for Sheep Ear-Angle
                  Extraction from Ambient Pasture Video Using Foundation-Model
                  Annotation},
  year         = {2026},
  howpublished = {\url{https://github.com/antonemking/SamSeesSheep}},
}
```
