# DATA_FORMAT — SamSeesSheep Schema Documentation

Documents the two data formats in this project: `review.json` v2 (the labeling artifact) and the YOLO-pose export format (the training artifact). Covers keypoint semantics, v-flag meanings, flip_idx convention, train/val splitting logic, and dataset versioning.

---

## review.json v2 — The labeling artifact

Each labeled video produces one `review.json` at `data/labels/{video_id}/review.json`. This is the source of truth for reviewed keypoints.

### Schema

```json
{
  "video_id": "7e53dfab",
  "video_filename": "IMG_3412.MOV",
  "reviewed": true,
  "frames": [
    {
      "frame_idx": 0,
      "frame_width": 512,
      "frame_height": 384,
      "instances": [
        {
          "head_bbox": {
            "x": 120,
            "y": 80,
            "w": 200,
            "h": 240
          },
          "keypoints": [
            {"x": 210.5, "y": 150.2, "v": 2},
            {"x": 180.3, "y": 110.8, "v": 2},
            {"x": 240.1, "y": 112.3, "v": 2},
            {"x": 170.9, "y": 85.4, "v": 2},
            {"x": 255.6, "y": 88.1, "v": 2}
          ]
        }
      ]
    }
  ]
}
```

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `video_id` | string | 8-char hex identifier, derived from video content hash |
| `video_filename` | string | Original filename of the uploaded clip |
| `reviewed` | boolean | Whether the video has been fully reviewed |
| `frames` | array | One entry per sampled frame (2 fps) |

### Frame object

| Field | Type | Description |
|---|---|---|
| `frame_idx` | int | 0-based frame index within the clip's 2 fps sample |
| `frame_width` | int | Frame width in pixels (after resize) |
| `frame_height` | int | Frame height in pixels (after resize) |
| `instances` | array | One entry per detected sheep head in this frame |

### Instance object

| Field | Type | Description |
|---|---|---|
| `head_bbox` | object | `{x, y, w, h}` bounding box in pixel coordinates (top-left origin) |
| `keypoints` | array | 5 keypoint objects (see below) |

### Keypoint slots

Five keypoints per instance, always in this order:

| Index | Name | Description |
|---|---|---|
| 0 | nose | Nose tip |
| 1 | L_ear_base | Left ear base (point on ear mask closest to head centroid) |
| 2 | R_ear_base | Right ear base |
| 3 | L_ear_tip | Left ear tip (point on ear mask farthest from head centroid) |
| 4 | R_ear_tip | Right ear tip |

Left/right is from the camera's perspective (image-space). Each keypoint has three fields:

| Field | Type | Description |
|---|---|---|
| `x` | float | Pixel x-coordinate |
| `y` | float | Pixel y-coordinate |
| `v` | int | Visibility/status flag |

### v-flag semantics

| v | Meaning | Export behavior |
|---|---|---|
| **0** | Keypoint absent — the body part is not in frame (mask missing) | Emitted as `0 0 0` in YOLO .txt |
| **1** | Auto-placed by SAM 3 — unreviewed, a machine guess | Emitted as `0 0 0` in YOLO .txt (unless `pseudo=true`) |
| **2** | Human-reviewed — confirmed or hand-corrected by the reviewer | Emitted as `<px> <py> 2` in YOLO .txt |

**Reviewed vs. full instances:**

- An instance with *any* `v=2` keypoint counts as "reviewed."
- An instance where all 5 keypoints are `v=2` counts as a "full" instance.
- An instance with some `v=2` and some `v=0` or `v=1` counts as a "partial" instance — typically means some body parts were out of frame or occluded.

The exporter trains on `v=2` keypoints only. Partially-reviewed instances export with only the `v=2` slots carrying signal; non-reviewed slots become `0 0 0`. The `?pseudo=true` query parameter promotes `v=1` to YOLO `v=2` — prototype-only. Never use pseudo labels for σ benchmarks: training on unreviewed machine guesses teaches the model to trust SAM's mask geometry as ground truth.

### Multi-sheep frames

Schema v2 handles multiple sheep in one frame via `instances[]`. Each detected sheep head gets its own entry with independent keypoints. A frame with three sheep has three entries in `instances[]`.

Schema v1 (superseded) had a flat `keypoints[]` array on the frame object, supporting only one sheep per frame. v1 reviews were migrated to v2 during the schema transition.

---

## YOLO-pose export format

Export output lands at `data/labels/exports/{dataset}/` where `{dataset}` is a versioned name like `sheep-pose-v0.4-yolo26n`.

### Directory structure

```
data/labels/exports/sheep-pose-v0.4-yolo26n/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── 7e53dfab_0000.jpg
│   │   ├── 7e53dfab_0001.jpg
│   │   └── ...
│   └── labels/
│       ├── 7e53dfab_0000.txt
│       ├── 7e53dfab_0001.txt
│       └── ...
└── val/
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

### data.yaml

```yaml
path: .
train: train/images
val: val/images
nc: 1
names: ['sheep_head']
kpt_shape: [5, 3]
flip_idx: [0, 2, 1, 4, 3]
```

When `val` is empty (all frames bucketed to train, e.g., `val_split=0` or only one exportable frame), `val:` points to `train/images` to keep the Ultralytics loader happy. At this dataset scale, σ-benchmark doesn't use held-out val anyway.

### Label file format (.txt)

One `.txt` per source image, named `{video_id}_{frame_idx:04d}.txt`. Each line corresponds to one sheep head instance in that frame:

```
class_id bx by bw bh  k0x k0y k0v  k1x k1y k1v  k2x k2y k2v  k3x k3y k3v  k4x k4y k4v
```

Where:

| Token | Description |
|---|---|
| `class_id` | Always `0` (single class: `sheep_head`) |
| `bx, by` | Bounding box center (x, y), normalized to [0.0, 1.0] |
| `bw, bh` | Bounding box width and height, normalized to [0.0, 1.0] |
| `k0x, k0y, k0v` | Keypoint 0 (nose): normalized x, normalized y, visibility |
| `k1x, k1y, k1v` | Keypoint 1 (L_ear_base) |
| `k2x, k2y, k2v` | Keypoint 2 (R_ear_base) |
| `k3x, k3y, k3v` | Keypoint 3 (L_ear_tip) |
| `k4x, k4y, k4v` | Keypoint 4 (R_ear_tip) |

Keypoint visibility in .txt:

| v | Meaning |
|---|---|
| 0 | Not labeled / not in frame |
| 2 | Labeled and visible (from `v=2` in review.json, or `v=1` if `pseudo=true`) |

Note: YOLO-pose format supports `v=1` (labeled but not visible), but this project does not use it. All usable keypoints export as `v=2`.

#### Example

For a frame with one sheep where all five keypoints were reviewed:

```
0 0.410156 0.456250 0.390625 0.625000  0.411133 0.391146 2  0.352148 0.288542 2  0.468945 0.292448 2  0.333789 0.222396 2  0.499219 0.229427 2
```

For a frame with one sheep where only nose and ear bases were reviewed (ear tips out of frame / `v=0`):

```
0 0.410156 0.456250 0.390625 0.625000  0.411133 0.391146 2  0.352148 0.288542 2  0.468945 0.292448 2  0.000000 0.000000 0  0.000000 0.000000 0
```

### Bounding box conversion

The source bbox in `review.json` is top-left + wh (`{x, y, w, h}`). The exporter converts to center-xywh and normalizes by frame dimensions:

```
bx = (bbox.x + bbox.w / 2) / frame_width
by = (bbox.y + bbox.h / 2) / frame_height
bw = bbox.w / frame_width
bh = bbox.h / frame_height
```

---

## flip_idx convention

`flip_idx` in `data.yaml` tells YOLO-pose how to re-index keypoints during horizontal-flip augmentation:

```yaml
flip_idx: [0, 2, 1, 4, 3]
```

This maps:

| Original index | Original name | Flipped index | Flipped name |
|---|---|---|---|
| 0 | nose | 0 | nose (symmetric — stays on midline) |
| 1 | L_ear_base | 2 | R_ear_base |
| 2 | R_ear_base | 1 | L_ear_base |
| 3 | L_ear_tip | 4 | R_ear_tip |
| 4 | R_ear_tip | 3 | L_ear_tip |

The nose is treated as symmetric (maps to itself) because it sits on the midline. Ear keypoints swap left ↔ right.

---

## Train/val hash-based split

The export pipeline uses deterministic MD5 hash bucketing rather than random shuffling. This ensures reproducibility across re-exports.

### Algorithm

1. For each `(video_id, frame_idx)` pair, compute hash:
   ```
   md5("{video_id}:{frame_idx}")
   ```
   Example: `md5("7e53dfab:0")` → `a1b2c3d4...`

2. Take the first 8 hex characters, interpret as integer, modulo 10000, divide by 10000:
   ```
   bucket = int(hash[:8], 16) % 10000 / 10000.0
   ```
   This produces a deterministic float in `[0.0, 1.0)`.

3. Classify:
   - `bucket < val_split` → **val** (default `val_split` = 0.2)
   - `bucket >= val_split` → **train**

### Minimum-val guarantee

If `val_split > 0` and all frames happen to bucket into `train` (possible with small datasets), the frame with the highest bucket value is promoted to `val`. This prevents the Ultralytics loader from crashing on an empty `val/images` directory. The promotion is deterministic (buckets are fixed per `(video_id, frame_idx)`).

### Design rationale

- **Deterministic**: Same frames → same split every export, regardless of when or where the export runs.
- **Per-frame, not per-video**: Frames from the same video can appear in both train and val. This tests generalization to unseen poses/angles from *known* scenes, matching the project scope (one flock, one camera, one geography).
- **md5 not sha256**: MD5 is faster and collision resistance is irrelevant — we only need stable bucketing. The hash key includes both `video_id` and `frame_idx`, eliminating any cross-video collision concern.

---

## Dataset versioning

Exported datasets follow the naming convention `sheep-pose-v{major}.{minor}`:

| Dataset | Training instances | Training videos | Model |
|---|---|---|---|
| `sheep-pose-v0.2` | 98 | 3 | yolo26n-pose |
| `sheep-pose-v0.3` | 313 | 6 | yolo26n-pose |
| `sheep-pose-v0.4` | 405 | 8 | yolo26n-pose |
| `sheep-pose-v0.5` | (v0.4 + new reviews) | 8+ | yolo26n-pose |
| `sheep-pose-v0.6` | (v0.5 + new reviews) | 8+ | yolo26n-pose |
| `sheep-pose-v0.7` | (v0.6 + new reviews) | 8+ | yolo26n-pose |

Each version is cumulative — it includes all reviewed frames from all previous versions plus newly-labeled clips. Training runs use unique suffix names (e.g., `sheep-pose-v0.4-yolo26n`) to avoid overwriting weights across A/B experiments. The reviewer produces data; the dataset version pins a specific snapshot of reviewed annotations for a training run.

### Versioning policy

- **New clips + new reviews → new minor version** (e.g., v0.4 → v0.5). The previous dataset is preserved for benchmark history.
- **Same data, different hyperparameters → use a suffix** (e.g., `sheep-pose-v0.4-yolo26n` vs `sheep-pose-v0.4-yolo11n`).
- **Re-export of same reviewed data → same dataset name, same hash split, same output** (the exporter cleans stale per-video files before writing, so a re-export balances cleanly with any newly-reviewed frames).
