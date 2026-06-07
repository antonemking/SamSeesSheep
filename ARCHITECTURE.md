# ARCHITECTURE — SamSeesSheep

Detailed technical architecture for researchers and engineers. Covers the full pipeline from video ingestion to edge inference, SAM 3 Video prompt strategy, keypoint schema, train/val split, ear-angle geometry, benchmark methodology, and monorepo layout rationale.

---

## Full pipeline

```
PHONE CAPTURE (1080p, 15–30 s, .MOV)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ LOCAL / CLOUD GPU POD                                       │
│                                                             │
│  FRAME EXTRACTION                                           │
│    pyav / imageio, 2 fps, ~512 px max dimension             │
│    → frame0000.jpg, frame0001.jpg, ...                      │
│         │                                                   │
│         ▼                                                   │
│  SAM 3 VIDEO — TEXT-PROMPTED SEGMENTATION                   │
│    prompts: "sheep head", "sheep ear", "sheep nose"         │
│    sessions: 3 (head + ear x2 + nose) on 24 GB GPU          │
│             or 2 (head + ear) on 6 GB GPU                   │
│    output: per-frame per-instance segmentation masks        │
│    model: facebook/sam3 (Hugging Face, ~3 GB)               │
│         │                                                   │
│         ▼                                                   │
│  KEYPOINT DERIVATION                                        │
│    from masks: nose centroid, ear base=closest-to-head-     │
│    centroid, ear tip=farthest-from-head-centroid            │
│    keypoints[]: 5 slots, image-space coords                 │
│    schema v2: instances[] per frame (multi-sheep)           │
│         │                                                   │
│         ▼                                                   │
│  LABELING UI (FastAPI + vanilla JS frontend)                │
│    human reviewer confirms/corrects every keypoint,         │
│    every instance, every frame                              │
│    v=2 on accept; v=1 on auto (SAM candidate)               │
│    keyboard: A=accept, Enter=save+advance, S=skip           │
│    persists to data/labels/{video_id}/review.json           │
│         │                                                   │
│         ▼                                                   │
│  EXPORT (POST /api/export/keypoints/all)                    │
│    v=2 keypoints → YOLO-pose format                         │
│    hash-based train/val split (default 80/20)                │
│    one .txt per image, N label lines per instance           │
│    output: data/labels/exports/sheep-pose-v0.N/             │
│         │                                                   │
│         ▼                                                   │
│  YOLO-POSE TRAINING (on pod GPU)                            │
│    yolo train model=yolo26n-pose.pt epochs=100 batch=8      │
│    imgsz=640, single class 'sheep_head', kpt_shape [5,3]    │
│    ~2.5 M params, ~6 min on RTX 4090, ~10 MB best.pt        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼ (rsync/sftp best.pt ~10 MB)
┌─────────────────────────────────────────────────────────────┐
│ LOCAL EDGE (GTX 1660 Ti, 6 GB)                              │
│                                                             │
│  INFERENCE (sheep-yolo/)                                    │
│    loads sheep-yolo/weights/<dataset>.pt                    │
│    YOLO-pose forward pass on 6 GB GPU                       │
│    produces keypoints for every detected sheep head         │
│    ByteTrack for temporal continuity,                         │
│    supervision PolygonZone for ROI filtering                │
│         │                                                   │
│         ▼                                                   │
│  σ BENCHMARK (sheep-yolo/scripts/bench_held_out.py)         │
│    residual σ = std(kpt − rolling_median_7_of_kpt)          │
│    measured on held-out clips (NCC < 0.23 vs training)      │
│    per-keypoint + derived ear-angle σ reported               │
│    pickle cache for per-model predictions                   │
│         │                                                   │
│         ▼                                                   │
│  OUTPUT ARTIFACTS                                           │
│    comparison videos (PyAV libx264 CRF 18)                  │
│    per-keypoint σ tables, ear-angle charts                   │
│    multi-lane flock ear-angle EKG (v0.7)                    │
└─────────────────────────────────────────────────────────────┘
```

## SAM 3 Video prompt strategy

SAM 3 Video (`facebook/sam3` on Hugging Face) is the annotation backbone — not the inference backbone. It is called with three text prompts, each run as a separate propagation session:

| Session | Prompt | Output |
|---|---|---|
| 1 | `"sheep head"` | Head mask per instance per frame |
| 2 | `"sheep ear"` | Left + right ear masks per instance per frame |
| 3 | `"sheep nose"` | Nose mask per instance per frame |

Session 3 (nose) requires a 24 GB GPU. On a 6 GB GPU, the pipeline drops to 2 sessions (head + ear) and nose keypoints fall back to head-mask geometry. Frame extraction runs at 2 fps with max dimension 512 px to keep SAM 3 memory within budget. Multiple instances (multiple sheep in one frame) are handled natively: SAM 3's text prompt returns all matching objects and tracks them independently via its internal tracker.

**Global singleton**: `backend/pipeline/video.py` holds one global `_video_model` with no locks. Two concurrent `/api/analyze` calls on the same pod will OOM. This is acceptable because labeling is a single-human activity — two analysts can't label the same pod simultaneously.

## Keypoint schema v2

Five keypoints per sheep head instance, stored in `review.json` as `frames[].instances[].keypoints[]`:

| Slot | Name | Description |
|---|---|---|
| 0 | nose | Nose tip (from nose mask centroid or head mask geometry) |
| 1 | L_ear_base | Left ear base — point on ear mask closest to head centroid |
| 2 | R_ear_base | Right ear base |
| 3 | L_ear_tip | Left ear tip — point on ear mask farthest from head centroid |
| 4 | R_ear_tip | Right ear tip |

Image-space coordinates (`x`, `y` in pixels), with left/right from the camera's perspective. Each keypoint carries a `v` flag:

| v | Meaning |
|---|---|
| 0 | Keypoint absent — mask missing, part not in frame |
| 1 | Auto-placed by SAM 3 — unreviewed, machine guess |
| 2 | Human-reviewed — confirmed or hand-corrected |

Only `v=2` keypoints enter the YOLO-pose training dataset during normal export. The `?pseudo=true` escape hatch promotes `v=1` to YOLO `v=2` at export time — prototype-only, never for σ benchmarks. An instance with *any* `v=2` keypoint counts as "reviewed"; partially-reviewed instances (some keypoints `v=2`, others `v=1` or `v=0`) are exported with only the `v=2` slots carrying signal.

### flip_idx convention

For YOLO-pose horizontal-flip augmentation, `data.yaml` declares:

```yaml
flip_idx: [0, 2, 1, 4, 3]
```

This maps: nose (0) → nose (0), L_ear_base (1) → R_ear_base (2), R_ear_base (2) → L_ear_base (1), L_ear_tip (3) → R_ear_tip (4), R_ear_tip (4) → L_ear_tip (3).

## Review data model

Each video gets a directory under `data/labels/{video_id}/`:

```
data/labels/{video_id}/
├── frames/
│   ├── frame0000.jpg
│   ├── frame0001.jpg
│   └── ...
├── review.json          # schema v2
└── (SAM 3 cache)        # masks, propagation state
```

`review.json` structure:

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
          "head_bbox": {"x": 120, "y": 80, "w": 200, "h": 240},
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

Multi-sheep frames have multiple entries in `instances[]`. Schema v2 replaced the earlier single-instance-per-frame model to handle clips with multiple animals.

## Train/val hash-based split

The export pipeline splits frames deterministically using MD5 hash bucketing:

1. For each `(video_id, frame_idx)` pair, compute `md5("{video_id}:{frame_idx}")`
2. Take the first 8 hex chars, modulo 10000, divide by 10000 → bucket in [0.0, 1.0)
3. Frames with bucket < `val_split` (default 0.2) go to val; rest go to train

This is deterministic and reproducible. A guard clause ensures `val` is never empty when at least 2 frames are exportable (if all frames bucket to train, the highest-bucket frame is promoted to val). This prevents the Ultralytics loader from crashing on an empty val/images directory.

The split is per-frame (not per-video), meaning frames from the same video can appear in both train and val. This tests generalization to unseen poses/angles from *known* scenes, which matches the project's scope (one flock, one camera, one geography). Cross-scene generalization is future work.

## Ear-angle geometry

Ear angle is computed in the labeler for QA visualization and in the benchmark for the welfare-relevant scalar metric:

1. **Head midline**: PCA on the head mask yields the dorsal head axis. Ear-midpoint position relative to head centroid disambiguates direction (ears mark the dorsal/poll end). Falls back to image-vertical "up" when no ears are available. When nose is detected, `_compute_anatomical_midline` (nose → ear-midpoint) is preferred over PCA.
2. **Ear direction**: For each ear, base = mask point closest to head centroid, tip = mask point farthest from head centroid. Direction = tip − base, normalized.
3. **Ear angle relative to head**: `arctan2` of ear direction relative to head midline. Image coords have y increasing downward, so the y component is negated before `arctan2`.
4. **Classification bands** (SPFES literature):
   - Green ≥ 30°: up/alert
   - Amber −10° to 30°: neutral
   - Red ≤ −10°: down/back

These bands are QA aids, not diagnostic cutoffs. Trust deltas, not absolutes.

## σ benchmark methodology

The `sheep-yolo/scripts/bench_held_out.py` script measures keypoint stability on a held-out clip. The methodology:

1. **Clip selection**: A clip whose normalized cross-correlation with every training video is < 0.23 — genuinely unseen.
2. **Target window**: A ~5 s segment where a target sheep's head is roughly stationary. Centroid std ~30 px. ROI padding excludes neighbouring sheep.
3. **Residual σ**: `std(kpt − rolling_median_7_of_kpt)`. Strips slow head drift (the sheep's actual motion); leaves the frame-to-frame jitter that matters for welfare estimation.
4. **Raw σ**: `std(kpt)` over the window. Expected to be ~45 px for all models (dominated by the sheep's actual head sway, not model noise). Serves as a sanity check.
5. **Ear-angle σ**: Derived from keypoints via the same geometry as the labeler. The headline metric.
6. **Detection rate**: Any YOLO detection (conf ≥ 0.25) in the ROI. Reports both total frames and in-ROI frames.
7. **Stock baseline**: `yolo26n.pt` (COCO-pretrained) run on the same clip. Reports boxes-but-no-kpts.

## Monorepo layout rationale

The repository combines labeling + training (root) with inference + benchmarking (`sheep-yolo/`) in one repo. This is intentional:

| Aspect | Rationale |
|---|---|
| **Shared data format** | Both sides read the same `review.json` schema and YOLO-pose export format. Splitting would create version-skew risk on the schema contract. |
| **Shared scripts** | `train_on_pod.sh` and `sync_weights_from_pod.sh` bridge both sides. Keeping them in one repo avoids cross-repo path assumptions. |
| **Benchmark proximity** | `bench_held_out.py` compares multiple model versions trained against data produced by the labeling pipeline. Colocation makes version-linking explicit. |
| **Single research artifact** | The repo is archived with a research paper. Two repos would require two archives; one repo tells the whole story. |

The boundary is clean: root owns labeling and training. `sheep-yolo/` owns inference and benchmarks. Only `best.pt` (~10 MB) crosses the boundary.

## Compute topology

```
┌──────────────┐     SSH/rsync      ┌────────────────────────┐
│   LAPTOP     │ ◄──────────────────►│  CLOUD GPU POD          │
│  (6 GB GPU)  │                    │  (RTX 4090 / L40S /     │
│              │  push clips,        │   H100, 24+ GB)         │
│  - labeling  │  trigger train,     │                         │
│    client    │  sync weights       │  - SAM 3 Video (3×)     │
│  - backups   │                     │  - labeling UI server   │
│  - inference │                     │  - yolo train            │
│  - bench     │                     │  - dataset storage      │
└──────────────┘                     └───────────┬────────────┘
                                                 │
                                        ┌────────▼───────────┐
                                        │  NETWORK VOLUME     │
                                        │  /mnt/labels        │
                                        │  (durable storage,  │
                                        │   survives Stop/    │
                                        │   Resume/Terminate) │
                                        └────────────────────┘
```

- Laptop GPU (6 GB GTX 1660 Ti) can run the reduced 2-session SAM 3 pipeline and full YOLO-pose inference, but cannot fit YOLO training (batch=8, imgsz=640).
- Cloud GPU pod (RunPod or Vast.ai) runs the heavy compute. Pod disk is ephemeral; the Network Volume mounted at `/mnt/labels` is the durable store for reviewed annotations.
- `data/labels/` on the pod is a symlink to the Network Volume mount. `scripts/start_pod_server.sh` refuses to boot if the mount is missing.

## Config and env management

Two env-file conventions coexist:

| File | Consumers | Purpose |
|---|---|---|
| `<repo>/.env.pod` | `push_clip.sh`, `pod_ssh.sh`, `backup_dataset.sh` | Pod IP, SSH port, HTTP proxy URL |
| `~/.sheep-yolo.env` | `train_on_pod.sh`, `sync_weights_from_pod.sh`, `fetch_dataset.sh` | Pod IP, SSH port, default DATASET |

On the pod, `.env.pod` sets `LABELS_VOLUME` (mount path) and `HF_HOME` (Hugging Face cache location, ideally on the network volume). Neither env file is committed.

## Key design decisions and trade-offs

- **SAM 3 as annotator, not inference engine**: SAM 3 Video's frame-to-frame tracking, occlusion handling, and zero-shot instance discovery make it the right tool for the labeling pipeline, where a human reviews every frame. It's too heavy for edge inference (~3 GB model, seconds per frame). The trained YOLO-pose model (~10 MB, <100 ms per frame) is the inference engine. The annotation cost is paid once per dataset version; inference is free forever.
- **2 fps sampling, 512 px max dimension**: Keeps SAM 3 within memory budget and per-clip processing time under 3 minutes. Faster than video rate is unnecessary for the labeling use case (sheep head pose changes slowly).
- **Hash-based split, not random**: Determinism matters for reproducible benchmarks. The `md5` of `(video_id, frame_idx)` is stable across re-exports. A new export run produces the same train/val assignment.
- **Single-class YOLO-pose**: The model detects one class (`sheep_head`) with 5 keypoints. Multi-class (e.g., separate head/ear/nose) would fragment an already-small dataset.
- **No re-identification (yet)**: Identity is per-track within a clip and human-confirmed. Re-identifying look-alike sheep across clips is unsolved — an ear tag is the honest answer.
