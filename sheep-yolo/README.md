# sheep-yolo

**Inference engine and sigma benchmark for trained sheep-head keypoint models from [SamSeesSheep](../).**

Runs YOLO-pose models (YOLO26n-pose, 2.5 M params, ~10 MB) on a 6 GB local GPU. Produces per-keypoint residual σ benchmarks on genuinely held-out clips and multi-sheep flock ear-angle charts. The models are trained in the parent repo via the SAM 3 → labeling → export → cloud-GPU pipeline; this directory is where they land and get measured.

## What's here

| Path | What |
|---|---|
| `weights/` | Trained `.pt` files synced from the pod (`sheep-pose-v0.N-yolo26n.pt`) |
| `scripts/` | Benchmark scripts, EKG renderer, supervision spike, dedup tooling |
| `backend/` | FastAPI inference server + UI (`python -m backend.main`) |
| `artifacts/` | Benchmark JSON, comparison MP4s, EKG renders (gitignored) |
| `test-clips/README.md` | Note pointing benchmark users to repo-level `../test-clips/` |

## Quickstart

```bash
cd sheep-yolo
uv sync
export YOLOE_MODEL=weights/sheep-pose-v0.7-yolo26n.pt
uv run python -m backend.main
# Open http://localhost:8000
```

The `YOLOE_MODEL` env var points to a trained YOLO-pose checkpoint. Stock `yolo26n.pt` has no keypoint head — it produces sheep bounding boxes but zero keypoints. Use a trained weight.

Benchmark scripts read repo-level `../test-clips/` from inside `sheep-yolo/`. Drop new `.mov`/`.mp4` files there and refresh.

## Benchmark script

`scripts/bench_held_out.py` is the canonical reproducible held-out sigma
benchmark. Superseded version-comparison scripts live under `scripts/archive/`
for provenance only.

```bash
uv run python scripts/bench_held_out.py IMG_3651          # 3-way: v0.2 vs v0.3 vs v0.4
```

The canonical script:
1. Runs inference across all specified model versions on the clip (cached to `artifacts/_cache/`)
2. Finds a motionless target window (sheep centroid σ ~30 px) via ByteTrack
3. Computes per-keypoint residual σ (detrended with rolling-median-7)
4. Derives ear-angle σ from keypoint geometry
5. Writes `artifacts/bench_report-*.json` + renders comparison MP4s

The residual σ metric strips slow head drift (property of the sheep) and isolates frame-to-frame jitter — the welfare-measurement-relevant signal.

## EKG renderer (multi-sheep flock)

```bash
uv run python scripts/render_ekg.py Test_Clip_Morning
```

Produces a synced multi-lane ear-angle chart — one trace per tracked sheep, with SPFES alert bands and circle-crop face thumbnails. Renders to `artifacts/synced-lanes-*.mp4`.

With the pro renderer:

```bash
uv run python artifacts/render_synced_pro_TCM.py
```

Additional per-sheep cropping and lane styling. See inline comments in the script.

## Sigma computation (standalone)

```bash
uv run python artifacts/compute_sigma.py
```

Reads the benchmark JSON and computes ear-angle σ from per-frame keypoint arrays — the same math as the bench scripts, without the inference pass.

## Supervision spike (LOR-123)

`scripts/render_supervision_spike.py` is a throwaway probe of the Roboflow [`supervision`](https://supervision.roboflow.com) toolkit (v0.28.0). It runs v0.7 YOLO-pose over a bounded sample of `../test-clips/IMG_3651.MOV` and renders an annotated MP4 into `artifacts/` (gitignored), exercising `sv.Detections` / `sv.KeyPoints.from_ultralytics`, supervision's video IO (`VideoInfo` / `get_video_frames_generator` / `VideoSink`), the `VertexAnnotator` / `EdgeAnnotator` keypoint annotators, and `PolygonZone` for ROI hit-counting. Ear-angle math is reused verbatim from `render_ekg.py`.

supervision is intentionally **not** a dependency of this repo — run it one-off:

```bash
uv run --with supervision scripts/render_supervision_spike.py            # first 90 frames
uv run --with supervision scripts/render_supervision_spike.py \
    --start-frame 380 --max-frames 90                                    # ewe inside the ROI
```

**Recommendation: optional artifact / reporting layer only.** The spike did not surface anything that justifies a rewrite.

- ✅ Cleaner video IO than the manual `cv2.VideoWriter` dance; `from_ultralytics` converters remove boilerplate; `PolygonZone` is a real upgrade over the inline center-in-box ROI test and is the piece worth promoting first if we build multi-region paddock reports.
- ❌ No keypoint *geometry* (we still own `ears()`/`ang()` — the actual signal); no equivalent of the crop + live matplotlib EKG panel; not worth a required dependency for the core paths.
- ⚠️ The first spike output can look worse than the EKG renderer if drawn with raw supervision keypoint annotators: they are generic drawing helpers and do not apply SamSeesSheep's confidence threshold for us. The default script now keeps showcase output confidence-filtered; `--raw-sv-keypoint-annotators` is only for API inspection.
- ⚠️ `sv.ByteTrack` is **deprecated** in 0.28.0 (tracking moving to a standalone `trackers` package). This spike does **no tracking**; the EKG renderer keeps Ultralytics' built-in ByteTrack. Do not migrate tracking here.

This README section is the durable tracked summary; generated MP4s and any expanded local notes stay ignored with the rest of `artifacts/` / `docs/`.

## Env knobs

See `backend/config.py` for the full list:

- `YOLOE_MODEL` — path to trained YOLO-pose checkpoint (default: `yoloe-v8l-seg.pt`, but trained checkpoints override via env)
- `MAX_FRAMES` — default 600
- `INFER_IMGSZ` — default 960
- `DETECTION_CONF` — confidence floor, default 0.25
- `TRACKER_YAML` — `bytetrack.yaml` (default) or `botsort.yaml`
- `TRACK_COVERAGE_FLOOR` — drop tracks seen for less than this fraction of frames (default 0.15)

## What this is / what this is not

**What this is:** the inference and benchmarking half of the SamSeesSheep pipeline. Trained YOLO-pose weights land here from the cloud GPU. Local inference produces (a) per-keypoint residual σ on held-out clips, (b) multi-sheep ear-angle charts, and (c) detection-rate baselines against stock YOLO.

**What this is not:** a trained sheep-parts model by itself. Not a welfare claim. Not a product. Not an OOB YOLOE probe — that was the original intent, but the investigation showed YOLOE cannot measure ear angle without fine-tuning. See the project [`README.md`](../README.md) and [`VALIDATION.md`](../VALIDATION.md) for scope and framing.

## Validation references

- McLennan & Mahmoud 2019 — Sheep Pain Facial Expression Scale (SPFES)
- Reefmann et al. 2009 — ear posture taxonomy
- Boissy et al. 2011 — cognitive sciences, ear postures, emotions in sheep

Thresholds used: `EAR_UP > 30°`, `EAR_DOWN < -10°` relative to dorsal head axis (nose → ear-midpoint). Same values as the parent pipeline.
