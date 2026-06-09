# Changelog

All notable changes to `sheep-seg` will be documented here.

This project follows [semantic versioning](https://semver.org/) at the milestone level: `MAJOR.MINOR.PATCH` where MAJOR = a validated capability gain, MINOR = an experiment delivered, PATCH = documentation or scope corrections.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The [`VALIDATION.md`](./VALIDATION.md) document is versioned alongside the code. Every release will note which sections of `VALIDATION.md` changed and why.

---

## [Unreleased]

### Added — 2026-05-13 — sheep-pose v0.4 and the first clean held-out benchmark

Third labeling round: 313 → 405 reviewed sheep-head instances across 6 → 8 videos. Same YOLO26n-pose architecture, same recipe, same compute as v0.2 and v0.3. Trained on the RunPod 4090 in ~6 min. Val pose mAP50-95 went 0.643 → 0.732.

Closed the v0.3 doc's outstanding IOU: a clean apples-to-apples held-out benchmark against `IMG_3651.MOV` — never pushed to the labeler, never reviewed, NCC < 0.23 vs every training video. All three model versions (v0.2 / v0.3 / v0.4) run against it, plus stock `yolo26n.pt` as the "off-the-shelf YOLO" baseline Post 2 promised.

Ear-angle residual σ on a 5-second motionless window: v0.2 6.7° / 6.1° → v0.3 4.8° / 4.2° → v0.4 4.1° / 4.1° (L / R). Stock yolo26n.pt produces zero keypoints — ear angle is unmeasurable without a sheep-pose keypoint head.

#### Added
- `sheep-yolo/scripts/bench_held_out.py` — 3-way held-out benchmark script. PyAV libx264 (CRF 18) video writer replaces cv2 mp4v default-bitrate output. Keypoint-distance dedupe handles overlapping NMS-survival detections. Pickle cache for per-model predictions so re-render iterations skip the ~3-min-per-model CPU inference.
- `docs/v0.4-benchmark.md` — full 3-way held-out report with per-keypoint σ tables, ear-angle σ, and stock-baseline numbers.
- `docs/v0.4-ear-angle-chart.png` — the flat-line chart Post 2 promised.
- Part 3 v2 LinkedIn draft — archived outside the current tree (full v0.2 → v0.3 → v0.4 curve framing). v1 preserved.
- `sheep-yolo/weights/sheep-pose-v0.4-yolo26n.pt` (gitignored; produced by `scripts/train_on_pod.sh`).
- `sheep-yolo/test-clips/README.md` — points bench users to repo-level `test-clips/` after the monorepo move without committing a fragile symlink.
- `CLAUDE.md` and `AGENTS.md` — operational notes for code agents (env-file split, pod paths, gotchas, label-count snippets).

#### Changed
- `README.md` — replaced the v0.3-vs-v0.2 hero section with a v0.4 section centered on the ear-angle chart and held-out framing.

### Added — 2026-05-20 — v0.5 training run + Vast.ai migration

Fifth YOLO26n-pose training run. Migrated GPU compute from RunPod to Vast.ai for cost flexibility. Same labeling workflow, same recipe, same held-out benchmark methodology. Left ear σ improved (4.06° → 3.66°), right ear σ slightly regressed (4.09° → 4.30°). ROI detections improved (149 → 153 of 155 frames).

#### Added
- `scripts/bootstrap_vast.sh` — Vast.ai instance bring-up script (CUDA drivers, uv, git clone, HF cache).
- `.env.pod.vast` — Vast.ai connection config template.
- `sheep-yolo/artifacts/bench_report-IMG_3651-v04v05.json` — v0.4 vs v0.5 held-out comparison.
- `sheep-yolo/artifacts/v0.4-vs-v0.5-IMG_3651.mp4` — side-by-side comparison video.
- `sheep-yolo/weights/sheep-pose-v0.5-yolo26n.pt` (gitignored).

### Added — 2026-05-27 — v0.6 training run

Sixth YOLO26n-pose training run. Mixed result: right ear σ at all-time best (3.55°) but left ear σ regressed to 4.65° (worst since v0.2). Demonstrates that more data doesn't monotonically improve all keypoints — per-keypoint coverage and per-session labeling consistency matter at least as much as instance count.

#### Added
- `sheep-yolo/artifacts/bench_report-IMG_3651-v04v05v06.json` — 3-way held-out comparison.
- `sheep-yolo/artifacts/v0.4-vs-v0.5-vs-v0.6-IMG_3651.mp4` — 3-way comparison video.
- `sheep-yolo/weights/sheep-pose-v0.6-yolo26n.pt` (gitignored).

### Added — 2026-06-06 — v0.7 training run + flock ear-angle monitor + dedup tooling

Seventh YOLO26n-pose training run. Left ear σ at 3.70° (second-best), right ear at 4.46°. Introduced keypoint-distance deduplication to handle overlapping NMS-survival detections (the "supervision spike" investigation confirmed ByteTrack's internal Kalman filter as the right dedup layer; supervision PolygonZone used for ROI filtering in bench renders). Added multi-sheep flock ear-angle monitor — six ewes tracked simultaneously with one live ear-angle lane each.

#### Added
- `sheep-yolo/scripts/render_ekg.py` — multi-lane flock ear-angle EKG renderer.
- `sheep-yolo/scripts/render_dedup.py` — keypoint-distance deduplication renderer.
- `sheep-yolo/scripts/render_supervision_spike.py` — supervision library integration spike.
- `sheep-yolo/docs/supervision-spike.md` — technical decision record for the spike.
- `sheep-yolo/artifacts/bench_report-IMG_3651-v04v05v07.json` — 3-way held-out comparison.
- `sheep-yolo/artifacts/v0.4-vs-v0.5-vs-v0.7-IMG_3651.mp4` — 3-way comparison video.
- `sheep-yolo/artifacts/synced-lanes-6ewes-IMG_3651.mp4` — 6-ewe flock ear monitor.
- `sheep-yolo/artifacts/flock-multi-ear-ekg-IMG_3651.mp4` — multi-sheep simultaneous EKG.
- `assets/v0.7-flock-ear-monitor.webp` — flock monitor hero still for README.
- `sheep-yolo/weights/sheep-pose-v0.7-yolo26n.pt` (gitignored).

### Changed — 2026-04-17 — Scope reset to visualization artifact

v0 descoped from *welfare instrument* to *visualization artifact*. A peer review flagged that the pipeline had drifted into capability-breadth rather than applied rigor — multiple segmentation backbones, multiple trackers, VLM orchestrators generating a constant, a depth-mesh feature off the critical path. The simplify branch cut ~3,700 lines to leave a single-backbone pipeline (SAM 3 Video → head-PCA midline → ear angles → chart). See `VALIDATION.md` §"Scope reset" for the claim implications.

### Removed

The removed-file entries below intentionally name files that no longer exist in
the current tree.

- `backend/pipeline/segment.py` — SAM 2.1 loader and all photo segmentation entry points. One backbone (SAM 3 Video) now.
- `backend/pipeline/depth.py`, `backend/pipeline/mesh3d.py` — Depth Anything V2 and Open3D Poisson mesh reconstruction. Not on the measurement path.
- `backend/pipeline/orchestrator.py`, `backend/pipeline/local_orchestrator.py` — Claude and Gemma 4 VLMs. The orchestrator was generating the constant prompt list `["sheep head", "sheep ear", "sheep nose"]` for every clip.
- `backend/pipeline/narrative.py` + `backend/pipeline/eup.py` — welfare-narrative generation and EUP% aggregation. Out of current scope.
- `backend/routes/session.py` + photo routes in `backend/routes/analyze.py` — photo flow, including `/api/upload`, `/api/analyze/batch|click|auto|multiclick|mesh|orchestrated|{photo_id}`, `/api/narrative`, `/api/session`, `/api/demo`.
- `backend/pipeline/video.py` — dropped the third SAM 3 Video session (nose), the alert-regions detector, and `_free_sam3_image_model`.
- Photo-flow frontend: model/subject radio toggles, click-to-segment UI, SAM 2.1 multi-click flow, segmentation result row, measurements panel, turntable, alert-region rendering, orchestrator scene panel.

### Added
- `_compute_head_midline_pca` in `backend/pipeline/ear_angle.py` — dorsal axis from head mask PCA with ear-midpoint disambiguation, replacing the nose-anchored anatomical midline.

### Deferred to future work
- Validation against documented stress events.
- Real-time / streaming mode (would need a causal tracker; batch is sufficient for this artifact).
- Continuous monitoring at water troughs / handling chute.

### Planned (Weekend 1 — Watch)
- Pick the species (sheep, per the literature) and document breed in `VALIDATION.md`
- 2 hours of unassisted observation of the flock with a notebook only
- 30 phone photos across varied lighting/angles/distances
- Run Meta SAM 3 against the 30 photos and measure segmentation success rate honestly
- One-page v0 plan committed (archived outside the current tree)

### Planned (Weekend 2 — Build)
- Architecture decision: tiny custom keypoint model vs. SAM 3 + published thresholds
- Working ear-angle extractor on a phone

### Planned (Weekend 3 — Measure)
- EUP% computed across 30-minute observation windows
- 6–10 documented stress events with timestamps

### Planned (Weekend 4 — Decide)
- Pass/kill against the 70% criterion
- `VALIDATION.md` updated with measured failure modes
- Public writeup either way

---

## [0.0.0] — 2026-04-11

### Added
- Project skeleton: `README.md`, `VALIDATION.md`, `LICENSE` (MIT), `.gitignore`, `CHANGELOG.md`
- Day-0 [`VALIDATION.md`](./VALIDATION.md) skeleton stating the claim space before any claims are made
- 4-weekend feasibility plan and explicit kill criterion documented in [`README.md`](./README.md)
- Anti-overclaim commitments in [`VALIDATION.md`](./VALIDATION.md) §"Anti-overclaim commitments"

### Decided
- v0 species: sheep (Sheep Pain Facial Expression Scale literature is best-validated)
- v0 signal: ear position only
- v0 metric: **EUP% (Ear-Up Percentage)** over a rolling 60-minute window
- v0 user: project author, on his own animals (no recruitment, no vet partner, no other farms in v0)
- v0 stack TBD pending Weekend 1 SAM 3 segmentation test
- v0 mode: public feasibility study, MIT-licensed, negative results published

### Not yet
- No code
- No data
- No measurements
- No claims

The repository exists at this point as a public commitment to the discipline of writing the validation contract before writing the system.
