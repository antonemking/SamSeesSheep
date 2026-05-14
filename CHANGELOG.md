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
- `docs/linkedin-post-draft-ep3-v2.md` — Part 3 v2 (full v0.2 → v0.3 → v0.4 curve framing). v1 preserved.
- `sheep-yolo/weights/sheep-pose-v0.4-yolo26n.pt` (gitignored; produced by `train_on_pod.sh`).
- `sheep-yolo/test-clips` symlink so the bench scripts find clips after the monorepo move.
- `CLAUDE.md` and `AGENTS.md` — operational notes for code agents (env-file split, pod paths, gotchas, label-count snippets).

#### Changed
- `README.md` — replaced the v0.3-vs-v0.2 hero section with a v0.4 section centered on the ear-angle chart and held-out framing.

### Changed — 2026-04-17 — Scope reset to visualization artifact

v0 descoped from *welfare instrument* to *visualization artifact*. A peer review flagged that the pipeline had drifted into capability-breadth rather than applied rigor — multiple segmentation backbones, multiple trackers, VLM orchestrators generating a constant, a depth-mesh feature off the critical path. The simplify branch cut ~3,700 lines to leave a single-backbone pipeline (SAM 3 Video → head-PCA midline → ear angles → chart). See `VALIDATION.md` §"Scope reset" for the claim implications.

### Removed
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
- One-page v0 plan committed to `docs/v0-plan.md`

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
