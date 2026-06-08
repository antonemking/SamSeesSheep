# SamSeesSheep: A Measurement Pipeline for Sheep Ear-Angle Extraction from Ambient Pasture Video Using Foundation-Model Annotation

**Antone King**  
Lorewood Labs  
Middletown, Delaware, USA  
antone@lorewood.dev

---

## Abstract

The Sheep Pain Facial Expression Scale (SPFES) \cite{mclennan2019} requires trained human observers to score pain frame-by-frame from video — a bottleneck that limits the scale and reproducibility of ovine welfare research. No automated system currently exists to extract the relevant ear-angle measurements from ambient pasture video. We present SamSeesSheep, a measurement pipeline that uses a foundation video segmentation model (SAM~3 Video) as an annotation tool, a human-in-the-loop review interface, and a lightweight edge-runnable keypoint detector (YOLO-pose, 2.5M parameters, ~10 MB) trained on the reviewed output. The pipeline produces per-keypoint residual standard deviation benchmarks on genuinely held-out clips. On a clip never seen by the labeler or trainer (NCC $<$ 0.23 versus all training videos), v0.4 achieves $\sigma = 4.06^{\circ}$ (left ear) and $4.09^{\circ}$ (right ear) for ear angle — a metric derived from five head keypoints (nose, left/right ear bases, left/right ear tips) — representing a 39\% reduction in jitter from the baseline v0.2 model. Stock YOLO (yolo26n.pt) produces zero keypoints on the same clip. The training progression from 98 to 405 reviewed instances across 3 to 8 training videos demonstrates monotonically improving keypoint stability. We explicitly do not claim pain detection or welfare scoring; the pipeline is a measurement instrument designed to lower the data-collection barrier for researchers who need reproducible ear-angle features from ambient footage.

---

## 1. Introduction

Sheep welfare assessment has advanced substantially through the development of facial expression scales. The Sheep Pain Facial Expression Scale (SPFES), validated by McLennan and Mahmoud \cite{mclennan2019}, identifies five facial action units — including ear posture — that correlate with pain states in clinical conditions such as foot rot, mastitis, and post-surgical recovery. Ear posture in particular has been established as a reliable indicator of emotional valence in sheep by multiple independent research groups \cite{reefmann2009, boissy2011}. The SPFES protocol, however, requires trained human observers to score each frame of video manually. For a single 30-second clip at even a modest frame rate, this represents hundreds of individual scoring decisions. Scaling this to continuous monitoring — across multiple animals, across days — is operationally infeasible without automation.

The gap is not in the biological validity of ear-angle features. It is in the measurement infrastructure. No automated system exists that can extract ear-angle measurements from ambient pasture video with quantified, reproducible accuracy. Building such a system requires solving a sequence of practical computer vision problems: detecting sheep heads in unconstrained outdoor conditions, localizing five anatomical keypoints per head (nose, left/right ear bases, left/right ear tips), tracking them across frames, and deriving a stable ear-angle signal from the keypoint geometry. Off-the-shelf object detectors provide sheep bounding boxes on approximately 35\% of frames but produce no landmark keypoints — the ear-angle signal is literally unmeasurable without a custom-trained keypoint head.

Foundation models have recently changed the economics of annotation. SAM~3 Video \cite{ravi2025sam3}, prompted with plain English phrases ("sheep head," "sheep ear," "sheep nose"), can segment every sheep in a video clip automatically and track them across frames. This transforms the labeling task from creation (drawing bounding boxes and placing landmarks from scratch on every frame) to review (confirming or correcting machine-generated candidates). A domain expert who knows what a sheep's ear looks like — but who is not a machine learning engineer — can produce a reviewed keypoint dataset in hours rather than weeks.

We exploit this transformation to build a measurement pipeline with the following structure: SAM~3 Video auto-segments sheep in short phone-captured clips; a labeling UI presents machine-generated keypoint candidates for human review; reviewed frames export to a YOLO-pose dataset; a small model (YOLO26n-pose, 2.5M parameters) trains on a cloud GPU; and the resulting weights (~10 MB) run inference on a consumer-grade edge GPU (6 GB). The pipeline produces per-keypoint residual standard deviation ($\sigma$) benchmarks on genuinely held-out clips, evaluated against a stock-YOLO baseline that produces zero keypoints.

Our contributions are:

1. A complete, open-source pipeline that converts ambient pasture video into a reviewed sheep-head keypoint dataset using a foundation model as the annotation engine and a human as the reviewer.
2. A trained edge-runnable keypoint detector (YOLO26n-pose, 2.5M parameters, $\sim$10 MB) that places five keypoints on every detected sheep head, achieving $\sim$$4^{\circ}$ ear-angle $\sigma$ on a held-out clip.
3. A reproducible benchmark protocol using a genuinely held-out clip (NCC $<$ 0.23 versus all training videos) with per-keypoint residual $\sigma$, derived ear-angle $\sigma$, and a stock-YOLO baseline.
4. A documented training progression (v0.2 $\rightarrow$ v0.4) across 98 to 405 reviewed instances showing monotonic improvement in keypoint stability, and subsequent versions (v0.5--v0.7) demonstrating both continued gains and a regression case that illustrates the role of per-keypoint labeling consistency.
5. A measurement instrument — not a pain detector, not a welfare scorer — designed to reduce the data-collection barrier for researchers who need reproducible ear-angle features from ambient pasture footage.

We are explicit about what this work is not. It is not a pain detection system. It is not a welfare assessment tool. It has not been validated against documented stress events. It runs on a single Katahdin hair-sheep flock on one homestead in Delaware, USA, with one smartphone camera, and has been annotated by a single non-veterinarian operator. Generalization to other breeds, flocks, geographies, or clinical contexts is future work and requires independent validation. This paper describes a measurement instrument — a tool that extracts ear-angle features with quantified stability — and the pipeline that built it.

---

## 2. Related Work

### 2.1 Ovine Facial Expression Scales and Ear Posture

McLennan and Mahmoud \cite{mclennan2019} developed the Sheep Pain Facial Expression Scale (SPFES), identifying five facial action units — orbital tightening, cheek tightening, ear posture, lip and jaw tension, and nostril and philtrum shape — that trained observers can score to assess pain in sheep. Their automated SPFES system used a machine learning approach with hand-crafted features and required manually annotated facial landmarks. The ear posture action unit, in particular, classifies ear position into three categories (forward/alert, neutral, and backward/flattened), with backward ear posture being a strong indicator of pain.

Reefmann et al. \cite{reefmann2009} and Boissy et al. \cite{boissy2011} independently established ear postures as indicators of emotional valence in sheep, finding that asymmetric or backward ear positions correlate with negative affective states. These studies provide the biological grounding for ear-angle measurement as a welfare-relevant feature, but they rely on trained human observers for frame-by-frame scoring.

The critical gap we address is the absence of an automated measurement pipeline. While the SPFES literature establishes the biological validity of ear-angle features, it provides no mechanism for extracting those features from ambient video at scale. The measurement step remains a manual, human-intensive bottleneck.

### 2.2 Foundation Models for Agriculture

The Segment Anything Model (SAM) family \cite{kirillov2023segment, ravi2024sam2, ravi2025sam3} has demonstrated remarkable zero-shot segmentation capabilities. SAM~3 Video extends this to the temporal domain, tracking objects across frames from text or click prompts. In agricultural applications, foundation models have been applied to crop monitoring, weed detection, and livestock identification \cite{xu2023livestock, wang2024agriculture}. However, most agricultural CV work uses foundation models for direct inference rather than as annotation tools for training domain-specific edge models. Our pipeline inverts this: SAM~3 Video is the annotation backbone, producing the labeled data that trains a lightweight deployment model.

### 2.3 Keypoint Detection and YOLO-Pose

YOLO-pose \cite{maji2022yolopose} extends the YOLO object detection architecture with a keypoint regression head, enabling simultaneous bounding box detection and keypoint localization in a single forward pass. The YOLO26n-pose variant (2.5M parameters) is particularly well-suited for edge deployment, fitting within the memory constraints of consumer GPUs. Keypoint detection has been applied to animal pose estimation in laboratory and controlled-farm settings \cite{mathis2018deeplabcut, graving2019deepposekit, pereira2022sleap}, but these tools assume either manual annotation workflows or controlled capture conditions. Our work differs in targeting ambient, unconstrained pasture video with a foundation-model-assisted annotation pipeline.

### 2.4 Precision Livestock Farming (PLF)

Precision livestock farming systems \cite{berckmans2014precision} use sensors and computer vision for continuous animal monitoring, targeting metrics such as feeding behavior, locomotion, and social interaction. Commercial PLF platforms (Cainthus, Connecterra) operate at enterprise scale with proprietary models and SaaS pricing. Our work occupies a different niche: an open-source, solo-operator-scale pipeline that demonstrates the feasibility of building a domain-specific keypoint detector from scratch using foundation-model annotation, with explicit quantification of measurement stability.

### 2.5 The Gap

No existing system provides an end-to-end pipeline that (a) uses a foundation model for automated sheep-head keypoint annotation, (b) incorporates a human-in-the-loop review step, (c) exports a standard-format dataset for training an edge-runnable model, and (d) publishes reproducible held-out benchmarks with per-keypoint $\sigma$, derived ear-angle $\sigma$, and a stock-model baseline. This is the gap we fill.

---

## 3. Method

### 3.1 Pipeline Overview

The SamSeesSheep pipeline consists of five stages connected by well-defined data-format contracts. Figure~\ref{fig:pipeline} (see the project ARCHITECTURE.md for the full diagram) summarizes the flow:

```
Phone capture (1080p, 15--30 s, .MOV)
  → Frame extraction (2 fps, ~512 px max dimension)
    → SAM~3 Video text-prompted segmentation
      → Keypoint derivation from masks
        → Human review in labeling UI
          → YOLO-pose dataset export
            → Cloud GPU training (RTX 4090, ~6 min)
              → Edge inference + σ benchmark (GTX 1660 Ti, 6 GB)
```

### 3.2 Video Capture and Frame Extraction

Video clips are captured with a smartphone at 1080p resolution, 30 fps, typically 15–30 seconds in duration. Frames are extracted at 2 fps using PyAV with a maximum dimension of 512 px (maintaining aspect ratio). The 2 fps sampling rate and 512 px resolution are chosen to keep SAM~3 Video within GPU memory budget (24 GB required for the full three-session pipeline; 6 GB for the reduced two-session variant) while providing sufficient temporal resolution for the sheep-head labeling task, where head pose changes slowly.

### 3.3 SAM~3 Video Segmentation and Keypoint Derivation

SAM~3 Video (`facebook/sam3` on Hugging Face, approximately 3 GB model size) is called with three text prompts, each run as a separate propagation session:

- Session 1: "sheep head" → head mask per instance per frame
- Session 2: "sheep ear" → left and right ear masks per instance per frame
- Session 3: "sheep nose" → nose mask per instance per frame

Session 3 (nose) requires a 24 GB GPU. On a 6 GB GPU, the pipeline drops to two sessions (head + ear) and nose keypoints fall back to head-mask geometry. SAM~3's internal tracker handles multiple instances natively, producing independent tracks for each sheep in multi-animal frames.

From the segmentation masks, five keypoint candidates are derived per instance per frame:

| Slot | Name | Derivation |
|------|------|------------|
| 0 | Nose tip | Centroid of nose mask, or anterior point of head mask when no nose mask is available |
| 1 | Left ear base | Point on left ear mask closest to head mask centroid |
| 2 | Right ear base | Point on right ear mask closest to head mask centroid |
| 3 | Left ear tip | Point on left ear mask farthest from head mask centroid |
| 4 | Right ear tip | Point on right ear mask farthest from head mask centroid |

Keypoints are stored in image-space pixel coordinates. Left/right is from the camera's perspective. Each keypoint carries a visibility flag ($v$): $v=0$ (absent — mask missing, part not in frame), $v=1$ (auto-placed by SAM~3, unreviewed), or $v=2$ (human-reviewed, confirmed or hand-corrected).

### 3.4 Labeling UI and Review Protocol

The labeling interface is a FastAPI backend serving a vanilla JavaScript frontend. For each frame, the reviewer sees the SAM~3-derived keypoint candidates overlaid on the source image. Controls: `A` accepts all keypoints for the current instance, dragging individual dots corrects placement, `Enter` saves and advances. The interface supports multi-instance frames (schema v2: `instances[]` per frame), allowing review of all sheep in a single clip pass.

A single operator (the project author, who is not a veterinarian) performed all reviews. Each video was reviewed in a single session. Reviewed keypoints are persisted to `review.json` files under `data/labels/{video_id}/`. An instance with any $v=2$ keypoint counts as "reviewed"; an instance with all five keypoints at $v=2$ counts as "fully reviewed." Partially reviewed instances (some keypoints out of frame or occluded) export with only the reviewed slots carrying signal.

### 3.5 YOLO-Pose Export and Training

Only $v=2$ keypoints enter the training dataset. The export pipeline produces YOLO-pose format: one `.txt` per source image, one line per instance, with class index 0 (`sheep_head`), normalized bounding box center and dimensions, and five normalized keypoint triples $(x, y, v)$. The `data.yaml` declares a single class with `kpt_shape: [5, 3]` and `flip_idx: [0, 2, 1, 4, 3]`, enabling horizontal-flip augmentation that correctly swaps left and right ear keypoints.

The train/validation split uses deterministic MD5 hash bucketing: for each `(video_id, frame_idx)` pair, `bucket = int(md5("{video_id}:{frame_idx}")[:8], 16) % 10000 / 10000.0`. Frames with `bucket < 0.2` go to the validation set; the remainder go to training. This is deterministic, reproducible, and per-frame (not per-video), meaning frames from the same video can appear in both train and val — appropriate for a single-flock, single-camera scope where generalization to unseen poses from known scenes is the target.

Training uses YOLO26n-pose (2.5M parameters) with 100 epochs, batch size 8, image size 640 px, on an NVIDIA RTX 4090 (24 GB) cloud GPU via RunPod. Training completes in approximately 6 minutes. The resulting `best.pt` is approximately 10 MB.

### 3.6 Edge Inference and Benchmark Protocol

Inference runs on the trained YOLO-pose model on a local NVIDIA GTX 1660 Ti (6 GB). The benchmark protocol, implemented in `sheep-yolo/scripts/bench_held_out.py`, evaluates keypoint stability on a genuinely held-out clip:

1. **Clip selection**: A clip whose normalized cross-correlation (NCC) with every training video is below 0.23 — ensuring no near-duplicate frames have leaked into the training set.
2. **Target window**: A 5-second segment where a target sheep's head is approximately stationary (head centroid standard deviation $\sim$30 px). An ROI filter excludes neighbouring sheep.
3. **Residual $\sigma$**: $\sigma_{\text{residual}} = \text{std}(\text{kpt} - \text{rolling\_median}_7(\text{kpt}))$. The rolling median of 7 frames strips slow head drift (the sheep's actual motion, consistent across all models) and isolates the frame-to-frame jitter that matters for welfare estimation.
4. **Raw $\sigma$**: $\text{std}(\text{kpt})$ over the window, dominated by the sheep's head sway ($\sim$45 px). Serves as a sanity check that all models track the same physical motion.
5. **Ear-angle $\sigma$**: Derived from keypoints via head-midline PCA on the head mask with ear-midpoint disambiguation for direction. Ear angle is computed as $\text{arctan2}$ of the ear direction vector relative to the head midline.
6. **Detection rate**: Any YOLO detection (confidence $\geq$ 0.25) within the ROI.
7. **Stock baseline**: `yolo26n.pt` (COCO-pretrained) run on the same clip, reporting detection count and keypoint count (expected: zero keypoints).

---

## 4. Dataset and Experimental Setup

### 4.1 Dataset Characteristics

The dataset comprises video clips of a single Katahdin hair-sheep flock (5 ewes plus lambs) on a homestead pasture in Middletown, Delaware, USA. Clips were captured with a single smartphone (iPhone, 1080p, 30 fps) under ambient outdoor lighting — varying sun angles, cloud cover, and shadow conditions across recording sessions spanning April–May 2026. No controlled lighting, no fixed camera mount. This is the ambient condition the measurement instrument is designed for.

### 4.2 Labeling Progression

Table~\ref{tab:dataset} summarizes the dataset growth across versions:

| Version | Reviewed instances | Training videos | val mAP50-95 (pose) | Residual $\sigma$ (mean, 5 kpts, px) |
|---------|-------------------|-----------------|---------------------|--------------------------------------|
| v0.2 | 98 | 3 | 0.479 | 10.89 |
| v0.3 | 313 | 6 | 0.643 | 8.90 |
| v0.4 | 405 | 8 | 0.732 | 7.70 |

The v0.2 baseline was trained on 3 video clips (98 reviewed instances). v0.3 added 3 new videos (215 additional instances). v0.4 added 2 more videos (92 additional instances). Each version is cumulative — it includes all reviewed frames from all previous versions plus newly labeled clips. The same model architecture (YOLO26n-pose), hyperparameters (100 epochs, batch 8, imgsz 640), and training infrastructure (RunPod RTX 4090) were used for all versions. The only variable is labeled-data scale and scene diversity.

### 4.3 Held-Out Clip

The held-out clip, `IMG_3651.MOV`, was filmed on the same pasture with the same flock but was never pushed to the labeling pod, never reviewed by the operator, and has NCC $<$ 0.23 versus every training video. The target window spans frames 367--521 (5.17 seconds at 30 fps), with a single dominant foreground sheep (ByteTrack ID 34, head size approximately 234 $\times$ 169 px, confidence 0.6+). The ROI `(1220, 80, 1560, 390)` is padded approximately half a head around the centroid range to exclude neighbouring sheep. The sheep's head centroid standard deviation within the window is approximately 30 px in both x and y — not perfectly still, but the calmest 5-second continuous segment available with a large foreground head.

### 4.4 Stock YOLO Baseline

`yolo26n.pt` (COCO-pretrained, identical backbone to the trained models but without a sheep-keypoint head) was run on the same held-out clip with confidence threshold 0.25. The model produces COCO-class `sheep` bounding box detections on 323 of 933 frames ($\sim$35\%), but zero keypoints — the model has no keypoint head for sheep anatomy. Ear angle is unmeasurable.

---

## 5. Results

### 5.1 Detection and Keypoint Stability

Table~\ref{tab:headline} reports the primary results on the held-out clip for all three model versions plus the stock baseline.

| Metric | Stock YOLO | v0.2 | v0.3 | v0.4 |
|--------|------------|------|------|------|
| Any detection (full clip, 933 frames) | 323 (35\%) | 909 (97\%) | 931 (99.8\%) | 933 (100\%) |
| In-ROI detection (window, 155 frames) | 0 kpts | 145 (94\%) | 150 (97\%) | 149 (96\%) |
| Residual $\sigma$ (mean, 5 kpts, px) | — | 10.89 | 8.90 | 7.70 |
| Raw $\sigma$ (mean, 5 kpts, px) | — | 49.44 | 46.90 | 46.60 |

All three trained models detect the target sheep on 94--97\% of window frames. The detection ceiling is high from v0.2 onward because this particular window features a single dominant foreground sheep — a relatively easy detection task compared to the crowded multi-sheep scenes that exposed v0.2's detection weakness in earlier benchmarks \cite{v03benchmark}. The keypoint stability story is where the training-scale signal lives.

Residual $\sigma$ drops monotonically with training scale: 10.89 px (v0.2) $\rightarrow$ 8.90 px (v0.3, $-18\%$) $\rightarrow$ 7.70 px (v0.4, $-13\%$ from v0.3, $-29\%$ from v0.2). Raw $\sigma$ remains flat at $\sim$46--49 px across all three models, confirming that the sheep's physical head sway dominates the raw measurement and that the residual metric correctly isolates model noise.

### 5.2 Per-Keypoint Residual $\sigma$

Table~\ref{tab:perkpt} reports residual $\sigma$ for each of the five keypoints.

| Keypoint | v0.2 (px) | v0.3 (px) | v0.4 (px) | $\Delta$ v0.2$\rightarrow$v0.4 |
|----------|-----------|-----------|-----------|--------------------------------|
| Nose | 11.02 | 9.96 | 8.00 | $-27\%$ |
| Left ear base | 9.30 | 6.16 | 6.03 | $-35\%$ |
| Right ear base | 9.09 | 7.73 | 7.03 | $-23\%$ |
| Left ear tip | 12.63 | 8.41 | 7.66 | $-39\%$ |
| Right ear tip | 12.39 | 12.26 | 9.78 | $-21\%$ |
| **Mean** | **10.89** | **8.90** | **7.70** | **$-29\%$** |

v0.4 improves on every keypoint. The largest gains are on ear tips — the keypoints that v0.2 historically struggled to place anatomically, often landing ear tips in mid-air rather than on the ear itself. Ear tips are the most geometrically distal keypoints from the head centroid and are therefore the most sensitive to mask quality from SAM~3 and to the reviewer's anatomical judgment. v0.4's ear-tip $\sigma$ of 7.66--9.78 px on a 234-px-wide head represents approximately 3.3--4.2\% of head width — a substantial improvement from v0.2's 5.3--5.4\%.

### 5.3 Ear-Angle Stability

Table~\ref{tab:earangle} reports the derived ear-angle residual $\sigma$, computed from keypoints via the head-midline PCA geometry described in Section~3.6. This is the welfare-relevant scalar: a stationary sheep should produce a flat ear-angle trace, and the deviation from flatness is the measurement noise floor.

| Version | Left ear $\sigma$ (°) | Right ear $\sigma$ (°) |
|---------|----------------------|------------------------|
| v0.2 | 6.71 | 6.07 |
| v0.3 | 4.82 | 4.21 |
| v0.4 | 4.06 | 4.09 |

v0.4 achieves approximately $4^{\circ}$ ear-angle jitter on both ears — a 39\% reduction (left) and 33\% reduction (right) from v0.2. For reference, the SPFES ear-posture classification bands span $40^{\circ}$ between the "alert" ($\geq 30^{\circ}$) and "down/back" ($\leq -10^{\circ}$) thresholds \cite{mclennan2019}. A $4^{\circ}$ noise floor is approximately 10\% of that decision range, meaning model noise alone is unlikely to trigger false band crossings in the absence of genuine ear movement.

The stock YOLO baseline produces zero keypoints and therefore zero ear-angle measurements — the metric is not merely noisy but unmeasurable without a custom-trained keypoint head.

### 5.4 Training Progression and v0.7 Qualitative Result

Subsequent versions (v0.5--v0.7) continued the labeling flywheel. v0.5 achieved left ear $\sigma = 3.66^{\circ}$ and right ear $\sigma = 4.30^{\circ}$ on the same held-out clip. v0.6 demonstrated a mixed result: right ear $\sigma = 3.55^{\circ}$ (best recorded) but left ear $\sigma = 4.65^{\circ}$ (regression to near-v0.2 levels), highlighting that per-keypoint labeling consistency — not just instance count — drives measurement stability. v0.7 recovered left ear $\sigma$ to $3.70^{\circ}$ (right ear $4.46^{\circ}$) and introduced a multi-sheep flock ear-angle monitor that tracks six ewes simultaneously with one live ear-angle lane each. This is a qualitative demonstration of the pipeline's scalability: the same 2.5M-parameter model, trained on the same single-flock dataset, tracks every sheep facing the camera at once with per-animal ear-angle time series.

### 5.5 v0.4 on Held-Out versus In-Distribution

v0.4's held-out residual $\sigma$ of 7.70 px approaches v0.3's in-distribution numbers on its own training clips (4.10--5.92 px, reported in the v0.3 benchmark \cite{v03benchmark}). The gap of approximately 1.8--3.6 px between in-distribution and held-out performance represents the generalization cost for an unseen scene, and the fact that this gap is quantifiable at all is a result that was unavailable from the earlier within-distribution benchmarks alone.

---

## 6. Discussion

### 6.1 What the Numbers Mean

The headline result — approximately $4^{\circ}$ ear-angle $\sigma$ on a held-out clip — establishes a noise floor for automated ear-angle measurement from ambient pasture video. Whether $4^{\circ}$ is "good enough" depends entirely on the application. For detecting gross ear-position changes (e.g., a sustained shift from $+30^{\circ}$ alert to $-10^{\circ}$ down/back, spanning a $40^{\circ}$ range), $4^{\circ}$ jitter is 10\% of the decision range and unlikely to generate false positives. For detecting subtle within-band changes (e.g., a $5^{\circ}$ shift within the neutral zone), $4^{\circ}$ jitter would drown the signal. The appropriate use of this instrument is therefore for coarse temporal patterns — sustained ear-position state changes over minutes, not frame-to-frame micro-movements.

The stock YOLO baseline is instructive. A practitioner hoping to extract ear-angle measurements from off-the-shelf models would obtain exactly zero data points. The keypoint head must be grown against the target animals — there is no pretrained shortcut.

### 6.2 Limitations

The following limitations, drawn from the project's VALIDATION.md contract, constrain the scope of any claim made on the basis of these results:

**Single flock, single breed, single geography.** The dataset comprises approximately 5 Katahdin hair-sheep ewes (plus lambs) on one homestead in Middletown, Delaware. No claim about these animals transfers to other breeds (wool sheep, horned breeds, breeds with cropped or floppy ears), other geographies, or other management conditions without independent validation.

**Single operator, single camera.** One non-veterinarian operator (the author) captured all video with one smartphone and performed all keypoint reviews. There is no inter-annotator agreement measurement, no independent ground truth, and no vet-validated keypoint placement. The operator's labeling consistency is measured only indirectly through the training progression — the fact that more data produces more stable models suggests consistency, but does not prove anatomical accuracy.

**Single held-out clip.** The benchmark protocol uses one held-out clip. A multi-clip ablation (3--5 clips spanning varied lighting, sheep count, and motion conditions) would be more defensible and is deferred to future work.

**Ambient-vs-clinical gap.** The SPFES ear-posture thresholds \cite{mclennan2019} that inform the ear-angle classification bands were validated in clinical pain conditions (foot rot, mastitis, post-surgical recovery) with trained veterinary observers. Generalizing those thresholds to ambient pasture observation — where sheep exhibit ear movements for reasons unrelated to pain (vigilance, social signaling, insect avoidance, wind response) — is an unresolved scientific question. Our measurement instrument can extract ear angles from ambient video; it cannot tell you what those angles mean.

**The measurement is not the diagnosis.** Ear angle is one geometric feature. Welfare is a multi-dimensional construct requiring veterinary assessment \cite{berckmans2014precision}. Pain is a clinical determination requiring multiple indicators including body condition, locomotion, feeding behavior, isolation, and vocalization. This pipeline measures one feature. It is not a welfare instrument.

### 6.3 Known Failure Modes

The following failure modes have been observed or are anticipated based on the pipeline design:

- **Dark fleece / low contrast**: SAM~3 segmentation degrades on black-faced sheep at dusk, and the trained YOLO-pose model inherits this weakness from its training data (which contains only light-fleeced Katahdin sheep).
- **Motion blur**: Handheld phone capture of moving sheep is the median case, not the edge case. Heavy motion blur degrades segmentation quality and keypoint stability.
- **Extreme angles**: The pipeline performs best on frontal and three-quarter views. Profile views and views where one ear is fully occluded reduce keypoint availability.
- **Occlusion**: In crowded frames, sheep occlude each other. The YOLO-pose model may detect the wrong sheep or place keypoints on occluded anatomy.
- **Wool-covered ears**: Heavy fleece can obscure ear shape entirely. The pipeline has not been tested on wool breeds.

### 6.4 What Validation Against Stress Events Would Require

The follow-up project that would turn this measurement instrument into a welfare-relevant tool requires: (a) documented stress events with timestamps (hoof trimming, tagging, separation, transport), (b) continuous pre- and post-event monitoring with the trained model, (c) a within-animal delta analysis (comparing each animal's ear-angle distribution before versus after the event), and (d) a pre-registered kill criterion — the project's VALIDATION.md specifies that if fewer than 70\% of documented stress events show a measurable change in ear-angle features, the welfare project terminates and the write-up of what failed becomes the deliverable. This paper describes the measurement instrument. That follow-up project would determine whether the measurements are welfare-informative.

### 6.5 Negative Results as Features

This work deliberately publishes negative results. The stock YOLO baseline (zero keypoints, 35\% detection) establishes that off-the-shelf models are not a viable shortcut. The v0.6 regression (right ear at all-time best 3.55° but left ear regressing to 4.65°) demonstrates that more data does not monotonically improve all keypoints — per-keypoint coverage and per-session labeling consistency matter at least as much as instance count. The v0.3 benchmark correction (admitting that the "held-out" clips were actually in v0.3's training distribution) is published alongside the corrected v0.4 benchmark \cite{v03benchmark}. These are not failures to suppress; they are evidence that the measurement methodology is working as designed.

---

## 7. Conclusion

We have presented SamSeesSheep, a measurement pipeline that converts ambient pasture video into quantified ear-angle features using a foundation-model annotation flywheel. The pipeline produces a small edge-runnable model (2.5M parameters, $\sim$10 MB) that places five keypoints on every detected sheep head and achieves approximately $4^{\circ}$ ear-angle residual $\sigma$ on a genuinely held-out clip — a metric that is literally unmeasurable with off-the-shelf object detectors. The training progression from 98 to 405 reviewed instances demonstrates monotonic improvement in keypoint stability, and the benchmark protocol provides a reproducible methodology for quantifying measurement quality.

We are explicit about the boundary: this is a measurement instrument, not a pain detector or welfare scorer. It measures ear-angle features with quantified stability. Whether those features are welfare-informative is a question for the follow-up validation study that this instrument is designed to enable.

Future work includes: (a) multi-clip held-out benchmarks spanning varied lighting, sheep count, and motion conditions; (b) cross-flock generalization testing on wool breeds and different geographies; (c) validation against documented stress events with the kill criterion described in Section~6.4; (d) inter-annotator agreement measurement; and (e) continuous monitoring deployment at water troughs or handling chutes, where sustained ear-position patterns — rather than single-frame snapshots — become the unit of measurement.

The code, dataset exports, trained weights, benchmark scripts, and full documentation are available under the MIT License at \texttt{https://github.com/antonemking/SamSeesSheep}.

---

## Acknowledgments

The author thanks the Katahdin ewes of Middletown, Delaware, for their patience during repeated filming sessions, and the open-source computer vision community for the foundation models and training infrastructure that made a solo-operator pipeline feasible.

---

## References

\bibitem{mclennan2019}
K.~M. McLennan and M.~Mahmoud.
Development of an automated pain facial expression detection system for sheep (SPFES).
\emph{Animals}, 9(4):196, 2019.

\bibitem{reefmann2009}
N.~Reefmann, F.~B.~Kasz{\`a}s, B.~Wechsler, and L.~Gygax.
Ear and tail postures as indicators of emotional valence in sheep.
\emph{Applied Animal Behaviour Science}, 118(3--4):199--207, 2009.

\bibitem{boissy2011}
A.~Boissy, A.~Aubert, L.~D{\'e}sir{\'e}, L.~Greiveldinger, E.~{Delval}, and I.~Veissier.
Cognitive sciences to relate ear postures to emotions in sheep.
\emph{Animal Welfare}, 20(1):47--56, 2011.

\bibitem{kirillov2023segment}
A.~Kirillov, E.~Mintun, N.~Ravi, H.~Mao, C.~Rolland, L.~Gustafson, T.~Xiao, S.~Whitehead, A.~C.~Berg, W.-Y.~Lo, P.~Doll{\'a}r, and R.~Girshick.
Segment anything.
In \emph{Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}, 2023.

\bibitem{ravi2024sam2}
N.~Ravi, V.~Gabeur, Y.-T.~Hu, R.~Hu, C.~Ryali, T.~Ma, H.~Khedr, R.~R{\"a}dle, C.~Rolland, L.~Gustafson, E.~Mintun, J.~Pan, K.~V.~Alwala, N.~Carion, C.-Y.~Wu, P.~Doll{\'a}r, and C.~Feichtenhofer.
SAM~2: Segment anything in images and videos.
\emph{arXiv preprint arXiv:2408.00714}, 2024.

\bibitem{ravi2025sam3}
N.~Ravi, V.~Gabeur, Y.-T.~Hu, R.~Hu, C.~Ryali, T.~Ma, H.~Khedr, R.~R{\"a}dle, C.~Rolland, L.~Gustafson, E.~Mintun, J.~Pan, K.~V.~Alwala, N.~Carion, C.-Y.~Wu, P.~Doll{\'a}r, and C.~Feichtenhofer.
SAM~3: Segment anything with temporal prompting.
\emph{arXiv preprint}, 2025.

\bibitem{maji2022yolopose}
D.~Maji, S.~Nagori, M.~Mathew, and D.~Poddar.
YOLO-pose: Enhancing YOLO for multi person pose estimation using object keypoint similarity loss.
In \emph{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops}, 2022.

\bibitem{mathis2018deeplabcut}
A.~Mathis, P.~Mamidanna, K.~M.~Cury, T.~Abe, V.~N.~Murthy, M.~W.~Mathis, and M.~Bethge.
DeepLabCut: markerless pose estimation of user-defined body parts with deep learning.
\emph{Nature Neuroscience}, 21(9):1281--1289, 2018.

\bibitem{graving2019deepposekit}
J.~M.~Graving, D.~Chae, H.~Naik, L.~Li, B.~Koger, B.~R.~Costelloe, and I.~D.~Couzin.
DeepPoseKit, a software toolkit for fast and robust animal pose estimation using deep learning.
\emph{eLife}, 8:e47994, 2019.

\bibitem{pereira2022sleap}
T.~D.~Pereira, N.~Tabris, A.~Matsliah, D.~M.~Turner, J.~Li, S.~Ravindranath, E.~S.~Papadoyannis, E.~Normand, D.~S.~Deutsch, Z.~Y.~Wang, et al.
SLEAP: A deep learning system for multi-animal pose tracking.
\emph{Nature Methods}, 19(4):486--495, 2022.

\bibitem{berckmans2014precision}
D.~Berckmans.
Precision livestock farming technologies for welfare management in intensive livestock systems.
\emph{Revue Scientifique et Technique (International Office of Epizootics)}, 33(1):189--196, 2014.

\bibitem{wang2024agriculture}
Y.~Wang, Z.~Chen, J.~Li, and S.~Liu.
Foundation models in agriculture: a survey.
\emph{Computers and Electronics in Agriculture}, 217:108556, 2024.

\bibitem{xu2023livestock}
B.~Xu, W.~Wang, L.~Guo, G.~Chen, Y.~Li, Z.~Cao, and S.~Wu.
Livestock monitoring with computer vision: a review.
\emph{Computers and Electronics in Agriculture}, 214:108286, 2023.

\bibitem{v03benchmark}
A.~King.
sheep-pose v0.2 vs v0.3 — comparison benchmark.
SamSeesSheep repository, \texttt{docs/v0.3-benchmark.md}, 2026.
\url{https://github.com/antonemking/SamSeesSheep}

---

\emph{This work was conducted on a single Katahdin flock in Middletown, Delaware, using one smartphone camera and one human reviewer (the author, who is not a veterinarian). All claims are bounded by the scope described in the project's VALIDATION.md contract. No pain detection. No welfare scoring. A measurement instrument.}
