# VALIDATION

**This document is the contract.** It states what `SamSeesSheep` measures, what it does not measure, and how confident you should be in either. It is the first artifact in this repository because every other artifact's claims should be readable through this lens.

This document will be updated as evidence accumulates. It will not be updated to make the project look better than it is.

---

## Scope (updated 2026-04-20)

The project went through two scope shifts since day 0. This section supersedes anything below it.

**What this repo now is:** a labeling + training pipeline for sheep head keypoints. SAM 3 Video auto-places candidate keypoints (nose, left/right ear bases, left/right ear tips) across every frame of a short clip. A human reviewer confirms or corrects them. Reviewed frames export to a YOLO-pose dataset. A small YOLO-pose model is trained against that dataset on a cloud GPU. The trained weights land in a separate repo (`sheep-yolo`) that owns inference and the σ-on-motionless-sheep benchmark.

**What this repo explicitly is not:**

- A welfare instrument. Ear-angle thresholds are literature-grounded reference bands, not diagnostic cutoffs.
- A pain detector. No claim here translates into "this sheep is in pain."
- A validated dataset against stress events. That is the follow-up project, with its own capture protocol and kill criterion.
- The inference path. `best.pt` leaves this repo and runs elsewhere.

**What's explicitly deferred:**

- Validation against documented stress events (hoof trim, tagging, separation, startle).
- EUP% as a pass/fail metric against the kill criterion below — that test gates on a trained YOLO-pose model producing stable measurements, which is downstream of this repo.
- Any welfare, pain, or cross-flock claim.

**How to read the kill criterion below.** It was written day 0 and still stands, but the subject of the test is now the YOLO-pose model trained on data produced here — not SAM 3's per-frame segmentations and not the ear-angle chart. The chart is a labeling-QA tool; it isn't the thing under evaluation.

### Scope history

- **v0 (day 0)** — Intended as a welfare-scoring pipeline against SPFES thresholds. Unrealistic at our evidence level.
- **v0.1 (2026-04 simplify branch)** — Descoped to "visualization artifact": SAM 3 Video → per-frame ear angles → smoothed EKG chart. No welfare claim. Clean but narrow.
- **v0.2 (2026-04 current)** — Re-scoped again after peer review. SAM 3 is the **annotation backbone** (not the inference backbone). The project's outputs are (a) a growing reviewed-keypoint dataset and (b) a trained YOLO-pose model small enough to run at the edge. The visualization chart persists as a labeling-QA view.

---

## TL;DR

| Aspect | Trustworthy? |
|---|---|
| **Anything at all** (as of day 0) | **Not yet measured.** This document is the day-0 skeleton. |
| **Pain detection** | **No.** This project does not detect pain. It will, at best, detect *ear-position features* whose threshold values come from the published welfare literature. |
| **Welfare scoring** | **No.** "Welfare" is a multi-feature construct that requires veterinary assessment. This project measures one geometric feature on one species under one set of conditions. |
| **Generalization to other flocks / breeds / farms** | **No.** v0 runs on five sheep on one Delaware homestead. Any claim that generalizes beyond that requires evidence this project does not yet have. |
| **Pairwise comparison** (same animal, same camera, same lighting, before vs. after a documented event) | **Pending.** This is the only claim space the v0 design can hope to support. |

---

## What this project claims to measure

**One thing**: the geometric position of a sheep's ears in a single frame, expressed as an aggregate metric over a rolling time window.

The metric is **EUP% (Ear-Up Percentage)** — the fraction of frames in a rolling 60-minute window in which the animal's ears are detected in the "up / alert" position vs. any other position.

That is the entire claim. Not pain. Not welfare. Not stress. Ear-position percentage over a time window.

The ear-position thresholds and the up/down classification will be drawn from the published literature on ovine ear posture as a welfare/affect signal — *not* invented by this project. References will be added to this document as they are used.

## What this project explicitly does NOT claim to measure

- **Pain.** The published grimace-scale literature on sheep (SPFES) was validated by trained veterinary observers in clinical pain conditions: foot rot, mastitis, post-surgical recovery. Generalizing those facial expressions to ambient pasture observation is a real and unresolved gap. Even if EUP% correlates with documented stress events on this homestead, that is not a pain claim.
- **Welfare** as a construct. Welfare is multi-dimensional and requires expert veterinary assessment. EUP% is one signal that may correlate with one dimension under some conditions.
- **Affect, mood, emotion.** These are anthropomorphizations that the project will not use even informally.
- **Disease.** EUP% is not a clinical screening tool. Off-feed, lameness, isolation from herd, vocalization, body condition score, and direct veterinary examination are the actual disease indicators.
- **Cross-flock generalization.** v0 runs on five sheep of one breed on one homestead with one camera in one geography. No claim made on these five animals transfers to other animals without independent validation.
- **Cross-breed generalization.** Wool sheep, hair sheep, breeds with cropped ears, breeds with floppy ears, breeds with horned heads occluding the ear — all are independent generalization questions.
- **Goats.** This project is sheep-only. Goats are not a sub-case of sheep; they are independent foragers with different facial musculature, different ear morphology, and different behavioral baselines. A project on goats would be a separate VALIDATION.md.
- **Velocity, kinematics, gait, body posture, vocalization.** Not measured. Out of scope for v0.

## What kind of dataset this is

| Aspect | v0 Reality |
|---|---|
| **Animals** | 5 sheep |
| **Breeds** | (to be filled in Weekend 1) |
| **Geography** | One pasture, Delaware, USA |
| **Camera** | One smartphone, handheld |
| **Lighting conditions** | Whatever the sky is doing on the day of capture |
| **Seasons covered** | (whichever weekend the data is collected) |
| **Time of day covered** | (to be documented) |
| **Annotators** | One: the project author, who is not a veterinarian |
| **Independent ground truth** | None (v0). Any vet-validated ground truth is a v1 milestone. |

**What this dataset can support a claim about**: "On these five animals, on this homestead, on these days, under these lighting conditions, the system extracted ear-position features at the following accuracy."

**What this dataset cannot support a claim about**: anything that uses the word "sheep" without the qualifiers above.

## Known and anticipated failure modes

These are predictions, not measurements. They will be validated or invalidated and moved to the appropriate section as evidence comes in.

- **Black-faced or dark-fleeced sheep at dusk** — segmentation contrast falls off, foundation models likely degrade
- **Wool-covered ears** — heavy fleece can occlude the ear shape entirely
- **Horns / cropped ears** — geometric features the published literature assumed are present may be absent or distorted
- **Motion blur** — handheld phone capture of moving animals is the median case, not the edge case
- **Extreme angles** — published grimace scales assume frontal or profile views; pasture footage will not provide that
- **Single-frame snapshots vs. continuous observation** — temporal smoothing matters for EUP%; a single frame is not the unit of measurement
- **Anthropomorphic projection by the annotator** (me) — I am not a vet. My labels are subject to projection bias and will be flagged as such
- **The ambient-vs-clinical gap** — the literature these thresholds are drawn from was developed in clinical pain conditions; generalizing to ambient pasture observation is unverified by definition

## How the numbers will be reported

When this project reports a number, it will report it with:

1. **Units**: EUP% has the unit "percent"; segmentation failure rate has the unit "percent"; etc.
2. **The dataset slice it was measured on**: which animals, which days, which lighting, which time-of-day window
3. **The comparison baseline**: against what was the number measured (random? human annotator? a different time window for the same animal?)
4. **The claim type**: whether this is a *delta* (within-animal change) or an *absolute* (cross-animal claim)

Numbers without all four of those will not be reported. Period.

This follows the **trust deltas, not absolutes** principle: pairwise comparisons within the same animal under the same camera and same conditions cancel out the largest sources of systematic error in single-camera CV systems. A within-animal "EUP% dropped 30% during the documented isolation event" is a defensible delta. An across-animal "average sheep EUP% is 65%" is a population claim this dataset cannot support.

## What HAS been validated

| Item | Method | Result |
|---|---|---|
| (nothing yet — this is day 0) | — | — |

## What is being validated next (Weekend 1 deliverables)

| Item | Method | Status |
|---|---|---|
| SAM 2.1 segmentation success rate on sheep faces in real pasture conditions | 30 phone photos across varied lighting/angles/distances; manual scoring | Pending |
| Annotator baseline (me) for ear-up vs. ear-down on still frames | 100 frames double-labeled with 1-week gap; intra-rater agreement | Pending |
| Behavioral baseline observation of the 5 sheep | 2 hours unassisted notebook observation | Pending |

## Anti-overclaim commitments

This project commits, structurally, to the following anti-overclaim norms. Each one will be enforced via the README and this document, and any deviation in code or documentation should be reported as an issue.

1. **No use of the word "pain"** outside of references to the published literature. The system does not detect pain.
2. **No use of the word "welfare"** as a measured outcome. Welfare is a vet-assessed construct.
3. **No emoji-style emotional labels** ("happy," "sad," "stressed") on individual animals. Use the metric name.
4. **No cross-flock claims** without an explicit second-flock dataset and a separate row in the validated-items table above.
5. **No marketing copy** that contradicts this document. If marketing copy contradicts this document, the copy is wrong.
6. **All numbers reported with units, slice, baseline, and claim type** as described above.
7. **Negative results published.** A killed v0 is a deliverable, not a failure to suppress.

## Who should NOT use this

- Anyone who needs a clinical pain diagnosis. See your veterinarian.
- Anyone running a commercial flock who needs validated welfare assessment. PLF tools designed for that exist; see Berckmans et al. and the European PLF literature.
- Anyone whose decisions about animal care would be substantially driven by an unreviewed metric from a phone app. The point of this project is to find out whether the metric is meaningful, not to act on it before the answer is known.

## Who this is for

- A small-flock owner who wants to understand how foundation-model CV is being applied to animal welfare and what the honest limits look like
- A computer vision practitioner who is curious whether the foundation-model commodity layer is real for animal applications
- A precision livestock farming researcher looking for a cleanly documented negative result on sheep grimace operationalization (if v0 doesn't clear)
- The author's future self, six months from now, looking to remember exactly what the v0 dataset could and could not support

---

## Document version

**v0.2.0** — Scope reset. Pipeline simplified to SAM 3 Video (head + ear) with head-PCA midline and SPFES threshold bands. Photo flow, SAM 2.1, depth/mesh, VLM orchestrators, and narrative generation have been removed from the critical path (see `CHANGELOG.md`). Validation against documented stress events is now an explicit future-work item, not a v0 deliverable.

**v0.1.0** — Pipeline built. SAM 2.1 hiera-small segmentation + Depth Anything V2 depth mesh operational. Ear angle extraction via PCA with McLennan SPFES thresholds. No systematic measurements yet — claim space is set, measurement begins Weekend 3.

This document will be versioned alongside the code. Every release will increment the version of this document and note what changed in [`CHANGELOG.md`](./CHANGELOG.md).
