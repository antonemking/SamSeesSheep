# Architecture options: A / B / C, recommendation, kill criteria

**Date:** 2026-11-01. **Decision owner:** Antone King. **Status:** recommendation, not yet adopted.
**Constraints this is judged against:** one founder plus coding agents; farm footage on an external drive
and phone; the Label Studio loop for sheep faces already built (locally, in the git-excluded
`fun/sheep-masks/skin/` workspace); strength in inference and vision, not in running SaaS; a small
Delaware farm as the proving ground; the 2026 laptop crash that destroyed the v0.x training data.

## The three options

### A. Thin opinionated layer on Ultralytics + Label Studio

A Python package (working name `herdkit`) that owns only the livestock-specific parts of the loop:
species schemas, capture manifests, task making (clip → native-resolution crops → pre-labels → Label
Studio), label pull with visibility semantics, versioned datasets on Hugging Face, trainer-native
exports, and the evaluation layer (judge sets, field report cards, adaptation curves). Ultralytics is an
optional dependency behind a trainer adapter; Label Studio Community is the labeling UI.

```
 capture ─► herdkit ingest ─► herdkit tasks ─► Label Studio (OSS) ─► herdkit pull
                                                                       │
     HF Hub dataset revision ◄─ herdkit version ◄──────────────────────┘
              │
              ├─► herdkit export ─► Ultralytics `yolo train`        (adapter, AGPL side)
              │                 └─► DeepLabCut 3 top-down / RTMPose  (adapter, LGPL code, our weights)
              ▼
         herdkit eval ─► judge scores · field report card · adaptation curve ─► model card
```

### B. Fork / rebrand the YOLO training stack with first-party livestock schemas

Vendor `ultralytics` into a livestock-branded package, add livestock heads (ID embeddings, multi-animal
association), ship it as "the livestock YOLO".

### C. Model zoo + dataset SDK + hosted annotation (open-core)

Open zoo and SDK, plus a hosted service: multi-tenant labeling, storage, cloud training, private
evaluation, billing. Roboflow's shape, narrowed to livestock.

## Scorecard

| | A. Thin layer | B. Fork | C. Open-core + hosted |
|---|---|---|---|
| Time to first external user | **Weeks** (most code exists locally) | Months (fork, then the same loop anyway) | Months (auth, storage, billing before any user) |
| License exposure | Code MIT; YOLO-path weights inherit AGPL-3.0 in Ultralytics' reading (see inventory); a non-Ultralytics path exists through DeepLabCut 3 (LGPL code) and RF-DETR (Apache-2.0) | **Whole product is AGPL-3.0** — a fork of an AGPL codebase cannot be relicensed | Hosted service over AGPL code triggers AGPL §13 network-use obligations unless an Ultralytics Enterprise license or a non-Ultralytics trainer is used |
| Maintenance load | Low: track upstream releases through an adapter | **High**: every Ultralytics release (they ship new model families roughly yearly) becomes a merge | High: on-call for a SaaS with farm footage in it |
| Differentiation | In content: schemas, data, benchmarks, field report cards | Near zero: users prefer upstream | Hosted UI is commodity (Roboflow, Ultralytics Platform, Label Studio Enterprise, CVAT Cloud) |
| Revenue optionality | Services, commercial-clean weights, later hosting | Licensing a fork of someone else's code | Subscriptions, if anyone pays |
| Fit with "Antone + agents" | **Good** | Poor | Poor until there are users |
| Survives Ultralytics shipping a livestock feature | Yes: we become their livestock data/benchmark supplier | No | Partly |

## Recommendation

**Adopt A now, built as the open half of C. Kill B outright. Gate the hosted half of C on evidence.**

Concretely:

1. A is the product for the next 90 days. Its public face is a model zoo (Hugging Face model repos with
   field report cards), datasets (Hugging Face dataset repos with DOIs and datasheets), and the
   `herdkit` package. That *is* C's open half; nothing has to be rebuilt to go further.
2. The trainer is an adapter, never the core. Ship two adapters by day 90: Ultralytics (community
   default, continuity with v0.x) and DeepLabCut 3 top-down with its RTMPose configs (LGPL-3.0 code,
   actively maintained; weights we train are ours to license). That is both the path for closed
   deployments and the architecture comparison the 200-label benchmark needs anyway. MMPose would have
   been the obvious Apache-2.0 choice, but it has had no release since 2024-07-12
   ([repo](https://github.com/open-mmlab/mmpose)); MegaDetector V6 shows the other escape route, shipping
   MIT and Apache-2.0 weight variants beside its AGPL ones
   ([MegaDetector](https://github.com/agentmorris/MegaDetector)).
3. Hosted anything (labeling, training, evaluation) waits for the gates below.

Why not B, in one line: it converts our only real advantage (livestock content) into a maintenance
liability on someone else's license, and Ultralytics' release cadence guarantees the fork is stale within a
year.

## Kill criteria

Each criterion has a date, a measurement, and the action it triggers. "Kill" means stop investing, write
down why, and keep what was learned as research artifacts in this repo.

### For the recommendation (A)

| # | When | Measurement | If it fails |
|---|---|---|---|
| K1 | Day 14 (2026-11-15) | The 76 judge heads are hand-marked | Stop adding code; nothing downstream is scoreable |
| K2 | Day 35 (2026-12-06) | v1.0-a (≥ 60 heads) beats the seeded-template baseline on eyes and nostrils on the judge set | Narrow the schema to ears + eyes; dense 36-point claim is withdrawn |
| K3 | Every training run | Ear-angle σ on IMG_3877's published window ≤ 2.84° (v0.7) | Block the release; regressions are reported, never shipped |
| K4 | Day 90 (2027-01-30) | On the second flock's held-out clip, our pretrained base beats a COCO-initialised model trained on the same heads, with non-overlapping 95% bootstrap intervals on per-keypoint pixel error at 50 **and** 100 heads | **Kill the model-zoo thesis.** If our base doesn't help, the platform is tooling, and tooling alone is commodity |
| K5 | Day 90 | Adding the second species changes ≤ ~200 lines of `herdkit` core | Stop calling it a platform; it is a sheep project with good habits |
| K6 | Day 90 | One person outside Lorewood has run the loop and posted a number | Do not build anything hosted; reassess whether the user exists |
| K7 | Day 90 | WCS reality check, on the causes with ≥ 20 real tagged frames (today glare, smudge, dust): the synthetic onset order matches the order of real per-cause drops, and the degraded-copy model beats the re-encoded control on the real holdout for at least two of the three | Drop synthetic sweeps from the field report card; keep only real-failure holdouts |
| K8 | Any time | Ultralytics or Roboflow ships an open livestock keypoint zoo with comparable schemas and benchmarks | Pivot to supplying them: contribute datasets/benchmarks upstream, stop shipping our own trainer adapters |
| K9 | First commercial conversation | Buyer needs closed deployment | Ship the DeepLabCut 3-trained weights (with an RF-DETR or other non-Ultralytics head detector), or price an Ultralytics Enterprise license into the deal. Never ship AGPL weights into a closed product without one |
| K10 | Month 9 (2027-07-31) | ≥ 2 paid adaptation projects, ≥ 1 repeat customer | **Kill the company framing.** Keep the open project and the services sideline; stop describing it as a platform business. The precedents are real: OneCup AI, Plainsight and Iris Data Science left or stalled in livestock vision (strategy memo §3.2) |

### Gates that open the hosted half of C

All four must hold, measured, not felt:

1. ≥ 3 external groups have adapted a model with the kit (K6 three times over).
2. ≥ 1 of them has asked, in writing, to pay for hosted labeling/training/evaluation.
3. Self-hosting Label Studio or GPU access is the stated blocker, not model quality.
4. The licensing path for hosting is resolved (a non-Ultralytics trainer in production, or an
   Ultralytics Enterprise license priced in).

Until then the "hosted" offer is a person: adaptation as a service, run by us with the same open tools.

### For B (already triggered)

B is killed on day 0 by two facts that do not depend on execution: AGPL-3.0 cannot be relicensed by a
fork, and a one-person team cannot track Ultralytics' release cadence. Reopen only if a livestock-specific
architectural change (for example an ID-embedding head) proves necessary *and* cannot be shipped as an
Ultralytics-compatible plugin or upstream contribution.

## Licensing map (what each layer carries)

| Layer | Component | License | Consequence |
|---|---|---|---|
| Our code | `herdkit`, this repo | MIT | Anyone can embed it |
| Labeling UI | Label Studio Community | Apache-2.0 | Self-host freely; Enterprise features are not needed for the MVP |
| Trainer (default) | Ultralytics | AGPL-3.0 or Ultralytics Enterprise | Weights trained with it are AGPL in Ultralytics' reading; fine for open research, a blocker for closed products |
| Trainer (outside Ultralytics) | DeepLabCut 3 top-down (RTMPose / HRNet configs) | LGPL-3.0 ([repo](https://github.com/DeepLabCut/DeepLabCut)) | The path for closed deployments: LGPL covers the library, not the weights we train. Check each backbone's pretrained-weight license |
| Head detector (outside Ultralytics) | RF-DETR N/S/M/L | Apache-2.0 ([LICENSE](https://github.com/roboflow/rf-detr/blob/develop/LICENSE)) | Replaces the YOLO head detector in a fully non-Ultralytics stack. Other RF-DETR sizes are under Roboflow's PML 1.0; avoid those |
| Not used | MMPose | Apache-2.0 | No release since 2024-07-12; dependency rot risk |
| Pretrained comparison | DeepLabCut SuperAnimal | Code LGPL-3.0; weights "academic, non-commercial purposes only" ([model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped)) | Benchmark comparator only, never a shipped base |
| Our datasets | `sheep-face-36`, WCS robustness subset | Decision for Antone: CC BY 4.0 recommended; WCS source is CDLA-Permissive-1.0 ([LILA](https://lila.science/datasets/wcscameratraps)) | CC BY maximises adoption; CC BY-NC keeps a commercial lever but cuts PLF-startup use |
| Second-species data | Wang et al. 2026 cattle/horse/sheep, 18 keypoints | CC BY 4.0 data, MIT code ([repo](https://github.com/yww0411/Livestock-keypoint-detection)) | Usable for open and commercial weights with attribution |

Exact license URLs and the Ultralytics wording on trained weights are in
[`ultralytics-cattle-inventory.md`](ultralytics-cattle-inventory.md).
