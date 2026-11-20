# "Almost Roboflow, built in": the thinnest slice

**Date:** 2026-11-01. **Status:** proposal. Most of the code exists in the local, git-excluded
`fun/sheep-masks/skin/` workspace and is promoted into the repo in weeks 1–2 of the
[90-day wedge](90-day-wedge.md).

## The blunt version

The generic annotate → version → train → deploy loop is a commodity. Roboflow sells it
([pricing](https://roboflow.com/pricing): free tier with 10 credits a month, Core from $39/month), Ultralytics
sells its own (see [`ultralytics-cattle-inventory.md`](ultralytics-cattle-inventory.md)), and the
labeling part is free and open: Label Studio Community is Apache-2.0
([LICENSE](https://github.com/HumanSignal/label-studio/blob/develop/LICENSE)), CVAT is MIT
([LICENSE](https://github.com/cvat-ai/cvat/blob/develop/LICENSE)). A small team that rebuilds that loop
loses to all of them.

What none of them do is know anything about animals. Roboflow's own loop, checked stage by stage on
2026-11-01, is built for boxes and is thin exactly where livestock keypoints need it:

| Roboflow fact (verified) | Why it matters for livestock faces and pose | Source |
|---|---|---|
| Keypoint training is "RF-DETR Two-Stage", licensed PML 1.0, with weight export listed as **No** | You cannot take the model to a barn Pi without Roboflow's runtime | [supported models](https://docs.roboflow.com/models/supported-models), [train](https://docs.roboflow.com/models/train/train-a-model) |
| Every YOLO family it trains (v5 to 26) is AGPL-3.0; commercial self-hosting of those weights needs an Enterprise add-on | Same AGPL wall we have, plus a contract | [model licenses](https://docs.roboflow.com/models/model-licenses.md), [pricing](https://roboflow.com/pricing) |
| Auto Label on keypoint projects only works with a keypoint model already trained in Roboflow | No cold-start pre-labels for a new species schema | [auto label](https://docs.roboflow.com/datasets/annotate/annotate/ai-labeling/auto-label) |
| Keypoint export formats: `coco` and `yolov5pytorch` only | No documented YOLO-pose `flip_idx` handling for left/right anatomy | [export](https://docs.roboflow.com/datasets/versions/dataset-versions/exporting-data) |
| Keypoint evaluation is OKS-mAP; no per-keypoint pixel error, PCK or temporal jitter | Cannot answer "how many degrees of ear-angle noise?" | [evaluate](https://docs.roboflow.com/models/evaluate/evaluate-trained-models) |
| New workspaces "allow data licensing and model training on private projects by default" (admins can opt out going forward) | A farm's herd footage is a data-rights question; ours stays in the owner's private or gated Hugging Face repo | [pricing](https://roboflow.com/pricing) |
| Universe keypoint datasets for livestock are small (about 8 to 1k images) with undocumented schemas; no livestock face-landmark set was found for any species | The content gap is real, not assumed | Universe search (pages return 403 to automated fetches; counts from search snippets) |

The livestock-specific value sits in **four seams between the stages**, and those are the only parts we
build:

| Seam | What goes wrong without it (evidence from this repo) | What we build |
|---|---|---|
| **1. Capture provenance** | Splits leak when nobody records which animal, camera or day a frame came from. The first WCS failure-transfer run held out camera-*days*, so 106 of 133 "unseen" holdout frames came from cameras the model had trained on, and holdout recall came out *above* clean-val recall (`experiments/failure-transfer/wcs/README.md` on the `claude/failure-transfer` branch; the camera-level fix in `prepare_wcs.py` is not yet committed) | A clip manifest: sha256, flock, animal ear-tag IDs, camera, distance, time, weather, judge/holdout flag. Every later stage reads it |
| **2. Anatomy-aware pre-labels** | Eyes and nostrils are 4–8 px on a 512 px full frame and cannot be labeled or learned there (`CLAUDE.md` decision 2). Generic auto-label tools prompt for a class, not for a 36-point face | Native-resolution head crops plus pre-labels from the previous model, a measured species template, and SAM 2.1 for eyes, nostrils and outline (`prelabel.py`) |
| **3. Split discipline** | A per-frame hash split measures "unseen poses from *known* scenes" (`DATA_FORMAT.md`), which flatters every number | Splits by video, animal, flock or camera, enforced by the tool; judge clips refused by name *and* content hash (`make_tasks.py` already does this) |
| **4. Evaluation that matches the farm** | mAP on a random val split says nothing about ear-angle jitter, a dirty lens, or a new flock | Judge-set scorer (per-group pixel error, NME, PCK, consecutive-frame jitter, misses), the ear-angle σ gate, the field report card (`experiments/failure-transfer/`), the adaptation learning curve |

Everything else is someone else's software.

## Build / depend / defer, stage by stage

| Stage | Roboflow equivalent | Ultralytics equivalent | **We** | Status |
|---|---|---|---|---|
| Capture | Upload | Upload | **Build** `herdkit ingest`: clip manifest (seam 1), judge refusal | New, small |
| Task making | Upload + auto-label | Upload + SAM annotate | **Build** `herdkit tasks`: frames (PyAV, native res) → head boxes → crops → pre-labels → Label Studio tasks | Exists as `make_tasks.py` + `ls_setup.py` |
| Labeling UI | Roboflow Annotate | Platform annotate | **Depend** on Label Studio Community (Apache-2.0). Build only the config generator. Label Studio's `KeyPointLabels` has no skeleton edges and no visibility flag ([tag docs](https://labelstud.io/tags/keypointlabels)), so our config carries occlusion as a per-point choice and `herdkit pull` maps it to v=1. CVAT (MIT) has skeletons, per-point occluded/outside flags and an Ultralytics YOLO-pose export ([skeletons](https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/skeletons/), [format](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)); it is the fallback adapter if Label Studio's keypoint UX becomes the bottleneck | Config generator exists |
| Review / QA | Review queue | — | **Build** small checks: flip-pair sanity, skeleton-length outliers, unreviewed-point counts, second-labeler sampling | New, small |
| Pull | Export | Export | **Build** `herdkit pull`: Label Studio → canonical labels with visibility semantics (v=2 reviewed, v=1 occluded/estimated, v=0 absent) | Exists as `ls_export.py` |
| Version | Dataset versions | Dataset versions | **Depend** on Hugging Face dataset repos: git-backed ([repositories](https://huggingface.co/docs/hub/repositories)), loadable by tag or commit (`revision=`, [loading](https://huggingface.co/docs/datasets/loading)), with cards and gated access ([cards](https://huggingface.co/docs/hub/datasets-cards), [gating](https://huggingface.co/docs/hub/datasets-gated)). **Build** `herdkit version`: immutable manifest = schema version + split assignment + label hashes + proof judge clips are absent + datasheet. Doubles as off-laptop backup. Hugging Face has no augmentation stage; augmentation lives in the training recipe, where it belongs | New |
| Export (data) | Format export | YOLO format | **Build** schema-driven exporters: YOLO-pose (`kpt_shape`, `flip_idx`), COCO keypoints (the format Wang et al. 2026 and most benchmarks ship), DeepLabCut 3 project format | YOLO exporter exists (`train/export_yolo.py`) |
| Train | Roboflow Train | Platform cloud training | **Depend** on Ultralytics `yolo train` and DeepLabCut 3 (top-down, RTMPose configs; LGPL-3.0, actively maintained) through adapters. Not MMPose: no release since 2024-07-12. **Build** recipes only. No hosted training. Watch RF-DETR Keypoint (Preview, Apache-2.0, currently pretrained on COCO person only; [repo](https://github.com/roboflow/rf-detr)) as a second commercial-clean candidate | `train/train.sh`, pod scripts exist |
| Evaluate | Model evaluation | Validation metrics | **Build** — this is the product: `herdkit eval` + `herdkit adapt` (seam 4) | `score.py`, `bench_held_out.py`, `experiments/failure-transfer/` exist |
| Export (model) | Deploy targets | `model.export()` | **Depend** on Ultralytics export (ONNX, CoreML, NCNN, TFLite). **Build** the model-card generator that embeds the field report card | New, small |
| Deploy | Inference server, Workflows | Platform deploy | **Defer.** A Pi recipe exists (`pi-deploy/README-PI.md`) | — |
| Active learning | Active learning hooks | — | **Build in month 2**: the DeepLabCut outlier queue (low per-keypoint confidence, frame-to-frame jumps, residual from rolling median) orders the next labeling batch | New |

## The whole command surface

Eight commands. If the MVP needs a ninth, something above was not thin enough.

```bash
herdkit ingest CLIP.MOV --flock de-katahdin --animals 0412,0419 --camera iphone15-1m
herdkit tasks   --schema sheep-face-36 --batch b03          # crops + pre-labels -> Label Studio project
herdkit pull    --batch b03                                 # marks -> canonical labels (v=0/1/2)
herdkit version sheep-face-36 --split per-video --tag v1.0  # immutable manifest + datasheet -> HF revision
herdkit export  sheep-face-36@v1.0 --to yolo|coco|dlc
herdkit train   sheep-face-36@v1.0 --recipe yolo26n-pose-320   # thin adapter; also --recipe dlc3-rtmpose-s-320
herdkit eval    best.pt --judge sheep-face-judge --field-card  # judge scores + ear σ gate + field report card
herdkit adapt   --base sheep-face-36@v1.0 --labels my-flock/ --tiers 25,50,100,200   # the DLC promise
```

`herdkit` is a placeholder name (free on PyPI as of 2026-11-01). Rename before the first public release;
do not put "Ultralytics" or "YOLO" in the product name.

## A species is a file

The platform claim — a second species is a vertical, not a new repo — rests on this. A schema is one
YAML file plus an optional pre-labeler prompt set. Sketch (sheep-face-36, truncated):

```yaml
name: sheep-face-36
version: 1.0.0
species: [ovis_aries]
unit: head_crop            # never full frames for faces
crop: {shape: square, margin: 0.3, train_px: 320, source: native_resolution}
detector: sheep-ears-5@0.7   # bootstraps head boxes and slots 0-4
keypoints:                   # slots 0-4 byte-identical to v0.7 (sheep-ears-5)
  - {id: 0, name: nose,        group: skull}
  - {id: 1, name: L_ear_base,  group: ears, flip: R_ear_base}
  - {id: 3, name: L_ear_tip,   group: ears, flip: R_ear_tip}
  # ... 33 more: 4 per ear, 4 per eye, 2 per nostril, 4 mouth, 4 skull, 9 silhouette
visibility: {2: reviewed_visible, 1: occluded_estimated, 0: absent}
oks_sigmas: pending          # filled from the 50-head double-labeling study, per keypoint
splits: {unit: video, also_allowed: [animal, flock]}
judge: {clips_sha256: [...], refuse: true}
prelabel: [previous_model, template_tps, sam2.1_tiny:eyes,nostrils,outline]
references: [SFLW (Hewitt & Mahmoud 2019), CatFLW (Martvel et al. 2024), SPFES action units (McLennan et al. 2016)]
```

The second species (cattle, weeks 10–13 of the wedge) must be added as a new file of this shape plus an
importer: `cattle-body-18`, built on the CC BY 4.0 cattle subset of Wang et al. 2026, with
`unit: body_crop` instead of `head_crop`. If that needs changes to `herdkit` core beyond ~200 lines, the
registry is not general and kill criterion K5 says so. The `oks_sigmas: pending` line is deliberate: COCO derived its per-keypoint OKS
constants from 5,000 redundantly annotated val images, measuring how consistently humans click each
keypoint ([COCO keypoint evaluation](https://cocodataset.org/#keypoints-eval);
[page source](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/keypoints-eval.htm)).
We found no sheep-face schema with annotator-derived sigmas. The double-labeling study produces ours,
and that table is itself a citable contribution.

## What makes it sticky (without lock-in)

- **Weights that only make sense with our schemas.** A pretrained base that adapts to a new flock with
  ~200 labels is the reason to start here instead of from COCO. Kill criterion K4 in
  `architecture-options.md` tests exactly this.
- **Report cards people cite.** If every livestock model card in the field carries a judge-set score and a
  field report card in our format, the format is the standard.
- **Dataset versions with DOIs.** Researchers cite DOIs; startups pin revisions.
- **Pre-annotations inside the tools people already use.** MegaDetector became the default largely by
  meeting users in AddaxAI and Timelapse rather than asking them to switch (strategy memo §3.6). Our
  base models emit Label Studio pre-annotations now and CVAT pre-annotations later, so they become the
  first stage of other labs' pipelines.
- **Exports to everything.** Stickiness comes from content, not from trapping data. Every version exports
  to YOLO, COCO and DeepLabCut formats; a user can leave at any time, which is why they will not.

## Deferred, with the trigger that would un-defer it

| Deferred | Un-defer when |
|---|---|
| Hosted labeling canvas | The four hosted-C gates in `architecture-options.md` all hold |
| Accounts, teams, billing | Same |
| Cloud training UI | Same, and an Apache-2.0 trainer is the production default |
| Inference server / deploy workflows | A paying integrator needs it and Ultralytics' or Roboflow's runtimes cannot serve the model |
| Generic classes ("label anything") | Never. Unregistered schemas are out of scope by design |
| Video-level behaviour labels | Keypoint line stable (`CLAUDE.md` step 8) and a temporal schema exists |
| SPFES scoring block | Keypoint line stable, plus clinical footage and an external scorer |

## Build cost estimate

Measured against what exists locally (`make_tasks.py` 149 lines, `ls_setup.py` 119, `prelabel.py` 99,
`ls_export.py` 49, `score.py` 124, `train/export_yolo.py` 57, `schema.py`, plus the committed
`experiments/failure-transfer/` and `sheep-yolo/scripts/bench_held_out.py`): roughly one to two agent-weeks
to promote, generalise to the schema registry and add `ingest`, `version` and the COCO exporter;
another week for `adapt` and the field-card generator. The labeling hours in the 90-day plan dominate the
schedule, not this code.
