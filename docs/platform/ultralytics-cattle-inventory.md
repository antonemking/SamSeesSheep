# Ultralytics cattle / livestock inventory

**Verified:** 2026-11-01, by loading each URL below (docs, GitHub files, Platform pages and their
embedded metadata). `ultralytics` package on `main` at the time: v8.4.197. Where something could not be
verified it is listed at the end, not filled in.

## Verdict

**There is no Ultralytics "cattle section" to take.** On ultralytics.com, cattle and livestock appear in
marketing pages and blog posts that ship no models or data. The docs (665 English URLs in the
[sitemap](https://docs.ultralytics.com/sitemap.xml)) have no livestock page. The pretrained livestock
capability is generic: COCO's `horse`/`sheep`/`cow`, YOLOv8-only Open Images models with Cattle, Goat,
Pig and Chicken classes, and open-vocabulary YOLOE. No animal pose weights are released.

Two recent facts matter more than anything in the "section":

1. **The "Roboflow built in" half already exists, and it is Ultralytics' flagship.**
   [Ultralytics Platform](https://docs.ultralytics.com/platform/) (announced 2026-03-19) does SAM-assisted
   annotation, custom keypoint skeletons, immutable dataset versions, cloud GPUs, 22 export formats and
   hosted deployment, for $0 or $29 per seat per month. Ultralytics HUB shut down on 2026-07-31.
2. **Ultralytics started seeding livestock content in late August 2026.** Its own Platform account
   re-hosted seven third-party livestock *detection* datasets between 2026-08-27 and 2026-08-31 (dates
   from each page's metadata). None has keypoints; three are non-commercial. That is a signal they are
   looking at the vertical, which is why kill criterion K8 in
   [`architecture-options.md`](architecture-options.md) exists.

A livestock fork of Ultralytics plus a hosted loop would compete with Ultralytics' main product, on
Ultralytics' code, under Ultralytics' license. The open gap is livestock keypoint and face data, schemas,
protocols and benchmarks.

---

## 1. Livestock on ultralytics.com (marketing and blog)

| Page | URL | Type | What it contains | Downloadable asset? |
|---|---|---|---|---|
| Computer vision in agriculture (© 2026; `/solutions/ai-in-agriculture` 308-redirects here) | https://www.ultralytics.com/solutions/computer-vision-in-agriculture | Marketing | "Livestock monitoring" card; the only livestock evidence cited is a third-party poultry counter (Wu et al. 2025, [PMC11939168](https://pmc.ncbi.nlm.nih.gov/articles/PMC11939168/), no Ultralytics authors, data "available on request") | No. The only linked demo model is wheat-head detection |
| Monitoring animal behavior using Ultralytics YOLOv8 (2024-05-30) | https://www.ultralytics.com/blog/monitoring-animal-behavior-using-ultralytics-yolov8 | Blog | Cow posture, pigs, goats (illustrative) | No |
| Computer vision and YOLO11's role in animal monitoring (2025-01-07) | https://www.ultralytics.com/blog/role-of-computer-vision-and-ultralytics-yolo11-in-animal-monitoring | Blog | "can be custom-trained… such as cattle, poultry" | No |
| The changing landscape of AI in agriculture (2024-10-30) | https://www.ultralytics.com/blog/the-changing-landscape-of-ai-in-agriculture | Blog | Generic livestock ideas | No |
| How to train YOLO models to detect animals in the wild (2025-04-15) | https://www.ultralytics.com/blog/how-to-train-ultralytics-yolo-models-to-detect-animals-in-the-wild | Blog | African Wildlife tutorial; no livestock | No |
| Stride equine gait customer story (2026-05-06) | https://www.ultralytics.com/customers/on-device-equine-gait-analysis-in-under-1-minute-ultralytics-yolo | Case study | YOLO26m-pose trained by the customer on "thousands of manually annotated images of horses" | No (customer's private model and data) |

Five more agriculture and animal blog URLs are listed in the
[site map](https://www.ultralytics.com/sitemap.xml) and were not opened.

## 2. Livestock datasets on the Ultralytics Platform account (the closest thing to a "cattle section")

URL pattern: `https://platform.ultralytics.com/ultralytics/datasets/<slug>`
([listing](https://platform.ultralytics.com/ultralytics/datasets), 317 datasets). License = the page's
`license` metadata field. All are **detection only**, with no trained models attached.

| Slug | Content | Images | Classes | License (Platform) | Created | Commercial use? |
|---|---|---|---|---|---|---|
| [`opencows2020`](https://platform.ultralytics.com/ultralytics/datasets/opencows2020) | Holstein-Friesian, indoor + UAV (Univ. Bristol) | 3,706 | cow | Research-Only; Bristol upstream: non-commercial government licence | 2026-08-30 | **No** |
| [`cows2021`](https://platform.ultralytics.com/ultralytics/datasets/cows2021) | Overhead walkway cows (Bristol) | 10,402 | cattle torso | Research-Only; same upstream licence | 2026-08-31 | **No** |
| [`cattle-detection-and-counting-in-uav-images`](https://platform.ultralytics.com/ultralytics/datasets/cattle-detection-and-counting-in-uav-images) | UAV pasture cattle (Univ. Tokyo) | 667 (328 labeled) | cattle | CC-BY-NC-ND-4.0 | 2026-08-31 | **No** (and no derivatives) |
| [`cherry-chevre`](https://platform.ultralytics.com/ultralytics/datasets/cherry-chevre) | Goats (INRAE) | 6,134 | goat | "Other" (upstream not checked: bot-blocked) | 2026-08-29 | Unknown |
| [`goat-image-dataset`](https://platform.ultralytics.com/ultralytics/datasets/goat-image-dataset) | Goat face parts (NWAFU) | 1,680 | ear, eye, face, mouth (boxes) | CC-BY-4.0 (upstream not checked) | 2026-08-31 | Yes, with attribution, if upstream agrees |
| [`automatic-monitoring-pigs`](https://platform.ultralytics.com/ultralytics/datasets/automatic-monitoring-pigs) | Top-view barn pigs | 667 | pig | CC-BY-4.0 (upstream not checked) | 2026-08-27 | Yes, with attribution, if upstream agrees |
| [`sheep-detection`](https://platform.ultralytics.com/ultralytics/datasets/sheep-detection) | Web photos of sheep (Kaggle/MakeML) | 201 | sheep | CC0-1.0 (upstream not checked) | 2026-08-28 | Yes, but toy-sized |

Related on the same account: `ena24` (camera traps, includes horse and chicken; "Other"), `meat-cut`
(beef cuts, CC-BY-4.0), `bee-image` (CC-BY-NC-ND-4.0). The account's 12
[solutions models](https://platform.ultralytics.com/ultralytics/solutions) include nothing animal.

**Read-across for us:** the goat face-parts set (eye/ear/mouth boxes, CC BY 4.0 on Platform) is the most
useful item here for a face line: it can seed pre-labels for a goat face schema. The two Bristol cattle
sets are fine for a research benchmark and unusable for commercial weights.

## 3. Pretrained models that already "see" livestock

| Asset | Where | Livestock coverage | License | Use to us |
|---|---|---|---|---|
| COCO class list | [`ultralytics/cfg/datasets/coco.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), [docs](https://docs.ultralytics.com/datasets/detect/coco/) | **14 bird, 17 horse, 18 sheep, 19 cow.** No goat, pig or chicken | YAML AGPL-3.0 | The baseline every livestock claim must beat; YOLO26 COCO detectors n–x span 40.9–57.5 mAP50-95 |
| Open Images V7 detectors | [docs](https://docs.ultralytics.com/datasets/detect/open-images-v7/), [`open-images-v7.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/open-images-v7.yaml) | **YOLOv8 only** (`yolov8{n,s,m,l,x}-oiv7.pt`, 18.4–36.3 mAP50-95). Classes include 5 Alpaca, 71 Bull, 99 Cattle, 108 Chicken, 174 Duck, 219 Goat, 225 Goose, 255 Horse, 346 Mule, 387 Pig, 452 Sheep, 560 Turkey | Weights AGPL-3.0 / Enterprise | Zero-shot detector for goats, pigs and poultry that COCO lacks; old architecture |
| YOLOE-26 (open vocabulary) | [docs](https://docs.ultralytics.com/models/yoloe/) | `set_classes(["goat","pig"])` with arbitrary names | AGPL-3.0 / Enterprise | Zero-shot baseline and a pre-labeler for new species |
| YOLO26 family | [models](https://docs.ultralytics.com/models/), [YOLO26](https://docs.ultralytics.com/models/yolo26/) | Released January 2026; seven tasks incl. pose. YOLO27 is waitlist-only ([page](https://www.ultralytics.com/yolo/yolo27)) | AGPL-3.0 / Enterprise | Our v0.x line is YOLO26n-pose |
| Pose weights | [pose task](https://docs.ultralytics.com/tasks/pose/) | `yolo26*-pose.pt` predict the 17 **human** COCO keypoints. **No animal pose weights** | AGPL-3.0 / Enterprise | Starting point only; the animal keypoint head is ours to train |

Objects365 and LVIS YAMLs list cow, sheep, horse, pig, goat and more
([Objects365.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml),
[lvis.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml)); no
released detector for either was confirmed.

## 4. Animal datasets in the Ultralytics docs

| Dataset | Docs / YAML | Size | Keypoints | License as stated | Livestock? |
|---|---|---|---|---|---|
| African Wildlife | [docs](https://docs.ultralytics.com/datasets/detect/african-wildlife/), `cfg/datasets/african-wildlife.yaml` | 1,504 images, 4 classes | — | "The source dataset does not specify a license." | No |
| Dog-Pose | [docs](https://docs.ultralytics.com/datasets/pose/dog-pose/), `cfg/datasets/dog-pose.yaml` | ~8.5k images | 24, `kpt_shape [24,3]`, no `flip_idx` | Platform: Research-Only; StanfordExtra annotations MIT ([repo](https://github.com/benjiebob/StanfordExtra)); Stanford Dogs image terms unverified | No; the template for a quadruped pose dataset card |
| Tiger-Pose | [docs](https://docs.ultralytics.com/datasets/pose/tiger-pose/), `cfg/datasets/tiger-pose.yaml` | 263 images | 12, `kpt_shape [12,2]`, no visibility | Annotations AGPL-3.0 (frames from a YouTube video) | No; toy-sized |

**No livestock keypoint or pose dataset exists** in the docs catalog
([pose datasets](https://docs.ultralytics.com/datasets/pose/): COCO-Pose, COCO8-Pose, Dog-Pose, Hand
Keypoints, Tiger-Pose) or among the 317 Platform datasets, and nothing in
[`cfg/datasets/`](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/datasets).

## 5. "Solutions" that apply on a farm

All generic, none livestock-specific ([index](https://docs.ultralytics.com/solutions/)). Each is
something we *use*, not build:

| Solution | URL | Farm use |
|---|---|---|
| Object counting | https://docs.ultralytics.com/guides/object-counting/ | Head counts through a gate or race (in/out across a line) |
| Region counting | https://docs.ultralytics.com/guides/region-counting | Per-pen counts |
| Heatmaps | https://docs.ultralytics.com/guides/heatmaps | Time at feed bunk or water |
| TrackZone | https://docs.ultralytics.com/guides/trackzone | Track only inside a paddock ROI |
| Speed estimation | https://docs.ultralytics.com/guides/speed-estimation | Rough activity proxy; built for traffic, needs calibration |
| Object cropping | https://docs.ultralytics.com/guides/object-cropping | Head-crop pipelines |
| Tracking | https://docs.ultralytics.com/modes/track/ | Six trackers; default `tracktrack.yaml` ([default.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)); ReID optional |

What is missing for livestock is not a solution but a measurement: none of these pages gives a
count-error or ID-switch protocol for a flock walking through a gate. That protocol is ours to define.

## 6. Ultralytics Platform: the "Roboflow built in" that already exists

| Capability | Source | Note for us |
|---|---|---|
| Launch | [announcement, 2026-03-19](https://www.ultralytics.com/news/introducing-ultralytics-platform-the-smartest-way-to-annotate-train-and-deploy-vision-ai) | — |
| HUB shut down 2026-07-31 | [HUB README](https://github.com/ultralytics/hub); [v8.4.115 "Deprecate HUB in favor of Ultralytics Platform"](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.115) | Anyone who built on HUB was migrated or left; Roboflow is courting them ([blog](https://blog.roboflow.com/migrate-from-ultralytics-hub/), seen in search) |
| Annotation: manual for all tasks; SAM 2.1 / SAM 3 / SAM 3.1 click-to-mask for detect/segment/OBB **but not pose**; YOLO pre-labels including pose; **custom keypoint skeleton templates** (built-ins: Person, Hand, Dog, Face, Box); video sampled at ≤ 1 fps, max 100 frames per video, no cross-frame tracking | [annotation docs](https://docs.ultralytics.com/platform/data/annotation/), [annotate page](https://www.ultralytics.com/annotate) | A user *could* define a sheep-face skeleton here. What they cannot get is our template, our pre-labeler, our crop policy, or native-resolution head crops from 30 fps video |
| Versions: "immutable NDJSON snapshots"; new datasets default to **Public** on Free and Pro, Private on Enterprise | [datasets docs](https://docs.ultralytics.com/platform/data/datasets/) | Farm footage defaults to public unless you pay for Enterprise. A real data-rights point in our favour |
| Cloud training: 26 GPU types, $0.24–$7.39/h from a credit balance | [cloud training](https://docs.ultralytics.com/platform/train/cloud-training/) | Commodity |
| Deploy: 22 export formats, shared API and dedicated endpoints | [deploy](https://docs.ultralytics.com/platform/deploy/) | Commodity |
| Pricing: Free $0 ($25 one-time credits), Pro $29/seat/month, Enterprise custom; "Platform plans and Ultralytics software licenses are separate"; only Enterprise includes the Enterprise License | [pricing](https://www.ultralytics.com/pricing) | Paying for Pro does not lift AGPL |

## 7. Licensing, precisely

| Item | Source | Text / finding |
|---|---|---|
| Package license | [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE); `pyproject.toml` `license = "AGPL-3.0"` | GNU AGPL v3 |
| Ultralytics' position on trained weights | [ultralytics.com/license](https://www.ultralytics.com/license) (FAQ) | "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default." / "The AGPL-3.0 License covers the training code and the models produced by that training code." The FAQ applies this even to training from scratch, and lists "R&D projects that are not fully open-sourced" as needing an Enterprise License; academic research is on the free list |
| Enterprise terms | [enterprise license](https://www.ultralytics.com/legal/enterprise-software-license) | Licensee owns "custom trained Models created by Licensee using the Software"; no sublicensing; one-year auto-renewing term; price unpublished |
| COCO | [terms of use](https://cocodataset.org/#termsofuse) | Annotations CC BY 4.0; "Use of the images must abide by the Flickr Terms of Use" |
| Open Images V7 | [facts and figures](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) | Annotations CC BY 4.0 (Google); images "listed as having a CC BY 2.0 license", with no warranty per image |

The weights position is Ultralytics' *interpretation*; whether AGPL reaches weights or internal use is a
question for counsel, not for this memo. What matters operationally:

- **Our published v0.2–v0.7 weights are already AGPL-3.0** (`docs/hf-model-card.md`), which matches
  Ultralytics' position. Nothing to fix for open research.
- **Any closed commercial deployment** of a YOLO-trained livestock model needs an Enterprise License or a
  different trainer. Hence the DeepLabCut 3 adapter in the 90-day plan and decision D4 in the strategy
  memo.
- **Our own code stays MIT** if Ultralytics is an optional extra behind an adapter, not a vendored copy.

## 8. Community look-alikes (not Ultralytics)

| Asset | URL | License | Note |
|---|---|---|---|
| CowcatcherAI/Calvingcatcher | https://huggingface.co/CowcatcherAI/Calvingcatcher | agpl-3.0 | Calving detector, updated 2026-10-15 |
| nsagatov1/cattle-detection-yolov8 | https://huggingface.co/nsagatov1/cattle-detection-yolov8 | agpl-3.0 | Aerial cattle |
| rmayormartins/yolov5-farm-cattle | https://huggingface.co/rmayormartins/yolov5-farm-cattle | ecl-2.0 | Cattle counter |
| KZHIwEI/mdpianimal-models | https://huggingface.co/KZHIwEI/mdpianimal-models | agpl-3.0 | Individual-ID models |
| Ultralytics Hugging Face org | https://huggingface.co/Ultralytics | — | First-party, but **nothing livestock** (YOLO models; COCO8, Tiger-pose, medical and signature sets) |

The pattern: single-purpose cattle detectors from individuals, all AGPL by inheritance, none with a
schema, a held-out protocol or a robustness card. That is the status quo the platform replaces.

## 9. Depend, fork, or reimplement

| Asset | Decision | Reason |
|---|---|---|
| `ultralytics` trainer, exporters, trackers, solutions | **Depend** (optional extra, behind an adapter) | Best-in-class, fast-moving, AGPL; vendoring it would make our code AGPL and stale |
| YOLO26 pose/detect weights as initialisation | **Depend** for the open line | Already our v0.x base; outputs inherit AGPL |
| COCO / OIV7 / YOLOE detectors | **Depend** as baselines and pre-labelers | They define "what stock YOLO already does" in every benchmark |
| Platform livestock datasets | **Cite, don't build on** for commercial weights; Bristol sets usable in research benchmarks only | Mixed and partly non-commercial licenses |
| Dog-Pose / Tiger-Pose | **Reference only** (dataset-card and YAML shape) | Not livestock; research-only images |
| Platform annotation / versions / cloud training | **Do not compete; interoperate** (export Ultralytics-format YAMLs so a Platform user can import our datasets) | Their flagship product |
| Anything with "Ultralytics" in the name | **Never** | Their company name and brand |

## 10. Not verified

- Stanford Dogs image terms (site unreachable); upstream licenses for CherryChèvre, the Tokyo UAV cattle
  set, Sheep Detection, the NWAFU goat set and the pigs set. Only Platform labels were read, except
  Bristol's two sets, which were confirmed non-commercial upstream.
- Whether `yolo26*-objv1-150.pt` (named in `ultralytics/utils/downloads.py`) is a released Objects365
  detector with livestock classes.
- What the `ul33-full-train` tag on several Platform datasets means.
- Contents of the five agriculture/animal blog posts listed only in the site map.
- Whether a `.pt` file (not just exports) can be downloaded on the Free plan.
- The OpenCows2020 page's visible text does not show a license; "Research-Only" is in the page metadata.
