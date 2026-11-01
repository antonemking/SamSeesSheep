# Livestock vision platform: decision memo

**Date:** 2026-11-01 · **For:** Antone King · **Status:** recommendation; decisions listed in §9
**Working name:** TBD. The package is called `herdkit` below as a placeholder. Do not put "Ultralytics"
or "YOLO" in the name.
**Companion docs:** [inventory](ultralytics-cattle-inventory.md) ·
[MVP slice](roboflow-built-in-mvp.md) · [90-day wedge](90-day-wedge.md) ·
[architecture + kill criteria](architecture-options.md)

---

## 1. The five-minute pitch

> **Farm video is the hard normal case of computer vision, and nobody owns it.**
>
> General detectors already find a cow or a sheep: both are COCO classes, and every COCO-trained YOLO
> checkpoint ships with them. What they cannot tell you is anything a farmer, vet or welfare scientist
> needs: where the ears are pointing, which animal it is, whether the model still works when the lens is
> fogged at 7 a.m., and whether it will work on *your* herd rather than the one it was trained on.
> So every precision-livestock lab rebuilds the same stack: an open labeling tool, a keypoint schema
> invented for that one paper (we counted 7, 9, 15, 18, 22 and 38 points in recent work), a YOLO
> fine-tune under AGPL, a split that can leak the same animal into train and val, no test against field
> conditions, and data that is never released (§3.4).
>
> **We are building the open reference layer for livestock vision:** species schemas, pretrained bases
> that adapt to a new herd with about 200 labels, and a field report card on every model that says how it
> breaks in haze, dust, smudge and glare. Thin tooling turns Label Studio and any trainer into a livestock
> dataset loop with splits that cannot leak. It runs *on top of* Ultralytics and Label Studio, not
> instead of them.
>
> **Why us:** we already have an open sheep keypoint model line with a measurement benchmark, not just
> mAP (ear-angle σ 2.84° on a held-out clip; weights on Hugging Face with a DOI), an explicit
> anti-overclaim contract, a working 36-point face pre-labeling loop, a field-robustness harness, and a
> farm. We found no open livestock face-landmark model or leaderboard anywhere (§3.3).
> Ultralytics and Roboflow are horizontal. Species schemas and farm-condition benchmarks are vertical
> content with a small market per vertical; that is exactly what a horizontal company will not build and
> a vertical one can own.
>
> **Wedge:** sheep faces first, because there we can be the reference within 90 days. Cattle second, from
> public data, as a schema *file* rather than a new repo — that is the platform test.
> **Proof by 2027-01-30:** a public dataset, a demo anyone can open, and one stranger who adapted our
> model to their own animals and posted the number.
> **Kill switch:** if our pretrained base does not beat a COCO-initialised model trained on the same 200
> labels from a new flock, the model-zoo thesis is dead and we say so.
>
> **Business, later, and unproven:** adaptation as a service for camera startups, integrators and
> animal-health companies that need objective measurements, delivering weights trained outside
> Ultralytics so they can ship them closed. Hosting only after users ask to pay for it. Several
> livestock-vision companies have pivoted out or stalled (§3.2); the plan does not assume a buyer exists.

### We are NOT building

| vs Ultralytics | vs Roboflow | vs everyone |
|---|---|---|
| A training framework or trainer | A hosted labeling canvas | Hardware, cameras, collars, connectors |
| A YOLO fork or a "livestock YOLO" brand | Hosted dataset storage and a versioning UI | Farm-management software |
| New general-purpose architectures | General AutoML / hosted training | Pain, welfare or disease diagnosis claims |
| Export runtimes or an inference server | A deploy server or workflow builder | A dairy lameness or BCS product |
| A general model hub | A general dataset marketplace (Universe) | Support for arbitrary, unregistered classes |

---

## 2. Pressure test of the seed thesis

The seed: *"Take the cattle section from Ultralytics and make an Ultralytics-for-livestock with almost a
Roboflow built in."* Four parts of it do not survive contact with the sources; one part is right.

| Part of the seed | Verdict | Evidence |
|---|---|---|
| "Take the cattle section from Ultralytics" | **Nothing to take.** Ultralytics' livestock material is marketing and blog posts with no models or data; the docs have no livestock page; no animal pose weights exist. The only concrete first-party livestock asset is seven third-party *detection* datasets re-hosted on Ultralytics' Platform account on 2026-08-27 to 08-31, three of them non-commercial | [`ultralytics-cattle-inventory.md`](ultralytics-cattle-inventory.md) §1–4 |
| "…make an Ultralytics-for-livestock" (a framework) | **Wrong shape.** Ultralytics states "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default", including R&D use that is not open-sourced; a fork cannot relicense that. It ships a new model family about yearly (YOLO26 in January 2026, YOLO27 already on a waitlist), which a one-person fork cannot track. And "Ultralytics" is their company name | [license FAQ](https://www.ultralytics.com/license); [YOLO26](https://docs.ultralytics.com/models/yolo26/); [YOLO27](https://www.ultralytics.com/yolo/yolo27) |
| "…with almost a Roboflow built in" | **Already built, twice.** Ultralytics Platform (launched 2026-03-19, HUB shut down 2026-07-31) does SAM-assisted annotation, custom keypoint skeletons, immutable versions, cloud GPUs and deployment for $0–$29 per seat; Roboflow does the same; Label Studio and CVAT give labeling away. The generic loop is a commodity. Both hosted loops are weak exactly where livestock keypoints need help (no SAM for pose on Platform; Roboflow's keypoint trainer is non-exportable PML 1.0 with OKS-only evaluation), and both default farm data toward sharing (Platform datasets are Public on Free/Pro; Roboflow workspaces allow data licensing by default) | [`roboflow-built-in-mvp.md`](roboflow-built-in-mvp.md); inventory §6 |
| "Strategize to take over the area", cattle first | **Cattle is the market, not the entry.** US cattle outnumber sheep 17 to 1 (86.2 M vs 4.99 M head), but cattle vision revenue sits with hardware-integrated vendors (§3), and the public cattle datasets Ultralytics re-hosted are research-only. Enter cattle as content and adaptation services; prove the method on sheep | §3, §7; [USDA cattle](https://esmis.nal.usda.gov/sites/default/release-files/795748/catl0126.txt), [USDA sheep](https://esmis.nal.usda.gov/sites/default/release-files/795751/shep0126.txt) |
| "Broad YOLO stays broad; we own the vertical" | **Right, if "own" means content.** No livestock keypoint or pose dataset exists in Ultralytics' docs catalog or among its 317 Platform datasets; no livestock face-landmark dataset turned up on Roboflow Universe for any species; Universe livestock keypoint sets are 8 to ~1k images with undocumented schemas. The vertical's *definitions* are unclaimed | inventory §4; MVP doc, Roboflow table |

**One timing signal.** Ultralytics re-hosting livestock datasets in late August 2026 means the window is
open but someone is looking at it. If they ship livestock keypoint content, kill criterion K8 turns us
into their supplier rather than their competitor. Plan for that outcome; it is not a bad one.

**The sharpened thesis:** own the livestock *definitions* — schemas, splits, benchmarks, field report
cards, adapted weights — and let Ultralytics, Roboflow and Label Studio own the machinery. Distribute
*through* them (Ultralytics-compatible weights and dataset YAMLs, a Universe mirror for discovery,
Hugging Face as the canonical home), not against them.

---

## 3. Market map

All entries were checked on 2026-11-01. Vendor accuracy figures are company claims, not independent
results. Entries marked (search) were seen only in search results and are weaker evidence.

### 3.1 Commercial livestock vision: where the money is, and who holds it

| Company | What | Wins at | Threat to us | Source |
|---|---|---|---|---|
| **CattleEye (GEA)** | 2D camera over the dairy exit lane + RFID: lameness 0–100 and daily BCS into GEA herd software | GEA's dealer network; "digital custodian for more than 200,000 cows"; processor/retailer customers (Arla, Danone) | Closed, dairy-only, gait and BCS. Proves a cheap 2D camera plus subscription sells. Acquired by GEA in March 2024 (this corrects §0.7 of the 2026-10-26 research memo, which found no post-2021 news) | [Fortune, 2025-06-26](https://fortune.com/2025/06/26/ai-farming-cows-cattleeye-health-environment); [GEA](https://gea.com/pl/campaigns/cow-welfare-cattleeye) |
| **Nedap SmartSight** | Locomotion score for every cow at every milking | Installed base of collars and herd software, now adding vision | **Real threat in dairy.** "the first step in an ongoing innovation strategy" (announced 2025-06-10) | [Nedap](https://nedap-livestockmanagement.com/knowledge-base/igniting-the-era-of-ai-powered-vision-technology-with-smartsight/) |
| **DeLaval BCS** | 3D camera over the cow's rear at each pass, integrated with DelPro | Milking-robot and sort-gate installed base | Hardware-bundled incumbent | [product PDF](https://delaval.com/globalassets/inriverresources/pdfs/b/bodyconditiescore-systeem-bcs2.pdf) |
| **HerdVision** | Camera over parlour or race: BCS, mobility; beef weight estimates | UK channel: "now available exclusively from Wynnstay"; £5,900 with first-year subscription | UK dairy/beef niche; the "camera + distributor + compliance" pattern | [Scottish Farmer](https://www.thescottishfarmer.co.uk/features/25352248.wynnstay-brings-herdvision-tech-improve-cow-health); [FWi](https://www.fwi.co.uk/livestock/assessing-performance/how-beef-units-could-improve-efficiency-with-camera-tech) |
| **MyAnIML** | Solar edge camera at water points; claims early warning for pinkeye and BRD | Off-grid range beef | Same edge shape we need; closed | [American Cattlemen, 2026-01-20](https://americancattlemen.com/myaniml-launches-ai-camera-system-that-predicts-cattle-disease/) |
| **HerdSense (The Main Branch + Valmont)** | Cameras on Valmont poles flagging feedlot anomalies | Valmont hardware and channel | Feedlots are the cattle market-size argument, and closed systems are already in trials there | [The Main Branch](https://themainbranch.com/?p=1445) |
| **Fancom eYeGrow**, **Asimetrix PigVision** | Pig weight from overhead / pen cameras | Barn-equipment channels; Pork Checkoff validation for Asimetrix | Pig weight is a solved commercial niche | [Fancom](http://www.fancom.com/blog/eyegrow-weighing-system-pigs); [Pork Checkoff](https://porkcheckoff.org/research/accurate-and-rapid-assessment-of-pig-body-weight/) |
| **Cumberland Scout / Faromatics ChickenBoy** | Ceiling robot in broiler houses: climate, distribution, dead birds | Poultry-equipment OEM channel (AGCO agreed to sell the unit to AIP in 2024) | Poultry vision lives inside equipment OEMs | [Provisioner](https://www.provisioneronline.com/articles/114093-cumberland-introduces-scout-robot-to-promote-poultry-health-and-performance?v=preview); [DTN](https://www.dtnpf.com/agriculture/web/ag/equipment/article/2024/07/25/agco-selling-grain-protein-unit-aip) |
| **Roboflow livestock vertical** | Counting, weight, lameness, posture, calving/farrowing on existing barn cameras | Horizontal tooling + the largest pile of community livestock datasets | **The most direct threat to an open livestock stack.** No livestock customer is named; no species schemas, no field benchmark | [roboflow.com/ai/poultry-monitoring](https://roboflow.com/ai/poultry-monitoring) |
| **neXtgen Agri** | Sheep face *ID* to match lambs to ewes, planned at ~$4–5 per lamb | The only priced sheep-face use case found | Identity, not expression or landmarks; no published results after the 2023 trial | [Westpac](https://www.westpac.com.au/news/in-depth/2023/04/ewe-must-be-joking-sheep-facial-recognition-tech-on-the-way/); [The Land](https://www.theland.com.au/story/8208412/nz-agtech-company-to-trial-sheep-facial-recognition-in-australia/) |

**The pattern:** commercial livestock vision is a camera bolted onto an existing hardware or
distribution channel (GEA, Nedap, DeLaval, Wynnstay, Valmont, barn-equipment OEMs), sold into dairy,
feedlot, pig or poultry operations, with a processor or retailer welfare scheme as the reason to buy.
None of them will open a schema, a dataset or a benchmark, and none of them is a customer for one.

### 3.2 The casualties (read these before believing the market exists)

| Company | What happened | Source |
|---|---|---|
| OneCup AI (BETSY, ranch-camera cattle ID) | `onecup.ai` now 301-redirects to onekind.ai, which sells "AI-Originated Clinical Endpoints for Animal Health" for pet studies. No cattle on the site. Pivot inferred from the redirect; no announcement found | [onekind.ai](https://onekind.ai/) |
| Plainsight | Counted cattle at JBS USA in 2021; its 2026 homepage sells "Physical Operations Intelligence" for manufacturing and retail, no livestock | [NVIDIA blog, 2021](https://blogs.nvidia.com/blog/plainsight-cattle-management-ai/); [plainsight.ai](https://plainsight.ai) |
| Iris Data Science (sheep face ID) | Worked technically in the yards; commercial path "stalled at the customer discovery stage" because the business case was too weak | [Country-Wide](https://country-wide.co.nz/keeping-an-eye-on-the-future/) |
| Cainthus | Absorbed into Ever.Ag; the old product URL now serves calving and feed products with no Cainthus brand (acquisition 2022, search only) | [Ever.Ag](https://www.ever.ag/dairy/maternity-warden/) |

Where the large rounds go is wearables, not cameras: Halter raised a $220M Series E at a $2B valuation in
March 2026 for virtual-fence collars ([TNW](https://thenextweb.com/news/halter-series-e-virtual-fencing-cattle));
Nofence raised over $35M in September 2025 and claims to be the only virtual fence for goats and sheep
([AgFunder](https://agfundernews.com/nofence-plots-us-european-expansion-for-virtual-fencing-following-oversubscribed-series-b-raise)).
Even MSD Animal Health lists "SenseHub Ovine" as "Coming Soon!" ([Allflex](https://allflex.global/livestock-monitoring)).

**Blunt reading:** the open gap is empty partly because the buyer is unclear. A sheep vision *product*
sold to farmers is the business case Iris could not make. Our plan therefore does not sell a product to
farmers. The OneKind pivot hints at a buyer that does pay for measurement — animal-health companies that
need objective trial endpoints — and that is a hypothesis to test in phase 1 (§7), not a claim.

### 3.3 Open and academic: what exists, and the gap

| Asset | What | License | Why it is not the answer | Source |
|---|---|---|---|---|
| DeepLabCut SuperAnimal-Quadruped | 39-keypoint quadruped model incl. sheep and horses | Code LGPL-3.0; weights "academic, non-commercial purposes only" | Cannot be a commercial base; no face model. DLC itself is very active (v3.0.2, 2026-09-23) | [model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped); [releases](https://github.com/DeepLabCut/DeepLabCut/releases) |
| MMPose (RTMPose, AP-10K support) | The standard pose toolbox | Apache-2.0 | **Dormant:** last release v1.3.2 on 2024-07-12, last commit 2025-08-04 | [repo](https://github.com/open-mmlab/mmpose) |
| AP-10K | 10,015 wild-animal images, 17 body keypoints, includes sheep | CC BY 4.0 | Wild scenes, body points, no faces, static since 2023 | [repo](https://github.com/AlexTheBad/AP-10K) |
| **Wang et al. 2026, Livestock-keypoint-detection** | 2,229 images of cattle, horses and sheep; 18 keypoints with visibility; edge latency benchmarks; few-shot transfer to AP-10K sheep (*Comput. Electron. Agric.* 240:111230) | Data CC BY 4.0; code MIT | **The closest thing to our benchmark.** A single-paper repo (6 stars, no releases, weights not confirmed); body points, not faces | [repo](https://github.com/yww0411/Livestock-keypoint-detection); [DOI](https://doi.org/10.1016/j.compag.2025.111230) |
| SwinePose | 3,778 frames, 85 videos, 27 pigs, 15 keypoints, **per-video split**, SLEAP/HRNet/RTMPose/ViTPose baselines | Data CC BY 4.0; code MIT | One species, one site — but it is the split discipline done right | [Zenodo](https://zenodo.org/records/19358700); [repo](https://github.com/AryaanVedak/SwinePose) |
| Bristol OpenCows2020 / Cows2021 | Holstein ID and detection | Non-commercial government licence | Research only | [OpenCows2020](https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17); [Cows2021](https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7) |
| MegaDetector / PyTorch-Wildlife | Animal/person/vehicle detector for camera traps; "runs in more than 80 conservation programs" | Project MIT; V6 ships MIT and Apache-2.0 weight variants beside AGPL ones | No livestock or pose — but the playbook to copy (below) | [repo](https://github.com/agentmorris/MegaDetector); [docs](https://microsoft.github.io/MegaDetector/) |
| Google SpeciesNet | 2,000+ label species classifier on MegaDetector output | Code Apache-2.0 | Wildlife; livestock labels not confirmed | [repo](https://github.com/google/cameratrapai) |
| OpenCV Zoo | Edge model zoo | Apache-2.0 | No animal models | [repo](https://github.com/opencv/opencv_zoo) |
| Bhujel et al. survey | 58 public livestock CV datasets; cattle about half; detection dominates; "limited quantity of high-quality annotated datasets" | — | The catalogue confirms the gap | [arXiv 2406.10628](https://arxiv.org/abs/2406.10628) |

**What is genuinely empty:** an open, maintained, commercially licensed, multi-species livestock
keypoint model zoo with a shared benchmark — and, narrower still, any open livestock *face-landmark*
schema with released models. We found no livestock leaderboard at all. One 2026 MDPI paper on sheep facial
landmarks ([Agriculture 16(2):151](https://www.mdpi.com/2077-0472/16/2/151)) could not be opened (HTTP 403)
and must be read before we claim "first".

### 3.4 The status quo the platform replaces

| Who | Stack | What got lost | Source |
|---|---|---|---|
| CSIRO, cattle behaviour (CVB) | YOLOv7 + BoT-SORT pre-labels, corrected in CVAT | Nothing — "only 29%" of 136,598 boxes needed adjustment. This is our pre-label-then-review loop, done by hand | [arXiv 2305.16555](https://arxiv.org/html/2305.16555) |
| FLI/FBN calves (Frontiers, 2026-01-16) | Label Studio + YOLOv8s-pose | 7 keypoints used of a possible 30; data "on request" | [Frontiers](https://www.frontiersin.org/journals/animal-science/articles/10.3389/fanim.2025.1718641/full) |
| Grecheneva et al. (*Veterinary World*, 2026-10-03) | YOLOv8l-pose, 38-keypoint whole-body cattle schema, 12k images, animal-level holdout | No dataset or weights release stated | [Vet World](https://veterinaryworld.org/Vol.19/October-2026/3.php) |
| YOLO-BCD sheep posture (Sensors, 2025) | LabelMe + custom YOLO | "not publicly available due to ongoing study" | [PMC12074352](https://pmc.ncbi.nlm.nih.gov/articles/PMC12074352/) |
| Russello et al., dairy lameness (2024) | T-LEAP pose, 9 keypoints | No dataset link on the abstract page | [arXiv 2401.05202](https://arxiv.org/abs/2401.05202) |
| Pig pose (Mendeley, 2025-12) | 22 keypoints, COCO format, CC BY 4.0 | Open, but 200 images | [Mendeley Data](https://data.mendeley.com/datasets/sn49zt6jpw/1) |
| Label Studio → YOLO-pose glue | Keypoints "are not linked to a detection" — open since 2023-08-21 | Every multi-animal pose lab writes its own exporter | [ml-backend #324](https://github.com/HumanSignal/label-studio-ml-backend/issues/324) |
| Field-level review (Animals, 2025) | YOLO across animal phenotyping | Datasets "suffer from small scale, inconsistent annotation standards, and poor cross-species generalizability" | [PMC12023960](https://pmc.ncbi.nlm.nih.gov/articles/PMC12023960/) |

**In one line:** every lab picks an open labeling tool, invents its own keypoint schema (7 in the calf
study, 9 in Russello, 15 in SwinePose, 18 in Wang et al., 22 in the Mendeley pig set, 38 in Grecheneva),
fine-tunes Ultralytics YOLO-pose under AGPL, splits however it likes, and does not
release the data. A shared schema, a leak-proof split and a benchmark are what is missing; more tooling
is not.

### 3.5 Who could crush us, and how

| Threat | How | Our answer |
|---|---|---|
| **Roboflow** | Publishes a "livestock pose" model on Universe and wins on distribution overnight | Be the content it would cite: schemas, benchmark, field cards. Mirror our datasets on Universe for discovery |
| **Ultralytics** | Adds one `cow-pose.yaml` to its docs; it becomes every student's default | Get our schemas into that conversation first (an upstream docs PR for `sheep-face-36` is a cheap move); K8 |
| **DeepLabCut / EPFL** | Ships a "SuperAnimal-Livestock" and owns academic mindshare | Its weights are non-commercial, so it cannot be the commercial default. Use DLC 3 as a trainer adapter; cite, don't fight |
| **GEA, Nedap, DeLaval** | Own the dairy camera mount, RFID and herd software | Do not compete in dairy. They cap the dairy upside; they do not threaten an open layer |
| **Foundation models (SAM 3)** | Shrink the labeling moat | A labeling tool alone does not survive this; a schema, a benchmark and adapted weights do |

### 3.6 The analogue to copy: MegaDetector, not Roboflow

MegaDetector became the default for camera traps by doing a few specific things: one narrow upstream task
everyone needs; free, with the project under MIT; meeting users inside the tools they already ran
(AddaxAI, Timelapse); processing users' backlogs for free; public training data on LILA; a public
"who's using it" list; and a downstream model (SpeciesNet) built on its output. Its V6 release ships MIT
and Apache-2.0 weight variants next to AGPL ones, which is a working precedent for escaping the Ultralytics
license while staying in the YOLO world ([MegaDetector](https://github.com/agentmorris/MegaDetector)).

Our equivalent: a permissively licensable livestock head/face detector and keypoint base that emits
Label Studio (later CVAT) pre-annotations, so it becomes the first stage of every lab pipeline in §3.4;
versioned species schemas; a per-video, field-robustness benchmark with a public table, with Wang et al.
2026 and SwinePose invited as its first external entries; and free adaptation runs for the first external
groups, the way MegaDetector processed backlogs.

---

## 4. Why we can win, and why we might not

**Assets that exist today (all checkable in this repo or on Hugging Face):**

| Asset | Evidence |
|---|---|
| Open sheep keypoint model line v0.2→v0.7 with a training-scale study | [`antking1/sheep-pose-yolo26n`](https://huggingface.co/antking1/sheep-pose-yolo26n), DOI [10.57967/hf/9129](https://doi.org/10.57967/hf/9129); `docs/v0.7-benchmark.md` |
| A measurement benchmark, not just mAP: ear-angle residual σ on near-stationary sheep, two held-out clips, bootstrap CIs | `sheep-yolo/scripts/bench_held_out.py`, `sheep-yolo/artifacts/*.json` |
| An anti-overclaim contract the welfare community will respect | `VALIDATION.md` |
| A field-robustness harness with real-failure holdouts and a public-data run on cattle/goat/sheep camera traps | `experiments/failure-transfer/` (on the unpushed `claude/failure-transfer` branch) |
| A working 36-point face labeling loop (task maker that refuses judge clips, SAM 2.1 pre-labels, Label Studio job, scorer) | Local `fun/sheep-masks/skin/`; promoted in week 1–2 |
| A farm, a flock, a Pi camera recorder | `pi-deploy/` |

**Liabilities, stated once:**

- One person. Labeling hours, not code, are the bottleneck (the 90-day plan budgets ~18–25 h of Antone's time).
- The v0.x training data is gone (2026 laptop crash). v1.0 starts from new footage.
- No institution. Credibility has to come from artifacts and an external replication, as DeepLabCut's
  did before it had one (see `docs/research/spfes-deeplabcut-deepercut-memo.md` §4).
- Every published weight file is AGPL-3.0 because it was trained with Ultralytics. Fine for research; a
  blocker for any closed product until the non-Ultralytics trainer path (DeepLabCut 3, decision D4) exists.
- The sheep market is small. USDA counts 4.99 million sheep and 2.51 million goats in the US on
  2026-01-01, against 86.2 million cattle (9.57 million milk cows)
  ([sheep and goats](https://esmis.nal.usda.gov/sites/default/release-files/795751/shep0126.txt),
  [cattle](https://esmis.nal.usda.gov/sites/default/release-files/795748/catl0126.txt)). Sheep is the
  proof, not the revenue.

---

## 5. Task taxonomy: what is platform (P0) and what is application

A task is **platform** if every downstream livestock product needs it and it must be measured the same
way everywhere. It is **application** if it encodes a buyer's decision or needs domain validation we
cannot do.

| Task | Tier | Why | Who already covers it |
|---|---|---|---|
| Animal detection (body box) | P0, but thin | Every pipeline starts here; COCO already has `cow`, `sheep`, `horse`. Our value is the field report card, not the class | Stock YOLO (COCO), Open Images models |
| **Head detection + face keypoints** (sheep-face-36) | **P0, the wedge** | Ear posture, eyes and mouth are the welfare-relevant signals; nothing open exists at this density for sheep | Nobody open (SFLW is 850 stills, on request) |
| Body pose (quadruped skeleton) | P1; **the second-species test** | Gait, lameness proxies, posture. Wang et al. 2026 gives an open (CC BY 4.0) cattle/horse/sheep 18-point set to test the schema registry on | DeepLabCut SuperAnimal (non-commercial weights), AP-10K, Wang et al. 2026, SwinePose (pigs) |
| Multi-animal tracking | P1 (eval protocol only) | ID switches in flocks are the failure everyone hits | Ultralytics' six trackers (default TrackTrack) and Roboflow `trackers` (Apache-2.0); we add the flock-specific metric |
| Individual re-identification | P2 | High value for cattle (coat patterns), data-hungry, privacy-adjacent | Academic (Bristol cattle-ID line); commercial vendors |
| Counting through a gate or line | Application | Ultralytics already ships line/region counting; we supply a detector that survives the field and a count-error protocol | Ultralytics solutions |
| Behaviour (grazing, lying, rumination) | Application | Needs temporal labels and a buyer question | Sensor and collar vendors |
| Body condition score proxies | Application | Needs 3D or top views and vet scoring | DeLaval and other vendors |
| Pain / welfare scales (SPFES) | Application, downstream | Needs clinical footage and an external trained scorer; never a platform claim | Academic only |
| Segmentation | Tool, not a task | SAM is the labeling backbone; we do not ship segmentation models | Meta SAM 2.1 / SAM 3 |

**P0 for the next 90 days is two rows:** head detection + sheep-face-36, and the field report card on
detection. Everything else waits for the day-90 review.

---

## 6. The robustness hill, productized

Antone's research hill — models break in the field for reasons lab mAP never sees — becomes a product
feature in one form: **no model ships without a field report card.**

A field report card is a fixed section of every model card:

1. **Clean numbers** on a split that cannot leak (per video, animal, flock or camera).
2. **Per-cause synthetic curves** for the five lens-side effects already implemented in
   `experiments/failure-transfer/degrade.py` (haze, dust, smudge, water spots, glare): onset, s50 and
   AUC, against a JPEG-95 noise floor. Each seed fixes one pattern and severity only scales it, which
   Albumentations' random-per-call effects cannot give (README, "What already exists").
3. **Real-failure holdout numbers**: frames that actually failed in the field, tagged by cause, never
   trained on, with bootstrap intervals that resample whole clips or cameras.
4. **The reality check** linking 2 to 3: does the synthetic breaking point predict the real per-cause
   drop? Until that check passes for a cause, the synthetic number is labelled "simulator only".

What exists today: the harness, the five effects, a consistency-mode smoke run on the two judge clips
(haze onset 0.19, smudge 0.21, glare 0.26, dust 0.43, water spots 0.63; label-free, so a statement about
answer stability, not accuracy), and the WCS public-data pipeline with 133 real degraded frames tagged by
cause. The corrected camera-level WCS run is week 1–2 work.

What the hill does *not* yet cover, and should not be claimed: scene-side effects (low sun, shadows,
rain, snow, mud on animals), animal-side shift (breed, fleece length, horns, lop ears), and night IR.
Those enter the card as real-failure holdouts first, synthetic effects only if the reality check earns
them a place.

The flywheel: users tag their own failures in the `failures.csv` format, those become holdout rows on
their side, and the recipe they produce augments their adaptation run. That is the one data-contribution
loop worth building, because it is the one users get value from on day one.

---

## 7. Go-to-market

**Who the first user is.** Not farmers: farmers do not install Python packages, and selling a sheep
vision product to farmers is the business case Iris Data Science could not make (§3.2). The first users
are the people in §3.4 who currently rebuild the stack — precision-livestock and animal-welfare
researchers, and engineers at small ag-tech companies — the same kind of "method-hungry field"
DeepLabCut served in neuroscience. Users are not buyers; the buyer question is separate.

**Who might pay — hypotheses, ranked, none tested:**

1. *Camera and ag-tech startups* building closed products on someone else's footage (the MyAnIML /
   HerdSense shape in §3.1) that need a model adapted to their animals and cannot ship AGPL weights.
2. *Animal-health companies* that need objective, repeatable measurements as trial endpoints. The
   evidence is indirect: OneCup's domain now sells "AI-Originated Clinical Endpoints for Animal Health"
   for pets (§3.2).
3. *Processor and retailer welfare schemes*, the reason CattleEye and HerdVision customers buy (§3.1) —
   reached through those vendors, not directly.

Phase 1 exists to find out which of these is real. If none is by month 9, this is an open research
project with a services sideline, and the docs should say so.

**Sheep first or cattle first?** Sheep first, cattle second, and be clear about why:

- Sheep is where we can be the reference in 90 days: we have the model line, the schema, the footage
  pipeline and no open competitor at face density.
- Cattle is where the money is, but cattle vision revenue today sits with hardware-integrated vendors in
  dairy parlours and feedlots (§3). A solo open-tools company does not win a dairy lameness contract.
  We enter cattle as *content* — a cattle schema, a robustness benchmark on public cattle data, and
  adaptation services — not as a product competing with those vendors.
- The product is the adaptation, not the species. Sheep proves it; cattle shows it generalises.

**Sequence.**

| Phase | Offer | Price | Trigger to move on |
|---|---|---|---|
| 0 (days 1–90) | Open dataset, weights, demo, adaptation notebook, benchmarks; **free adaptation runs for the first external groups** (MegaDetector's free-backlog move, §3.6) | Free | All-"Go" day-90 review |
| 1 (months 4–9) | Adaptation as a service: we label and adapt to a customer's footage with the open tools and deliver weights trained outside Ultralytics (DeepLabCut 3, or Ultralytics with an Enterprise License priced in) plus a field report card on their data | Fixed-fee projects | ≥ 2 paid projects, ≥ 1 repeat (kill criterion K10) |
| 2 (only if gated) | Hosted adaptation and private evaluation | Subscription | The four hosted-C gates in `architecture-options.md` |
| Throughout | Edge pack: Pi / Jetson recipes and exports | Free | — (distribution, not revenue) |

**Distribution through incumbents, not against them.** Ultralytics-format weights and dataset YAMLs so
`YOLO("…")` just works; a Roboflow Universe mirror of the public dataset for discovery; Hugging Face as
the canonical home with DOIs; a CV4Animals workshop paper and the arXiv preprint for the research user;
model cards that carry the field report card so the format spreads with the weights.

**What would make this a company rather than a project.** Two integrators paying for adaptation in phase
1. Until then, it is a credible open project with a services option, and that is the honest description.

---

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Labeling hours do not materialise | High | Hours budgeted per week in the 90-day plan; K1 stops coding if the judge set is not marked |
| Our base does not beat COCO-init at 200 labels | Medium | That is K4; the result is published either way |
| Ultralytics or Roboflow ships livestock content | Medium (Ultralytics re-hosted livestock datasets in Aug 2026; Roboflow already markets a livestock vertical) | K8: become their supplier; our datasets and benchmarks remain citable |
| AGPL blocks the first commercial deal | High if unaddressed | DeepLabCut 3 adapter ships by day 90; K9 |
| No buyer exists (OneCup, Plainsight and Iris left or stalled) | Medium–high | Users before buyers; K10 ends the company framing at month 9 if no one pays |
| A competitor with farm access out-labels us | Medium | Competing on data volume is a loss; compete on schema, split discipline and evaluation, which volume does not buy |
| Repo is public, so this strategy is public | Certain | Nothing here depends on secrecy; execution and data are the moat |

---

## 9. Decisions for Antone

| # | Decision | Recommendation |
|---|---|---|
| D1 | Architecture | **A now, as the open half of C; B killed; hosted C gated** (`architecture-options.md`) |
| D2 | Second species for the platform test | **Cattle body pose from Wang et al. 2026** (CC BY 4.0 data, COCO keypoints with visibility, published baselines to reproduce), added as a `cattle-body-18` schema file; time-boxed to 10 working days. Not the Bristol sets (non-commercial) and not a dairy product |
| D3 | Dataset license for `sheep-face-36` | **CC BY 4.0** (adoption) unless you want a commercial lever (CC BY-NC) |
| D4 | Trainer outside Ultralytics | **DeepLabCut 3 top-down** (LGPL-3.0 code, v3.0.2 released 2026-09-23, ships [`rtmpose`](https://github.com/DeepLabCut/DeepLabCut/tree/main/deeplabcut/pose_estimation_pytorch/config/rtmpose) configs and CSPNeXt/HRNet backbones) as the second adapter, never with SuperAnimal weights in a product. Not MMPose: Apache-2.0 but no release since 2024-07-12. RF-DETR (Apache-2.0) is the candidate head detector for a fully non-Ultralytics stack. Check each backbone's pretrained-weight license before a commercial release |
| D5 | Ultralytics Enterprise license | Do not buy now. Price it when K9 triggers |
| D6 | Name | Pick before the first public release (targeted Fri 2026-12-18); no "Ultralytics"/"YOLO" in it |
| D7 | Where the platform lives | `herdkit/` starts as a package inside this repo for speed (weeks 1–2). At naming (D6) it moves to its own repo, and SamSeesSheep becomes its first species vertical and demo, depending on it. This strategy package moves with it. SamSeesSheep stays the sheep repo, not the platform brand |

---

## 10. Outside-judge checklist

| Judge criterion | Where it is met |
|---|---|
| Every competitor claim has a URL | §3 tables; `ultralytics-cattle-inventory.md` |
| Ultralytics cattle inventory cites specific pages/paths | `ultralytics-cattle-inventory.md` |
| Architecture recommendation has kill criteria | `architecture-options.md` K1–K10, hosted-C gates |
| 90-day wedge names a stranger-openable artifact | `90-day-wedge.md`: the Hugging Face Space (F) plus dataset, notebook, benchmark card |
| No fake users, counts or certifications | No user, partner or customer is claimed anywhere; all candidates are labelled "not contacted" |
