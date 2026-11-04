# Mozilla Human-Scale AI — plan for a fundable on-farm vision stack

**Status:** planning only. Nothing in this folder is the application.
**Audience:** Antone King. Written so a skeptical Mozilla reviewer can be anticipated, not flattered.
**Deadline:** Friday, 20 November 2026, 11:59 p.m. Pacific (UTC-8). No extensions ([FAQ](https://www.mozillafoundation.org/en/what-we-do/grantmaking/incubator/human-scale-ai/call-for-proposals/faqs/), last updated 15 October 2026).
**Form:** [Initial Project Proposal](https://mozillafoundation.tfaforms.net/f/humanscaleai). Primary review happens here. A few applicants are invited to a full application. Submitted forms cannot be edited.
**Cohort:** up to $300,000 and 12 months of support starting April 2027. Possible further $200,000 and two years (up to $500,000 over three years), judged later on technology, product-community fit, public benefit, and a roadmap. Seven projects, plus three preselected anchors. Non-dilutive. Virtual. Worldwide, subject to sanctions rules.
**Sources read for this plan:** the live CFP, the FAQ, the eligibility PDF, the live form’s field list, this repository as of `849b0ba` (30 October 2026), the public GitHub API, and the public Hugging Face model API. The attached post image is a grainy human eye and a small server. It contains no application rules. The CFP does.

This document is an AI-assisted planning memo. Mozilla’s AI policy allows tools for brainstorming and clarity. The submitted answers have to be your words and your ideas. Do not paste these files into the form.

---

## Recommendation

**CONDITIONAL.**

On the packet that exists today, an outside reviewer should not advance this past the Initial Project Proposal. The sheep-head pipeline is real, small, and unusually honest. The thing Mozilla says it is buying is a working technology already in use by people other than its builder, with early product-community fit. That evidence is missing.

Submit on 20 November only if, by the end of 18 November, both of these are true:

1. One named person who is not you has used the published weights or the labeler on real footage, and you have their permission to describe that use in concrete terms (who, what they ran, what happened).
2. A two-minute video shows the model running on pasture footage, including one failure you are willing to say out loud.

If (1) is not true, the recommendation flips to **NO-GO** for this deadline. Fill nothing. The form’s own screening flags are built to drop “working prototype” plus “none of these yet.” Better prose will not move those flags. A v0.8 training run will not move them either.

**Single highest-leverage action:** book one real external use this week and get a quotable result. Everything else in the 21 days is downstream of that.

Fellowship is a legal parallel application (the FAQ says you may apply to both). It is the wrong story to hide inside. The tech is past “an idea.” The hole is users. A Fellowship application does not repair an Incubator packet that claims deployment it does not have.

---

## Thesis a reviewer can repeat

**A few-megabyte vision model that stays on the farm, scored by whether it still works in mud, mist, night, and occlusion. Sheep-head keypoints are the working wedge, not the product line.**

The durable outcome this grant would buy, if the condition above is met:

1. A small on-farm vision runtime whose daily path does not require a frontier model, a hyperscale API, or a copy of the farmer’s footage in someone else’s cloud.
2. Public building blocks other farms and labs can reuse: a failure-mode scorecard, the teacher-to-student training recipe, the weights, and the anti-overclaim contract.
3. Evidence, collected in the open, that handlers and small-flock people will actually use that runtime. Livestock first. A crop head only after a real crop dataset or a real crop partner exists.

SamSeesSheep / SPFES ear angle is the demo that proves the recipe. It is not the brand. The brand is agricultural vision that survives the farm. OpenBarn is shut down and does not appear in the application under any name.

---

## What was verified, and what was not

### Verified from this clone, the paper, and public APIs (30 October 2026)

| Fact | Where it lives |
|---|---|
| Pipeline runs: phone clip → SAM 3 Video text prompts → human review (`v=2`) → YOLO-pose export → train → local inference | `README.md`, `ARCHITECTURE.md` |
| Runtime model is YOLO26n-pose, 2.5 M parameters, ~6 MB `best.pt` | `README.md`, `paper/paper.tex` |
| Training recipe held still across v0.2–v0.7: 100 epochs, batch 8, imgsz 640. Data scale is the variable: 98 → 523 reviewed instances, 3 → 11 videos | `README.md`, `docs/v0.7-benchmark.md` |
| Held-out ear-angle residual σ: HO-1 (`IMG_3651`) v0.7 is 3.70° / 4.46° (L/R). HO-2 (`Test_Clip_Morning`) v0.7 σ_avg is 2.84°. The paper treats 2.84° as an easier clip, not a model upgrade. v0.4 on HO-2 is 2.61° | `docs/v0.7-benchmark.md`, `paper/paper.tex` |
| Stock `yolo26n.pt` produces zero keypoints. README: sheep boxes on ~35% of frames. Paper discussion also says 12–35%. Do not quote both as one number | `README.md`, `paper/paper.tex` §Discussion |
| Inference benchmarks ran on a GTX 1660 Ti, 6 GB, described as a **stand-in**. “Low-power edge deployment is the design target (a roadmap goal)” | `paper/paper.tex` lines 121–122 and §Edge Inference |
| README tagline was corrected on purpose: inference is “locally on a desktop GPU,” and an earlier “at the barn” wording was removed (`e7ae901`) | `git log` |
| One flock: ~5 Katahdin ewes plus lambs, one iPhone, Middletown, Delaware, April–May 2026 ambient light, one non-veterinarian annotator (you). Held-out clips are the same animals | `VALIDATION.md`, paper limitations |
| σ is precision (frame-to-frame jitter), not accuracy against an independent landmark | paper limitations |
| Welfare, pain, stress-event validation, cross-flock, cross-breed, inter-annotator agreement: explicitly unclaimed. A 70% stress-event kill criterion exists for a **future** study | `VALIDATION.md` |
| Known failure modes (dark fleece / dusk, motion blur, extreme angles, occlusion, wool ears) are written down. `VALIDATION.md` still labels that list “predictions, not measurements” | `VALIDATION.md` §Known and anticipated failure modes; paper §Known Failure Modes |
| Code license MIT. Weights AGPL-3.0 because Ultralytics. DOI `10.57967/hf/9129`. Hub repo `antking1/sheep-pose-yolo26n` | `LICENSE`, `README.md`, `docs/hf-model-card.md` |
| Weights, raw clips, and `data/` are gitignored. A fresh clone has no `sheep-yolo/weights/`, no `data/`, no `test-clips/` | `.gitignore`, this workspace |
| val pose mAP 0.479 / 0.643 / 0.732 for v0.2–v0.4 is disclosed as unrecoverable RunPod logs | `docs/paper-substantiation.md` |
| Public GitHub `antonemking/SamSeesSheep`: created 13 April 2026, last push 26 September 2026, **1 star, 0 forks, 0 subscribers, 4 open issues, all opened by `antonemking`, 0 comments**. The issue titles mention a “farmer Look/Skip demo.” The repo description says “Built for shepherds, shepherdesses, anyone with fluffy sheep” | GitHub API, 30 Oct 2026 |
| One GitHub release, `paper-artifacts-v0.7` (12 June 2026). Asset `sheep-pose-caches-v0.7.tar.gz` download count **1** | GitHub API |
| Hugging Face: **83 downloads, 0 likes**, created and last modified 12 June 2026 | HF API, 30 Oct 2026 |
| No `CONTRIBUTING`, no code of conduct, no security policy. `docs/CLOUD.md` is cited from `AGENTS.md` and is not in this tree | repo search |
| Check-count drift: `paper/README.md` says the verifier is 44/44. The model card says 59/59 | those two files |
| VLM orchestrators (Claude, Gemma) were **removed** from the pipeline in April 2026 | `CHANGELOG.md` |

### Named in planning context, not established by the repo

Treat all of these as `[CONFIRM]` before they touch the form.

- A farm Raspberry Pi used historically for capture. No Pi, ONNX, ncnn, or Jetson path is in the tree. Do not claim on-device farm inference.
- A local external-drive dataset with train / never-train / inbox splits, plus a Label Studio plan. Not in git. This plan did not open the drive.
- Lorewood Labs as a public or legal entity. The repo copyright is “Antone King.” Lorewood was removed from the README and the license header on purpose.
- Any consulting income. It is not project revenue, not a partner, and not a community signal. Leave it out of the form.
- Whether the preprint was actually submitted to arXiv. The repo says “arXiv-ready.” No arXiv id is in the tree.
- Whether anyone besides you has run the labeler, the weights, or a failure capture.
- Current legal ability to receive a USD grant, and whether any Mozilla staff relationship creates a conflict under the eligibility PDF.

### Explicitly out of bounds

- OpenBarn, under that or any other name. It is not a live product.
- A crop model. The only “crop” hits in code are image crops. The paper mentions crop monitoring as someone else’s related work.
- A second farm, a veterinarian partner, a handler panel, letters of support. None are in the repo. `VALIDATION.md` records the day-0 decision: “no recruitment, no vet partner, no other farms in v0.”

---

## Outside-judge rubric

Scale: 0 absent, 1 trace a reviewer will discount, 2 real but narrow, 3 plausible for this cohort if the rest of the packet is honest, 4 strong, 5 anchor-level. Grades use the words you asked for: **deployed / pilot / aspirational / missing**.

“Deployed” here means a person the project is for can use it in their setting without you in the loop. By that definition, nothing in this repo is deployed. The pipeline is a **pilot of one**: you, your flock, your desktop GPU.

| Criterion (CFP wording) | Grade | Score | Evidence the reviewer can open | Risk if you write it hotter than the grade |
|---|---|---|---|---|
| **Human-Scale AI** — “a compelling expression… reflecting the Small, Intentional, and Tailored principles” | Mixed. Small is pilot. The other two are aspirational | **2** | See the three rows below. The umbrella score is the weakest principle, because the CFP says projects should embody **all three** | Checking all three boxes and then describing a solo lab demo reads as a slogan |
| **Small** — “small models and less compute… local or low-power devices,” and the choice must make the product “more capable, accessible, affordable, private, or practical” | **Pilot** for model size. **Aspirational** for the device it must run on | **3** | 2.5 M params, ~6 MB, no frontier API on the runtime. CFP’s own text example of “small” is a 7 B text model. This is orders of magnitude under that, which is the right comparison for vision. Training is a ~6 minute 4090 job. Benchmarks are a 6 GB desktop card standing in for the edge | Calling it on-farm or on-Pi. The paper already refused that claim. SAM 3 (~3 GB, 24 GB GPU for the nose session, global singleton, no lock) is still required to **make** a new dataset |
| **Intentional** — “insight into how their data is collected, stored, transmitted, and used” and agency to “configure, adapt, fork, own, or govern” | **Aspirational** | **1** | Design intent is strong: raw video stays with the flock owner; only a 6 MB weight file needs to move; v=2 review is a human gate. MIT code can be forked. Weights are on the Hub under AGPL | There is no control surface a second farmer can see. Labeling today is a cloud GPU pod you SSH to. Labels are not public. There is no export/delete story written for anyone but you. Open source is not, by Mozilla’s eligibility PDF, automatically a charitable purpose or user agency |
| **Tailored** — “specific problems, communities, and contexts,” especially resource-constrained or underserved, and “in active conversation with the people who will use the technology” | **Pilot** for one homestead. **Missing** for a community, and **missing** for crops | **2** | The system is shaped by one Katahdin pasture: phone video, multi-sheep frames, ear tips that stock YOLO cannot see, a noise floor that saturates after v0.4. That is real tailoring. It is also a sample of one, and the tailor is the only user | “Livestock and crops,” “shepherds” (the GitHub description), or “underserved farmers” as a class. You have not talked to that class in any artifact this review can check |
| **Technical viability** — “novel technology with real momentum,” capacity, fit to problem, budget, and grant term | **Pilot** | **3** | Seven published iterations in about two months (April–June 2026), a stock-model baseline, bootstrap CIs, a window-sensitivity sweep, a written regression (v0.6), and a verification script. Novelty is the recipe (foundation model as annotator, tiny model as runtime, pasture noise floor published), not a new detector architecture. DeepLabCut / SLEAP already do animal pose. CattleEye already sells commercial cattle vision | Momentum optics: Hub frozen 12 June, GitHub last push 26 September. Solo capacity is proven for the wedge and unproven for a two-domain product. $300 k is enough; the term is long enough for a scorecard and one more head, not for a platform |
| **Problem-community clarity** — “who it is solving it for,” “evidence of dialogue,” “a real-world need” | **Aspirational** | **1** | The need is concrete in your own pasture: off-the-shelf detectors do not yield ear angle; lab pose tools assume controlled capture; commercial PLF assumes a vendor cloud. Dialogue evidence is you watching your own sheep. Day-0 `VALIDATION.md`: “no recruitment, no vet partner, no other farms” | A problem essay with no second voice. Reviewers have been told to look for dialogue |
| **Product-community fit** — users returning, contributors, community shaping, partners, downstream adoption. “You don’t need to have achieved” it. You do need “early evidence” | **Missing** | **1** | You return to your own tool. That is the entire positive. Public counters: 1 star, 0 forks, 0 issue comments from anyone else, 0 Hub likes, 83 Hub downloads, 1 release-asset download. Downloads are not users. `sheep-yolo/README.md` says “Not a product.” The honest form checkbox today is **“None of these yet.”** | Checking Users, Contributors, Partners, or Downstream because the dropdown “Less than 100” can hide a count of one. A reviewer who clones the repo will catch it |
| **Impact and values** — public benefit, agency, transparency, openness, community control | **Aspirational**, with real open artifacts | **2** | MIT pipeline, public weights, public negative results, a validation contract that forbids pain/welfare claims. Benefit beyond you is still potential. Eligibility PDF: “Touching on a good cause, or using open source licenses, isn’t enough on its own if the underlying purpose is commercial.” | A commercial PLF pitch, or a claim that publishing weights already delivered community control |
| **Sustainability potential** — “a plausible path… that aligns with its values,” and openness to different models | **Aspirational** | **1** | You have shipped in public for six months without a company wrapped around it. That is a values signal. There is no written path for who maintains the scorecard when the grant ends, and no second maintainer | A SaaS revenue slide. Also the reverse error: pretending a solo repo survives on vibes. Follow-on $200 k explicitly re-checks product-community fit |

**Holistic read.** Cohort design also wants a mix of geographies, communities, and technical approaches. Anchors already cover offline education LLMs (Tiles), household encrypted memory (Family Intelligence), and shared renewable compute (DAIR). A small vision model for farms is not a duplicate of those three. That is an opening, not a qualification. Mix does not excuse a missing user.

**What would move the scores before 20 November.** One external use moves product-community fit from 1 toward 2, and is the difference between CONDITIONAL and NO-GO. It does not make this a 4. A filmed failure on your own flock moves Small/Tailored honesty; it does not create a community. Do not expect the rubric to become “strong” in three weeks. Expect it to become admissible.

---

## Red team against “What we’re not looking for”

CFP list, applied to this repo as it sits.

| They said they do not want | Where this project stands | Survive? |
|---|---|---|
| “Concepts without working technology, including ideas that exist only as wireframes or prototypes.” | The sheep pipeline works on your clips. The form’s stage list is `Concept or wireframe only` / `Working prototype` / `Deployed with early users (<100)` / `Deployed at scale (100+)`. Honest box today: **Working prototype**. That box sits next to the CFP sentence that brushes off prototypes. This is the screen | Only if you can truthfully move to “Deployed with early users” via a real pilot, or the 12 November session says a solo operator on their own animals counts. Ask. Do not assume |
| “Standalone governance, policy, or research projects without a working technology component.” | You have working tech. The paper is a supporting artifact. Do not lead with the preprint | Yes, if the form’s primary output stays “technology platform or tool,” with the paper secondary |
| “Technologies without users, contributors, or another meaningful signal of early product-community fit.” | This is the current miss. See rubric | No, unless the external-use condition is met |
| “Generic chatbot applications that apply existing AI capabilities to a new topic.” | VLM narrative code was deleted. Do not reintroduce a chat layer to look modern | Yes, as long as the proposal never acquires a chatbot |
| “Core functionality depends on a frontier-model or a slimmed-down version of a frontier model (e.g. Llama 1B or 3B).” | Form question, narrower than the CFP: “Does your project’s core functionality depend on calling the **API** of a frontier AI model?” Hint lists GPT, Claude, Gemini/DeepMind, Llama. Runtime answer is **No**. Parameter dropdown’s smallest bucket is “10 billion or fewer.” Both YOLO26n-pose (2.5 M) and a ~3 GB SAM 3 checkpoint fall in that bucket. Write the 2.5 M figure in prose so the bucket does not make you look like a 7 B app | Conditional. If a reviewer decides SAM 3 is the product, you look like a wrapper on a large foundation model. The proposal has to make the runtime the product and the teacher replaceable |

FAQ, same teeth: “we are looking for a working technology that people can actually use — not just an idea or wireframe.” And: “if you are using a large model, your application should clearly justify why.”

Internal form flags visible in the form HTML (names only; the scoring weights are not published): `Short Form Flag: Project Stage`, `Early Signals`, `Human-Scale AI Pillars`, `Project Outputs`, `Work in the Open`, `Charitable Purpose`; `Hard Stop Flag: Countries`; `Priority Flag: Video Link`. Assume “concept only,” “none of these yet,” no open outputs, no charitable purpose, or a sanctioned country shortens or stops the review. Assume a missing video lowers priority rather than auto-killing, because the CFP calls the video “optional but strongly encouraged.” Send the video anyway.

---

## Small, Intentional, Tailored — for farm vision, not for an LLM

The CFP’s size examples are 7 B–32 B text and voice models. Ignore that scale except as contrast. Agricultural vision does not need a language model to see a sheep or a leaf. Putting Llama in this stack to sound like Tiles would trip the frontier exclusion and make the product worse: a chat box does not survive mud on a lens.

### Small

What is actually small:

- The thing a farm would run every day: YOLO26n-pose, 2.5 M parameters, about 6 MB, one forward pass, benchmarked on a 6 GB consumer GPU.
- Training cost for a new head, once labels exist: on the order of six minutes on a 4090 at batch 8. Re-training is a burst, not a data center.
- No runtime call to GPT, Claude, Gemini, or Llama. The April 2026 changelog removed those calls. Keep them out.

What is not small, and must be labeled as teacher-only:

- SAM 3 Video, `facebook/sam3`, about 3 GB, up to three propagation sessions, nose session wants 24 GB, one global model, no lock (`ARCHITECTURE.md`). Two concurrent analyzes can OOM a 4090. That is acceptable for a single labeler and fatal as a multi-farm product.
- Frame extraction at 2 fps and ~512 px is a memory budget, not a deployment architecture.

The Small argument that matches the CFP’s “makes the product better” clause:

- A 6 MB file can live on the farm. A 3 GB foundation model that needs a rented GPU cannot be the daily path.
- The small model is also the one that **works** on this task. Stock YOLO, the general model, yields zero keypoints. Specialization beat scale. That is the Human-Scale claim in one result. Quote it with the baseline, the clip, and the “zero keypoints” outcome.
- Energy and privacy follow from the same choice. Do not invent a watt-hour number you have not measured.

Year-1 technical commitment, if funded: measure the runtime on one named low-power device you actually possess, and publish the frame rate, the failure, and the power setup. Until that measurement exists, the form says “desktop GPU stand-in, edge is the design target.”

### Intentional

Write the data path you have, then the path the grant would build. Do not merge them.

**Today.**

- Capture: your iPhone, your pasture.
- Labels: a pod you start by hand. `scripts/start_pod_server.sh` is not auto-started. Labels are meant to sit on a network volume. They are not in git, which is good for animal privacy and bad for “open dataset” claims.
- Review: you click `v=2`. Auto keypoints (`v=1`) do not enter normal training. `pseudo=true` exists and is documented as prototype-only.
- What crosses the network after training: `best.pt`, about 6 MB. That sentence in `CLAUDE.md` is the best Intentional fact you have.
- What a stranger can fork: the MIT code and the AGPL weights. What they cannot see: your video, your `review.json`, your animals.

**What a second farmer would need before “agency” is true.**

- A one-page data map: what stays on the farm (clips, frames, review files, local inference logs), what may leave (a weight file, a benchmark JSON with no frames), who can delete the copy, how long a rented GPU is allowed to hold frames.
- Inference that does not require the labeling pod. The pod is an authoring tool.
- A way to adapt the model to their animals without sending the flock to you. That is the point of the tiny retrain. It is also the point you have not productized.
- Governance that is more than a license: a label schema they can change, a failure scorecard they can add a condition to, and a written rule that ear-angle bands are not a diagnosis.

AGPL on the weights is a real constraint for some downstream builders. Disclose it. Do not apologize for it in the form, and do not pretend MIT covers the checkpoints.

### Tailored

The context that already shaped the tech:

- Hair sheep, not wool. Ear tips are visible. Wool ears are an untested failure, correctly written down.
- Handheld 1080p phone, no mount, no controlled light. April–May only. Morning vs afternoon is the lighting experiment you actually have (HO-2 vs HO-1), and the paper says the σ gap is mostly clip difficulty.
- Multi-sheep frames forced schema v2 (`instances[]`) and a flock monitor. That came from the pasture, not from a COCO checklist.
- The metric that matters on a calm animal is residual jitter, because a handler will stare at a trace and trust noise. You measured that and then showed it plateau. That choice is tailored. mAP is not the farm question.

The context that has **not** shaped the tech yet, and must not be claimed as if it had:

- Handlers, shearers, vets, or a second flock telling you which failure loses them the morning.
- Night, mist, lens smudge, mud, dust, season beyond spring, fixed cameras at a trough or chute.
- Any crop, any growth stage, any spray or scouting decision.
- Resource constraint as lived experience of a community. Your constraint is real (one homestead, consumer GPU, rented training bursts). It is not a proxy for “underserved farmers” as a population.

Tailored, said in a way that survives `VALIDATION.md`: “Built for one Katahdin homestead in Middletown, Delaware, by the person who lives there. The grant year is for learning which of those choices survive contact with a second site and with the failures this spring dataset does not contain.”

---

## Farm failure modes → system choices

Grades: **measured** (a number in this repo), **observed** (you have seen it, no published count), **untested** (named, not run).

| Failure | Grade in this repo | What the system does today | What the scorecard would add |
|---|---|---|---|
| Lighting swing, sun / cloud / shadow, spring | Observed across April–May. HO-1 vs HO-2 is a lighting-and-size contrast, and the paper refuses to call it a model win | Train on ambient phone clips. No controlled light protocol | Paired clips, same animal, same camera, two light conditions, σ and detection rate side by side |
| Morning vs afternoon | Measured only as two different clips of the same flock | Reported honestly as clip difficulty | Keep that honesty. Add a same-hour control so light is separable from head size |
| Motion blur | Untested as a number. Called the median case, not the edge case | 2 fps labeling subsample hides blur the runtime will see at 30 fps | Score the runtime at full frame rate on a moving animal. Publish the drop |
| Occlusion, crowded frames | Observed enough to motivate dedup and ROI filters. No rate | ByteTrack, keypoint-distance dedup, PolygonZone in spikes | Occlusion rate: fraction of frames where the ear tip is on the wrong sheep or missing |
| Profile / extreme angle | Described. Best on frontal and three-quarter | Keypoint `v=0` when the part is absent. That flag is the right representation | Report availability (`v=0`) separately from jitter. A missing ear is not a 0° ear |
| Dark fleece, dusk, low contrast | Predicted. Training animals are light-fleeced Katahdins. SAM 3 called out as degrading; the student inherits the hole | Nothing. You do not have the animals | Either borrow a dark-faced clip from someone who agrees to it, or write “untested” in the scorecard. Do not simulate with a brightness slider and call it dusk |
| Night | **Missing** | No IR, no night clip in the docs | A night row that is allowed to say “model fails.” A failure row is a Mozilla-shaped output |
| Mist, lens smudge, mud, dust | **Missing** | Not in `VALIDATION.md`’s failure list even as a prediction | Physical protocol: clean lens vs smudge vs mud film, same scene. This is the strange, question-magnet experiment. It is also cheap. It has not been done |
| Season / phenology | **Missing.** Spring 2026 only | Cannot claim coat, grass, or light changes across a year | A calendar commitment: one repeat capture in winter on the same fence line, even if the model falls over |
| Wool ears, horns, cropped ears | Untested. Correctly out of claim | Katahdin-only weights | A second breed is a generalization test, not a feature launch |
| Domain shift to another flock | Missing. Both held-out clips are the same five animals | “Held-out” means unseen frames, not unseen sheep | Say that in the form in one sentence so a reviewer does not have to discover it in the limitations section |
| Crops: growth stage, weed vs crop, disease, dew, soil splash | **Missing.** No model, no images, no partner | Do not map sheep keypoints onto plants in the proposal | A crop head is allowed in month 7–9 only if a person with plants has shown up. Otherwise the public output is the protocol, and someone else runs it on plants |

System choices the scorecard should lock, because they are already the architecture:

1. **Teacher is not the product.** SAM 3 proposes keypoints. A human sets `v=2`. The farm runs the student.
2. **The student is replaceable per task.** Sheep head is head #1. A crop head is a new label schema and a new 6 MB file, same export and same scorecard. Do not build a general farm foundation model.
3. **Report the noise floor, the drop under a named failure, and the stock-model baseline.** A single mAP will waste the reviewer’s trust. Your own v0.6 regression is the proof you know how to publish a drop.
4. **Keep welfare off the scorecard.** SPFES bands can stay in the labeler as literature reference. They are not the grant metric. The kill criterion in `VALIDATION.md` remains a later study with its own capture protocol.

---

## Anchors and near neighbors

Do not imitate their stack. Steal the shape of the proof: a working system, a specific community, data that does not leave, and a public block someone else can pick up.

| | Tiles Privacy Technologies | Family Intelligence | DAIR | This project, honestly |
|---|---|---|---|---|
| CFP’s own description | Offline education tools, open materials, governance. Quantized open weights through llama.cpp, encrypted P2P inference, identity on device | Household shared AI memory no outside server can read. Family Trunk assistant. Hub stores ciphertext. Family devices search locally. Archive exportable | Renewably powered shared compute so small orgs can run models without hyperscalers. NZ cluster, NVIDIA B300s, Kaskazi Foundry, isolation between orgs | A 2.5 M sheep-head pose model and the labeling flywheel that made it. Measured on one Delaware flock. Desktop GPU. No second user on record |
| Small | Small enough to run offline. Still a language model. They were preselected; the Llama 1 B/3 B exclusion is aimed at applicants, and their core is a local education stack, not “call GPT” | Local reasoning. The hub is not the model | **Not small models.** Small organizations sharing serious GPUs. Different principle | Actually smaller than the anchor LLM stacks. Weaker on “runs where the user is” |
| Intentional | Identity stays on device | Cryptographic refusal of server-side reading, plus export | Workload isolation | A sentence about weights being the only thing that crosses the wire. No crypto, no export button, no second user to hand it to |
| Tailored | Schools and offline classrooms | One household | Values-aligned small orgs | One homestead. The tailoring is physical (pasture, phone, hair sheep), not yet social |
| What to learn | The demo is the offline path, not the benchmark table | “No outside server can read this” is a sentence a reviewer remembers | Public benefit can be shared infrastructure, open-sourced, without you owning a cloud | Your memorable sentence is the failure scorecard plus the 6 MB file. You do not have their deployment proof |
| What not to copy | llama.cpp, a chat tutor, a governance framework with no farm runtime | An encrypted assistant for the barn | A GPU cluster proposal. DAIR already occupies “shared compute,” and your daily path should avoid that dependency | — |

**CattleEye and the PLF category.** Commercial cattle vision (lameness, condition) is the thing farmers will mention. You are not in that category. You do not have a lameness model, a herd dashboard, or a deployment. The difference worth writing, once, is: their pattern is a vendor model and the footage leaving the farm; your pattern is a tiny model the farm can hold, judged by a public failure protocol. Do not claim you outperform them. Do not use their product screenshots. Do not say “open-source CattleEye.”

**DeepLabCut, SLEAP, DeepPoseKit.** They own lab animal pose and careful annotation. Your difference is pasture video, a foundation model used only as a teacher, and a published residual-σ protocol with a stock baseline. Say “we use a different annotation economy and a different test,” not “we replace DeepLabCut.”

---

## Twelve-month arc (April 2027 – March 2028)

The cohort starts April 2027. November 2026 through March 2027 is unpaid runway. Do not promise those months as grant deliverables. Do use them so April does not start from a silent repo.

Success Mozilla asked you to define: stronger technology, deeper product-community fit, a sharper sustainability path, and something public others can use. Follow-on money re-checks those, plus public benefit.

### Before the cohort (only if you submit)

- The external use in the recommendation, written down.
- A two-minute video.
- A one-page data map that matches reality.
- GitHub description and issues #1–#4 reconciled with `VALIDATION.md`, so a reviewer does not meet a “farmer Look/Skip demo” that the validation contract does not recognize.
- A failure-mode capture sheet: the rows in the table above, with “untested” already filled in. Empty rows are allowed. Invented numbers are not.

### Months 1–3 (Apr–Jun 2027) — score the wedge you have

- Freeze sheep-head v0.7 as the reference student. New training only to support a failure row, not to chase σ on HO-1. The paper already says the curve plateaued.
- Publish **Farm Failure Scorecard v0**: a spec, a JSON schema, and results on your flock for at least four rows. Two of those rows should be ones you expect to fail (night, or mud/smudge, or heavy blur).
- Run the same weights on one low-power device you can name after the run. Publish fps and the first thing that breaks.
- Write the data map into the repo. Daily inference does not start the labeling pod.
- Sit with the external person from November at least twice. Write down what they refused to trust. That note is a product-community artifact.

**Leave behind:** scorecard spec + your results, including the bad rows. MIT.

### Months 4–6 (Jul–Sep 2027) — one new head, or a documented refusal

Pick one, based on who actually showed up. Do not pick both.

- **Second livestock site or breed,** if a person with animals agrees to a capture protocol and a review pass, or
- **One crop head** (a single decision a grower already makes with their eyes: for example presence of a named stress in a named crop), if a grower and a pile of their frames exist.

If neither person exists, the deliverable is a two-page recruitment log and a scorecard that still only covers your flock. That is an acceptable public result. Shipping a fake crop demo is not.

Also in this window: one second annotator on a subset, even a small one, so “single operator” stops being a permanent shrug. Pay them. Inter-annotator agreement is a number you do not have.

**Leave behind:** either a second 6 MB head with the same scorecard, or the log that says the second head did not earn its data.

### Months 7–9 (Oct–Dec 2027) — the runtime a handler can start

- Capture on the farm, infer on the farm, log on the farm. A clip leaves the farm only when someone chooses to train or to publish a frame.
- A delete/export note a non-programmer can follow.
- Mozilla Festival showcase, if the cohort still does one: the scorecard and a failure, not a welfare readout and not a chatbot. The CFP calls the showcase their demo day. Show the 6 MB model missing an ear in the dark.

**Leave behind:** a setup path that does not mention RunPod in the first screen, plus the festival demo’s source clips or a reason they cannot be public.

### Months 10–12 (Jan–Mar 2028) — decision memo

- Does ear angle deserve another year, under the existing kill criterion, with a pre-registered stress protocol? Default is no, unless a vet or handler has asked for that study in writing.
- Is the thing other people reused the scorecard, the weights, or neither? Count named reusers, not Hub downloads.
- Sustainability memo in the open: maintainer, fiscal home, what the $200 k would be for, or a recommendation to stop at the public blocks.

**36-month picture, only as a hypothesis.** A handful of farms run the scorecard on their own animals. One crop group has reused the recipe without you owning their images. A second person can cut a release. If that did not happen, the charitable residue is the scorecard and the weights, and you say so. The CFP wants projects still standing in five or ten years. The honest version of that sentence is: the blocks remain usable even if the organization stays one person.

### What $300,000 has to unlock

The form asks this. The answer is time and other people’s attention, not a cluster.

- Your time off the unrelated consulting cycle, enough to do the field protocol. Say that you need the time. Do not attach a private rate or a client name.
- Stipends for handlers or a second annotator. This is the product-community budget line.
- Short GPU rentals for student retrains. Not a standing SAM 3 service for every farm.
- Travel or cameras only in service of the second site.
- Write-up, video, and the festival.

**Eligibility trap, read before you pick Applicant Type.** The [grant eligibility PDF](https://assets.mofoprod.net/network/documents/Mozilla_Foundation_Grant_Eligibility_Guidelines_tNhKrgM.pdf) says grants to **for-profit** entities cannot be general operating support and, as an example, may not buy “computers, servers, GPUs” over certain amounts. A Lorewood Labs application that depends on buying GPUs can fail even if the story is good. An **individual** charitable project is a cleaner fit for this work (scientific research and public education, benefit flowing outward). Fiscal sponsorship is the FAQ’s suggested path if you want an organization without 501(c)(3) status. Confirm the equipment rule for individuals at the 12 November session before you write the unlock paragraph. This plan is not legal advice.

Charitable purpose that fits the PDF: advance scientific research on whether small vision models survive farm conditions, and educate the public with the protocol, the weights, and the negative results. Open source by itself is not the purpose. A product whose primary purpose is commercial gain is ineligible. Keep paid farm software, if you ever want it, outside the grant scope.

### Open blocks Mozilla’s own “success” section is asking for

The CFP: “More building blocks: Projects leave behind open source code and documentation that helps developers build Human-Scale systems faster.” Also “examples of better technology,” “new design patterns.”

Ship these, in this order:

1. **Farm Failure Scorecard** — spec, schema, and a reference implementation that emits detection rate, keypoint availability, and residual σ under named conditions. Domain-agnostic columns, sheep as the first filled sheet.
2. **Teacher → student note** — how to use a large vision model only as an annotator, then delete it from the runtime. This is the pattern Tiles does not cover, because Tiles’ small model is still the language model.
3. **The weights and the verifier** — already started. Fix the 44-vs-59 check-count drift before you cite either number.
4. **Data-residency one-pager** — the map above. Family Intelligence’s analogue, without pretending you built encryption.
5. **Negative-result sheet** — stock model at zero keypoints, v0.6 left-ear regression, post-v0.4 plateau, “held-out” defined as same flock. You already do this. Make it a first-class grant output so it does not look like hedging.

Do not promise a public video dataset of the flock as the openness plan. Animal identity, farm location, and neighbor land are in those frames. Openness here is the protocol, the weights, the benchmark JSON, and selected frames you are willing to show. Say that limitation. It is an Intentional feature.

---

## Risks

| # | Risk | Likelihood if you write the form this month | Effect | Mitigation |
|---|---|---|---|---|
| 1 | Product-community screen. Honest checkbox is “None of these yet.” | High | Short-form or reject | External use, or do not submit |
| 2 | “Working prototype” read as the CFP’s excluded prototype | High | Same | Ask on 12 Nov. Do not tick “Deployed with early users” for a sample of one author |
| 3 | Proposal contradicts `VALIDATION.md` | High if you draft from the locked hill instead of the repo | Reviewer trusts the repo and drops the file | Every number and every user claim traces to a file or a named person |
| 4 | SAM 3 perceived as the core model | Medium | Frontier / “large model” FAQ miss | Runtime is the product. Teacher is justified in one paragraph and kept off the daily path |
| 5 | Livestock-and-crops scope reads as vapor | High if both are in the title | Technical-viability and tailored scores fall | Title the project as on-farm robustness. Sheep is the wedge. Crops are a gated milestone |
| 6 | For-profit applicant plus a hardware shopping list | Medium | Eligibility, not just scoring | Read the PDF. Prefer individual or fiscal sponsor. Ask about GPUs |
| 7 | Solo executor versus a two-domain year | High | You will ship the familiar wedge and miss the community | The budget buys stipends and time, not a second research program |
| 8 | Clone test. No weights, no clips, issues that sound like a farmer product | Medium | Reviewer concludes the README overreaches | HF link in the first screen. Issue tracker cleaned. Description edited |
| 9 | Number drift (35% vs 12–35%, 44 vs 59 checks, mAP that cannot be re-pulled) | Medium | Looks careless next to an honesty brand | Quote σ from `docs/v0.7-benchmark.md` only. Leave mAP out of the form |
| 10 | Welfare or pain language creeps back | Medium | Contradicts your own contract and sounds like CattleEye | Ban those words in the form except to say you do not score them |
| 11 | AI-shaped prose | Medium | FAQ: applications that appear entirely generated get further review | You rewrite every answer. Disclose the tools you actually used |
| 12 | Video shows a polished demo and hides the failure | Medium | Misses the only memorable claim | One failure on camera |
| 13 | GitHub quiet since 26 September, Hub quiet since 12 June | Medium | “Real momentum” is a viability question | A small public commit that is true (data map, scorecard template, issue cleanup) beats a silent November |
| 14 | OpenBarn named by habit | Low if you use a checklist | Instant vapor signal | The word does not appear in the form |
| 15 | Sanctions / country hard stop | Low for a Delaware applicant | Hard stop | Applicant location and “regions served” stay honest. Do not claim you serve the world |
| 16 | Info-session registration still “coming soon” on the CFP page as fetched 30 Oct 2026 | Medium | You miss the only live Q&A | Check the CFP on 9 Nov and 12 Nov morning. Email `incubator@mozillafoundation.org` only if registration is still absent on 10 Nov |

---

## Questions for the 12 November info session

12 November 2026, 5 p.m. UTC / 12 p.m. ET / 9 a.m. PT. One hour. Registration was “coming soon” on the CFP page at the time of this plan. Ask these. Do not pitch.

1. Does the builder’s own repeated use on their own animals count as “deployed with early users,” or does the project-stage flag require someone outside the team?
2. The form lists “Working prototype.” The CFP dismisses prototypes. Which of those controls the short-form flag?
3. A large open vision model used only as an offline annotator, with a 2.5 M model as the runtime and no frontier API: does that satisfy the large-model justification, or do you treat the annotator as core functionality?
4. Do Hugging Face download counters count as downstream adoption, or only a named person who used the artifact?
5. For a for-profit, the eligibility PDF restricts computers, servers, and GPUs. Does that restriction apply to an individual grantee? Is a short cloud GPU rental an allowed cost?
6. Can the 12-month public output be a failure-mode protocol plus one livestock wedge, with a second domain gated on a real partner, or do you expect both domains in the proposal title?
7. If farm video should not be published, is a protocol, weights, and benchmark JSON an acceptable “work in the open” package?
8. Is an unlisted two-minute phone video of the model, including a failure, enough for the video priority flag?
9. Agricultural vision is not represented by the three anchors. Is that a mix you still want, or is the cohort already aimed elsewhere?
10. One external pilot note by submit date: is that “early evidence,” or will it still read as none?

Write down the answers the same day. If question 1 comes back as “someone outside the team,” and you do not have that person, stop the submission on 13 November.

---

## Day by day, 30 October – 20 November 2026

Rule for the whole stretch: no retraining, no new architecture, no crop prototype, no application prose before 9 November. The scarce object is a person.

| Date | Action | Stop if |
|---|---|---|
| Fri 30 Oct | Read this plan. Write a private list of five people who have animals, handle yours, or would run a weight file. Not a mailing list | — |
| Sat 31 Oct | Ask three of them for a specific session: run the Hub weight on a clip, or sit a labeling review. Record yes / no / silence | — |
| Sun 1 Nov | Write the true data map (phone → pod → `v=2` → weight file → desktop GPU). One page. It becomes the Intentional answer | — |
| Mon 2 Nov | Second ask to anyone who has not replied. Read GitHub issues #1–#4 against `VALIDATION.md`. List what a reviewer would misunderstand | — |
| Tue 3 Nov | If someone said yes: schedule the session inside this week. If everyone said no: start the NO-GO note. Do not open the form | Three no’s |
| Wed 4 Nov | External session, or a written decline log. Capture their words. No coaching the quote | Session cancelled and no alternate |
| Thu 5 Nov | Decide the checkbox pair in writing: stage + early-signal. If the signal is still “none of these yet,” the decision is NO-GO unless the 12 Nov answer saves you. Park the form | — |
| Fri 6 Nov | Only if the session happened: cut a rough two-minute capture. Model on, one failure, you talking. Ugly is fine | No session |
| Sat 7 Nov | Repo hygiene list, not a refactor: GitHub description, issue titles, HF link visibility, 44-vs-59, README first paragraph still says desktop GPU. Fix only what would contradict the form | — |
| Sun 8 Nov | Draft the one-sentence thesis in your own words. Read it against `VALIDATION.md` out loud. Cut every sentence the contract forbids | — |
| Mon 9 Nov | Check whether info-session registration opened. Start outline answers in `INITIAL-PROPOSAL-OUTLINE.md` only for facts you can point to | — |
| Tue 10 Nov | If registration is still missing, mail `incubator@mozillafoundation.org` and ask for the link. Keep drafting the challenges answer. That answer is where you are strongest: solo, one flock, failures unmeasured, fit missing | — |
| Wed 11 Nov | Eligibility choice drafted: individual, fiscal sponsor, or for-profit, after a pass through the PDF. List the equipment you will **not** ask the grant to buy if the for-profit rule applies | — |
| Thu 12 Nov | Attend. Ask the ten questions. No demo unless they invite one | Missed session: email the ten questions the same day |
| Fri 13 Nov | Apply their answers to the checkbox decision. If the path is closed, stop. Tell yourself in one paragraph | They required an external user and you do not have one |
| Sat 14 Nov | Rewrite every long answer in your voice. Delete this plan’s phrasing where it shows | — |
| Sun 15 Nov | Contradiction pass: form vs `VALIDATION.md` vs README vs paper limitations. Second person reads it if the external user will | — |
| Mon 16 Nov | Upload the video unlisted. Paste GitHub, HF, DOI, video URL into a scratch doc. Watch the video once with the sound on | — |
| Tue 17 Nov | Enter the form. Save a resume link. Do not submit | — |
| Wed 18 Nov | Freeze. Re-read “what we’re not looking for” and the five signal checkboxes. This is the last day the recommendation can flip to NO-GO without a scramble | Any `[CONFIRM]` still load-bearing |
| Thu 19 Nov | Buffer. Fix a broken link or a video caption. Do not add scope | — |
| Fri 20 Nov | Submit before 11:59 p.m. Pacific only if 18 Nov stayed CONDITIONAL. If you are unsure at 9 p.m., do not submit. Late work is discarded. There is no extension | — |

---

## What not to claim

- Pain, welfare, stress, mood, or a clinical score. `VALIDATION.md` already bans the marketing version.
- That σ is accuracy, or that 2.84° means v0.7 beat v0.4.
- Cross-flock, cross-breed, cross-season, night, mist, smudge, mud, dust.
- On-farm, on-Pi, or “at the barn” deployment. The repo walked that back.
- A crop model, a multi-species platform, or OpenBarn.
- Users, partners, letters, downstream builders, or “83 users.”
- That GitHub issues are a community. They are yours, and the titles over-suggest a farmer demo.
- A public dataset of the videos.
- Recoverable mAP logs.
- That SAM 3 is small, or that the product calls it every frame.
- An arXiv publication, unless you have the id.
- Lorewood as the applicant, unless counsel and the PDF agree.
- Any dollar figure for unrelated consulting.
- That this plan, or an AI draft of the form, is your application.

---

## How this plan should be used

`INITIAL-PROPOSAL-OUTLINE.md` is the form, section by section, with the claims pre-weakened to what you can stand behind.

`GAP-CHECKLIST.md` is the gate. If the external-use box is unchecked on 18 November, you do not submit.

The work that wins a slot, if anything does, is a handler or a second builder who has touched the 6 MB model, and a video of that model failing in a way you intend to measure. The work that loses a slot is a platform narrative wrapped around a single-flock prototype.
