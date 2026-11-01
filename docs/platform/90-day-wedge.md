# 90-day wedge: the proof calendar

**Window:** Mon 2026-11-02 (day 1) → Sat 2027-01-30 (day 90).
**Owner:** Antone King, with coding agents. **Status:** proposal; nothing below has happened yet.
**Read with:** [`livestock-ultralytics-strategy.md`](livestock-ultralytics-strategy.md) (why),
[`roboflow-built-in-mvp.md`](roboflow-built-in-mvp.md) (the tooling slice),
[`architecture-options.md`](architecture-options.md) (kill criteria).

## What a stranger can open on day 90

One flagship and three supporting artifacts. If the flagship is not live on day 90, the wedge failed,
whatever else shipped.

| # | Artifact | Where it will live | What a stranger does with it |
|---|---|---|---|
| **F** | **Sheep-face demo + benchmark page.** Drop a sheep photo or short clip; get 36 face landmarks, ear angles, and the model's field report card, with links to the dataset and the benchmark table | Hugging Face Space (Gradio, free CPU tier), linked from the repo README | Sees the model work on *their* image in under a minute, without installing anything |
| 1 | `sheep-face-36` v1.0 dataset: native-resolution head crops, 36-point labels, per-video splits, datasheet, judge set marked test-only | Hugging Face dataset repo with a DOI | Downloads it, trains anything on it, reports a number against the judge set |
| 2 | Adaptation notebook: "bring ~200 labeled heads of your animals, get an adapted model and a learning curve" | Colab notebook in the repo, using the `herdkit` package (placeholder name; free on PyPI as of 2026-11-01) | Runs the DeepLabCut-style promise on their own flock |
| 3 | Livestock field-robustness benchmark: WCS camera-trap cattle/goat/sheep, real degraded frames tagged by cause, camera-level holdout, baseline numbers | Hugging Face dataset card + `experiments/failure-transfer/wcs/` | Scores their own detector against haze, dust, smudge and glare that actually happened |

What is deliberately **not** on the list: a hosted labeling app, a sign-up page, a pricing page, a cattle
product. Those are gated behind the day-90 review (see "Day-90 review" below).

## The one external criterion

By day 90, **one person outside Lorewood has run the loop on animals we have never seen and reported the
number in public** (a GitHub Discussion or issue on the repo, or a co-authored line in the benchmark
table). Either of these counts:

- (a) they run the adaptation notebook on their own sheep or goat footage and post the learning curve; or
- (b) they supply footage of a second flock (with consent and a short data agreement), label or co-label it,
  and that flock becomes the held-out second-flock benchmark.

Candidates to approach in week 2 (none contacted yet; these are leads, not partners):
University of Delaware Cooperative Extension, whose agriculture and animals agent holds statewide
responsibility for small ruminants ([directory page](https://www.udel.edu/content/udel/en/academics/colleges/canr/cooperative-extension/directory-contacts/dan-severson);
[small-ruminant programme impact sheet](https://www.udel.edu/content/dam/udelImages/canr/pdfs/extension/impacts/ImpactInfographic_SmallRuminants_FINAL.pdf)),
University of Maryland Extension sheep and goat programme
([extension.umd.edu](https://extension.umd.edu/programs/agriculture-food-systems/program-areas/animal-science/sheep-and-goat-program)),
the SPFES authors (McLennan, University of Chester; Mahmoud, Cambridge — see
`docs/research/spfes-deeplabcut-deepercut-memo.md` §4.4 on the `claude/failure-transfer` branch), and
nearby flock owners Antone already knows. One "yes" is enough. Zero by day 60 triggers the fallback in
week 9.

Separately, invite the authors of the two open livestock keypoint benchmarks that already do things right
— Wang et al. 2026 (cattle/horse/sheep) and SwinePose (pigs, per-video split; [Zenodo](https://zenodo.org/records/19358700))
— to be the first external rows in the benchmark table. That is MegaDetector's playbook (strategy memo
§3.6) and it costs one email each. Not contacted yet.

## Calendar

Labeling hours are the bottleneck, not code. Every week below names Antone's hours explicitly so the plan
cannot quietly assume an annotator who does not exist.

### Weeks 1–2 (Nov 2 – Nov 15): ground truth and durability

| Deliverable | Who | Done when |
|---|---|---|
| Hand-mark the 76 judge heads (IMG_3877 + IMG_4425) in the existing Label Studio job | Antone, ~4–5 h | `judge/gt.json` exists; the provisional baseline in the local skin workspace becomes a real one |
| Promote `schema.py`, `make_tasks.py`, `prelabel.py`, `ls_setup.py`, `ls_export.py`, `score.py`, `train/export_yolo.py` from the git-excluded `fun/sheep-masks/skin/` into a repo package (`herdkit/`), with a schema registry and the judge-clip refusal kept | agents, ~1 week | `herdkit tasks` + `herdkit pull` + `herdkit eval` run from the repo on a non-judge clip; nothing under `fun/` is committed |
| Private Hugging Face dataset repo created; every labeling batch is pushed as a revision | agents | The 2026 laptop-crash failure mode (labels on one disk) cannot recur |
| Capture new footage under the protocol in `CLAUDE.md` (frontal + both profiles, 1 m and 3 m, ear-tag ID per clip, two times of day) | Antone, ~2 h on farm | ≥ 6 clips, ≥ 5 animals, logged in a clip manifest |
| Re-run the WCS failure-transfer experiment with the camera-level holdout (the fix in `prepare_wcs.py` is written but not yet committed on the `claude/failure-transfer` branch) | agents + pod | `results/{sweep,reality,eval}` exist; the reality-check verdict is written down, good or bad |
| Send the external-user asks (week 2) | Antone | Messages sent; replies logged |

**Gate 0 (day 14):** judge ground truth exists and the loop runs from the repo. If the judge heads are not
marked by day 14, stop adding code — everything after this is unscoreable.

### Weeks 3–5 (Nov 16 – Dec 6): first model on the new schema

| Deliverable | Who | Done when |
|---|---|---|
| Label 60 heads (pre-labeled with v0.7 + template + SAM 2.1) | Antone, ~2–3 h | Batch pushed as a dataset revision |
| Train v1.0-a (YOLO26n-pose, 320 px crops, per-video split) | agents + pod | Scored on the judge set: per-group pixel error, NME, PCK@5/10 px, jitter, misses |
| Ear regression gate on IMG_3877's published window | agents | σ_avg ≤ 2.84° (v0.7) or the regression is reported, not hidden |
| Pre-label the next 200 heads with v1.0-a | agents | Tasks in Label Studio, ordered by the outlier queue (low per-keypoint confidence first) |

Thanksgiving (Nov 26) sits in week 4; nothing is scheduled that week beyond training runs.

**Gate 1 (day 35):** v1.0-a beats the seeded-template baseline on eyes and nostrils on the judge set. If it
does not, the dense-face premise at 320 px is wrong: fall back to ears + eyes (13 points) and say so in the
dataset card.

### Weeks 6–9 (Dec 7 – Jan 3): public release #1

| Deliverable | Who | Done when |
|---|---|---|
| Review 200 more heads; train v1.0-b | Antone ~5–7 h; agents | ≥ 260 reviewed heads across ≥ 8 clips |
| Double-label 50 heads with a second person | second labeler, ~2 h | Inter-labeler pixel error per keypoint reported next to model error (the DeepLabCut human-variability baseline); the per-keypoint spread becomes the schema's OKS sigmas |
| Post the v0.x arXiv preprint (rebuild the corrected `.tex`) | Antone | arXiv ID in the README |
| **Release:** `sheep-face-36` v1.0 dataset (DOI), v1.0-b weights with a field report card, the HF Space demo (artifact F) | agents, Antone approves | All four links resolve for a logged-out browser |
| WCS robustness benchmark card published (artifact 3) | agents | Card states single-annotator cause tags and the camera-level split |

Christmas and New Year fall in weeks 8–9; the release is targeted for **Fri Dec 18** so the holiday weeks
are slack, not critical path.

**Fallback if no external user has said yes by day 60:** run the second-flock benchmark on the best
available non-Lorewood sheep footage with a documented license and say plainly in the benchmark that it
is not an external user.

### Weeks 10–13 (Jan 4 – Jan 30): the platform claim and the product claim

| Deliverable | Who | Done when |
|---|---|---|
| **Second-species test** (time-boxed to 10 working days): add `cattle-body-18` as a schema *file* plus an importer for the cattle subset of Wang et al. 2026 ([repo](https://github.com/yww0411/Livestock-keypoint-detection); data CC BY 4.0, COCO keypoints with visibility). It deliberately differs from the sheep line on species *and* unit (whole body, not head crop) | agents | Measured: lines changed in `herdkit/` core vs in the new schema/config files (target ≤ ~200 core lines); and our pipeline reproduces their published keypoint AP within its reported spread. Reproducing someone else's number on their data is the credibility check |
| **200-label adaptation benchmark** on the second flock: learning curve at 25/50/100/200 heads, per-video split, YOLO-pose vs one non-Ultralytics alternative (DeepLabCut 3 top-down with its RTMPose configs), forgetting re-scored on IMG_3877 | agents + pod; labels from the external user or Antone | Curve published with bootstrap CIs; the "200" is wherever it bends, not assumed |
| External user's number in public (the one external criterion) | external user | Linked from the benchmark table |
| Day-90 review written against the kill criteria in `architecture-options.md` | Antone + agents | A one-page go / narrow / kill decision, committed |

## Day-90 review: what gets decided

| Question | Go | Narrow | Kill |
|---|---|---|---|
| Does a pretrained livestock base beat a COCO-initialised YOLO-pose trained on the same 200 second-flock heads? | Clearly better at 25–100 heads | Only at 25 heads | No difference → the zoo has no value; keep the tooling as research code |
| Was the second species a file, not a fork? | Core change ≤ ~200 lines | 200–1,000 lines | Needed a new pipeline → stop calling it a platform |
| Did a stranger run it? | External number posted | Ran with heavy hand-holding | Nobody → do not build anything hosted |
| Did the synthetic field sweep predict real failures (WCS reality check)? | Breaking points line up with real per-cause drops | Some causes only | No relation → drop synthetic sweeps from the report card; keep only real-failure holdouts |

Only an all-"Go" review opens the next tier (hosted adaptation runs for paying integrators; see
`architecture-options.md` option C gates).

## Hours budget (Antone)

| Work | Hours |
|---|---|
| Judge set marking | 4–5 |
| Farm capture | 2–4 |
| 260 training heads (pre-labeled, review-mode) | 8–10 |
| Outreach, approvals, preprint | 4–6 |
| **Total over 13 weeks** | **~18–25** |

If that budget is not available, cut the second-species test first, then the DeepLabCut 3 comparison. Never cut
the judge set or the external criterion; they are the proof.
