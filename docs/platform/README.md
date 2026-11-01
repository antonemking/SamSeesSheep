# Livestock vision platform — strategy package

Decision-ready docs for one question: should SamSeesSheep's tooling grow into the open reference layer
for livestock computer vision, and if so, what gets built in the next 90 days? Written 2026-11-01.
SamSeesSheep is the first species wedge and the demo; it is not the platform's brand.

**Read in this order** (about 20 minutes in total; the first page of the memo alone is about 5):

| # | Doc | What it answers |
|---|---|---|
| 1 | [`livestock-ultralytics-strategy.md`](livestock-ultralytics-strategy.md) | The decision memo. Five-minute pitch and the "we are NOT building" table on page 1; pressure test of the seed thesis; market map; task taxonomy; robustness as a product; go-to-market; decisions D1–D7 |
| 2 | [`ultralytics-cattle-inventory.md`](ultralytics-cattle-inventory.md) | What Ultralytics actually ships for cattle and other livestock, with URLs, file paths and licenses, and what we may depend on, fork, or must reimplement |
| 3 | [`roboflow-built-in-mvp.md`](roboflow-built-in-mvp.md) | The thinnest "almost Roboflow" slice: the four livestock-specific seams we build, the stages we depend on, the eight-command surface, and what is deferred |
| 4 | [`architecture-options.md`](architecture-options.md) | Options A / B / C, the recommendation, kill criteria K1–K10, and the gates for anything hosted |
| 5 | [`90-day-wedge.md`](90-day-wedge.md) | The calendar, 2026-11-02 → 2027-01-30: what a stranger can open on day 90, the one external-user criterion, Antone's hours |

**The recommendation in four lines.** Own livestock *definitions* (schemas, leak-proof splits,
benchmarks, field report cards, adapted weights); depend on Ultralytics, Label Studio and Hugging Face
for the machinery. Architecture A now, as the open half of C; never fork the trainer. Sheep faces prove
it by mid-December; a cattle schema added as a file proves it is a platform. If our pretrained base
does not beat COCO initialisation on a new flock at 200 labels, the model-zoo thesis is killed on day 90.

**Ground rules these docs follow.** Every external claim carries a URL that was loaded while writing.
No users, partners, customers or certifications are claimed; outreach candidates are labelled
"not contacted". Repo paths refer to this repository; a few (the failure-transfer experiment, the
2026-10-26 research memo, the `CLAUDE.md` direction section) live on the unpushed
`claude/failure-transfer` branch, and the sheep-face-36 tooling lives in a local, git-excluded
workspace until it is promoted in week 1–2.
