# LinkedIn Post Draft — SamSeesSheep, Part 3 (v2)

*(v2 — rewrite of `linkedin-post-draft-ep3.md` after v0.4 trained and the
clean held-out 3-way benchmark on IMG_3651 finished. v1 of this draft
assumed only v0.2 → v0.3; bracketed placeholders are now filled in with
real numbers from `bench_report-IMG_3651-3way.json`. Original draft kept
in `linkedin-post-draft-ep3.md` as history.)*

---

Two posts ago I said the next one would have *the* benchmark: a clip of a sheep that isn't moving, run through the model, and the ear-angle line should be flat. Off-the-shelf YOLO would produce noise. A model trained on my own hand-labeled frames should produce the line. If it didn't, I'd publish that too.

Here's the line.

[IMAGE — HERO: `ear_angle_lines-IMG_3651.png` (chart). Per-frame ear angle for v0.2 (orange), v0.3 (green), v0.4 (magenta) on a 5-second window of a stationary sheep in a clip none of the three models has ever seen. v0.2 bounces 6–7°. v0.4 holds within ~4°.]

This morning I sat at the kitchen table for about ninety minutes and labeled another 60 frames of sheep faces. By lunch the dataset had gone from 313 reviewed sheep-head instances to 405. By 1:07pm a 4th-generation model was finished training. Six minutes on a rented 4090.

I'm telling that story in minutes because for once the AI is not the slow part. I am. Three labeling sessions, three retrains, three months apart. Same model architecture. Same training recipe. The only thing that changed was how many ear tips I'd clicked.

---

## What was actually different between the runs

Same model architecture across the board: **YOLO26n-pose**, 2.5 million parameters. The smallest member of the family. The one designed to run on a phone or a tiny camera.

Same 100 epochs of training. Same image size. Same batch size. Same loss weights. Same `yolo train` command, copy-pasted.

What changed three times was the labels:

- **v0.2:** 98 reviewed sheep-head instances. 3 videos. Single trough, single time of day.
- **v0.3:** 313 instances. 6 videos. Added a low-light evening clip and a couple of shots from across the pen.
- **v0.4:** 405 instances. 8 videos. Added two more clips from the same week.

That is the whole experiment. Same model, more data, what happens.

## The benchmark numbers

Fine-grained keypoint accuracy (mAP50-95 for pose, if you want the term) on the validation split:

**v0.2: 0.479 → v0.3: 0.643 → v0.4: 0.732**

A 53% relative improvement over the full curve. v0.3 over v0.2 was the big jump (3.2× more data). v0.4 over v0.3 was smaller but real (1.3× more data, +14% relative). Diminishing per unit of labeling, but the curve hasn't flattened.

## The number that matters for welfare

mAP is a benchmark number. The whole point of these models is to measure ear angles on a sheep that's just standing there — and the welfare signal collapses if the model is jittering frame to frame on an animal that hasn't moved. So I measure the angle directly, in degrees, on a window where the sheep is mostly still.

This time, for the first time, the test was on a **clip none of the three models has ever seen** — never pushed to the labeler, never reviewed, never in any training set. The v0.3 round used hero clips whose 2-fps-sampled frames had ended up in v0.3's training set; I called that out in the writeup at the time. This time the held-out is real.

Ear-angle residual σ (pure jitter after subtracting slow head sway with a rolling median):

```
            v0.2      v0.3      v0.4      Δ v0.2→v0.4
left ear    6.71°     4.82°     4.06°       −39%
right ear   6.07°     4.21°     4.09°       −33%
```

Per-keypoint pixel-level σ tells the same story underneath — v0.4 puts every keypoint within ~7 px of where the rolling median says it should be, on a 234-px-wide head. That's about 3% of head size. The ear tips, the keypoints that matter most for welfare, improved the most. Full per-keypoint table and methodology in [`docs/v0.4-benchmark.md`](https://github.com/antonemking/SamSeesSheep/blob/main/docs/v0.4-benchmark.md).

[VIDEO — SUPPLEMENTARY: `v0.2-vs-v0.3-vs-v0.4-IMG_3651.mp4`. 3-up. v0.2 (orange), v0.3 (green), v0.4 (magenta). For people who want to see the curve, not just the endpoints.]

## What "off-the-shelf YOLO" actually does

I owed Post 2 the off-the-shelf comparison. The honest answer is *stronger* than I'd framed it.

Stock `yolo26n.pt` — the COCO-trained checkpoint you get with `pip install ultralytics` — finds a bounding box around 35% of the sheep in this clip. That's it. Box only. **Zero keypoints.** No nose, no ear bases, no ear tips. The ear-angle line doesn't have *noise* in the stock baseline; it has *no value at all*, because the measurement is undefined without a sheep-pose keypoint head.

That's the part of this that doesn't get talked about enough. You don't actually have a "noisy welfare model" before you do the labeling work. You have a vacuum. The keypoints are a thing you have to grow against your own animals. There's no shortcut, no prompt, no zero-shot, no foundation model that knows where this Katahdin ewe's left ear tip is. You sit at the kitchen table and you click.

## The lesson I keep relearning

In a narrow domain — one species, one farm, one camera angle — the model architecture is not the hard part. The hard part is labels. And the labels are not the hard part because labeling is intellectually difficult; the labels are the hard part because labeling is the thing nobody else can do for me. Nobody else knows which of those animals is Ulysses, which is his daughter, which is the ewe that always crowds the trough. The model I'm building is the model of *my flock*, and the only training signal that helps is the one I sit down and produce.

What changed across these three rounds is not a better architecture or a smarter loss function. I made coffee, sat down, and clicked through five hundred ear tips. Three times. Each time the model got measurably better. Each time the welfare-relevant jitter got tighter. Each round costs maybe a dollar in cloud GPU time and a Sunday morning of attention.

This is not a story about AI. It is a story about a farmer paying attention to his animals, with a model riding shotgun and getting better every time he does.

The model can run on a $50 camera at the barn. I can't. So the model is what gets to watch the barn while I'm doing something else.

---

## Caption / hook options

Pick one for the LinkedIn header:

1. "Three labeling sessions. Three retrains. Same tiny model. Same training script. Same compute. Here's what 4× the labels did to the welfare-relevant jitter — measured on a clip none of the three models has ever seen."

2. "Ninety minutes of labeling, six minutes of training, the cost of a coffee. The model got measurably better at watching my sheep. Round three of the unsexy lesson."

3. "Same 2.5M-parameter model. 98 → 313 → 405 reviewed sheep heads. On a held-out clip: keypoint jitter dropped 29%. The data was the whole story."

## Closing engagement question

(Pick one; #2 stays the most on-brand for AI Farmer):

- "What's the smallest model you've gotten away with on a narrow problem?"
- "For everyone reaching for the next big architecture: when's the last time you doubled your labels and just… watched what happened?"
- "Is there a domain where you've watched the labels-vs-architecture tradeoff flip? Curious where this stops working."

---

## Editorial notes for me (delete before posting)

- **All three model versions are real and on disk** at `sheep-yolo/weights/`. The benchmark is reproducible: `python sheep-yolo/scripts/bench_held_out.py IMG_3651`. Full numbers in `docs/v0.4-benchmark.md` and `sheep-yolo/artifacts/bench_report-IMG_3651-3way.json`. The ear-angle chart is `sheep-yolo/artifacts/ear_angle_lines-IMG_3651.png` (also in `docs/v0.4-ear-angle-chart.png` for the README).
- **Post 2 IOU is closed.** The exact promise was: "Off-the-shelf YOLO will produce noise around it. The version trained on my hand-labeled frames should produce the flat line." Stock yolo26n.pt actually produces *zero keypoints*, which is a stronger statement than "noise" — call this out explicitly in the post.
- **Held-out caveat I deliberately did NOT bury.** The v0.3 hero post had to caveat that its test clips were in v0.3's training set. This post should foreground "first truly held-out benchmark" because it makes the σ numbers stronger, not weaker.
- **What this post does NOT show, by design:**
  - The earlier hero v0.2-vs-v0.3 "47% → 99% detection rate" delta. Those clips were crowded, multi-sheep, and in v0.3's training set. That delta was real but partly a function of scene difficulty, not pure model gap. IMG_3651 has a single dominant subject — all three models find it ~94–97% of the time. Detection rate isn't the story here. Jitter is.
  - That v0.4 ear-angle is ready as a clinical welfare instrument. It is *not*. Validation against documented stress events is a separate undertaking — see `VALIDATION.md`.
- **Going forward.** Each future post compares the new version against the previously-posted version only. v0.4 vs v0.5, v0.5 vs v0.6, on a fresh held-out clip each time. This v0.2/v0.3/v0.4 3-way is the one-time bridge that pays back the "clean held-out is next" IOU from the v0.3 post.
- **Counter-intuitive datapoint, for comments:** pose precision/recall both moved monotonically through the three versions, but raw σ stayed flat (~47 px across all three). That's not a contradiction — raw σ is dominated by slow sheep drift across the window, which is the same for any model. Subtracting the rolling median is what exposes the model-quality signal. Mention it only if a CV person digs in.
- **Economics line for the AI Farmer brand:** *each retrain cost about a dollar in cloud GPU time and a Sunday morning of attention.* Keep this visible if there's room — small, hands-on, accessible AI work on a real farm is the whole brand.
