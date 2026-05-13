# LinkedIn Post Draft — SamSeesSheep, Part 3 (v2)

*(v2 — rewrite of `linkedin-post-draft-ep3.md` after v0.4 trained and the
clean held-out 3-way benchmark on IMG_3651 finished. v1 of this draft
assumed only v0.2 → v0.3; bracketed placeholders are now filled in with
real numbers from `bench_report-IMG_3651-3way.json`. Original draft kept
in `linkedin-post-draft-ep3.md` as history.)*

---

This morning I sat at the kitchen table for about ninety minutes and labeled another 60 frames of sheep faces. Five keypoints per sheep — nose, both ear bases, both ear tips — across two new clips I'd shot at the trough earlier in the week. By lunch the dataset had gone from 313 reviewed sheep-head instances to 405. By 1:07pm a 4th-generation model was finished training. Six minutes on a rented 4090.

[VIDEO — HERO: `v0.2-vs-v0.4-IMG_3651.mp4`. 5-second 2-up. v0.2 on the left (orange keypoints, 98 labels). v0.4 on the right (magenta, 405 labels). Same frames. A clip neither model has ever seen.]

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

mAP is a benchmark number. The whole point of these models is to measure ear angles on a sheep that's just standing there — and the welfare signal collapses if the model is jittering frame to frame on an animal that hasn't moved.

So we measure σ. Pixel-standard-deviation of each keypoint across a multi-second window of a sheep just standing in place, after subtracting slow head sway with a rolling median (because slow sway is the sheep, not the model). What's left is pure frame-to-frame jitter. The lower it is, the more usable the downstream signal.

This time, for the first time, the test was on a **clip none of the three models has ever seen** — never pushed to the labeler, never reviewed, never in any training set. The v0.3 round used hero clips that were technically in v0.3's training distribution, and I called that out in the writeup at the time. This one is honest about the held-out setup.

Per-keypoint residual σ on a 5-second motionless window:

```
                v0.2     v0.3     v0.4     Δ v0.2→v0.4
nose            11.0 px   10.0 px   8.0 px      −27%
L-ear-base       9.3 px    6.2 px   6.0 px      −35%
R-ear-base       9.1 px    7.7 px   7.0 px      −23%
L-ear-tip       12.6 px    8.4 px   7.7 px      −39%
R-ear-tip       12.4 px   12.3 px   9.8 px      −21%
─────────────────────────────────────────────────────
mean            10.9 px    8.9 px   7.7 px      −29%
```

v0.4's keypoint jitter on a held-out 234-px-wide sheep head is **about 3.3% of head size**. That's a usable number for ear-angle estimation. v0.2's was about 4.6% of head size on the same sheep, in the same frames, with the same threshold. The ear tips — the keypoints that matter most for welfare — improved the most (−39% / −21% from v0.2 to v0.4).

[VIDEO — SUPPLEMENTARY: `v0.2-vs-v0.3-vs-v0.4-IMG_3651.mp4`. 3-up. v0.2 (orange), v0.3 (green), v0.4 (magenta). For people who want to see the curve, not just the endpoints.]

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

- **All three model versions are real and on disk** at `sheep-yolo/weights/`. The benchmark is reproducible: `python sheep-yolo/scripts/bench_held_out.py IMG_3651`. Full numbers in `docs/v0.4-benchmark.md` and `sheep-yolo/artifacts/bench_report-IMG_3651-3way.json`.
- **Held-out caveat I deliberately did NOT bury.** The v0.3 hero post had to caveat that its test clips were in v0.3's training set. This post should foreground "first truly held-out benchmark" because it makes the σ numbers stronger, not weaker.
- **What this post does NOT show, by design:**
  - The earlier hero v0.2-vs-v0.3 "47% → 99% detection rate" delta. Those clips were crowded, multi-sheep, and in v0.3's training set. That delta was real but partly a function of scene difficulty, not pure model gap. IMG_3651 has a single dominant subject — all three models find it ~94–97% of the time. Detection rate isn't the story here. Jitter is.
  - That v0.4 ear-angle is ready as a clinical welfare instrument. It is *not*. Validation against documented stress events is a separate undertaking — see `VALIDATION.md`.
- **Going forward.** Each future post compares the new version against the previously-posted version only. v0.4 vs v0.5, v0.5 vs v0.6, on a fresh held-out clip each time. This v0.2/v0.3/v0.4 3-way is the one-time bridge that pays back the "clean held-out is next" IOU from the v0.3 post.
- **Counter-intuitive datapoint, for comments:** pose precision/recall both moved monotonically through the three versions, but raw σ stayed flat (~47 px across all three). That's not a contradiction — raw σ is dominated by slow sheep drift across the window, which is the same for any model. Subtracting the rolling median is what exposes the model-quality signal. Mention it only if a CV person digs in.
- **Economics line for the AI Farmer brand:** *each retrain cost about a dollar in cloud GPU time and a Sunday morning of attention.* Keep this visible if there's room — small, hands-on, accessible AI work on a real farm is the whole brand.
