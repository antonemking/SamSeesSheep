# Sheep triage with TypeSafe Jev

Experiment: can TypeSafe Jev triage tracked sheep from their ear positions?

```
video clip
  → YOLO-pose + ByteTrack (sheep-pose v0.7, five keypoints)
  → one ear pack per track (code does the maths)
  → TypeSafe Jev, one call per track, three questions in parallel:
      triage  Choice  look | skip | cannot   (SPFES ear protocol) ← the decision
      look    Noul    confidence gate on a typed look
      reason  Choice  which ear sign, from a fixed list
  → typed look, noul >= threshold (default 0.5) → look
    typed look, noul <  threshold             → cannot (not sure)
    typed skip → skip, typed cannot → cannot, whatever the noul
  → one pen call: walk_now | later | fine
```

Jev sees text. It does not see frames. This does not run on a Pi, and it does not score pain or welfare. `look` means the ear numbers are worth a human glance, `skip` means both ears looked even and steady, and `cannot` means the face or ears were not seen well enough to judge.

Keypoints, image-space left/right, same order as the labeler: `nose`, `L_ear_base`, `R_ear_base`, `L_ear_tip`, `R_ear_tip`. No eye keypoints in v0.7.

## Pose-only smoke (no API key)

Plumbing check, no weights and no clip:

```bash
python experiments/sheep-triage-jev/run_pipeline.py --synthetic --pose-only
```

Real clip, from the repo root. Needs the sheep-yolo env (Ultralytics) and a local source video:

```bash
uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py \
  --clip test-clips/Test_Clip_Morning.mov \
  --pose-only
```

`test-clips/` is gitignored. `IMG_3651.MOV` is the other held-out clip. `assets/synced-lanes-6ewes-pro-Test_Clip_Morning.mp4` is a rendered showcase, not a source clip — do not point `--clip` at it.

If `TYPESAFE_API_KEY` is unset, the pipeline stays pose-only even without `--pose-only`.

## Full Jev triage

Create a key at the TypeSafe console. Export it in the shell. Do not put it in a file that gets committed.

```bash
export TYPESAFE_API_KEY=...   # never commit this
uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py \
  --clip test-clips/Test_Clip_Morning.mov
```

The same key works on the synthetic fixture (no weights) if you only want to check the API:

```bash
export TYPESAFE_API_KEY=...
python experiments/sheep-triage-jev/run_pipeline.py --synthetic
```

Each kept track is one POST to `https://api.typesafe.ai/v1/systemone` with model `jev-latest` and three questions: the `triage` Choice, the `look` Noul and the `reason` Choice. One more POST asks the pen question; `--no-pen` skips it. `--pose-only` skips every call even when the key is set.

## SPFES ear check (blog notes)

McLennan et al. (2016), the Sheep Pain Facial Expression Scale (SPFES), scores five facial areas. The v0.7 model sees the nose and ears only, so this is the **ear slice** of SPFES and nothing more: unusual ear carriage, left/right asymmetry, and frequent ear posture change. It has no eyes, cheeks, lips or nostrils, it says nothing about gait or lameness, and it has not been checked against a person scoring the same sheep. A Look is a prompt to glance at the face, not a pain score.

What Jev sees per track (the `ear_pack` in `tracks.json`, also under `jev_state`):

| Field | Meaning |
|---|---|
| `left_ear` / `right_ear` `.median_deg` | Typical ear angle. About 90° is out to the side, lower points forward toward the nose, higher points back |
| `.std_deg` | Frame-to-frame spread of that ear's angle (flutter) |
| `.measured_fraction` | Share of facing frames where that ear's tip was found |
| `asymmetry_deg` | Median per-frame \|left − right\| |
| `facing_fraction` | Share of frames with the nose and both ear bases visible |
| `n_frames`, `duration_s` | How long the track is. Not a sign on its own |

Angles are `null` when an ear was measured on fewer than 3 frames. Speed, centroid path and track coverage stay in `tracks.json` but are not sent to Jev.

The protocol text every question carries (`SPFES_EAR_PROTOCOL` in `jev_client.py`, question set `spfes-ear-v1`):

> Ear items of McLennan's SPFES sheep facial scale, read from head-pose numbers. Signs worth a person's look: unusual ear carriage (an ear held far forward, far back toward the neck, or so flat that only one ear can be measured while the face is toward the camera), clear asymmetry between the left and right ears, or frequent ear posture change across frames. Cannot check: the face was mostly not toward the camera, or the ears could not be measured on most facing frames. Walking, speed and a short time in view are not signs.

The questions: `triage` asks "should a person look at this sheep, skip it, or can it not be checked?"; `look` asks "does this sheep show an ear sign worth a person's look?"; `reason` picks one of `flutter`, `asymmetry`, `carriage`, `one_ear_missing`, `steady`, `not_facing`, `ears_unmeasurable`. Code keeps the reason consistent with the decision and words it for the farmer with the track's own numbers. The pen question gets the look/skip/cannot counts and reason counts and picks `walk_now`, `later` or `fine`.

### The decision

Jev's typed `triage` class is the decision. The `look` noul is only a confidence gate on a typed look:

| Jev's typed class | noul | Final decision | Farmer sees |
|---|---|---|---|
| look | ≥ `--threshold` (0.5) | look | Look, with Jev's reason |
| look | < `--threshold` | cannot, reason `not_sure` | Not checked: "Not sure enough to call — the ear signs were weak or mixed." |
| skip | any | skip | Skip |
| cannot | any | cannot | Not checked, with Jev's reason |

The noul never turns a skip or a cannot into a Look, and a Look never comes from the noul alone. Each row in `triage.json` keeps `typed_class`, `noul`, `threshold`, the final `decision` and `gated` side by side, so the blog can show every call the gate changed; `summary.txt` prints typed and final counts and how many typed looks the gate turned into cannot. `not_sure` is set by code only; Jev never picks it.

`--triage book` also writes a fixed-rule baseline beside Jev in `triage.json` (`book` on every track, `book_rules` at the top) and a `book vs final` line in `summary.txt` that compares the book with the final, gated decision. The rules, first match wins: face toward the camera on under half the frames → cannot; neither ear measured on half the facing frames → cannot; ear spread ≥ 15° → look; asymmetry ≥ 25° → look; an ear's median below 75° or above 140° → look; one ear measured and the other not → look; otherwise skip. Jev still drives the badges and the page.

Re-ask Jev about a run you already tracked, without running YOLO again. `--from-run` needs a folder made with `--demo` (it reads `frames.json`) and writes a new folder:

```bash
export TYPESAFE_API_KEY=...   # this shell only; never commit it, never put it on the Pi
python experiments/sheep-triage-jev/run_pipeline.py \
  --from-run experiments/sheep-triage-jev/runs/img3877-demo/ \
  --triage book \
  --out experiments/sheep-triage-jev/runs/img3877-spfes/
```

Add `--demo` (in the sheep-yolo env) to rebuild the replay and glance list for the new answers; the source clip must still be at the path recorded in the old `tracks.json`. Fresh from the clip:

```bash
uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py \
  --clip sheep-yolo/test-clips/IMG_3877.MOV \
  --triage book --demo \
  --out experiments/sheep-triage-jev/runs/img3877-spfes/
open experiments/sheep-triage-jev/runs/img3877-spfes/glance-list.html
```

## Farmer demo

`--demo` adds a replay with a Look / Skip / Not checked badge on every sheep and a glance list of which sheep to look at first, in the same `--out` folder. Local Mac smoke, from the repo root, with the v0.7 weights in place (see Weights):

```bash
export TYPESAFE_API_KEY=...   # this shell only; never commit it, never put it on the Pi
uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py \
  --clip sheep-yolo/test-clips/IMG_3877.MOV \
  --demo \
  --out experiments/sheep-triage-jev/runs/img3877-demo/
open experiments/sheep-triage-jev/runs/img3877-demo/glance-list.html
```

`sheep-yolo/test-clips/` is gitignored, so copy `IMG_3877.MOV` there yourself or point `--clip` at another local source clip you have the rights to show. `--max-frames 900` gives a 30 s cut if you want a quicker first pass.

The page is farmer language only: the pen call on top ("Walk the pen now." / "No rush — look them over on your next walk-through." / "The pen looks fine for now."), then Look / Skip / Not checked, when each sheep is in view, one plain-English reason, and a Jump button that seeks the replay to that moment. Look cards come first. When Jev answers `cannot`, the card reads Not checked with the reason ("Couldn't see its face well enough — it was mostly turned away."); a typed look that fails the noul gate reads Not checked with "Not sure enough to call — the ear signs were weak or mixed." If every sheep comes back Look, the page and the replay both say so; they never invent a Skip. Without a key, or with `--pose-only`, badges read "Not checked" and the page says "Not sorted yet". Run folders from before the SPFES check (`look` / `dont`) still render.

Offline smoke with no key, weights, or clip. Every badge reads "Not checked":

```bash
uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py \
  --synthetic --pose-only --demo --out experiments/sheep-triage-jev/runs/synthetic-demo/
open experiments/sheep-triage-jev/runs/synthetic-demo/glance-list.html
```

Rebuild either piece of an existing run folder without re-tracking:

```bash
python experiments/sheep-triage-jev/glance_list.py experiments/sheep-triage-jev/runs/img3877-demo/
uv run --project sheep-yolo python experiments/sheep-triage-jev/render_overlay.py \
  experiments/sheep-triage-jev/runs/img3877-demo/
```

`annotated.mp4` is H.264 through PyAV (already in the sheep-yolo env), so it plays and seeks in Safari and Chrome. If PyAV cannot encode H.264 the renderer says so and writes mp4v instead; QuickTime plays that, browsers may not. `--demo-width` (default 1280) caps the replay width.

## Weights

Default checkpoint: `sheep-yolo/weights/sheep-pose-v0.7-yolo26n.pt`.

`*.pt` is gitignored, so a fresh clone does not have it. The file is about 6 MB on Hugging Face (`antking1/sheep-pose-yolo26n`):

```bash
mkdir -p sheep-yolo/weights
curl -L -o sheep-yolo/weights/sheep-pose-v0.7-yolo26n.pt \
  https://huggingface.co/antking1/sheep-pose-yolo26n/resolve/main/sheep-pose-v0.7-yolo26n.pt
```

Pass `--weights` to use another file.

## What a run writes

`experiments/sheep-triage-jev/runs/<timestamp>-<clip>/` (gitignored):

| File | Contents |
|---|---|
| `tracks.json` | Per-track visibility, ear-angle median/spread, motion, the `ear_pack`, and the exact JSON state Jev sees |
| `triage.json` | Per track, side by side: Jev's `typed_class`, the raw `noul`, `threshold`, the final `decision` (`look` / `skip` / `cannot`), `gated` (the noul gate changed the call) and `reason`; then the `triage` and `reason` choice probabilities, model, token usage, and `book` with `--triage book`. Top level: `pen`, `decision_rule`, `question_set`. Pose-only runs leave `decision` null |
| `summary.txt` | One line per track (typed class, noul, final call, book call), then typed and final counts with the gate count, book counts and book-vs-final agreement, and the pen call |
| `annotated.mp4` | `--demo` only. Replay with pose boxes, ear-angle lines (blue left, purple right, degrees at the tip) and a Look / Skip / Not checked badge per track from `triage.json` |
| `glance-list.html` | `--demo` only. Look cards first, each with a jump into `annotated.mp4` |
| `frames.json` | `--demo` only. Per-frame boxes and keypoints, so the replay can be re-rendered |

Tracks seen on fewer than `--min-frames` frames (default 2) are dropped and listed as `dropped_track_ids`. Detection confidence defaults to 0.25 and keypoint confidence to 0.4, matching the ear-angle renderer.

Useful knobs: `--max-frames`, `--threshold`, `--model`, `--fps`, `--out`, `--triage book`, `--no-pen`, `--from-run`.

## Tests

No API key, no weights, no network (Jev is a local fake and the tests fail if anything reaches for urllib):

```bash
python experiments/sheep-triage-jev/test_triage.py
```

The four replay tests need OpenCV and are skipped without it. Run them in the sheep-yolo env:

```bash
uv run --project sheep-yolo python experiments/sheep-triage-jev/test_triage.py
```

## Still open

- No API key ships with the repo. The SPFES questions (`spfes-ear-v1`) and the pen question have not been sent to the live API yet; the tests use a fake Jev that answers in the documented response shape.
- The 0.5 gate on typed looks is a starting point, not a tuned operating point. Noul probabilities are calibrated by TypeSafe in general; they are not calibrated on this flock.
- Nothing here measures whether `look` agrees with a person, or with a full SPFES score. That eval is future work.
- The book thresholds (15° spread, 25° asymmetry, 75°–140° band, half the frames) are guesses from the ear-angle geometry, not from labelled SPFES data. Tune them after a real IMG_3877 run.
- The farmer demo is checked on synthetic fixtures only. Nobody has watched a real-clip replay yet.
- Jev-Omni and multimodal Gemma are out of scope. So is Pi / Pico deployment.
