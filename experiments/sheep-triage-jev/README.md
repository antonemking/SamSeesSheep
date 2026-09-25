# Sheep triage with TypeSafe Jev

Experiment: can TypeSafe Jev triage tracked sheep from pose features?

```
video clip
  → YOLO-pose + ByteTrack (sheep-pose v0.7, five keypoints)
  → one numeric summary per track
  → TypeSafe Jev Noul ("should a person look?")
  → look  if noul >= threshold (default 0.5)
    dont otherwise
```

Jev sees text. It does not see frames. This does not run on a Pi, and it does not score pain or welfare. `look` means the pose summary is worth a human glance. `dont` means skip the track.

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

Each kept track is one POST to `https://api.typesafe.ai/v1/systemone` with model `jev-latest` and a single Noul named `look`. `--pose-only` skips that call even when the key is set.

## Farmer demo

`--demo` adds a replay with a Look / Skip badge on every sheep and a glance list of which sheep to look at first, in the same `--out` folder. Local Mac smoke, from the repo root, with the v0.7 weights in place (see Weights):

```bash
export TYPESAFE_API_KEY=...   # this shell only; never commit it, never put it on the Pi
uv run --project sheep-yolo python experiments/sheep-triage-jev/run_pipeline.py \
  --clip sheep-yolo/test-clips/IMG_3877.MOV \
  --demo \
  --out experiments/sheep-triage-jev/runs/img3877-demo/
open experiments/sheep-triage-jev/runs/img3877-demo/glance-list.html
```

`sheep-yolo/test-clips/` is gitignored, so copy `IMG_3877.MOV` there yourself or point `--clip` at another local source clip you have the rights to show. `--max-frames 900` gives a 30 s cut if you want a quicker first pass.

The page is farmer language only: Look / Skip, when each sheep is in view, one plain-English reason from its pose numbers, and a Jump button that seeks the replay to that moment. Look cards come first. If every sheep comes back Look, the page and the replay both say so; they never invent a Skip. Without a key, or with `--pose-only`, badges read "Not checked" and the page says "Not sorted yet".

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
| `tracks.json` | Per-track visibility, ear-angle median/spread, motion, and the exact JSON state Jev would see |
| `triage.json` | `look` / `dont`, the raw `noul`, model, token usage. Pose-only runs leave `decision` null |
| `summary.txt` | One line per track |
| `annotated.mp4` | `--demo` only. Replay with pose boxes, ear-angle lines (blue left, purple right, degrees at the tip) and a Look / Skip badge per track from `triage.json` |
| `glance-list.html` | `--demo` only. Look cards first, each with a jump into `annotated.mp4` |
| `frames.json` | `--demo` only. Per-frame boxes and keypoints, so the replay can be re-rendered |

Tracks seen on fewer than `--min-frames` frames (default 2) are dropped and listed as `dropped_track_ids`. Detection confidence defaults to 0.25 and keypoint confidence to 0.4, matching the ear-angle renderer.

Useful knobs: `--max-frames`, `--threshold`, `--model`, `--fps`, `--out`.

## Tests

No API key, no weights:

```bash
python experiments/sheep-triage-jev/test_triage.py
```

The two replay tests need OpenCV and are skipped without it. Run them in the sheep-yolo env:

```bash
uv run --project sheep-yolo python experiments/sheep-triage-jev/test_triage.py
```

## Still open

- No API key ships with the repo. Full triage is unverified until someone runs it with `TYPESAFE_API_KEY`.
- The 0.5 cut is a starting point, not a tuned operating point. Noul probabilities are calibrated by TypeSafe in general; they are not calibrated on this flock.
- Nothing here measures whether `look` agrees with a person. That eval is future work.
- The farmer demo is checked on synthetic fixtures only. Nobody has watched a real-clip replay yet, and the plain-English reason thresholds (ear spread 15°, half the frames, half a box width per second) are guesses.
- Jev-Omni and multimodal Gemma are out of scope. So is Pi / Pico deployment.
