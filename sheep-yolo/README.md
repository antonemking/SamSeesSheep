# sheep-yolo

Out-of-the-box YOLO benchmark for sheep ear-angle welfare monitoring — v2 of
[sheep-seg](../sheep-seg).

## Why a v2

`sheep-seg` used SAM 3 Video at inference time: strong masks, but slow (≈5–8
s/frame) and VRAM-hungry, which capped usable clips at ~10 seconds on a GTX
1660 Ti. This repo tests whether the Ultralytics-pitched alternative —
**YOLOE-seg with text prompts for sheep head/ear/nose, no fine-tuning** —
actually works on our footage.

The architectural tradeoff is discussed in `sheep-chat.md`. Short version:

- **SAM 3** is an annotation-time tool. Keep it for labeling.
- **YOLO** is the inference-time tool. This repo is a straight OOB probe — no
  training — to see how far the open-vocab path gets us on real flock
  footage before we commit to a training pipeline.

If YOLOE finds zero `sheep ear` detections, that's a valid result: the OOB
model doesn't know this anatomy and fine-tuning becomes the next task. The UI
surfaces that outcome clearly instead of silently failing.

### First-run result on `footage/Screencast from 2026-04-14 11-00-08.webm`

Probing YOLOE-v8l-seg at conf ≥ 0.10 on frame 0:

| prompt        | best confidence | notes                         |
|---------------|----------------:|-------------------------------|
| `sheep`       | 0.94            | finds both whole animals      |
| `sheep head`  | 0.30            | detected, below default conf  |
| `nose`        | 0.17            | marginal                      |
| `sheep ear`   | —               | **no detections at any conf** |

**Conclusion: out-of-the-box YOLOE cannot measure sheep ear angle on this
footage.** Whole-sheep detection works; parts don't. This matches what
Ultralytics hinted at in `sheep-chat.md` — the SOTA open-vocab model is fine
for common classes but specialization (fine-tuning on ~300–500 labeled sheep
images) is required for anatomical parts. Next step per the chat: SAM-based
auto-annotation → YOLO26-pose training in a sibling repo.

## Architecture

```
upload video → YOLOE-seg (text: head / ear / nose) → Ultralytics ByteTrack
              → match parts to heads → ear-angle geometry (same as v1)
              → annotated MP4 + per-sheep ear-angle timeseries + SPFES bands
```

- One model, one pass. No SAM at inference.
- `model.track(persist=True)` gives stable per-sheep track IDs — ByteTrack is
  built in, so there's no ear-locking logic like v1 had.
- The ear-angle math (`backend/pipeline/ear_angle.py`) is a direct port of
  v1's geometry so chart bands stay comparable.

## Run

```bash
cd ~/dev/lorewood-advisors/sheep-yolo
python -m venv .venv && source .venv/bin/activate
pip install -e .
# First run downloads yoloe-v8l-seg.pt (~350 MB) into the ultralytics cache.
python -m backend.main
# open http://localhost:8000
```

Env knobs (`backend/config.py` for full list):

- `YOLOE_MODEL` — default `yoloe-v8l-seg.pt`. Use `yoloe-v8s-seg.pt` for faster/smaller.
- `MAX_FRAMES` — default 600. Long clips are now cheap.
- `INFER_IMGSZ` — default 960.
- `TRACKER_YAML` — default `bytetrack.yaml`. `botsort.yaml` if ReID helps.

## UI

- Upload a video.
- Pick **Track all sheep** (default) or **Click to select one**.
- Run → annotated MP4 plays + one ear-angle trace per tracked sheep on the
  chart (solid = left, dashed = right) with SPFES up / neutral / down bands.

## What this is not

Not a trained sheep-parts model. Not a welfare claim. Not a product. See
`docs/linkedin-post-draft.md` and `sheep-chat.md` for the framing.

## Validation references

- McLennan &amp; Mahmoud 2019 — Sheep Pain Facial Expression Scale (SPFES)
- Reefmann et al. 2009 — ear posture taxonomy
- Boissy et al. 2011 — ear posture as emotional valence indicator

Thresholds used: `EAR_UP > 30°`, `EAR_DOWN < -10°` relative to dorsal head
axis (nose → ear-midpoint). Same values as v1.
