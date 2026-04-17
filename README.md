# SamSeesSheep

**A computer vision pipeline for measuring ear posture in small-flock sheep. Built and validated against a single Katahdin flock in Middletown, DE. Generalization to other breeds and conditions is future work.**

Upload a clip, pick an animal, watch its left and right ear angles over time against published welfare thresholds.

> Part of an ongoing series applying AI to small-flock animal welfare.

```mermaid
graph LR
    A[Video clip] --> B[SAM 3 Video<br/>head + ear] --> C[Per-frame<br/>ear angles] --> D[Smoothed<br/>EKG chart]
```

## Scope — read this first

This is a **visualization artifact**, not a welfare instrument.

- **What it does:** segment a sheep's head and ears in every frame of a short clip, compute per-frame ear angles, render a smoothed timeline with SPFES-referenced threshold bands.
- **What it does not do:** detect pain, score welfare, generalize across flocks, or validate against documented stress events. Validation against stress events is future work.

The thresholds are drawn from clinical studies ([McLennan & Mahmoud 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6523241/), [Reefmann et al. 2009](https://www.sciencedirect.com/science/article/pii/S0168159109001610), [Boissy et al. 2011](https://www.sciencedirect.com/science/article/pii/S0031938411000369)). Applying them to ambient pasture observation is a real and unresolved gap. [`VALIDATION.md`](./VALIDATION.md) is the contract.

## The pipeline

```
Video clip
   │
   ▼
Frame extraction (2 fps, 768 px max)
   │
   ▼
SAM 3 Video — text-prompted video tracker
   prompts: "sheep head" + "sheep ear"   (two SAM 3 Video sessions)
   │
   ├─► Click on frame 0 picks the subject head to lock tracking to
   │
   ├─► Ears attached to the tracked head are kept; other animals' ears rejected
   │
   ▼
Head-PCA midline — long axis of the head mask, disambiguated by ear position
   │
   ▼
Per-frame ear angles (base → tip direction vs. head-horizontal)
   │
   ▼
3-frame median smoothing + SPFES threshold bands
   │
   ▼
Annotated GIF + live EKG-style chart
```

| Component | Model | Size | Runs on |
|---|---|---|---|
| Segmentation + tracking | SAM 3 Video (`facebook/sam3`) | ~3 GB | GPU (GTX 1660 Ti, 6 GB) |
| Ear angle | PCA on head + ear masks | No model | CPU |

One backbone. No ByteTrack, no nose prompt, no mesh reconstruction, no VLM orchestrator — those were earlier experiments that didn't earn their place on the measurement path.

## Run it

```bash
git clone https://github.com/antonemking/SamSeesSheep.git
cd SamSeesSheep
uv sync
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Drop a sheep video. Click the animal you want to track. Wait 1–3 minutes.

**Requirements:** Python 3.12+, CUDA GPU (6 GB+). First run downloads SAM 3 (~3 GB).

## Reading the chart

- **Y axis:** ear angle relative to the head's dorsal axis. Positive = up/forward, negative = back/down.
- **Bands:** green ≥ 30° (up/alert), amber −10°–30° (neutral), red ≤ −10° (down/back), per SPFES.
- **Two traces per sheep:** left ear (blue), right ear (orange). Asymmetry shows up as trace divergence ([Boissy 2011](https://www.sciencedirect.com/science/article/pii/S0031938411000369)).
- **Presence bar:** green = tracked, grey = animal left the frame.
- **Smoothed line + raw dots:** the line is a 3-frame rolling median; the dots are honest about what's under the smoothing.

**Trust deltas, not absolutes.** A within-animal EUP% change before/after a documented event is a defensible delta. Cross-animal averages are not claims this system can support.

## Roadmap / future work

- **SAM 3 → YOLO keypoint distillation.** Current pipeline runs SAM 3 Video per clip (~3 GB VRAM, ~1–3 min per 20-frame clip). Next architectural step is to use SAM 3 masks as pseudo-labels for a small YOLO keypoint model (head + ear tips), so inference runs at video framerate on edge hardware. SAM 3 stays the annotation backbone; YOLO becomes the deployment backbone.
- **Validation against documented stress events** (hoof trim, tagging, separation, startle). This is the welfare-instrument project — a separate undertaking with a capture protocol and a kill criterion.
- **Continuous monitoring** at water troughs, feeders, and the handling chute.
- **Real-time mode.** Current pipeline is batch; a streaming variant would need a causal tracker (ByteTrack on top of SAM 3 obj_ids) — deferred until the offline artifact proves out.

**Kill criterion for the welfare project (not this artifact):** if fewer than 70% of documented stress events show measurable EUP% change, the welfare project ends and the writeup of what failed is itself the deliverable.

## References

- McLennan, K.M. & Mahmoud, M. (2019). [Development of an Automated Pain Facial Expression Detection System for Sheep](https://pmc.ncbi.nlm.nih.gov/articles/PMC6523241/). *Animals*, 9(4), 196.
- Reefmann, N. et al. (2009). Ear and tail postures as indicators of emotional valence in sheep. *Applied Animal Behaviour Science*.
- Boissy, A. et al. (2011). Ear postures as indicators of emotional valence in sheep. *Physiology & Behavior*.
- Ravi, N. et al. (2024/25). [SAM 2 / SAM 3](https://ai.meta.com/sam). Meta AI.

## Built by

[Antone King](https://github.com/antonemking) — applying AI to agriculture and movement science.

---

*Read [`VALIDATION.md`](./VALIDATION.md). It's the contract.*

[MIT License](./LICENSE) — Animal welfare research belongs in the commons.
