# SamSeesSheep

**Segment a sheep's face from a smartphone photo. Extract ear position. Build a depth mesh. All from a single image on an 11-acre Delaware homestead.**

> Part 1 of an ongoing series applying AI to small-flock animal welfare.

<!-- Replace with your actual demo screenshot or GIF -->
![Demo pipeline: photo → segmentation → face extraction → ear overlay → depth mesh](docs/demo-pipeline.png)

---

## The problem

Small ruminants are prey animals. They hide illness. By the time you notice something's off, you're often already behind. If you don't have a quality livestock vet nearby, that window gets even smaller.

Industrial operations have precision livestock farming tools — cameras, sensors, per-animal analytics. Small-flock owners have nothing.

## The research

In 2019, McLennan and Mahmoud at the Universities of Chester and Cambridge published a framework for [automated sheep pain detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC6523241/) using the **Sheep Pain Facial Expression Scale (SPFES)**. The system detects sheep faces, maps facial landmarks (ears, eyes, nostrils, mouth), and scores pain indicators automatically.

The SPFES has been validated against clinical conditions like foot rot and mastitis. Ear position alone — whether ears are up/alert, neutral, or pinned back — is one of the strongest single-frame indicators.

Meanwhile, Meta's Segment Anything Model (SAM) made foundation-level segmentation free. The gap is the join: **nobody has used the foundation-model shift to put a published welfare signal into a homesteader's hands.**

## The pipeline

### 1. Upload a photo

Smartphone photo of a sheep's face. No special hardware. Whatever lighting the sky gives you.

<!-- Replace with your upload screenshot -->
![Upload](docs/step-1-upload.png)

### 2. Click 3 points: face center, each ear tip

SAM segments the head using all 3 points as positive prompts. Each ear is isolated using SAM's positive/negative prompt capability — ear tip as positive, face center as negative — so the model cleanly separates each ear from the face.

<!-- Replace with your 3-click screenshot showing F, 1, 2 dots -->
![Click to segment](docs/step-2-click.png)

### 3. Face extraction + ear segmentation

The head mask extracts a clean face on black background. Ear masks are color-coded and angles are computed relative to the head midline using PCA.

**Thresholds from published literature:**
- &gt; 30&deg; above horizontal → **up/alert**
- -10&deg; to 30&deg; → **neutral**
- &lt; -10&deg; → **down/back** (potential pain indicator)

<!-- Replace with your face extraction + ear overlay side-by-side -->
![Segmentation results](docs/step-3-segmentation.png)

### 4. Depth mesh

Depth Anything V2 estimates monocular depth from the cropped head region. Poisson surface reconstruction builds a smooth 2.5D mesh. The turntable rocks &plusmn;40&deg; to show facial topology.

<!-- Replace with your turntable screenshot or GIF -->
![Depth mesh turntable](docs/step-4-depth-mesh.png)

## Architecture

```
Smartphone photo
       │
       ▼
   SAM (facebook/sam-vit-base)
   3-point prompt: face + ear tips
       │
       ├──► Head mask ──► Face extraction (clay render)
       │
       ├──► Ear masks ──► PCA angle extraction ──► EUP% metric
       │
       └──► Depth Anything V2 ──► Poisson reconstruction ──► 2.5D mesh
```

| Component | Model | Size | Runs on |
|---|---|---|---|
| Segmentation | SAM vit-base | 375 MB | GPU (GTX 1660 Ti) |
| Depth estimation | Depth Anything V2 Small | 50 MB | GPU |
| Mesh reconstruction | Open3D Poisson | CPU | CPU |
| Ear angle extraction | PCA + ellipse fitting | No model | CPU |

## What this measures (and doesn't)

Read [`VALIDATION.md`](./VALIDATION.md) first. It's the contract.

**Measures:** Geometric position of sheep ears in single frames. EUP% (Ear-Up Percentage) as an aggregate metric over time.

**Does not measure:** Pain, welfare, emotion, disease, cross-flock generalization, cross-breed generalization, or anything in goats. The thresholds come from clinical validation studies. Applying them to ambient pasture observation is a real and unresolved gap. This project says so out loud.

**Dataset:** 5 sheep, one breed, one homestead, one phone, one non-veterinary annotator. Trust deltas, not absolutes.

## Run it yourself

```bash
git clone https://github.com/antonemking/SamSeesSheep.git
cd SamSeesSheep
uv sync
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Upload a sheep photo. Click the face, then each ear tip.

**Requirements:** Python 3.12+, CUDA GPU recommended (runs on CPU but slow). SAM and Depth Anything models download automatically on first run (~425 MB total).

## What's next

This is a feasibility study, not a product. The 4-weekend plan:

1. **Weekend 1 — Watch.** Observation + SAM segmentation quality on real photos. **(done)**
2. **Weekend 2 — Build.** Working ear-angle extractor + depth mesh pipeline. **(done, this repo)**
3. **Weekend 3 — Measure.** EUP% across documented stress events.
4. **Weekend 4 — Decide.** Pass or kill against the criterion.

**Kill criterion:** If fewer than 70% of documented stress events show measurable ear-position change, the project ends. The writeup of what failed is itself the deliverable.

## References

- McLennan, K.M. & Mahmoud, M. (2019). [Development of an Automated Pain Facial Expression Detection System for Sheep](https://pmc.ncbi.nlm.nih.gov/articles/PMC6523241/). *Animals*, 9(4), 196.
- Reefmann, N. et al. (2009). Ear and tail postures as indicators of emotional valence in sheep. *Applied Animal Behaviour Science*.
- Kirillov, A. et al. (2023). [Segment Anything](https://segment-anything.com/). Meta AI.
- Yang, L. et al. (2024). [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2). 

## Built by

[Antone King](https://github.com/antonemking) — AI architect by day, farmer by dawn. 11 acres, 5 sheep, a herd of goats, and the conviction that the most interesting AI applications of the next decade won't be in the consumer cloud. They'll be on commodity phones in places where the dataset is whatever you can carry out to the pasture.

First public artifact of [Lorewood Advisors](https://github.com/lorewood-advisors) — applied AI for agriculture, built from the bottom of the market upward.

---

*Read [`VALIDATION.md`](./VALIDATION.md). It's the contract.*

[MIT License](./LICENSE) — Animal welfare research belongs in the commons.
