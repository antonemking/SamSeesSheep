# arXiv paper source — SamSeesSheep

This directory is the **canonical, arXiv-ready** version of the paper. It
supersedes the working Markdown draft at `../docs/archive-paper.md`.

## Build

```bash
pdflatex paper
bibtex   paper
pdflatex paper
pdflatex paper
```

Requires `natbib` (for `unsrtnat`), `graphicx`, `booktabs`, `authblk`,
`caption`, `hyperref` — all standard in TeX Live / arXiv's compiler. No exotic
packages, no `Date.now`-style nondeterminism. Uploading the contents of this
directory (`paper.tex`, `paper.bib`, `figures/`) as a tarball to arXiv compiles
as-is.

## Contents

| File | What |
|---|---|
| `paper.tex` | The paper. |
| `paper.bib` | 13 references (copy of repo-root `paper.bib`). |
| `figures/hero.png` | Static frame from `synced-lanes-6ewes-pro-Test_Clip_Morning.mp4` (Fig. 1). |
| `figures/earangle-img3651.png` | Ear-angle traces, HO-1 (`IMG_3651`) v0.2/v0.3/v0.4/v0.7 (Fig. 3). |

## Reproducing every number

From the repo's `sheep-yolo/` directory:

```bash
python scripts/verify_paper_claims.py          # 44/44 checks vs committed JSON artifacts (works on a fresh clone)
python scripts/gen_bench_Test_Clip_Morning.py  # regenerates the HO-2 JSON — needs the _cache/ archive (see below)
```

The benchmark JSONs are committed, so `verify_paper_claims.py` runs on a fresh
clone. The per-frame prediction caches (`sheep-yolo/artifacts/_cache/*.pkl`) and
the trained weights are **gitignored for size** and ship as a tagged GitHub
release asset (Zenodo DOI to be minted); the `gen_*`/`bench_*` regeneration
scripts need that archive. Three figures are reported from outside the repo and
are **not** checkable by the verification script (disclosed in the paper's Data
and Code Availability section): the validation mAP values (RunPod Ultralytics
training logs), the in-distribution σ range (prior v0.3 benchmark report), and
the held-out clip-similarity screening (performed at capture time; held-out clips
were additionally never pushed to the labeler).

## What changed from the Markdown draft

This version implements the fixes from `../docs/fable5-peer-review.md`:
monotonicity bounded to v0.2→v0.4 with an explicit post-v0.4 plateau; the
cross-clip 2.84° framed as a noise floor rather than a v0.4→v0.7 improvement;
the stock-YOLO baseline reported as sheep-class detection on both clips; a
bootstrap-CI table and statistical caveat; resolved figure/table cross-refs; a
data-availability statement; and acknowledgments to Ultralytics, Meta AI/SAM,
and the ewes.

A subsequent substantiation pass (see `../docs/paper-substantiation.md`) added
meaningful held-out clip IDs (HO-1/HO-2) with a capture-file mapping table,
corrected the HO-2 stock-baseline description (the dominant stock class is
COCO `sheep`, not "dog"), dropped the unverifiable `NCC < 0.23` threshold, and
rewrote Data Availability to reflect that the prediction caches and weights ship
as a release archive rather than in the git tree. The verification harness was
extended to 44 checks (and a silently-skipped detection-rate check was fixed).
