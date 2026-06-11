# Paper substantiation audit

Final pre-submission traceability pass over `paper/paper.tex`. Every number and
load-bearing factual claim is classified against the artifact that backs it.

**Verdict.** After this pass, every headline number in the paper is either
(a) auto-checked against a committed JSON by `sheep-yolo/scripts/verify_paper_claims.py`
(**44/44 pass on a fresh clone**), (b) backed by a committed repo artifact that the
harness does not parse, or (c) explicitly disclosed as an external / attestation
figure. One figure is flagged for an author decision (model size; see §C and
§Open author actions).

```
cd sheep-yolo && python scripts/verify_paper_claims.py   # -> OVERALL: PASS (44/44)
```

## Status legend

| Status | Meaning |
|---|---|
| **A** | In a committed JSON under `sheep-yolo/artifacts/` **and** auto-checked by `verify_paper_claims.py`. Reproducible on a fresh clone, no downloads. |
| **B** | Backed by a committed repo artifact (JSON field or dated benchmark doc) the harness does not parse, or a deterministic arithmetic derivation of A-values. Verifiable by inspection. |
| **C** | Not derivable from repo artifacts: produced off-repo (training logs, prior-doc figures), an author attestation, or contradicted-by-artifact and flagged. Disclosed in the paper's Data Availability. |

---

## A — Auto-checked numbers (44/44)

All values below are read from the named committed JSON and compared to the paper
by the verifier. Tolerance is 2% or 0.02 absolute (exact for integer counts).

| Claim (paper §/table) | Artifact | Status |
|---|---|---|
| HO-2 v0.7 σ_L 2.39°, σ_R 3.29°, σ_avg 2.84° (abstract, Tab. 10) | `bench_Test_Clip_Morning.json` | A |
| HO-2 v0.7 primary field 2.84° (not the 2.81° qa_roi variant) | `bench_Test_Clip_Morning.json` | A |
| HO-2 v0.7 in-ROI detection 148/150 (98.7%) (§4.3, Tab. 13) | `bench_Test_Clip_Morning.json` | A *(was silently skipped — fixed)* |
| HO-2 v0.2/v0.3/v0.4 residual σ mean px 6.73 / 4.22 / 3.85 (Tab. 9) | `bench_report-Test_Clip_Morning-v02v03v04.json` | A |
| HO-2 v0.2/v0.3/v0.4 ear-angle σ L,R (Tab. 12) | same | A |
| HO-2 stock sheep-class 118, any-class 986, keypoints 0 (§5.5, Tab. 9) | same (`stock_baseline`) | A *(new check)* |
| HO-1 v0.2/v0.3/v0.4 residual σ mean px 10.89 / 8.90 / 7.70 (Tab. 2, 4) | `bench_report-IMG_3651-3way.json` | A |
| HO-1 v0.2/v0.3/v0.4 raw σ mean px 49.44 / 46.90 / 46.60 (Tab. 6) | same | A |
| HO-1 v0.2/v0.3/v0.4 ear-angle σ L,R (Tab. 3, 8, 11) | same | A |
| HO-1 stock sheep-class 323, any-detection 933, keypoints 0 (§5.5, Tab. 6) | same (`stock_baseline`) | A *(any-detection new check)* |
| HO-1 v0.6 ear-angle σ L 4.65°, R 3.55° (§5.4, Tab. 11) | `bench_report-IMG_3651-v04v05v06.json` | A |
| HO-1 v0.4/v0.5/v0.7 ear-angle σ L,R (Tab. 8, 11) | `bench_report-IMG_3651-v04v05v07.json` | A |
| HO-1 v0.4–v0.7 residual σ mean px 7.70 / 7.83 / 8.09 / 7.90 (Tab. 4, 11) | `bench_residual_px-IMG_3651-v05v06v07.json` | A *(new artifact + check — see §Resolutions)* |

The bootstrap-CI method string (block=10, 5000 resamples, seed 12345) is recorded
in `bench_bootstrap_ci.json`.

## B — Repo-backed, not auto-checked

| Claim (paper §/table) | Where it lives | Note |
|---|---|---|
| Bootstrap 95% CIs, Tab. 7 (HO-1 v0.2/v0.4/v0.7; HO-2 v0.2/v0.4) | `bench_bootstrap_ci.json` (committed) | Cross-checked by hand; matches the JSON exactly. Harness does not parse the `[lo,hi]` arrays. |
| HO-1 σ_avg per version 6.39/4.52/4.07/3.98/4.10/4.08 (Tab. 11) | arithmetic mean of the A-checked L,R per version | Deterministic derivation of A values. |
| HO-1 full-clip detection 909/931/933 (97/99.8/100%) (Tab. 2) | `docs/v0.4-benchmark.md` (committed); regenerable from caches | Not in the 3-way JSON. |
| HO-2 full-clip detection 929/986/986 (Tab. 9) | derivable from the v0.2–v0.4 caches | Doc-level figure. |
| In-ROI window detection rates other than the one A-checked (HO-1 145/150/149; HO-2 132/149/150) | the `detection_rate` fields of the committed 3-way / TCM JSONs | Present in committed JSON, just not asserted by the harness. |
| Per-keypoint residual σ tables (Tab. 5, 10) | `…-3way.json`, `…-v02v03v04.json`, `bench_residual_px-…json` | Committed; harness checks the means, not each of the 30 cells. |
| Per-kpt "% of head width" (3.3–4.2%, 1.4–1.8%, 5.3–5.4%) (§5.2, 5.6.2) | arithmetic from per-kpt σ (B) and head width (234/316 px) | Derivation. |
| Window geometry: frames, ROI, head sizes 234×169 / 316×185, centroid σ (§4.3, Tab. 14) | `window`/`roi`/`target_track_id` fields in the committed JSONs; head sizes from capture-time calibration | ROI + window are in committed JSON; head sizes are calibration metadata (not committed — `artifacts/calibration/` is gitignored). |

## C — External / attestation / flagged

| Claim (paper §) | Disposition |
|---|---|
| **val mAP50-95 0.479 / 0.643 / 0.732** (Tab. 1) | **External, unrecoverable.** From Ultralytics `results.csv` on the RunPod GPU; only weights crossed back to the laptop. The RunPod instance has since been terminated and is **not recoverable**, so the pose-run logs cannot be retrieved — these values are a permanent C-class disclosure. Disclosed in Data Availability ("only the weights, not the logs, return to the laptop"). *(The only local `results.csv` is for the unrelated `_topdown/v0.4-crops` A/B, not these pose runs.)* |
| **In-distribution 4.10–5.92 px** (§5.6, "Held-Out vs In-Distribution") | **Prior-doc figure.** Quoted from `docs/v0.3-benchmark.md` (committed) lines 34–35; the underlying `bench_report-IMG_358*/360*.json` are training-set data held off-repo. Cited `\citep{v03benchmark}` and disclosed as such (`\label{sec:indist}`). |
| **NCC < 0.23 vs every training video** | **Dropped — unverifiable.** No code computes NCC and no matrix was ever saved; the held-out clips' raw frames and the training-video frames are both off-repo (test-clips/ holds only a README; data/ is gitignored), so NCC cannot be computed from this repo. Removed from the paper entirely; softened in README, ARCHITECTURE.md and `bench_held_out.py` to the substantiated screening (never pushed to the labeler + visually distinct). See §Resolutions. |
| **"Never pushed to the labeler"** | **Author attestation.** Enforced by per-file sha256 + per-pod dedup in `.pushed_clips.tsv`, but that file is gitignored, so a reader cannot independently re-derive it. Framed as an attestation in Data Availability. |
| Clip-similarity screening "at capture time" | **Author attestation.** Disclosed in Data Availability. |
| **Model size "~10 MB"** (abstract, §1, §3.5, contribution 2, Fig. 1, conclusion) | **Flagged for author.** The on-disk `best.pt` is **6.04 MB** (6,037,991 bytes), consistent across v0.2–v0.7. "~10 MB" matches the FP32 parameter footprint (2.5 M × 4 B), not the deliverable file. Left unchanged to preserve repo-wide consistency (README/VALIDATION/CHANGELOG all say ~10 MB) and because the FP32 framing may be deliberate. **Recommend** stating "~6 MB on disk" (line 298 explicitly says "`best.pt` is ~10 MB", which the artifact contradicts). Data Availability deliberately cites no weight size to avoid an internal contradiction. |
| Hero video "2036×640 px, 7.5 s" (Fig. 1) | From the archived `.mp4`; verifiable once the archive is published. |
| Clip provenance: 5 Katahdin ewes, Middletown DE, iPhone 1080p/30fps, April–May 2026; HO-2 "61 MB / 986 frames / ~30 fps" | Dataset attestation; standard for a dataset paper. Not machine-checkable. |
| "~6 min" training time, "2.5 M parameters" | External/architecture spec (RunPod; YOLO26n-pose). |
| SPFES 40° band, ≥30° / ≤−10° thresholds | Literature (`\citep{mclennan2019}`). |

---

## Resolutions applied this pass

1. **§5.5 stock-baseline correction (factual error fixed).** The paper claimed the
   HO-2 stock detections were "predominantly COCO 'dog'." The committed cache
   (`stock-yolo26n__Test_Clip_Morning.pkl`) shows `other_class_counts = {sheep(18):
   5551, horse(17): 3, person(0): 1, dog(16): 1}` — the dominant detected class is
   **COCO sheep**, and "dog" appears once. (The peer-review note mis-decoded class
   18 as "dog"; it propagated into the paper.) Rewrote §5.5 and the Tab. 9 caption to
   report only the committed, load-bearing figures (sheep-class 12% / 35%, 0
   keypoints) and dropped the class-composition editorializing. Added harness checks
   for the HO-2 stock fields (118 / 986 / 0).

2. **Residual-σ px gap closed (C → A).** The v0.5/v0.6/v0.7 residual-σ px values
   (7.83 / 8.09 / 7.90 in Tab. 4 and Tab. 11) were in no committed artifact. Added
   `scripts/gen_residual_px.py`, which recomputes them from the caches (and exactly
   reproduces the committed v0.2/v0.3/v0.4 values), saved
   `bench_residual_px-IMG_3651-v05v06v07.json`, and added 4 harness checks (v0.4 as a
   cross-source anchor).

3. **Verifier bug fixed (35 → real checks).** The HO-2 detection-rate check read
   `in_roi_target_detection_rate` from `data["v0.7"]` instead of top-level `data`, so
   it silently never ran — the documented "36/36" was actually 35 effective checks.
   Fixed it to run, and extended the harness to **44 checks**.

4. **NCC threshold dropped.** Removed `NCC < 0.23` from the paper (it was already
   absent from `paper.tex`; confirmed and kept out), and softened the four other
   live assertions: `README.md`, `ARCHITECTURE.md` (×2), `sheep-yolo/scripts/bench_held_out.py`.
   Dated/archival docs that still mention it (`CHANGELOG.md`, `docs/v0.4-benchmark.md`,
   `docs/archive-paper.md`) are left as historical snapshots.

5. **Data Availability rewritten for accuracy.** The old text claimed the prediction
   caches are "in `sheep-yolo/artifacts/_cache/`" — but `_cache/*.pkl` and the
   weights are **gitignored** (only `artifacts/*.json` is whitelisted). Rewrote it as
   a layered story: committed JSONs (clone-and-verify, 44/44) → companion release
   archive for caches+weights (regenerate JSONs; **GitHub release + Zenodo DOI
   placeholder**) → raw clips + dataset on request. Updated `paper/README.md` to match.

6. **Clip identities renamed.** `IMG_3651` → **HO-1** (afternoon, foreground) and
   `Test_Clip_Morning` → **HO-2** (morning, background) throughout the prose, with a
   new mapping table (Tab. 14, `\label{tab:clipmap}`) tying each paper ID to its
   capture filename. Artifact and script filenames keep the original IDs (the table
   is the bridge). See [Job 2 below](#clip-identity--data-availability-job-2).

7. **One stray "welfare" claim removed.** §5.3 called ear angle "the welfare-relevant
   scalar" (an author characterization, not a literature citation) — changed to "the
   pipeline's output scalar." All other "welfare"/"pain" occurrences are literature
   context or explicit non-claims, per the VALIDATION.md contract.

## Clip identity & Data Availability (Job 2)

| Paper ID | Capture file | Lighting | Target | Artifacts using the original ID |
|---|---|---|---|---|
| **HO-1** | `IMG_3651.MOV` | Afternoon | Foreground ewe, track 34 | `bench_report-IMG_3651-3way.json`, `…-v04v05v0{6,7}.json`, `bench_residual_px-IMG_3651-v05v06v07.json`, caches |
| **HO-2** | `Test_Clip_Morning.mov` | Morning | Background ewe near fence, track 215 | `bench_Test_Clip_Morning.json`, `bench_report-Test_Clip_Morning-v02v03v04.json`, `synced-lanes-6ewes-pro-Test_Clip_Morning.mp4`, caches |

**Data Availability plan (decisions taken; see Open author actions to confirm):**

- **Shared in git (clone-and-verify):** code, the benchmark `artifacts/*.json`, the
  paper source + figures, docs. `verify_paper_claims.py` passes 44/44 on a fresh
  clone with no downloads.
- **Companion archive (caches + weights):** `_cache/*.pkl` (~3 MB total) and the
  per-version `best.pt` weights are gitignored for size; publish them as a **tagged
  GitHub release** asset (e.g. `paper-v0.7`) and mint a **Zenodo DOI** from that
  release. With the caches a reader regenerates every committed JSON from scratch.
  *Default chosen: caches-only public + GitHub release; DOI left as `<DOI placeholder>`.*
- **On request:** the raw held-out clips (HO-1, HO-2) and the full reviewed-keypoint
  dataset — withheld for size and to limit redistribution of identifiable homestead
  footage. A reader with the clips + released weights can regenerate the caches
  themselves; without them, every published number is still reproducible from the
  caches onward.

## Open author actions

1. **Mint the archive DOI.** Tag a GitHub release with `_cache/*.pkl` + the
   `sheep-pose-v0.{2..7}-yolo26n.pt` weights, then mint a Zenodo DOI and replace
   `<DOI placeholder>` in `paper.tex` Data Availability.
2. **Confirm clip-sharing policy.** Default is caches-only (clips on request). If the
   footage is privacy-clear and size is acceptable, the held-out clips can also go in
   the release; update Data Availability's "on request" paragraph accordingly.
3. **Decide the model-size figure.** `best.pt` is 6.04 MB on disk; the paper says
   "~10 MB" in 6 places (and README/VALIDATION/CHANGELOG). Recommend changing to
   "~6 MB on disk" (optionally noting the ~10 MB FP32 footprint) repo-wide, or
   confirm the FP32 framing is intended.
4. **val mAP stays external — logs unrecoverable.** The RunPod instance is
   terminated, so the pose-run `results.csv` cannot be recovered; the
   0.479/0.643/0.732 values in Table 1 remain a permanent disclosure (the weights
   on the laptop are the only surviving training output). No action available
   beyond the existing Data Availability disclosure.
