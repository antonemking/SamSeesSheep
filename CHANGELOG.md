# Changelog

All notable changes to `sheep-seg` will be documented here.

This project follows [semantic versioning](https://semver.org/) at the milestone level: `MAJOR.MINOR.PATCH` where MAJOR = a validated capability gain, MINOR = an experiment delivered, PATCH = documentation or scope corrections.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The [`VALIDATION.md`](./VALIDATION.md) document is versioned alongside the code. Every release will note which sections of `VALIDATION.md` changed and why.

---

## [Unreleased]

### Planned (Weekend 1 — Watch)
- Pick the species (sheep, per the literature) and document breed in `VALIDATION.md`
- 2 hours of unassisted observation of the flock with a notebook only
- 30 phone photos across varied lighting/angles/distances
- Run Meta SAM 3 against the 30 photos and measure segmentation success rate honestly
- One-page v0 plan committed to `docs/v0-plan.md`

### Planned (Weekend 2 — Build)
- Architecture decision: tiny custom keypoint model vs. SAM 3 + published thresholds
- Working ear-angle extractor on a phone

### Planned (Weekend 3 — Measure)
- EUP% computed across 30-minute observation windows
- 6–10 documented stress events with timestamps

### Planned (Weekend 4 — Decide)
- Pass/kill against the 70% criterion
- `VALIDATION.md` updated with measured failure modes
- Public writeup either way

---

## [0.0.0] — 2026-04-11

### Added
- Project skeleton: `README.md`, `VALIDATION.md`, `LICENSE` (MIT), `.gitignore`, `CHANGELOG.md`
- Day-0 [`VALIDATION.md`](./VALIDATION.md) skeleton stating the claim space before any claims are made
- 4-weekend feasibility plan and explicit kill criterion documented in [`README.md`](./README.md)
- Anti-overclaim commitments in [`VALIDATION.md`](./VALIDATION.md) §"Anti-overclaim commitments"

### Decided
- v0 species: sheep (Sheep Pain Facial Expression Scale literature is best-validated)
- v0 signal: ear position only
- v0 metric: **EUP% (Ear-Up Percentage)** over a rolling 60-minute window
- v0 user: project author, on his own animals (no recruitment, no vet partner, no other farms in v0)
- v0 stack TBD pending Weekend 1 SAM 3 segmentation test
- v0 mode: public feasibility study, MIT-licensed, negative results published

### Not yet
- No code
- No data
- No measurements
- No claims

The repository exists at this point as a public commitment to the discipline of writing the validation contract before writing the system.
