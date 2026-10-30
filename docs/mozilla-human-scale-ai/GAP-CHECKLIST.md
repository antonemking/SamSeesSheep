# Gap checklist — before the 20 November 2026 submit

Gate for the Initial Project Proposal. Sources: the live CFP, the FAQ (updated 15 October 2026), the eligibility PDF, the form as fetched 30 October 2026, and this repo at `849b0ba`.

**Rule:** an unchecked gate item means you do not submit. A checked evidence item means you can point to a file, a URL, or a named person. `[CONFIRM]` means you still have to look. Do not check a box to clear the list.

---

## Gate (all three, or NO-GO)

- [ ] **External use.** One named person who is not you has run `sheep-pose-v0.7-yolo26n` or the labeler on real footage. You have, in writing: their name, the date, what they ran, what happened (including a miss), and permission to put that in the form.
- [ ] **Checkbox match.** Project stage is either “Working prototype” with a 12 November answer that this still gets a full read, or “Deployed with early users” backed by the bullet above. “None of these yet” is checked on early signals unless a single signal box matches the session. You have not checked Contributors, Community, or Partners.
- [ ] **Video.** Unlisted URL, about two minutes, model actually running, one failure visible, your voice. Link pasted in the form.

If the first box is empty on Wednesday 18 November, stop. The recommendation in `PLAN.md` flips to NO-GO.

---

## Users, community, and the clone test

Public counters as of 30 October 2026. Re-pull them the week you submit. Do not memorize them into the form as adoption.

- [ ] GitHub `antonemking/SamSeesSheep`: stars, forks, and issue commenters re-checked. Baseline here is 1 star, 0 forks, 0 non-author comments.
- [ ] Hugging Face `antking1/sheep-pose-yolo26n`: downloads and likes re-checked. Baseline 83 downloads, 0 likes. You have a sentence that says downloads are not users.
- [ ] Release `paper-artifacts-v0.7` asset download count re-checked. Baseline 1.
- [ ] Issues #1–#4 read. All four were opened by you. Titles mention a “farmer Look/Skip demo” and a triage experiment. `[CONFIRM]` what is actually in the bodies. Relabel or close anything that implies a farmer product `VALIDATION.md` does not recognize.
- [ ] GitHub repo description no longer says the project is built for shepherds in general. Suggested replacement direction: one Katahdin flock, a small pose model, limits in `VALIDATION.md`.
- [ ] No `CONTRIBUTING` file. `[CONFIRM]` whether you will add a half-page “how to file a real issue / how weights are AGPL” before submit. A new file is optional. A misleading issue list is not.
- [ ] You can name zero partners. No letter of support is attached unless the person in the gate box wrote one. Do not solicit a generic endorsement.

---

## Deployment proof (what you may say)

- [ ] The form says inference was benchmarked on a desktop GPU (GTX 1660 Ti, 6 GB), described in `paper/paper.tex` as a stand-in for a future low-power device.
- [ ] The form does not say “at the barn,” “on the Raspberry Pi,” or “on device in the pasture.” The README was edited in `e7ae901` to remove the barn claim. Planning notes about a farm Pi are `[CONFIRM]` and absent from the repo. Leave them out.
- [ ] You have re-read `sheep-yolo/README.md` line that this is not a product, and the form does not call it a shipped product.
- [ ] SAM 3 is described as the labeling teacher (local checkpoint, ~3 GB, 24 GB class GPU for the nose session), not as the daily model.
- [ ] Frontier-API answer is No. Parameter answer is “10 billion or fewer,” with 2.5 M written in prose.
- [ ] You did not invent a parameter count for SAM 3.

---

## Screenshots, video, links

- [ ] Video uploaded and opened in a private window. Audio is intelligible. A failure is on screen.
- [ ] Still frames in the video match artifacts you can point to (`docs/labeling-ui.png`, `docs/v0.7-ear-angle-chart.png`, or a fresh capture from a clip you actually have). No mock UI.
- [ ] Links resolve: GitHub repo, HF model, DOI `10.57967/hf/9129`.
- [ ] `[CONFIRM]` arXiv. The repo says “arXiv-ready.” If there is no id, the form links the GitHub PDF and does not say arXiv.
- [ ] No homepage URL. GitHub `homepage` was empty. `lorewood.dev` was removed from the README. Do not put it back for the form.
- [ ] The word OpenBarn does not appear anywhere in the form, the video description, or the GitHub description.

---

## Numbers you will quote, and numbers you will not

Quote only if you re-open the file that day.

- [ ] v0.7 training scale: 523 reviewed instances, 11 videos. `README.md`, `docs/v0.7-benchmark.md`.
- [ ] HO-1 v0.7 ear-angle σ: left 3.70°, right 4.46°. Same doc.
- [ ] HO-2 v0.7 σ_avg 2.84° (left 2.39°, right 3.29°), with the sentence that v0.4 on that clip is 2.61° and the gap is clip difficulty.
- [ ] Stock model: zero keypoints. `[CONFIRM]` which detection-rate sentence you use. README says sheep boxes on ~35% of frames. The paper also says 12–35%. Pick one source or say “a minority of frames, zero keypoints.”
- [ ] σ described as precision (self-consistency), not accuracy. Paper limitations.
- [ ] Held-out defined as unseen clips of the **same** flock.
- [ ] val mAP 0.479 / 0.643 / 0.732 left out. Logs unrecoverable per `docs/paper-substantiation.md`.
- [ ] Verifier count reconciled. `paper/README.md` says 44/44. `docs/hf-model-card.md` says 59/59. Re-run `sheep-yolo/scripts/verify_paper_claims.py` or cite neither.
- [ ] Failure rows you have **not** measured are written as untested: night, mist, lens smudge, mud, dust, season beyond April–May 2026, wool, second farm, any crop. `VALIDATION.md` still calls the existing failure list predictions.
- [ ] No crop metric, no crop model, no crop partner.

---

## Repo hygiene a reviewer will hit in ten minutes

- [ ] Fresh clone story is true in the form: no `data/`, no `test-clips/`, no `sheep-yolo/weights/` (gitignored `*.pt` and `weights/`). The HF link is how they run a model.
- [ ] `VALIDATION.md` anti-overclaim list checked against every form field. Banned as outcomes: pain, welfare, mood, disease, cross-flock, goats, gait.
- [ ] `docs/CLOUD.md` is referenced from `AGENTS.md` and is missing. `[CONFIRM]` ignore for the form. Do not cite it.
- [ ] Last GitHub push before this plan was 26 September 2026. Hub model last modified 12 June 2026. `[CONFIRM]` any November commit is a truth fix (description, issues, data map), not a new model.
- [ ] License sentence is ready: MIT code, AGPL-3.0 weights.
- [ ] Dataset location. Planning context says clips live on a local drive with a never-train split. That drive is not in this clone. `[CONFIRM]` the never-train split still holds before you describe HO-1 and HO-2 as never labeled. The paper’s claim is that those clips were never pushed to the labeler.

---

## Eligibility and money

- [ ] Eligibility PDF read: https://assets.mofoprod.net/network/documents/Mozilla_Foundation_Grant_Eligibility_Guidelines_tNhKrgM.pdf
- [ ] Applicant type chosen: individual preferred in `INITIAL-PROPOSAL-OUTLINE.md`. `[CONFIRM]` you can legally receive USD and sign the grant agreement.
- [ ] For-profit path not chosen, or you have written advice that the GPU/computer restriction will not break the project. Lorewood Labs is `[CONFIRM]` and is not the repo’s copyright line.
- [ ] Charitable-purpose paragraph says the public gets the protocol, the weights, and the limits. It does not say the grant builds a commercial product.
- [ ] No consulting client, retainer, or personal rate appears in the form.
- [ ] Conflict check: `[CONFIRM]` no immediate family member in a Mozilla staff or contractor role.
- [ ] Regions served: United States / the Delaware homestead. Not “global.”
- [ ] Availability checkbox: `[CONFIRM]` you can do the monthly sessions from April 2027.
- [ ] $300 k paragraph buys time, stipends, and short rentals. It does not buy a cluster. `[CONFIRM]` rental-vs-purchase answer from 12 November.

---

## Info session, 12 November 2026

5 p.m. UTC / 12 p.m. ET / 9 a.m. PT. CFP page said registration was “coming soon” when fetched 30 October 2026.

- [ ] Registration link checked on Monday 9 November and again the morning of the 12th.
- [ ] If the link is still missing on Tuesday 10 November, mail incubator@mozillafoundation.org and ask for it.
- [ ] The ten questions in `PLAN.md` asked, or emailed the same day if you could not attend.
- [ ] Answers written down before any checkbox changes.
- [ ] If they say an external user is required and you do not have one, the form is abandoned on 13 November.

---

## Voice, disclosure, and the final read

- [ ] Every long answer rewritten by you. This folder’s sentences are not in the form verbatim.
- [ ] AI disclosure is Yes if any tool touched the drafting, which includes this planning pass. The “how” box matches what you actually did.
- [ ] Search the form for: `pain`, `welfare`, `OpenBarn`, `Raspberry Pi`, `our users`, `crops we`, `Llama`, `chatbot`, `partnered`, `arXiv` (without an id), `83 users`, `at scale`.
- [ ] Saved-form resume link stored somewhere you control. The FAQ says Mozilla cannot recover that link or the password.
- [ ] Submit is a separate action from save, done before 11:59 p.m. Pacific on Friday 20 November, only with the gate closed.
- [ ] You know a submitted form cannot be revised. If it is not final at 9 p.m. Pacific on the 20th, you do not send it.

---

## Explicitly not required for this deadline

These are grant-year work. Doing them in November does not fix the gate.

- Retraining v0.8.
- A crop demo.
- ONNX / Pi port.
- A public video dataset.
- A fiscal-sponsor contract, unless you already decided the applicant is not you personally. The FAQ allows you to apply before 501(c)(3) status exists.
- A full budget. The initial form, as fetched, asks what the $300 k unlocks. It does not ask for a spreadsheet.
