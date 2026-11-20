# Initial Project Proposal — outline

**Form:** https://mozillafoundation.tfaforms.net/f/humanscaleai
**Deadline:** Friday, 20 November 2026, 11:59 p.m. Pacific. A submitted form cannot be edited.
**How to use this file:** section order matches the live form as read on 30 October 2026. Sentences under “Spine” are facts you can say. Anything marked `[CONFIRM]` is not yet a fact. Rewrite every spine in your own voice before it goes in the form. Do not paste this outline. Mozilla’s AI policy treats a fully generated application as a review problem. Disclose whatever help you actually used, in the disclosure box at the end.

**Recommended title (working):** Farm Failure Scorecard — small vision models that stay on the farm.

**Recommended one-sentence summary:** I train few-megabyte vision models on a real homestead and measure whether they still work when the farm gets hard — mud, mist, night, occlusion — starting from a sheep-head model that already runs on a desktop GPU.

If the external-use gate in `GAP-CHECKLIST.md` is open on 18 November, do not fill this form.

---

## 0. Applicant information and eligibility

### Applicant type

Options on the form: Individual; Nonprofit or NGO; Fiscal sponsorship; For-profit or other organization; Governmental unit; Other.

**Spine:** Apply as an **individual** carrying out a charitable project, unless a fiscal sponsor is already signed. `[CONFIRM]` legal name, mailing address, and that you can receive USD by check, ACH, or wire and sign a grant agreement.

**Do not default to a for-profit company.** The eligibility guidelines say for-profit grants are not general operating support and, as an example, may not buy computers, servers, or GPUs over certain amounts. `[CONFIRM]` at the 12 November session whether an individual grantee can rent a GPU for short training runs.

Lorewood Labs is `[CONFIRM]`. It is not the copyright holder in this repository. Do not list it until you have decided it is the applicant.

### Regions served

**Spine:** United States. The working system has been used on one pasture in Middletown, Delaware. Do not select a world region you have not served.

### How you heard about it

`[CONFIRM]` the true channel (the X post, a person, search). Referrals are not a review criterion. Still be accurate.

### Availability checkbox

You can check this only if you can attend roughly one or two hour-long virtual sessions a month from April 2027 through March 2028, plus the Festival showcase. `[CONFIRM]` against the consulting calendar before you check it.

### Eligibility checkbox

Check only after you have read the PDF: https://assets.mofoprod.net/network/documents/Mozilla_Foundation_Grant_Eligibility_Guidelines_tNhKrgM.pdf

There is a country hard-stop. A Delaware applicant is not in the sanctioned-country category. `[CONFIRM]` no conflict of interest with Mozilla staff.

---

## 1. Project overview (the screening page)

These fields feed short-form flags. They matter more than the prose.

### Proposed project title

Farm Failure Scorecard.

Avoid “SamSeesSheep” as the title. The repo can be the link. Avoid “welfare,” “pain,” “OpenBarn,” and “livestock and crops platform.”

### Project summary

**Spine:** A 2.5-million-parameter sheep-head model (~6 MB) already locates five keypoints in ambient pasture video from one Katahdin flock in Middletown, Delaware. Stock detectors find no keypoints on the same clips. The grant year publishes a failure-mode scorecard for that model — lighting, blur, occlusion, night, lens contamination — and keeps daily inference on hardware the farm holds. A second head (another flock, or one crop task) happens only if a second person brings data.

`[CONFIRM]` character or word limit, if the form enforces one. The HTML fetched on 30 Oct 2026 showed no `maxlength`.

### Which principles — select all that describe the project

Options: Small, Intentional, Tailored.

**Spine:** Check all three, then use the long answers to show they are not equally proven. Small is demonstrated by the 2.5 M runtime. Intentional is a data path you can describe and have not yet handed to anyone else. Tailored is one homestead, not a market.

If the 12 November session says the flag expects all three to be true in production, revisit this with the rubric in `PLAN.md` before you submit.

### Project stage — select one

Options:

- Concept or wireframe only
- Working prototype
- Deployed with early users (less than 100 users)
- Deployed at scale (100+ users)

**Today’s honest selection:** Working prototype.

**Selection if the gate is met:** Deployed with early users (less than 100 users), and the traction essay names the person, the date, and what they ran.

**Never:** Deployed at scale. **Never:** Concept or wireframe only — the pipeline runs, and that box throws away the only technical asset you have.

`[CONFIRM]` after 12 November whether “working prototype” is auto-short-formed. The CFP text dismisses prototypes. See `PLAN.md`.

### Early signals — select all that apply

Options, verbatim:

- Users: People are using your technology and coming back to it.
- Contributors: People outside your core team contribute code, content, data, or expertise.
- Community: The people you serve help shape your product, priorities, or governance.
- Partners: Organizations are piloting, deploying, collaborating on, or funding your work.
- Downstream adoption: Other builders use what you’ve created in their own work.
- None of these yet.

**Today’s honest selection:** None of these yet.

You use your own tool. That does not match “people,” “outside your core team,” “organizations,” or “other builders.” GitHub on 30 October 2026: 1 star, 0 forks, 4 issues opened by you, 0 comments. Hugging Face `antking1/sheep-pose-yolo26n`: 83 downloads, 0 likes. The release asset has 1 download. Do not check Downstream because of the download counter.

**If the gate is met:** check the one box that matches what they actually did (Users or Downstream). Leave the others empty. Name them in the traction answer.

### Approximately how many users

Options start at “Less than 100.” There is no “1” and no “just me.”

**Spine:** If you are still the only operator, do not let this dropdown speak for you. Either the project is not ready to submit, or you pick Less than 100 and the very next sentence of the traction answer is: “One operator, the author, on his own flock.” `[CONFIRM]` which of those two you are doing. Prefer not submitting over the quiet dropdown.

If the external person counts, the number you write in prose is the real count, not “under 100” as a story.

### Primary outputs — select all that apply

Options include: Technology platform or tool; Research publication; Policy framework; Community organizing tools; Educational resources; Creative or artistic work.

**Check:** Technology platform or tool. Research publication (the preprint is in `paper/paper.pdf`; `[CONFIRM]` it is not on arXiv until you have an id).

**Leave unchecked:** policy, community organizing, creative work. Educational resources only if you mean `SETUP.md` and `VALIDATION.md` and are willing to call them that in one sentence. They are docs for a developer, not a curriculum.

### Accessibility or disability-inclusive design

Options range from “central… built with and for people with disabilities” to “Not at this time.” The form says every answer is still eligible.

**Spine:** Not at this time. Or “We have considered it, but it is not yet a focus,” if that is truer after you look at the labeler. Resource-constrained farms are not a disability claim. Do not borrow one.

### Frontier API

Question: “Does your project’s core functionality depend on calling the API of a frontier AI model?” Hint: GPT, Claude, Gemini/DeepMind, Llama.

**Spine:** No.

The runtime does not call those APIs. Changelog (April 2026) removed a Claude/Gemma orchestrator that was only emitting the prompt list `sheep head`, `sheep ear`, `sheep nose`. SAM 3 runs from a local checkpoint as an annotator, not as an API product. Say that in the Small answer so “No” is not a dodge.

### Largest parameter count of any model core to functionality

Options: 10 billion or fewer; then 10 B bands; More than 40 billion; My project does not use AI models directly.

**Spine:** 10 billion or fewer.

Write “2.5 million parameters” in the Small essay. Do not select “does not use AI models directly.” `[CONFIRM]` do not invent a parameter count for SAM 3. The architecture note says the checkpoint is about 3 GB. If you call SAM 3 core, it still sits in this same dropdown bucket, and you owe the justification the FAQ asks for.

### Charitable purpose

Question: does the project serve a charitable purpose and create meaningful public benefit?

**Spine:** Yes, as scientific research and public education: an open protocol and a small model others can run, with the benefit outside any product revenue. Open source alone is not the purpose (eligibility PDF). The primary purpose of the funded work is not commercial gain.

Then the short public-benefit box:

**Spine:** Beyond me, a developer or a small-flock owner gets a few-megabyte model, a written limit on what it measures, and a scorecard for farm failures that lab benchmarks skip. They can run it without sending a frontier model their footage. Today that benefit is the published code and weights. The grant is what makes the scorecard and a second person’s use real.

### Work in the open — select all that apply

Options: Open sourcing code or other assets; Publishing educational materials; Publishing research papers; Openly sharing datasets; Public workshops or presentations; Another public-facing resource; None of the above.

**Check now:** Open sourcing code (MIT pipeline). Publishing research papers (preprint in repo; `[CONFIRM]` arXiv). Another public-facing resource (Hugging Face weights, AGPL-3.0, DOI `10.57967/hf/9129`).

**Do not check:** Openly sharing datasets. Video, frames, and `review.json` are not in the git tree, on purpose. Public workshops: none yet. `[CONFIRM]` only check workshops if one is scheduled before submit, which is unlikely and not required.

---

## 2. Project details (long answers)

Write these yourself. The spines are the claims that survive the repo. Target length: enough to be specific, short enough that a reviewer finishes them. Lead with the thing, then the limit.

### What technology are you building, and how does it address this cohort’s theme?

**Spine:**

I am building a small on-farm vision stack. The piece that already runs is SamSeesSheep: short phone clips of my Katahdin flock, a large vision model used only to propose keypoints, a human review step, and a 2.5 M-parameter YOLO-pose model (~6 MB) that places nose and ear points and reads an ear angle. On held-out clips from the same pasture, ear-angle jitter sits around 3–4 degrees. A stock YOLO on those clips produces zero keypoints.

That split — large model as a labeling teacher, tiny model as the thing that actually runs — is the Human-Scale bet. The theme asks for AI sized to a place and a job. A farm does not need a general model in a data center to answer “where are the ears in this frame,” and it cannot depend on one when the lens is muddy.

The incubator year is not a new sheep paper. It is a public failure scorecard for this kind of model: lighting, blur, occlusion, night, smudge and mud. Sheep is the first sheet, because it is the one I have measured. Crops are a later sheet if a grower shows up with plants. I am not shipping a chatbot, and I am not scoring pain or welfare. `VALIDATION.md` in the repo is the limit I will keep.

**Leave out:** OpenBarn. Pi deployment. Any user you have not met. The phrase “at the barn.”

### Small: models, hardware, infrastructure, and why they fit

**Spine:**

- Runtime: YOLO26n-pose, 2.5 M parameters, about 6 MB, imgsz 640 in training, benchmarked at inference on a GTX 1660 Ti (6 GB). The paper calls that card a stand-in. Low-power hardware is the design target and is not yet the measured deployment. `[CONFIRM]` no Pi result before you name a Pi.
- Why this size: the keypoint head has to be trained on the animals. A general detector does not grow one. Once trained, the file is small enough to copy to the farm and leave there. Re-training a head, given labels, took about six minutes on an RTX 4090 at batch 8. I do not need a standing cluster.
- Teacher, not runtime: SAM 3 Video, local checkpoint around 3 GB, text prompts “sheep head,” “sheep ear,” “sheep nose.” The nose pass wants a 24 GB GPU. Two jobs at once can OOM the machine. That is tolerable for labeling and unacceptable as the farm’s daily path. The grant keeps SAM off that path.
- No frontier API. A earlier VLM step was removed because it only repeated a fixed prompt list.
- Why the fit is practical: bandwidth at the farm, a machine I can unplug, and a model I can retrain when the season changes without renting a foundation-model endpoint.

**Numbers to cite, with sources:**

- 523 reviewed instances, 11 training videos, v0.7. Source: `README.md`.
- HO-2 σ_avg 2.84° (left 2.39°, right 3.29°) for v0.7, and the note that v0.4 on the same clip is 2.61°, so the lower number is the clip. Source: `docs/v0.7-benchmark.md`.
- HO-1 v0.7 left 3.70°, right 4.46°. Source: same file.
- Do not cite val mAP. The training logs were on a pod that was terminated (`docs/paper-substantiation.md`).
- Do not cite “44/44” or “59/59” until you re-run `sheep-yolo/scripts/verify_paper_claims.py` and the model card match.

### Intentional: how users understand and control data

**Spine — say what is true today, in order:**

1. The only flock in the system is mine. I capture on one iPhone. I am the only annotator. I am not a veterinarian.
2. Frames and `review.json` stay out of the public repo. A fresh clone has no videos and no weights. Weights are on Hugging Face. Raw footage is not.
3. During labeling, clips go to a GPU pod I start by hand so SAM 3 can propose points. That is a real transmission. I should describe it, including that the pod is not a multi-tenant product.
4. After training, the artifact that has to move is `best.pt`, about 6 MB.
5. Training export keeps human-reviewed points (`v=2`). Unreviewed machine points (`v=1`) stay out unless someone deliberately passes `pseudo=true`, which the docs forbid for benchmarks.
6. The code is MIT and can be forked. The weights are AGPL-3.0 because they were trained with Ultralytics. A downstream builder inherits that.
7. What a second person cannot yet do: see a data map in the product, delete a cloud copy from a screen, or govern a roadmap. That is the incubator work, not a feature I will imply already exists.

**Add only if you have written the one-page map:** where a clip sits, who can delete it, what never needs to leave.

**Do not say:** end-to-end encryption, on-device identity, peer-to-peer inference, “no server ever sees the video.” The labeling pod sees video today.

### Tailored: how the people, problem, and context shaped the technology

**Spine:**

The context is one pasture and one breed. Katahdin hair sheep, about five ewes plus lambs, Middletown, Delaware, filmed April–May 2026 on a handheld iPhone, no rig, no controlled light. I live there. That is why the labeler is a review tool instead of a drawing tool: I will not hand-label every frame, and I will not trust an unlabeled foundation-model point.

The same context forced the schema. More than one sheep in the frame meant `instances[]` per frame, image-space left and right, and a flip index so augmentation does not swap ears silently. Ear angle is computed from tip, base, and a head midline because that is the measurement I can define. I publish residual jitter on a nearly still animal because a wiggling trace is how this tool would lie to me.

The limits are the same context. Both “held-out” clips are these animals. σ is precision, not accuracy against a second observer. There is no night footage, no winter, no wool breed, no second farm. Failure modes I have written down — dusk on dark fleece, motion blur, profile views, occlusion, wool — are mostly still predictions. `VALIDATION.md` says so. The scorecard exists to turn that list into numbers, including failures.

Nobody else has shaped the roadmap yet. `[CONFIRM — replace this sentence only if the external session happened, and quote what they changed.]`

Crops have not shaped anything in the repo. I will not describe a crop model.

### Traction / early product-community fit

**Spine if the gate failed:** do not submit.

**Spine if you are the only user and you were told that still counts:**

I am the user. I have trained seven versions on my own flock and I use the held-out benchmark to decide whether a new version is real. Public interest I can point to without inventing users: the GitHub repo (1 star, 0 forks as of 30 October 2026), the Hugging Face weights (83 downloads, 0 likes — downloads are not people), and a preprint in the repo. No outside contributor, no partner, no letter. `sheep-yolo/README.md` says this is not a product. I am applying because the model runs and the missing piece is other people, which is what the year is for.

**Spine if the gate is met:**

On [date], [name, role, and whether they have animals or are a developer] ran [the v0.7 weight / the labeler] on [their clip / my clip, say which]. What happened: [one observed result, including a miss]. They said I could cite that. They have not become a community. They are the first evidence the CFP asked for.

`[CONFIRM]` name, date, artifact, permission. No permission, no sentence.

### Challenges and biggest blockers

This is the strongest honest section. Use it.

**Spine:**

- I am one person. The labeling server is a single SAM 3 process with no lock. It is not a service for a cohort of farms.
- Product-community fit is the blocker, not another point of mAP. The curve flattened after v0.4. v0.6 got worse on one ear and I published that. More of my own frames will not create a user.
- Every measured frame is one breed, one farm, one spring, one camera, one annotator. I do not have a night, mud, or smudge number. Claiming robustness before those rows exist would contradict the repo.
- Edge hardware is a goal. The benchmark machine is a desktop GPU.
- I will not turn ear angle into a welfare product inside this grant. The clinical gap is unsolved and the kill criterion for a stress study is a different project.
- Receiving the money cleanly: `[CONFIRM]` individual versus fiscal sponsor, and whether short GPU rentals are an allowed cost.
- The public issue tracker has notes that sound like a farmer demo. They are mine. I will clean that so the repo matches the validation contract. `[CONFIRM]` done before submit.

### Who is the team, and what experience do you have

**Spine:**

I am the team. Antone King, Middletown, Delaware. I built the labeling pipeline, ran the seven training iterations, and wrote the validation contract so the repo cannot claim pain detection or cross-farm generalization. The relevant experience is this repository, from April 2026 to the v0.7 benchmarks, plus living with the flock the data comes from.

`[CONFIRM]` any sentence about movement-science or other work you want in this box. If it is not needed to show you can execute, leave it out. Do not invent a co-founder, an advisor, or a vet.

The FAQ allows a small team if it can do the work. “Just me” is true. The $300 k answer is where you say the money buys other people’s time, not where you pretend they are already here.

---

## 3. Impact and openness

### Success in 12 months and in 36 months, and how you will measure it

**12 months (through March 2028), measurable:**

- Farm Failure Scorecard v0 published, with at least four named conditions run on the existing flock, at least two of them expected to be ugly (night, mud or smudge, or full-rate motion blur). Each row has detection rate, keypoint availability, and residual σ, plus the stock-model baseline where it applies.
- One recorded run on a low-power device you name only after the run, with frame rate and the first break.
- Daily inference documented without the labeling pod.
- Either one external person has used a build at least twice, or you publish a one-page log of who declined. Both are measurements.
- A second head (other flock or one crop task) only if the data showed up. Otherwise that milestone is explicitly skipped in public.

**36 months, as a hypothesis you will test, not a promise:**

- Someone you do not employ reuses the scorecard or the weights on animals or plants you do not own.
- A second person can cut a release.
- If those are false, the project’s public remainder is the scorecard, the weights, and the written limits, and you will say the organization did not grow.

**Measures you will not use:** mAP as the headline, Hub download counts as users, ear angle as a welfare KPI, a crop accuracy number you do not have.

### What you need to become sustainable, and the barriers

**Spine:**

I need a maintainer who is not only me, a fiscal home that can receive charitable funds without turning the work into a vendor product, and a reason for a second farm to keep labeling. The barriers are time, the absence of that second farm today, and the fact that an AGPL weight file does not pay for annotation. I am open to the model the cohort actually fits — fiscal sponsor, a tiny services practice that is outside the grant, or stopping at the public blocks. I am not proposing a surveillance product or a subscription wrapped around other people’s sheep.

`[CONFIRM]` strike “services practice” if you do not mean it. The eligibility PDF does not allow the grant’s primary purpose to be commercial gain.

### What would this $300,000 unlock?

**Spine:**

Time to run the failure protocol through a year of weather, stipends for a handler or a second annotator so the labels are not only mine, short GPU rentals to retrain the small model when a new head earns its labels, and the writing and video that make the scorecard usable by someone who was not in the pasture. It does not unlock a foundation-model API bill, a GPU purchase I may not be allowed to make, or a multi-species platform.

`[CONFIRM]` the equipment sentence against the 12 November answer.

### How you will work in the open

**Spine:**

The pipeline stays MIT at https://github.com/antonemking/SamSeesSheep. New scorecard code goes in the same repo. Weights stay on Hugging Face under AGPL-3.0, with the model card’s limits kept current. Benchmark JSON stays in the repo, as it does now, so a clone can check the numbers without my clips. I will not publish the flock videos as the openness plan. Farm location and individual animals are in those frames. Selected stills I am willing to show, the protocol, and the weights are the public bundle. Negative results stay in the repo, including regressions.

`[CONFIRM]` a Festival talk if you are selected. Do not promise a workshop tour.

### What novel approach does this introduce?

**Spine:**

Three things, each smaller than a platform claim:

1. Use a large segmentation model as a teacher a human corrects, then ship a 2.5 M student. The novelty is the split, not a new network. Animal-pose tools already exist for labs. They are not this annotation economy, and they are not tested as a farm failure scorecard.
2. Judge the student by residual jitter and by named farm failures, and publish the plateau and the regression. v0.6’s left ear got worse while the right ear improved. That result is in the repo so I cannot hide it.
3. Treat “footage stays on the farm” as an architecture constraint: the daily file is ~6 MB, and the teacher never has to run at the trough.

**Do not say:** first farm AI, first sheep model, replacement for DeepLabCut, or an alternative to CattleEye’s product. You can say commercial livestock vision generally takes the footage to the vendor, and this design is the opposite shape, without naming a company you have not benchmarked.

### Links

- Code: https://github.com/antonemking/SamSeesSheep
- Weights: https://huggingface.co/antking1/sheep-pose-yolo26n
- DOI: https://doi.org/10.57967/hf/9129
- Preprint PDF in the repo: `paper/paper.pdf` on that GitHub tree. `[CONFIRM]` an arXiv URL if one exists. Do not write “arXiv” without it.
- Video: `[CONFIRM]` unlisted URL, two minutes, model running, one failure, your voice.

There is no project homepage in the GitHub metadata (`homepage` empty on 30 October 2026). Do not invent a URL. A previous personal-site link was removed from the README on purpose.

### Video (optional, strongly encouraged, priority flag)

**Shot list, all of it true:**

1. Ten seconds of the pasture and what the camera actually is (phone).
2. The five points locked on a head, and the ear-angle trace on a still animal, so jitter is visible.
3. One failure: occlusion, blur, or a stock-model frame with no points. Say which.
4. You, in one sentence: the 6 MB model is the product, the big model is the labeling teacher, the footage is mine, and the grant is for measuring when this breaks and for getting it into someone else’s hands.

No soundtrack, no welfare dashboard, no crop B-roll, no Pi unless the Pi is really in the shot and running the model.

---

## 4. AI disclosure and terms

### Were AI or translation tools used?

`[CONFIRM]` at submit time. If you used any assistant to outline, edit, or translate, the answer is yes. This planning folder was drafted with an assistant. That alone means the truthful answer is yes unless you throw the outline away and write from scratch with no tool. Even then, if a tool touched the file, say yes.

### If yes, how?

**Spine you can adapt:** I used an AI assistant to organize the Mozilla criteria against my repository and to flag claims the repo does not support. I wrote the submitted answers myself. The technical claims are from my benchmarks and `VALIDATION.md`. The assistant did not invent users, partners, or results, and I removed anything I could not point to.

Make that true before you submit it. If a tool drafted a paragraph you kept, say that you edited it.

### Share with other funders / newsletter

Optional. Neither affects review. `[CONFIRM]` your preference. Default in this outline: sharing with other funders is reasonable for a charitable research project; the newsletter is your call.

### Privacy notice

Read it. Check it only if you agree. https://www.mozilla.org/en-US/privacy/websites/

---

## 5. Voice pass before you hit submit

Read the form once looking only for these strings. None of them should remain unless the sentence is “I do not do this.”

`pain` `welfare` `stress score` `OpenBarn` `Raspberry Pi` `on the barn` `our users` `farmers are using` `crops we detect` `Llama` `chatbot` `partnered with` `letter of support` `arXiv` (without an id) `83 users` `deployed at scale`

Then read `VALIDATION.md` §Anti-overclaim commitments and delete any form line that would violate it.
