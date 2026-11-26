# The first experiments of the next build: pre-registrations

*Written Nov 26, 2026, in the second review run, from `docs/review/NEXT_BUILD.md` §10. Nothing here has been run.
Each entry follows the repository's build routine (`CLAUDE.md`): the build step it serves, the question with a yes/no
answer and the decision each answer leads to, the measurement (built first and proven on known cases), the cases
that must pass and must fail, the controls, how many runs, and where the pass lines come from. Pass lines come from
controls, never from the results they judge. Each experiment needs Shreyas's go before any spend.*

*Revised the same night after an independent critique (`docs/review/CRITIQUE.md`; changes logged in REVIEW_LOG.md,
F44–F56): X0 added; X1, X3, X4 and X6 changed; X7 and X8 added.*

*Shared conventions:*
- *"Held out" means inputs (positions, flows, prompts, networks) never used to design the method or fit the
  candidate.*
- *Every run writes its full record (programs, calls, scores) to `results/next/<experiment>/` and is pushed as it
  lands.*
- *Free experiments come first in every entry: what can be re-scored or re-read from saved runs is done before
  anything new is paid for.*

---

## X0. The test against copies, on saved runs (main arm free; reader arm cheap)

**Build step it serves.** The anti-copy measure of the box (NEXT_BUILD §3.2 item 3). Every later claim of
"explanation, not copy" rests on it, and X1's AlgZoo arm uses it if it passes.

**Question.** Written as estimators of what their networks do, do the real explanations of small networks beat
sampling the network, where every saved copy, disguise and the shuffled-network account does not?
- **Yes**: answering without running becomes the anti-copy gate (NEXT_BUILD Q12).
- **No** (a must-pass account fails, or a must-fail case passes): report it, return to the intent, and keep copies out
  by restricted languages and must-fail sets alone, with that limit stated on every claim. Do not tune the lines.

**Cases (the T2 sets, `docs/glass/info_measure_T2.txt`).**
- Must pass: M4,3's account (c928, fidelity 0.990) and the lc_8_6 census rule (round 2 b, fidelity 0.922).
- Reported, but unable to decide anything: M2,2's account. It is the task rule, and the network matches it at
  fidelity 1.000, so every estimator, copy or not, gets every quantity below exactly right.
- Must fail: every saved copy and near-copy in the T2 table, the 18 disguises D1–D6, and the lc_3_4 shuffled-network
  account.

**Measurement (frozen before any scoring).**
- **What is estimated:** each network's accuracy against the task rule on the task's own input distribution, and on
  four sub-populations of it fixed now, before any estimate exists (for example: the top two values within 1% of
  each other; the maximum in the last position). Sub-populations stay inside the domain the accounts claim, so a
  must-pass account cannot fail for being asked about inputs it never covered, and rare ones are where sampling is
  expensive.
- **Estimators:** each must-pass account is written once as an estimator (a procedure that returns an estimate and
  counts its own floating-point operations) by a session that has the account and the task, and nothing else: not
  the network, its weights, its accuracies or any score. Running the account itself as a cheap sampler is allowed;
  a short faithful program is a cheaper sampler than the network, and that is part of what an explanation buys.
  Copies and disguises compute the network, so their estimator is sampling them. This step measures the
  estimator-writer as well as the account; that is the known cost of the measure, stated in every report.
- **Scoring, in ARC's own form (EVIDENCE_LADDER rung Y, "averaging the squared error"):** each estimator's mean
  squared error on each quantity, over 100 repeated runs if it is randomised (its squared error if it is
  deterministic), divided by Monte Carlo's mean squared error on the network at the same compute. Below one forward
  pass of compute, the Monte Carlo reference is the task-rule estimate (the best guess with no samples).
- **The line comes from the copies:** every must-fail program is run as a sampler and scored the same way, which
  gives the ratio a copy reaches by chance. An estimator "beats sampling" on a quantity when its ratio is below the
  5th percentile of the copies' ratios on that quantity.
- **Degenerate quantities:** where Monte Carlo's mean squared error at the estimator's compute is below 1e-6 (a
  sub-population the network almost never gets wrong), the quantity cannot separate anything; it is reported and
  left out of the decision. At least three of the five must remain, or the sub-populations are redrawn by the same
  blind rule before any estimate exists.
- **"Passes"** means: it beats sampling on every quantity left in the decision. The quantities are nested, so no
  independence is assumed; the chance rate is read off the copies' own results.
- **Reported, not a line:** a small student distilled from each network to the same fidelity as its account, run as
  a sampler. A short readable rule and a small opaque student of equal cost tie on this test; if one does, the
  account's advantage is readability, not computation, and the reader arm decides it.

**Reader arm (cheap; Flash calls, so Shreyas's go).** The same question with a language model as the estimator,
which removes the estimator-writing step.
- The reader may not run code. Its token budget per question is set below what hand-executing, on one input, the
  smallest opaque program in the comparison takes (the distilled student; computed from its forward-pass cost before
  any call), so neither the copy nor the student can be run within it.
- Four conditions per network: given the account; given a copy; given the distilled student as code; given only the
  task (the floor).
- **"Passes"** means: given the account, its estimates of the five quantities beat the task-only floor, the copy
  condition and the student condition by more than the floor's spread over 5 repetitions.
- The copy condition is reported on its own. If it beats the floor, the reader is reading structure off raw weights
  without executing them, which is itself a finding for flagship 4.

**Runs.** One scoring pass over saved material; no network is trained. The reader arm is 2 deciding networks (plus
M2,2, reported) × 4 conditions × 5 repetitions of short calls.

**Cost.** None for the main arm beyond a session's time; a few dollars of Flash calls for the reader arm.

---

## X1. The stronger-reader test (the R3 question, finally asked), where a short account is known to exist

**Build step it serves.** Choosing the model for the reasoning roles of the box (NEXT_BUILD §3.4): theorist,
formaliser, critic, director.

**Question.** On targets where a short, verified account is known to exist and the cheap reader failed, does a
frontier reader recover the account where DeepSeek V4.1 Flash does not, at the same step budget?
- **Yes** (the frontier reader recovers month rollover, and every planted network X2 supplies; Flash none): frontier
  models go in the reasoning roles from the start. If X2 has not run, month rollover alone decides, with 3 runs each
  as below.
- **No, both fail**: the bottleneck is the method (one leap to a whole account), not the model, which supports the
  new centre (NEXT_BUILD §4).
- **Both pass**: the interface and feedback, not the model, were the missing piece; cheap models can hold reasoning
  roles where the feedback guides them.

**Why these targets.** The first version of X1 used M8,5 and lc_6_5 alone, and the next version wrongly called them
known negatives. They stalled for the cheap reader, but both were decomposed exactly by SPD and the atlas (DIARY: an
"EXACT, sparsely used decomposition of M8,5 … faithful above the task rule"; ATLAS_RESULTS: lc_6_5 passes on
development), so a faithful account exists, though not a short program. The 7% cap on rule replacement (U17) was
measured on M16,10-class networks, not on these two. So they are an open arm here, informative but not decisive.

**Targets.**
- Known account, cheap reader failed:
  - GPT-2 month rollover (hand-built account 5 of 5 stages; blinded cheap reader 1 of 5);
  - X2's planted networks, if X2 has run, with their keys sealed.
- Open arms, reported but not deciding:
  - GPT-2 countdown (blinded cheap reader 1 of 5, the same as its prior-only control; the hand-built account reached
    only 2 of 5, `docs/RESULTS_LM.md`, so there is no complete account to recover);
  - M8,5 and lc_6_5 (above). A short account here is checked for copying first (X0's test and T2's S).
- Harness check: M4,3, which both readers passed before.

**Measurement (frozen before any run).**
- LM targets: the LM judge at a commit fixed now. It is first re-run on the hand-built month account and be-verb
  (must pass) and on the prior-only controls (must fail). If the hand-built month account does not pass on it, its
  re-certification comes first (PROJECT_REVIEW §8), and X1 waits.
- Planted networks: X2's scorer against the sealed key.
- Open AlgZoo arm: the version-2 certificate (`docs/glass/INTENT.md`), with X0's test added if X0 passed.
- **"Recovers"** means every stage passes (month rollover) or every planted mechanism is recovered (X2's scorer),
  while the reader's prior-only control does not. On the open arms, the stage count is reported.

**Must-fail cases.**
- The same readers on shuffled weights of each target.
- The saved disguised copies from round 2, re-scored.
- If any of these passes, the run is void and the measure goes back to INTENT.

**Controls.**
- The prior-only control for each reader (`--told --no-weights --one-shot`).

**Runs.**
- 3 per reader per target.
- 3 against 0 is the smallest split a one-sided exact test separates at p ≈ 0.05.
- Step caps as in the original runs: 150 for the LM targets, 80 for the AlgZoo arm.

**Free first.** Re-score every saved run under the frozen judge. This also checks that no saved run already passes.

**Precondition and cost.** A frontier-model key: this workspace has only DeepSeek V4.1 Flash (PLAN_SCALING).
Frontier API calls need a key and Shreyas's go.

---

## X2. A first discovery score for today's tools

**Build step it serves.** The calibration ladder (NEXT_BUILD §6 and §9, phase 1): the yardstick every later claim of
discovery is measured with.

**Question.** Can a puzzle-maker produce small networks that implement algorithms published nowhere, which behave as
specified and are not trivially readable, and what share of them does today's toolkit, used by a Claude session,
recover blind?
- **Yes, they can be made** (each behaves as specified on ≥ 99% of held-out inputs and passes the readability check
  below): the ladder exists, and today's recovery rate is the baseline the box must beat.
- **No**: weight-space-programmed networks are too readable, so the ladder uses trained-imitation networks instead,
  as described below.

**The networks.** Twelve in all:
- **Six written by weight-space programming.**
  - A frontier agent writes the weights of a small network for an algorithm invented for this test.
  - Examples of the kind (the real ones are invented at build time and sealed): "1 if the number of increasing runs is
    prime"; "the index of the element closest to the running median after removing the current maximum".
  - Its design notes and the algorithm are sealed in a file whose hash is committed before any recovery attempt.
- **Six trained with a planted rule.**
  - Train a small RNN or MLP on data where an invented rule decides the label in one region of the input space.
  - The network must compute that rule somewhere.
  - The rule is sealed the same way.
- **The readability check, so that a ladder network does not give its answer away.** Weights written as sparse
  near-integers can be read off their sign pattern (`main:docs/literature/aether/arxiv-2611.21449.md`). For the
  written networks:
  - apply a random orthogonal change of basis to every hidden layer (function-preserving where the architecture
    allows);
  - or distil the written network into a freshly trained student and use the student.

**Measurement.** A network counts as **recovered** when both hold:
- the recovered program equals the sealed rule on ≥ 95% of held-out inputs;
- its variables pass interchange at the sites they are placed (the calibrated judge).

**Must-pass.** One Tracr program with a known answer, through the same pipeline.

**Must-fail.**
- Two shuffled copies of ladder networks.
- One memoriser trained on random labels: the pipeline must report "lookup".

**Runs.** One blind recovery attempt per network, with a fixed budget per network. The answer keys are opened only
after every attempt has been scored.

**Cost.** CPU only for the networks. The puzzle-maker and the recovery sessions are agent time.

---

## X3. Does a rebuilt fluid model have something to be cheap about?

**Build step it serves.** Flagship 2's target (NEXT_BUILD §6): without it, "how is it cheap" cannot be asked.

**Question.** Does a video model trained on coarse-grained frames of a fine simulation, with a long step, beat the
coarse solver at matched cost? In other words, does it learn a closure or a long-step integrator worth decoding?
- **Yes** (network error below the coarse solver's by more than the spread between training seeds): the learned
  closure and integrator are the discovery target.
- **No**: our rebuild does not reproduce the published kind of gain; fix the training, the resolution gap or the
  step, and only then consider another physics target.

**Setup.**
- **Simulator.** The existing one (`kepler/targets/fluid/sim.py`, which accepts any grid size n).
  - The fine simulation runs at n = 256.
  - Frames are coarse-grained to 64 × 64 by spectral truncation.
- **Model.** The existing video architecture (`kepler/targets/fluid/model.py`), trained to step 4× and 16× the coarse
  solver's stable time step.
- **References,** all scored against the coarse-grained fine truth on held-out flows:
  - (a) the coarse pseudo-spectral solver at 64 × 64 with no closure;
  - (b) the same solver with a Smagorinsky eddy-viscosity closure, a standard known closure, as the reference
    library's best;
  - (c) the network.

**Measurement.**
- One-step error, and error after 20 steps.
- Cost per simulated time unit, measured on the same machine.

**Known answer.** Kochkov et al. 2021 (arXiv 2102.01010) showed, on 2D turbulence, that learned corrections on
coarse grids match baseline solvers with 8–10× finer resolution in each dimension, with 40–80× speed-ups. So "Yes" is
the expected outcome of a correct rebuild, and the experiment is a known-answer rung. "No" means our rebuild is wrong
(training, resolution gap, step) before it means the physics target is wrong; fix that first. The discovery comes
after: reading the learned correction, which that work did not do.

**Must-fail control.** A network trained on 64 × 64 frames of a 64 × 64 simulation (Kepler's current setup). It is
scored against the fine truth the same way, and must show the coarse solver's own error. This check says that any
advantage of the new model comes from the coarse-graining.

**Runs.** 3 training seeds per step size. The pass line is the seed spread.

**Cost.** A few GPU hours (Colab), with Shreyas's go.

---

## X4. One concept-invention cycle on LD2's quiet-position residual, run by a strong agent

**Build step it serves.** The definition stage of the loop (NEXT_BUILD §3.3 C, §3.8). X4 and X6 run on the same
target with the same scorer, so together they decide which engine runs that stage.

**Why this target.** Channel 22's causal evidence is in quiet positions: resetting it removes 32% of the
quiet-position residual (kepler/DIARY K4). The record's numbers are all LD2's, a convolutional network; the
tactical residual and the transformer t1 come later.

**Step 0: how much of the residual is knowledge (cheap, first).** The residual is LD2's value minus the classical
surrogate (boosted trees fitted to LD2's outputs, `kepler/core/residual.py`). Any decent network's value predicts
outcomes beyond a fit to 61 concepts, so a yes/no test would be foregone; the number that matters is the share.
- Measure: the share of the residual's variance explained by the game's outcome (and by a deep engine's evaluation,
  if one is available) beyond the surrogate, with a 95% interval, on held-out positions. Reported three ways: all
  positions, quiet positions, and channel 22's component (the part of the residual channel 22 tracks).
- Control: the same with the residual shuffled across positions.
- The share, not a yes/no, sets the label: what X4 names counts toward level K in proportion to the knowledge share
  of the part it explains, and the rest is reported as explaining LD2's errors.

**Question.** Can a strong agent, working through the cycle of §3.8, produce a definition for channel 22 of Leela LD2
that passes Kepler's frozen criteria and an intervention check? The cycle is: observer, simulation test, formaliser,
rivals separated by designed positions. Kepler's 59 one-shot programs found none.
- **Yes**: concept invention works on a real network, and the residual curve gets its first point.
- **No**: locate the break (the description, the program, or the alignment) before scaling anything.

**Data.** Regenerate with the existing scripts (`scripts/kepler/chess_build.py`, `chess_channels.py`), into `KDATA`.
The per-position data is not in the repo.

**The definition language (fixed now, shared with X6).** Python over the board using `python-chess`; search up to
two moves deep allowed; at most 5 numeric constants in all, counting every numeric literal and table entry in the
program (a search over thousands of programs tunes hard-coded numbers by selection, and a literal piece-square table
is a fitted table); no fitted sub-models (no trees, no regressions inside a program); no access to the network, its
weights or activations; no outside engines or tables. Only the language excludes boosted trees and linear maps of
earlier channels, since either would beat the 5-coefficient surrogate below.

**Measurement.** Kepler's frozen four criteria, unchanged (`results/kepler/chess/calculation_features.json` shows
their form):
- held-out gain on LD2 with a 95% interval above zero;
- the same on the replication network 744706;
- decodability above the untrained copy;
- not a restatement of a library concept.

Two new criteria, fixed now:
- steering channel 22 along the program's value moves the network's evaluation in the predicted direction on held-out
  positions, at least as much as the median effect of the accepted capture-resolution feature;
- the program beats the best sparse surrogate with the same number of constants (a linear model over the 64-feature
  classical library with at most 5 non-zero coefficients, fitted to channel 22 on the fit set).

**Must-fail.**
- The FEN-hash noise program, as before.
- A program fitted to channel 22 values shuffled across positions.
- Inside the language, the loophole a search would find: a legal program carrying a literal table tuned on the fit set
  (it must be rejected by the constant count).
- Outside the language, scored anyway to show what the language excludes: a boosted-tree fit to channel 22 over board
  features, and a linear map from earlier channels.

**Runs.**
- One cycle, as long as the strong agent needs within a two-hour cap.
- Every candidate is logged and counted, so the comparison with X6 is by number of programs tried.

---

## X5. The Helix skeleton on the insides of a network

**Build step it serves.** The loop's skeleton (NEXT_BUILD §3.3).

**Question.** Does Helix Probe's predict-first loop, pointed at a network's internals, recover planted mechanisms?
In this setting the probes are interventions and the model predicts internal states. The comparison is the atlas
pipeline on the same networks.
- **Yes** (all planted mechanisms recovered, with their triggers, at a budget the atlas also needs): the skeleton
  becomes the loop.
- **No**: keep the roles, and design selection rules for internal claims.

**Targets.** The atlas's planted RNN (5 mechanisms per matrix) and planted transformer (copy-previous-token), whose
true mechanisms are known exactly.

**Adapter.** One new environment file in `envs/`:
- a probe is an intervention on a named site plus an input;
- an observation is the network's states and output.

**Measurement.**
- Planted mechanisms recovered: the trigger rule matches the planted one on held-out inputs at ≥ 0.98, the bar the
  atlas met.
- The number of experiments used.

**Must-fail.** The same loop on the shuffled planted network must recover no mechanism.

**Runs.** 3 seeds per target. DeepSeek Flash, as in Helix Probe.

**Cost.** Small: Flash calls and CPU.

---

## X6. Helix volume on the definition the one-shot scientist missed

**Build step it serves.** The definition stage (NEXT_BUILD §3.3 C, §3.11): whether the Helix volume engine runs it,
and whether the network's signal gets a cheap model past its prior.

**Question.** Does Helix's figure-eight on a Flash-grade model find a board program that matches channel 22 on
held-out positions and passes X4's criteria, where Kepler's 59 one-shot programs found none? The loops are an
exploring loop that varies programs and a reasoning loop that tests one change at a time, with a memory of findings
and one frozen scorer. Two arms:
- **A, score-only:** the proposer sees its programs and their scores.
- **B, evidence shown:** the proposer also sees, each round, the 20 fit-set positions where its best program errs
  most, and minimal pairs that differ mainly in channel 22.

**The null, stated first.** Helix's own research case predicts that selection by a score alone stays near what the
model already proposes (`helix_evolve/docs/research/RESEARCH_CASE.md`). So arm A is expected to fail where Kepler
failed, and only a signal on the proposal side (arm B) might not.

**Decisions.**
- **Yes in B only**: the network's evidence, shown to the proposer, is what gets past the prior; the engine runs the
  definition stage with it.
- **Yes in A and B**: volume plus the score suffices.
- **Yes at Helix scale only**: volume was missing. **Yes at both scales**: the loop structure, not volume, was missing.
- **No anywhere**: the bottleneck is the definition language or the variable itself, and the director takes that
  stage. Compare with X4 on the same scorer.

**The scorer (frozen before the run, plain code, no model).**
- **Language.** X4's definition language, exactly.
- **Fit and selection.** Correlation with channel 22's activity on positions from a fit set; selection by the same
  measure on a validation set.
- **Final verdict.** On a test set of positions from games never used, scored once at the end with X4's criteria,
  including "beats the best sparse surrogate with the same number of constants".
- **Target.** A single decomposition feature that carries the residual if one is available, because a raw channel may
  mix several features. Channel 22 itself is the fallback, and the report says which was used.
- **Loopholes closed in the scorer itself, as Helix learned to do:** no access to the network or its activations; no
  position hashing; exact, deterministic evaluation; a CPU-time cap.

**Must-fail (run through the scorer before the search starts).**
- The FEN-hash noise program.
- A run whose target is channel 22's values shuffled across positions. It sets the null distribution: a real pass
  must beat the 99th percentile of the shuffled-target runs.
- X4's tuned-literal-table program, which the constant count must reject.
- X4's out-of-language fits (boosted trees; a linear map of earlier channels), to confirm that the language rejects
  them; the surrogate line alone would not.

**Budgets.**
- 59 programs, matched to Kepler.
- 2,500 programs, about Helix run C's scale.

**Runs.** 3 seeds per arm at each budget, 12 runs, because Helix's seed-to-seed spread is large
(`helix_evolve/docs/LEAN_RESULTS.md`; runs C and D ended 0.007 apart).

**Cost.**
- Flash calls at Helix scale: about one VM day per long run, as for Helix Evolve's runs; six long runs.
- Shreyas's go is needed.

---

## X7. The lying gate, in the smallest model that shows it

**Build step it serves.** Flagship 3's first L2 object, the gate (NEXT_BUILD §6, flagship 3).

**Question.** Is the lying mode routed through a small set of attention heads that read the chat-template tokens?
- **Yes** (a set of at most 10 heads carries more than half of the mode's effect on the lie rate): decompose and
  define the gate.
- **No**: the mode is spread out, as the sprint's direction result hinted (adding the mode's mean direction moved the
  detector by only 1–5% of the gap, liar/DISCOVERY); move to a decomposition of the template-token stream
  (transcoders).

**Model.** The smallest of the liar sprint's models that shows the mode under the sprint's failed-lie test; a
smaller open model if one passes the same test.

**Measurement (frozen).** The model's lie rate and the detector's score on the sprint's prompts, under path patching
from the template positions, one head at a time, then greedy sets. Every effect is predicted in writing before its
run.

**Controls.** A truthful-instruction prompt set (the heads must not change truthful answers); a fiction or role-play
set (side effects predicted in advance); random head sets of the same size (a found set must beat the 99th
percentile of 100 random sets).

**Cost.** GPU hours for a 7–8B model, or less for a smaller one; Shreyas's go.

---

## X8. Which open model reads weights above its controls?

**Build step it serves.** Flagship 4's subject (NEXT_BUILD §6, flagship 4).

**Question.** Does any open model, whose insides the toolkit can decompose, read small networks' weights above a
shuffled-weights control?
- **Yes**: it is flagship 4's subject (Q8).
- **No**: flagship 4 starts by training an open reader with verified rewards (NEXT_BUILD §3.6), then studies it.

**Protocol.** The shapes tasks of the replication recorded in `docs/literature/wsp/K1_STATE_OF_ART.md` (with its
shuffled control), and Singh's experiment-1 edit-prediction test (Spearman against the no-weights baseline). Code
sandbox mode and large output budgets, as K1 advises.

**Models.** Open models chosen before any run, plus DeepSeek V4.1 Flash as the known-weak reference (0 of 8 on shapes).

**Pass line.** Above the shuffled-weights control by more than that control's spread over 3 repetitions, and a
Spearman gain over no weights with a 95% interval above zero.

**Free first.** Re-read the replication's saved transcripts for which open models came closest and why (K1: DeepSeek
spends its output budget copying weight rows).

**Cost.** API calls or GPU time; Shreyas's go.

---

## How they fit together

| Experiment | Decides | Feeds |
|---|---|---|
| X0 | Whether answering without running separates explanations from copies | §3.2 item 3; every must-fail set |
| X1 | Which model holds the reasoning roles | §3.4 |
| X2 | Whether the calibration ladder can be built, and today's discovery score | Every later claim |
| X3 | Whether our rebuilt fluid target reproduces the known gain | Flagship 2 |
| X4 + X6 | Which engine runs the definition stage, and whether the network's evidence gets a cheap model past its prior | §3.3 C, §3.11 |
| X5 | Whether Helix Probe's loop is the box's skeleton | §3.3 |
| X7 | Whether the lying mode has a localisable gate | Flagship 3 |
| X8 | Whether flagship 4 has an open reader to study | Flagship 4 |

Recommended order, because each answer shapes the next:
1. X0: free, and the anti-copy test everything else uses.
2. X2 and X5: cheap, and they build the yardstick and test the skeleton.
3. X1: it decides who reasons (needs a frontier key).
4. X4 and X6 together, X4's step 0 first; X7 and X8 alongside.
5. X3: it opens flagship 2.
