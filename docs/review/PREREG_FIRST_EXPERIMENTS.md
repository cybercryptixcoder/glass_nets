# The first six experiments of the next build: pre-registrations

*Written Nov 26, 2026, in the second review run, from `docs/review/NEXT_BUILD.md` §10. Nothing here has been run.
Each entry follows the repository's build routine (`CLAUDE.md`): the build step it serves, the question with a yes/no
answer and the decision each answer leads to, the measurement (built first and proven on known cases), the cases
that must pass and must fail, the controls, how many runs, and where the pass lines come from. Pass lines come from
controls, never from the results they judge. Each experiment needs Shreyas's go before any spend.*

*Shared conventions:*
- *"Held out" means inputs (positions, flows, prompts, networks) never used to design the method or fit the
  candidate.*
- *Every run writes its full record (programs, calls, scores) to `results/next/<experiment>/` and is pushed as it
  lands.*
- *Free experiments come first in every entry: what can be re-scored or re-read from saved runs is done before
  anything new is paid for.*

---

## X1. The stronger-reader test (the R3 question, finally asked)

**Build step it serves.** Choosing the model for the reasoning roles of the box (NEXT_BUILD §3.4): theorist,
formaliser, critic, director.

**Question.** On dense networks where the cheap reader stalled, does a frontier reader produce an account that passes
the version-2 certificate, where DeepSeek V4.1 Flash does not, at the same step budget?
- **Yes** (frontier passes on at least one target, Flash passes on none): frontier models go in the reasoning roles
  from the start.
- **No, both fail**: the bottleneck is the method (one leap to a whole account), not the model, which supports the
  new centre (NEXT_BUILD §4).
- **Both pass**: the version-2 feedback, not the model, was the missing piece; cheap models can hold reasoning roles
  where the certificate guides them.

**Targets.**
- M8,5 (AlgZoo, 112 parameters) and lc_6_5 (longest cycle, 378 parameters): both stalled in rounds 1–2.
- M4,3 as the must-pass harness check: both readers passed it before.

**Measurement (frozen before any run).**
- The version-2 certificate as it stands (`docs/glass/INTENT.md`):
  - fidelity with a 95% interval against the task rule;
  - information S against the copy frontier (the T2 form, which placed every saved copy at S ≥ 0.91);
  - inside claims at intermediate sites, judged by the calibrated interchange tests.
- **"Passes"** means all three of these:
  - fidelity above the task rule by more than two standard errors;
  - S below the line set by the shuffled-weights runs (the lowest S any shuffled run reaches, minus its spread);
  - at least one intermediate-site claim that passes interchange.

**Must-fail cases.**
- The same readers on shuffled weights of each target.
- The saved disguised copies from round 2, re-scored.
- If any of these passes, the run is void and the measure goes back to INTENT.

**Controls.**
- The prior-only control for each reader (`--told --no-weights --one-shot`).

**Runs.**
- 3 per reader per target, for 12 runs in all.
- 3 against 0 is the smallest split a one-sided exact test separates at p ≈ 0.05.
- Step cap: 80, as in round 2.

**Free first.** Re-score every saved round-2 run under the frozen version-2 certificate. This also checks that no
saved run already passes.

**Cost.** Frontier API calls need a key and Shreyas's go.

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
- **No**: widen the resolution gap or the step, or choose another physics target.

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

**Must-fail control.** A network trained on 64 × 64 frames of a 64 × 64 simulation (Kepler's current setup). It is
scored against the fine truth the same way, and must show the coarse solver's own error. This check says that any
advantage of the new model comes from the coarse-graining.

**Runs.** 3 training seeds per step size. The pass line is the seed spread.

**Cost.** A few GPU hours (Colab), with Shreyas's go.

---

## X4. One concept-invention cycle on the chess residual, run by a strong agent

**Build step it serves.** The definition stage of the loop (NEXT_BUILD §3.3 C, §3.8). X4 and X6 run on the same
target with the same scorer, so together they decide which engine runs that stage.

**Question.** Can a strong agent, working through the cycle of §3.8, produce a definition for channel 22 of Leela LD2
that passes Kepler's frozen criteria and an intervention check? The cycle is: observer, simulation test, formaliser,
rivals separated by designed positions. Kepler's 59 one-shot programs found none.
- **Yes**: concept invention works on a real network, and the residual curve gets its first point.
- **No**: locate the break (the description, the program, or the alignment) before scaling anything.

**Data.** Regenerate with the existing scripts (`scripts/kepler/chess_build.py`, `chess_channels.py`), into `KDATA`.
The per-position data is not in the repo.

**Measurement.** Kepler's frozen four criteria, unchanged (`results/kepler/chess/calculation_features.json` shows
their form):
- held-out gain on LD2 with a 95% interval above zero;
- the same on the replication network 744706;
- decodability above the untrained copy;
- not a restatement of a library concept.

One new criterion, fixed now:
- steering channel 22 along the program's value moves the network's evaluation in the predicted direction on held-out
  positions, at least as much as the median effect of the accepted capture-resolution feature.

**Must-fail.**
- The FEN-hash noise program, as before.
- A program fitted to channel 22 values shuffled across positions.

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

**Build step it serves.** The definition stage (NEXT_BUILD §3.3 C, §3.11): whether the Helix volume engine runs it.

**Question.** Does Helix's figure-eight on a Flash-grade model find a board program that matches channel 22 on
held-out positions and passes Kepler's frozen criteria, where Kepler's 59 one-shot programs found none? The loops are
an exploring loop that varies programs and a reasoning loop that tests one change at a time, with a memory of
findings and one exact scorer.
- **Yes at Helix scale, no at Kepler's scale**: volume is what was missing, and the engine runs the definition stage.
- **Yes at both**: the loop structure, not the volume, was missing.
- **No at both**: the bottleneck is the library's base language or the variable itself, and the director takes that
  stage. Compare with X4 on the same scorer.

**The scorer (frozen before the run, plain code, no model).**
- **Fit and selection.** Correlation with channel 22's activity on positions from a fit set; selection by the same
  measure on a validation set.
- **Final verdict.** On a test set of positions from games never used, scored once at the end with Kepler's four
  criteria plus X4's steering criterion.
- **Programs.** Python using `python-chess`, with a CPU-time cap. Search up to two moves deep is allowed, so
  calculation can be expressed.
- **Loopholes closed in the scorer itself, as Helix learned to do:**
  - no access to the network or its activations;
  - no position hashing;
  - exact, deterministic evaluation.

**Must-fail.**
- The FEN-hash noise program.
- A run whose target is channel 22's values shuffled across positions. It sets the null distribution: a real pass
  must beat the 99th percentile of the shuffled-target runs.

**Budgets.**
- 59 programs, matched to Kepler.
- 2,500 programs, about Helix run C's scale.

**Runs.** 3 seeds at each budget, because Helix's seed-to-seed spread is large (`helix_evolve/docs/LEAN_RESULTS.md`).

**Cost.**
- Flash calls at Helix scale: about one VM day per long run, as for Helix Evolve's runs.
- Shreyas's go is needed.

---

## How the six fit together

| Experiment | Decides | Feeds |
|---|---|---|
| X1 | Which model holds the reasoning roles | §3.4 |
| X2 | Whether the calibration ladder can be built, and today's discovery score | Every later claim |
| X3 | Whether flagship 2 has a target | Flagship 2 |
| X4 + X6 | Which engine runs the definition stage: a strong agent's cycle, or Helix volume | §3.3 C, §3.11 |
| X5 | Whether Helix Probe's loop is the box's skeleton | §3.3 |

Recommended order, because each answer shapes the next:
1. X2 and X5: cheap, and they build the yardstick and test the skeleton.
2. X1: it decides who reasons.
3. X4 and X6 together: they decide the definition engine.
4. X3: it opens flagship 2.
