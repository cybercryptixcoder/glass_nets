# Independent critique of `docs/review/NEXT_BUILD.md` (Glass Networks)

*Reviewer: a fresh agent that did not see the document being written. Read-only. Every objection below was checked against the source files named. Version reviewed: branch `review-next-build` at commit `1c91c590` (includes the new §3.11–3.14 and the model-grade column in §3.4). Evidence repo: `cybercryptixcoder/glass_nets` branch `liar` (paths below are relative to it unless marked `helix_evolve/`).*

## 1. Verdict in five lines

1. The diagnosis (§1b, §2) is mostly right and well evidenced. The record does show hand-built, vocabulary-bound, decision-level instruments, and a reader that recalls or copies.
2. The record table and the lessons table are broadly accurate, but they report successes without their failed replications and caveats (mod-mul, the decompiler, month rollover, chess "level K", Helix Evolve's headline run).
3. The design is coherent as a research programme. Its weakest load-bearing part is the answer to the project's central unsolved problem, telling an explanation from a disguised copy.
4. **Single biggest weakness:** "compression against a shared library" is presented as a theory that defeats copies "by construction". The repo's own pre-registered test of that idea concluded "explanation = fewer bits than a copy is wrong as a general definition", and it rejected every real explanation. The volume-engine verdict in §3.11 then inherits the same hole, because its "exact, loophole-free" scorer is a truth test that a copy passes.
5. Several first experiments (X1, X4/X6, X3) cannot cleanly separate their outcomes as designed. Each can be fixed cheaply.

## 2. Objections, most severe first

### O1. Major (near-fatal for the stated claim): shared-library compression does not defeat disguised copies, and the record already says so

- **Claim attacked.** §2 Gap 2: "A copy cannot be short in such a vocabulary ... a disguised weight table is exactly such a primitive." Also: "Both ideas meet, by construction, the three conditions ... a copy in any encoding fails them". And §3.2 item 3: "an explanation is a point that dominates both [the copy and the task rule]".
- **Evidence.**
  - `docs/glass/DIARY.md` lines 412–431 (the 11:35 result of the pre-registered information-measure test): "M4,3's real account isn't shorter than its 32 weights. It reorganises them (the symmetry) rather than discarding them." It concludes: "'explanation = fewer bits than a copy' is wrong as a general definition ... a copy is bad not because it is long but because it tells you nothing without being run."
  - `docs/glass/info_measure_T2.txt`: the revised T2 measure fails **all three** must-pass cases. M2,2 scores 0.68, M4,3 scores 1.44 and the census rule scores 0.50 (each marked "NO"), and the file ends "PRE-REGISTERED CRITERION MET: False".
  - Gap 2 reports only the half that favours the idea ("placed every one of 30+ saved copies ... at S ≥ 0.91").
- **Internal contradiction.** §3.2 amortises the library "over every network **and input** it explains", and Gap 2 says "only structure that recurs (across inputs ...) gets cheap". A weight table is reused on every input, so under that wording a copy is amortised toward zero cost.
  - If amortisation runs across networks only, the doc's own L1 foundation becomes copy-priced: transcoders and SPD are "trained once per model" (§3.7), so they are per-network primitives that pay their full length.
  - The flagships have one network each, plus two replication nets for chess.
- **Second hole.** Generic compression primitives (low rank, quantisation, pruning) recur across every network, so an MDL abstractor (§3.3 E) would make them cheapest of all.
  - The copy frontier is defined as "weights rounded to 1, 2, … bits" (`docs/glass/EVIDENCE_LADDER.md`, rung X). That is a weak compressor, and a pruned or low-rank copy sits above it.
  - The units program cites Arora et al. 2018: noise-stable networks are compressible (`docs/glass/UNITS_FINDINGS.md`, F3 prior work). Those are exactly the tolerant networks the box can explain.
- **Why it matters.** This is the project's central unsolved problem (`docs/PROJECT_REVIEW.md` §6). The doc says it is solved "with a theory" when the repo's best evidence says that theory, in its simplest form, rejects true explanations.
- **Fix.**
  - Demote measure 3 to a reported signal, as the repo already did.
  - Adopt the framing the 11:35 entry reached: a copy "only answers questions by sampling". In practice that means ARC's matching-sampling yardstick (EVIDENCE_LADDER rung Y, not among the six measures), proof compression where the domain allows it, and prediction without running the network, against mechanism-free baselines (see O9).
  - Run condition (c) of `docs/glass/REPORT_ROUND2.md` (corrections §3) on the saved set, **including the must-pass M2,2 and M4,3**, before calling anything a theory.
  - Fix the "by construction" sentence: Gap 2 claims the conditions are met by construction, then says (c) "still has to be shown".

### O2. Major: proof compression cannot be computed on any flagship, so the flagships have no working anti-copy measure

- **Claim attacked.** Gap 2 and §3.2 item 4: proof compression "cannot be talked around".
- **Evidence.**
  - `docs/glass/EVIDENCE_LADDER.md`, rung 5: proof is "Not built; out of reach above tens of parameters (Gross et al.)".
  - `main:docs/literature/aether/arxiv-2410.07476.md` (Wu et al.):
    - the verifier is about 3× faster than brute force (0.75 s vs 2.20 s) and non-vacuous on about 55% of models;
    - "Loss bounds do not work";
    - "a vacuous bound is not strong negative evidence against an explanation".
  - "Faster than brute force" needs an enumerable domain. Chess positions, continuous fields and text are not enumerable.
  - The doc's own §6 table gives every flagship only "Probabilistic" guarantees.
- **Why it matters.** On the flagships, the anti-copy burden falls entirely on measure 3 (O1).
- **Fix.** Scope measure 4 to the calibration ladder in §3.2, and say what replaces it on the flagships. Matching sampling, and prediction without running the network, are the candidates.

### O3. Major: §3.11's verdict that definition synthesis has an "exact, loophole-free" score is wrong as written

- **Claim attacked.** §3.11 table: "Definition synthesis ... Exact, loophole-free score? Yes ... **This is the best fit in the whole box**". Also REVIEW_LOG F39: "meets all four conditions".
- **Evidence and argument.**
  - **The score is a truth test, which a copy passes perfectly.** Held-out agreement and intervention agreement are exactly the tests that `docs/PROJECT_REVIEW.md` §6 says a copy passes: "A copy of the network is perfectly true, so it passes every test of (a)."
    - A program that embeds fitted coefficients (a regression or boosted trees over square features) predicts a channel on held-out positions.
    - So does a program that re-implements the earlier layers.
    - The hash-noise control only catches memorisation of inputs, not fitted copies of the computation.
  - **The new rule widens the hole.** §3.11 now says "Definitions must be allowed in terms of variables already aligned inside the network" (line 979). A definition of channel 22 as a fitted map of earlier aligned channels is the network's own computation.
  - **The only anti-copy term is the unproven one.** That term is "length in library terms", which is measure 3 (O1). Two rows lower, the same table says the shared-library and proof measures are "still unproven as gates".
  - **The Helix analogy breaks.** Helix's programs are large optimisers whose readability is irrelevant: the record program `helix_evolve/results/records/v102_v102-helix_p2563.py` is 516 lines built on scipy L-BFGS-B. In definition synthesis, readability is the whole point.
  - **The score is not exact.** AC2's score is an exact mathematical quantity. A channel-agreement score is a sample estimate, so thousands of candidates overfit it. The new "selection overfitting" bullet concedes this.
  - **Freezing the scorer was already tried and beaten.** Glass-2 was frozen and still "passed the pre-registered must-fail control (shuffled lc_3_4, 0.81)" (`docs/glass/REPORT_ROUND2.md`, corrections §3). The new §3.12 claim ("safe exactly when the score is exact and closed to loopholes ... frozen by the director ... must-fail controls built in") restates the round-2 design.
- **Fix.**
  - Restrict the definition language: a stated budget of free numeric constants, no access to weights, no fitted sub-models.
  - Before the scorer exists, write must-fail programs that a volume search will find: a boosted-tree fit, a linear map of earlier channels, a pruned copy of earlier layers.
  - Make X6's pass line "beats a fitted surrogate of equal description budget", not just agreement.

### O4. Major: the "best home for Helix" argument inverts Helix's own research case, and the Helix evidence is reported without its caveats

- **Claim attacked.** §0: "here the network's own variables are an exact signal from outside that prior". §3.11: interpretability supplies what Helix's research case "says is missing".
- **Evidence.**
  - `helix_evolve/docs/research/RESEARCH_CASE.md` classifies scoring as "Re-weight | Selection", with reach "Stays near what the model has already written".
    - Its evidence: "19 of 19 fresh restarts peaked at or below 0.8825, the textbook construction", with an exact scorer present throughout.
    - The "signal from outside the prior" it cites is distillation from another model. The same row says reinforcement learning on verifiable rewards "does not reach beyond what the base model can already sample". A scorer is that kind of reward.
  - Kepler already put network-derived signal on the proposal side, with minimal pairs chosen by channel 22, and got 0 of 20 (`docs/kepler/DIARY.md`, K9).
  - On fairness of the Helix record (`helix_evolve/docs/RECORDS.md`, "What this does not control for"):
    - run C had "a summary of methods published" as input;
    - run C is "One of two seeds"; the second (run D, 0.941244) did not beat the human 0.941362;
    - without outside input, run B beat it by about 2e-6 (0.941364), after about 4,500 programs across warm-started runs (`helix_evolve/README.md`).
  - The doc's "Helix Evolve beat the best human construction ... that way" (§0) omits all three.
- **The four-condition "rule" is post hoc.** "Strong exactly when four things hold, and weak when any one fails" is contradicted by the doc's own evidence table. Helix Probe on FIND succeeded with at most 25 experiments per function, not thousands (`docs/handoff/03_ANALYSIS.md` §1).
- **Fix.**
  - Report the caveats.
  - Change "exactly when" to "in the cases we have".
  - Write Helix's own prediction (selection alone re-weights the prior) into X6 as the stated null.

### O5. Major: X1 cannot tell "model" from "method", because its targets have no short faithful account

- **Claim attacked.** §10 X1: run a frontier model on "the same dense targets that stalled (M8,5, lc_6_5)". If it fails, "the bottleneck is the method, not the model".
- **Evidence.**
  - `docs/glass/UNITS_FINDINGS.md`:
    - AlgZoo ε95 of about 0.001–0.016;
    - "Rules can replace at most 7% of a network before it falls below the task rule (U17)";
    - "AlgZoo above 32 parameters offers neither [exact structure nor tolerance]: no exact structure in 7 searches".
  - The R6 hand runs on lc_3_4 and lc_8_6 found no crisp code (`docs/glass/DIARY.md` 12:13, 12:26).
  - The doc's own §8 says "The box does not force a short program where the network has none".
  - Access: `docs/glass/PLAN_SCALING.md` says "Only DeepSeek V4.1 Flash is available on Luminal; there is no key for any other model in this workspace."
- **Why it matters.** Failure is predicted by the target, not by the method or the model. X1 cannot produce its "fails" decision honestly.
- **Fix.** Run X1 where a short account is known to exist and the cheap reader failed:
  - month rollover (reader 1 of 5 stages, hand-built 5 of 5);
  - countdown (reader 1 of 5, equal to prior-only);
  - X2's planted networks.
  - Settle model access first.

### O6. Major: the weights-versus-inference number is not like-for-like, and assumption 3 bundles two claims

- **Claim attacked.** §0 item 3: "weights alone predict edit effects at Spearman 0.17 against 0.97 from the hidden state". The same pairing appears in §1b and flagship 4.
- **Evidence.** `docs/literature/wsp/blog-csinva-wsp-autoresearch-systematic-tests.md` (lines 18, 27, 39):
  - 0.17 vs 0.11 is experiment 1: 300 edits, Claude Opus 5.5 reading weight text with no code and no forward passes.
  - 0.97 is an oracle in experiment 3: 48 special edits "whose effect is an exact formula given one unknown (the normalised final hidden state)". Experiment 3's oracle overall score is listed as 0.84.
  - It is a result about an LLM doing hand arithmetic. It says nothing against the weight-based instruments the doc keeps (SPD, the decompiler's embedding geometry, compact proofs).
- **The bundling.** Assumption 3 mixes:
  - (a) data source: activations versus weights;
  - (b) object: the decision versus the computation.
  - The record supports (a). The doc's exemplar for (b), month rollover, is itself a decision-level "is Q at S and used?" result. Q is the calendar position from a behaviour-model ladder, S is depth 8, and the judge compares decisions. Its comparison step is unexplained: "Not yet known. Which components compute the comparison itself after depth 8" (`lab/constructions/gpt2_month_rollover/DISCOVERY.md`).
- **Fix.**
  - Drop the 0.17/0.97 pairing, or quote each number with its own setting.
  - Restate assumption 3 as the decision-versus-computation claim.
  - Present month rollover as a partial trace with a black-box step.

### O7. Major: the cross-network library premise is supported by a cherry-picked result

- **Claim attacked.** §2 Gap 12, "Evidence that the lever exists": comparators "in 5 of 5 training runs at 8 neurons". The same premise underlies measure 3, §3.6 and §3.10 ("the public product becomes that library").
- **Evidence omitted.**
  - `docs/glass/UNITS_FINDINGS.md`:
    - "Units do not converge across training runs at 16 neurons (3–4 of 5)";
    - the pre-registered bar of recurring at all three sizes was not met;
    - the program's one-line conclusion is "approximate universality, exact individuality", and "In hypersensitive networks the individual details carry the decisions".
  - `docs/glass/ATLAS_RESULTS.md`, "Recurrence across training runs: not confirmed": 1 of 4 seeds beat chance against a bar of 3.
- **Fix.**
  - Report the negatives.
  - Make Q3 (does the library compress across networks?) an early gating experiment, before §3.10's product claim and before measure 3 depends on amortisation.

### O8. Major: the chess flagship's key premises are untested, and §1 overstates chess at "level K"

- **Evidence.**
  - **Architecture mismatch.** The 78% / 22% residual and channels 22 and 15 come from **LD2, a 128×10 SE ResNet** (`docs/kepler/FINDINGS.md` §4). Kepler calls it "a weak network's value residual" (`docs/kepler/DIARY.md`, K9). It solves 65% of puzzles. The flagship is t1, a transformer, and the look-ahead hypothesis comes from Jenner et al. on "a transformer treating squares as tokens" (`docs/kepler/RESEARCH.md` line 25).
  - **Wrong positions.** Channel 22's causal evidence is in **quiet** positions ("removes 32% of the residual in quiet positions"). §3.8 and X4 aim at the tactical residual using channel 22.
  - **Knowledge or error?** Nothing tests whether the residual is knowledge rather than the weak network's own errors. No ground-truth check exists in `results/kepler/chess/`. The surrogate is boosted trees on network outputs (`kepler/core/residual.py`). The residual curve (Q9, "the box's headline number") would bend just as well by naming LD2's systematic mistakes.
  - **Human transfer failed.** Kepler's test came back null for every rule, including the queen trade (−0.02 [−0.10, 0.06]), and the rules "are not a switch inside the network" (`docs/kepler/FINDINGS.md` §5).
  - §1 says "Level K was reached ... for chess behaviour rules", yet the table defines K as "knowledge a person can learn".
- **Fix.**
  - Add a cheap experiment: does LD2's (or t1's) residual over the classical surrogate predict deep-engine evaluation or game outcomes beyond the surrogate? If not, the residual is not "what the network knows that we do not".
  - Measure the 78% baseline on t1 before using it as the flagship reference.
  - Rate chess as B with a candidate K.

### O9. Major: "control" and "transfer" have no mechanism-free baseline, and the record shows one wins

- **Claim attacked.** §3.2 item 6: predict every effect of an edit; "Understanding that cannot be used to edit precisely is incomplete."
- **Evidence.** `docs/EXPERIMENT_LOG.md` E7. On held-out interventions, attribution patching (no explanation) beat the construction: 90.8% vs 82.3% overall, and 57.0% vs 13.9% when the decision changed. "It beats attribution patching only where the path to the output is nonlinear and the intervention is on its variables."
- **Fix.** Every edit-prediction and transfer score should be reported against the baselines that understand nothing:
  - attribution patching or linearisation;
  - "no change";
  - a surrogate fitted to behaviour.

### O10. Minor to major: record-table and lessons-table entries stated without their failures

Each item below was checked against its source.

- **Mod-mul "complete".** This was 1 of 2 completed blinded runs. The rerun reached 3 of 4 stages (`lab/constructions/INDEX.md`). Be-verb is a single run.
- **Countdown "stalled".** The blinded reader's 1 of 5 equals the prior-only control's 1 of 5. It is no better than memory.
- **Decompiler.** §1 lists the passes and omits the failures in `docs/decomp/RESULTS.md`:
  - blind non-abelian groups: 2 of 5 at v3.5, 0 of 5 at the first pre-registered v3.3;
  - challenge v3: 3 of 5;
  - factor mode: 2 of 8 strict, then 4 of 8.
- **Who found nested discrete logs.** The session found the mechanism with the tools, and "Automation: the decompiler should find this ... without being told" is still open (`docs/decomp/POW_MECHANISM.md`). §3.5's "It found nested discrete logarithms" overstates this.
- **§0 versus Gap 5.** §0's "Recover structure blind, without being told what to look for" contradicts Gap 5, which says the decompiler explains only in "circles, groups, and three closed forms".
- **Month rollover.** It was "certified under rules set that same morning; ... re-certify it on a judge fixed in advance" (`docs/PROJECT_REVIEW.md` §8). No record of re-certification was found. Near-ties (32% of decisions) are excluded. The doc calls it the one result that "held up" with none of this.
- **Helix Probe.** §3.3 says it "earned its place", and §4 says "Predict-first ... keep-if-better rules made a weak model strong". The source says the gap is "mostly behavioural" and lists "whether the FIND claim is about the loop's structure or about persistence" as undecided (`docs/handoff/03_ANALYSIS.md` §0 and §6, item 3). §3.11's own row is accurate, so the document contradicts itself.
- **§1b lessons "the next build should treat as known".** Several are thin:
  - the memoriser test passed on a pre-registered count of one mechanism ("passes, but thinly", ATLAS_RESULTS A41);
  - "models are not organised by our behaviour labels" rests on two whole-layer ablations that the source lists under "Not verified" (`docs/JOURNAL.md` E3 and E8; `docs/GOAL2_CALIBRATION_RESULT.md`), from a run whose only verified finds were word-table tricks.
  - Relabel these as candidate lessons.
- **E12 (game naming).** The "specific question itself supplied the grid layout and the fact that outputs are actions" (`docs/EXPERIMENT_LOG.md` E12).

### O11. Minor: flagship 4 may lack a subject to study

- **Claim attacked.** Flagship 4: "a mechanistic account, inside an open reader model".
- **Evidence.** In `docs/literature/wsp/K1_STATE_OF_ART.md`, open models mostly fail to read: DeepSeek 0 of 8, Qwen 0 of 8, GLM 2 of 8, Kimi 4 of 8. The readers that work (Opus 5.5, Astra) are closed.
- **Fix.** First measure which open model reads above controls. If none does, reorder the flagship as "train a reader, then open it".

### O12. Minor: the fluid target and X3

- **The doc understates its own point.** It says the network has "little to be cheap about". In fact FluidNet is much more expensive than its solver.
  - FluidNet: width 32, six 3×3 blocks on 64×64 (`kepler/targets/fluid/model.py`), about 0.25 billion multiply-adds per frame.
  - The solver: four RK4 pseudo-spectral steps on 64×64, roughly 80 small FFTs, on the order of 10 million operations.
- **X3's question has a published answer for this flow class.** [Kochkov et al. 2021](https://arxiv.org/abs/2102.01010) report learned 2D-turbulence solvers "as accurate as baseline solvers with 8–10x finer resolution" with "40–80x" speedups.
- **Fix.** Frame the X3 setup as a known-answer rung (reproduce that cheapness), and keep the discovery target as reading the learned correction. That is the open seat the new §6 literature table already names.

### O13. Minor: two of the four named flagships get no first experiment

Lying and weight-reading are deferred to phase 6, behind an unscoped decomposition build at 7–8B parameters. Every X1–X6 targets AlgZoo, puzzle networks, fluid or chess. Add one small, decision-bearing LLM experiment. The liar sprint's mechanism step that never ran, now posed as routing and gating, is the obvious candidate.

## 3. Claims checked and found correct

- **Kepler fluid.** 5 of 38 terms selected; coefficients within about 2%; advection fades at wavenumbers 9, 11 and 13; Biot–Savart correlation 0.974; velocity swap 0.86 against 0.28.
- **Fluid build.** 64×64 grid, dt 0.0125, FRAME_DT 0.05, so 4 substeps per frame.
- **Chess numbers.** 78% / 81% / 36%; plateau at 0.79–0.80; channel 22 removes 32% of the residual in quiet positions; 59 scientist programs, 0 accepted; the last round at low effort with a brevity instruction after the token-budget failure; calculation features add 1.2 points.
- **Atlas.** 5 of 6 held-out RNNs; A35 fail, A37 pass, A40 fail, A42 and A44 pass; the query/key gap is an open limit.
- **Liar.** Lying mode carried by the template tokens; within-model real-lie trace 0.86–0.88; single-direction steering closes at most about 12%.
- **Goals.** Goal 1: 2 of 12 removals. Goal 2: failed 4 of 5 criteria and gamed its score.
- **Judge churn.** 22 judge commits (PROJECT_REVIEW §7).
- **Copies under T2.** Every saved copy and disguise scores S ≥ 0.91, as stated (but see O1 for what is left out).
- **Helix Evolve.** Run C's figures; mutations set 75–98% of gains (AC1); notes did not help on average; the floating-point boundary exploit. Kepler's 59 programs are about one fortieth of 2,564.
- **Helix Probe.** 0.82–0.90 against 0.38; +0.10 at matched inputs; ARC-AGI-3 0 of 35; the theorist showed no measurable value.
- **Units.** ε95 for AlgZoo is about 1% against 0.79 for the modular nets; 5 of 5 seeds at 8 neurons.
- **Singh.** Each of 0.17/0.11 and 0.97 exists in the source. Only their pairing is wrong (O6).

## 4. What is missing that would change the plan

1. **A copy-resistant yardstick for continuous or huge domains.** ARC's matching sampling is in the repo (EVIDENCE_LADDER rung Y; PROJECT_REVIEW §10 option b) but is absent from the six measures. Without it, or something like it, the flagships have no anti-copy test (O1, O2).
2. **A test that a residual is knowledge, not error,** before the residual curve becomes the headline metric (O8).
3. **The fitted-copy must-fail set for every volume scorer,** written before the scorer exists (O3).
4. **Model access and a compute budget as explicit preconditions.** The strong roles in §3.4 need access the record says this workspace lacked. Transcoders at Leela or 7–8B scale need GPUs, and SPD already took hours at 227k parameters. CLAUDE.md requires asking before spending.
5. **An ordering change.** Run Q3 (cross-network compression) and the fitted-copy test early. Measure 3, the library-as-product claim (§3.10) and the §3.13 flywheel all depend on them, and the record leans negative on the first.
