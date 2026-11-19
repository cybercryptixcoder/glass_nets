# The units program: basic units and their composition (started Nov 18, 2026, 19:10 ET)

*Mandate (Shreyas, 19:06):*
- Plan and run micro experiments continuously, in minutes not hours. Each one is chosen live from the previous
  result, and each one is there to learn something specific that a build needs.
- In parallel, build the next implementation FROM what the experiments show, never from a guess.
- Test every component end to end, then run on real models and look for discoveries of two kinds: (a) a network
  never interpreted before, decomposed and certified; (b) something new about how networks work in general (how
  units behave).
- Document every thought, decision, experiment and result.
- Keep going via scheduled check-ins. Send Shreyas updates; don't wait for approval.
- The routine in `CLAUDE.md` applies throughout.

## 1. The concept (Shreyas's hypothesis, Nov 18)

**Premises.**
1. Every network that maps inputs to outputs implements some mechanism (an algorithm); the weights are just a hard
   representation of it.
2. Every algorithm decomposes into simpler parts, down to **basic units**: fundamental operations.

**Hypothesis.**
- Networks build their behaviour by **composing** basic units: units form clusters, clusters act as tools, tools
  combine into more complex behaviour.
- If we can identify the units and how they're wired, we can interpret the network as "these units, wired like
  this", which sits between one rule for everything and a copy of every weight.

**Why it might generalise.**
- Networks are trained in similar ways, so they may converge on the same units, the way some operations are
  instrumentally necessary for many goals ("you can't do X without first being able to do Y").

**Refinement agreed on Nov 18.**
- Units are expected to be the same **at the level of function**, not at the level of exact weights. Two networks
  can implement "compare" with entirely different weights.
- A unit may be **spread over a cluster of weights**, not sitting in one weight or one neuron.
- So units are identified and tested **by what they do**, verified by intervention (the judge's rung-2 tests), not by
  matching weight patterns.

**How this connects to our problem.** Dense AlgZoo networks have no clean whole-network rule, and a faithful
account of their insides is a copy (`DIAGNOSIS_ABOVE_32.md`, `PLAN_SCALING.md`). Units plus wiring could be the
missing middle level.

## 2. What the literature says (research done Nov 18)

| Line of work | What it found | Bearing on the hypothesis |
|---|---|---|
| Olah et al., "Zoom In" (Distill 2020) | Three claims: features, circuits, universality. Curve detectors and similar circuits recur across vision models | The founding statement of the idea |
| Merullo et al., circuit component reuse (ICLR 2024) | GPT-2 solves two different tasks with ~78% of the same key attention heads; argues for "a small number of task-general algorithmic building blocks" | Units reused across tasks, inside one model |
| Gurnee et al., universal neurons (TMLR 2024) | 1–5% of GPT-2 neurons are universal across seeds; those are usually interpretable and fall into a few families | Some units universal, most not, at the single-neuron level |
| Chughtai et al., toy model of universality (ICML 2023) | All networks use a variant of one algorithm for group operations, but the exact circuits and their order are arbitrary | "Weak" universality: same functions, different wiring. Matches our refinement |
| Zhong et al., Clock and Pizza (NeurIPS 2023); unifying follow-up (NeurIPS 2025) | Small changes give different algorithms for modular addition; the follow-up argues both are the same abstract algorithm | Units may be the same at an abstract level, while the surface differs |
| Csordás et al., "Are neural nets modular?" (ICLR 2021) | Common networks fail to reuse submodules | Evidence against easy reuse |
| Huh et al., Platonic Representation Hypothesis (ICML 2024) | Representations converge across models, data and modalities | Convergence at the representation level |
| Braun et al., APD (2025); Bushnaq et al., SPD (2025, Goodfire) | Decompose the parameter vector into a sum of components (mechanisms spread over weights) with minimal description length. Works on toy models; SPD is more scalable; open library | Direct machinery for "a cluster of weights forms a unit" |
| Gao et al., weight-sparse transformers (OpenAI, Nov 2025) | Training with sparse weights gives circuits about 16× smaller; some transfer to dense models | Density is partly a training artefact; units may be blended, not absent |
| Driscoll et al., shared dynamical motifs (multitask RNNs, 2022–24) | Dynamical motifs (attractors, decision boundaries, rotations) are reused across tasks; they propose motifs as "a fundamental unit of computation, intermediate between the neuron and the network" | Almost exactly the hypothesis, for RNNs. Units as dynamical motifs |
| Maheswaranathan et al., universality and individuality (NeurIPS 2019) | Across thousands of RNNs, the geometry varies but the topological structure of fixed points and transitions is often universal | Function-level universality in RNNs. Our AlgZoo networks are RNNs |

**Summary.**
- Units at the level of FUNCTION have real support (strongest in RNNs: dynamical motifs, fixed-point structure).
- Units at the level of exact WIRING mostly don't recur.
- Nobody has tested this on AlgZoo-style dense networks, or used units to build certified whole-network
  constructions. That's our seat.

**Research still to do** (each item when an experiment needs it):
1. The SPD library: does it handle recurrent weight sharing and transformers this small? What are its outputs?
2. Fixed-point and dynamical-motif analysis methods for ReLU RNNs (Sussillo and Barak style, adapted to
   piecewise-linear networks).
3. Prior work on "neuron-level primitive libraries" and symbolic regression of single neurons.
4. Whether anyone has catalogued units across AlgZoo or similar zoos.

## 3. Experiments: a decision tree, chosen live

Each is minutes, free (no AI calls) unless stated, logged in DIARY with question → result → decision.

- **U1. Do single hidden units compute simple functions of the input history?**
  - Networks: all 2nd-argmax and argmedian RNNs (M4,3 … M16,10, argmedian 8_5 and 16_5).
  - Method: regress each hidden unit at each step on a small library of candidate primitives (x_t, running max,
    running 2nd max, running min, relu(x_t − max), relu(max − x_t), |x_t − max|, sign terms …).
  - Measures: variance explained with ≤ 2 primitives; which primitives; how often the same primitive appears across
    networks and tasks.
  - If units are mostly explained → U3 (are they causal?).
  - If not → U2 (distributed units).
- **U2. Distributed units.** Decompose hidden states into directions (sparse dictionary / ICA) or decompose weights
  (SPD), then repeat U1 on the components.
- **U3. Causal check.** For every primitive found, does an interchange intervention on that unit or direction
  behave as the primitive predicts (judge rung 2)? Only causally verified units count.
- **U4. Composition.** Can each unit's next value be written as a small expression of other units and x_t (the
  wiring)? And can the readout be written from units? If yes, the network is "units plus wiring".
- **U5. Recurrence across networks and tasks.** A table of verified units for every network. Which recur
  (convergent units), which are network-specific?
- **U6. The dynamical-motif view** (if U1–U4 are weak). Fixed points, regions and rotations of the piecewise-linear
  dynamics, compared across networks (Maheswaranathan, Driscoll).
- **U7. Transformers.** The same questions for the longest-cycle networks: attention-pattern units (pointer, copy,
  count, compare) and value-subspace units.

## 4. The build: decided by the experiments (intent first)

**Intended capability:** a constructor that explains a network as **verified units plus their wiring**, with each
unit certified separately (per-part confidence, like AlphaFold's pLDDT) and the whole certified by the v2
certificate.

Which parts get built depends on the results:
- **If U1/U3 succeed**, a **unit finder** that maps neurons or directions to library primitives and verifies them
  causally. Plus a **unit library** (verified primitives with code and tests). Then a reader mode where the
  construction is written in terms of verified units, and the reader explains only the wiring.
- **If units are distributed (U2)**, a decomposition stage first (SPD or a dictionary), then the same finder on the
  components.
- **If U5 shows recurrence**, the library grows across networks, and new networks are searched against it first.
  That's what makes the catalog compound.

Every component gets an end-to-end test on a known case (M4,3: whose units are known from ARC's notes) before use.

## 5. What would count as a discovery
- **(a) A network never interpreted before:** e.g. argmedian 8_5 / 16_5, or longest-cycle, decomposed into verified
  units plus wiring, passing the certificate.
- **(b) A general finding about units:** e.g. "the running-max unit appears in every 2nd-argmax and argmedian RNN
  above size X, implemented by a ReLU pair". Or the opposite, a clean negative. Either is recorded, not sidelined.
- Every claim gets a prior-work check first.

## 6. Operating mode
- Scheduled check-ins about every 25 minutes. Each one: read this file plus the DIARY tail, run the next micro
  experiments, update the docs, commit and push, send Shreyas a short update, schedule the next.
- State that must survive a context reset lives in: this file (plan and status), `DIARY.md` (every step),
  `CHECKLIST.md` (now).

## 7. Status
| Exp | Status | Result | Decision |
|---|---|---|---|
| U1 | done | Single-ReLU comparators against a running statistic (relu(max−x_t), relu(x_t−2nd), relu(x_t−min)) recur across 2nd-argmax and argmedian networks. Coverage of variance after the first step: M16,10 0.42 (shuffled 0.00), M8,5 0.24 (0.07), others ≤ 0.10. M2,2 and M4,3 are raw-input units | mixed → U3 on the explained part |
| U3 | done | FAIL: substituting all explained units keeps M16,10 23%, M8,5 76%, argmedian 16,5 82% (bar 95%); one unit at R² 0.98 can cost a third of decisions. Ablation shows the units matter (9–28% kept) | the units are real and used, but only approximate |
| U1b | done | The rank-based family (current input + sorted history) doesn't capture the residual: joint substitution 19–36% | the residual is not rank structure; position/timing is involved |
| U6a / U6a' | done | Clock (rotation ≈ k·360°/n): beyond chance only in M4,3, M8,5 and argmedian 8,5 (p ≤ 0.04); not general (3/9; Fisher p 1.2e-3 vs bar 1e-3) | dropped as a general unit |
| U2 | done | Task quantities are NOT linear directions. Argmedian networks hold the median value at 0.02–0.09 vs 0.60 in shuffled ones; M16,10 holds max 0.99 and 2nd 0.85, not positions | → U6b |
| P1 / P2' / P3 | done | prior work found (comparators known; a classifier tool exists); δ-units replicate R² 0.99 but don't substitute; ε95 = 0.1–1.6% for all but M2,2 | units must be exact to substitute → P4 exact-structure search |
| UNIT MAP | built | v2 passes K1–K3; maps for 11 networks; see §8 | → P1–P4 |
| U6b | done | Modes (ReLU on/off patterns) covering 90%: up to 18 (M8,5), 74 (M16,10), 114 (argmedian 16,5); compact only in small networks | exact mode decomposition not viable |

**First-sweep conclusion (≈19:13): "approximate universality, exact individuality".** Function-level units recur and matter, but dense networks' decisions hinge on detail that the units miss. No exact compact route was found (5 tried).

**Working conclusion (≈19:12).**
- Units exist at the level of function and recur across networks and tasks (comparators). That supports weak universality.
- In dense RNNs they are approximate, and the network's decisions hinge on the detail they miss.
- Candidate build (pending U2): a **unit map** per network. Every neuron or direction gets a verified unit label with per-part confidence (fit R², single-unit substitution fidelity, ablation importance) and total coverage, plus a cross-network unit library. This is the per-part-confidence certificate, and it doesn't need exact substitution.

## 4b. Build decided from the evidence (≈19:13): the UNIT MAP

**Intent.** For any RNN target, produce a map that labels each neuron at each step with the best unit from a fixed
library, plus three per-part confidence numbers:
1. **fit**: R² of the unit formula on held-out inputs;
2. **sufficiency**: decisions kept when only that neuron is replaced by its formula;
3. **importance**: decisions lost when that neuron is set to its mean.

It also reports **coverage** (variance-weighted share of neuron-steps with fit ≥ 0.9) against a shuffled-weights
control, and a cross-network **unit library** (which unit types recur where).

**Why this and not an exact construction:** U3, U1b, U2 and U6b show that no exact compact construction exists at
these levels for dense RNNs, while the units are real and matter (U1, U3 ablation). The map is honest about both.

**Known-answer tests, written before the code:**
- **(K1) M16,10.** At least 3 of units {1, 6, 7, 8} must be labelled as max−x_t comparators (the "running max" group
  singled out independently by ARC's Z. Chen and a LessWrong analyst), with coverage above the shuffled control.
- **(K2) M4,3.** All of its neurons are labelled as raw-input units (ARC's account: signed input channels), with fit
  ≥ 0.9.
- **(K3) Every shuffled network:** coverage of history units ≤ 0.1.

**Then run it on networks never interpreted before** (argmedian 8_5, 8_7, 16_5, 16_7) and publish the unit maps,
plus the cross-network library, as the first discovery of type (a) and (b).

## 8. Results of the unit-map build (≈19:23) and the next steps

**Unit maps** (`results/glass/units/maps/*.md`; v2 passes K1–K3). Coverage after the first step (history units +
raw/filter units; shuffled control in brackets):

| Network | History | Raw/filter | Notes |
|---|---|---|---|
| M2,2 / M4,3 | 0.00 | 1.00 | fully raw/filter units (matches ARC's signed channels) |
| M8,5 | 0.23 (0.08) | 0.10 | relu(max−x_t) ×2, relu(x_t−2nd), one filter unit |
| M16,10 | 0.37 (0.00) | 0.06 | units 1/6/7 relu(max−x_t), 8 relu(2nd−x_t), 4 relu(x_t−2nd): **matches ARC (Chen N7, N8; LessWrong N1/N6/N7/N8)** |
| M4,4 / M8,8 | 0.00 | ≤ 0.02 | the "amplifying" networks: no units found |
| argmedian 8_5 / 8_7 / 16_5 / 16_7 (never interpreted) | 0.01 / 0.04 / 0.16 / 0.02 | 0.13 / 0.11 / 0.28 / 0.22 | comparators recur (relu(x_t−min), relu(2nd−x_t), relu(x_t−max)) but most neurons are outside the library |

**Findings to date (provisional; prior-work check pending):**
1. **Convergent units.** Single-ReLU comparators against a running order statistic recur across sizes AND tasks
   (2nd argmax, argmedian). An automated map reproduces ARC's hand-found units in M16,10.
2. **Approximate universality, exact individuality.** In dense RNNs the units are approximate and the decisions
   depend on what they miss (U3: substitution collapses), and no exact compact decomposition was found by 5 routes.
3. **The library is task-biased.** It covers 2nd-argmax units far better than argmedian ones. Argmedian networks
   compute something else for most neurons.

**Next (in order; each a micro experiment with a pre-registered decision):**
- **P1. DONE:** comparators in M16,10 are known (Keane; a LessWrong commenter fits them with δ-dependent coefficients at R² 0.992); a correlation-based AlgZoo neuron classifier exists (candle-mi stoicheia). Argmedian above 32 params is unexplained publicly. Finding (1) for M16,10 is a replication; finding (2) and the argmedian maps remain open.
- **P2'. DONE:** δ-tables replicate LessWrong's fit (M16,10 units 1/6/7/8 at R² 0.97–0.999), but joint substitution still collapses (26–33%).
- **P3. DONE: precision requirement ε95.** Every network except M2,2 flips 5% of decisions under hidden-state noise of only 0.1–1.6% of each neuron's spread, including M4,3 (0.016), which is explainable because its account is EXACT. **Units help explain these networks only when they capture exact structure; approximate units can't be swapped in.**
- **NEXT (check-in plan):**
  - **(a) DONE (T4, unit map v3):** reproduces P3; M16,10 has 16/160 substitution-grade neuron-steps, M4,3 8/12.
  - **(b) P4 DONE, NEGATIVE:** exact signed pairs M4,3 50% vs 2–19% in the bigger networks; no exact redundancy. M4,3's exactness does not scale. (Original plan:) search for EXACT structure in bigger networks: exact linear invariants or reductions, like M4,3's symmetry, holding on subsets of neurons (local pairings, exact low-rank subspaces, neurons that are exact linear functions of others). Exact pieces are the only units that can be substituted.
  - **(c) U7 DONE (attention part), NEGATIVE.**
    - Known answers pass after a library fix (tracr_reverse: the reversal map; tracr_hist: same token + first position).
    - Trained lc networks: 8–12% labelled vs controls 0–12% (bar 50%).
    - U7b content lookup: determinism 0.29–0.54 vs controls 0.28–0.57.
    - Hard attention is equally common in shuffled networks.
  - **NEXT (check-in plan, after U7):**
    - (e) DONE, NEGATIVE (U7c).
      - Known answers pass.
      - Layer-0 decision-relevant residual full R² 0.71–0.88 (bar 0.9).
      - Relational share beats shuffled by ≥ 0.2 in only 2 of 4 networks.
      - Residual ε95 0.8–5% (Tracr 6–10%).
      - The U7 line is closed.
    - (f) DONE: `docs/glass/UNITS_FINDINGS.md`.
      - Includes P5: ε95 across families. Modular nets 0.79 vs AlgZoo median 0.010.
      - Two routes to an explanation: exact structure, or tolerance.
    - (g) DONE (CHANGES T5): the catalog shows each network's unit map.
      - The four argmedian networks now have cards ("unit map only").
      - Every AlgZoo RNN card has a Parts section: coverage vs control, ε95, per-neuron labels, and per neuron-step fit, sufficiency, importance, relative error and substitution grade.
      - All maps regenerated with v3 first.
    - (h) DONE (U8, seeds):
      - Units converge across runs at M8,5 (5/5 seeds, 0/5 controls) but not at larger sizes (3–4 of 5).
      - ε95 replicates in 15/15 seed networks.
    - **NEXT:**
      - (i) DONE, NEGATIVE (U9).
        - A sparse search over pairwise comparisons explains only 2–3% of the unexplained argmedian variance (33% in 16_5), against a bar of 50%.
        - One clean soft-rank unit was found in 16_5.
      - (j) DONE as U10 (tolerance intervention, inconclusive: under-trained, ε95 unchanged) and U11 (near-tie control: F3 stands, with ties a factor of about 2.5).
      - (k) DONE (U10b): not supported as pre-registered (3.2× rather than 5×; joint substitution at most 78%).
        - The trend is in the predicted direction (Spearman 0.89).
        - Tolerance saturates near 2.5% as accuracy falls.
      - (k2) NEXT: test "precision is the price of accuracy". Bigger networks (M16,5 / M32,5 trained the same way) should reach the same accuracy at a larger ε95. If they do, the trade-off is about capacity, and wider networks are more explainable at equal accuracy. Pre-register before running.
      - (l) NEXT: write the "approximate universality, exact individuality" result up as a short note for Shreyas, only once (k) is in.
      - Check-in chain note: if a check-in arrives whose text lists (e)–(g) as the work, it is the old duplicate. Do §8 NEXT, but do not schedule a successor (DIARY 21:16).
  - (Original U7 spec:)
    - **Attention units:** per head and query position, the share of attention mass that falls on each library unit's target set. Library: pointer j = x_i; inverse pointer {j: x_j = i}; same token {j: x_j = x_i}; self; a fixed position; uniform. A unit is labelled when it carries ≥ 90% of the mass on average.
    - **Residual units:** per position, the held-out R² of the decision-relevant residual (the top PCs that preserve decisions) regressed on token one-hots plus relational flags (fixed point, 2-cycle, preimage count).
    - **ε95** on the residual stream.
    - **Known-answer tests first:** the compiled Tracr transformers. tracr_reverse must be labelled pointer-to-(N−1−i) (as a fixed-position map); tracr_hist must be labelled same-token.
    - **Control:** shuffled lc networks.
    - **Decision.** If ≥ 50% of attention mass in the trained lc networks falls on labelled units → attention units are real → extend the unit map to transformers. Otherwise record it.
  - **(d) DONE:** no quantitative tolerance measure was found; consistent with the qualitative "almost all ablations hurt dramatically".
- **P2.** What do the unexplained argmedian neurons compute? A data-driven search, not a guessed library: fit each
  one with a sparse combination of pairwise comparisons relu(x_i − x_j) and the order statistics; then test
  sufficiency.
- **P3.** U7, transformers: unit maps for longest-cycle networks (attention units: pointer, copy, compare).
- **P4.** Put the unit maps in the catalog as per-part confidence entries (the AlphaFold-style certificate).
