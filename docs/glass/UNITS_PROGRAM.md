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
| U1 | starting 19:15 | | |
