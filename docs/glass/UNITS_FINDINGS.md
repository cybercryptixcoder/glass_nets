# What the units program found (consolidated; first written Nov 18, 2026, 21:07 ET)

This is a summary of every experiment in the units program, for Shreyas. Every number comes from a file in
`results/glass/units/`. Every decision rule was written in `docs/glass/DIARY.md` before the run it judged. The plan
and the literature review are in `UNITS_PROGRAM.md`.

## One-page summary (updated Nov 18, 2026, 22:57 ET)

**The question.** Shreyas's hypothesis: networks are built from basic units that are the same at the level of
function (not weights), possibly spread over clusters of weights, and similar training makes networks converge on the
same units. If so, "these units, wired like this" is a readable account of a network.

**What was tested.** About 90 networks:
- ARC's AlgZoo RNNs for 2nd argmax and argmedian at several sizes, including all 5 of ARC's training runs per size;
- AlgZoo longest-cycle transformers;
- my own noise-trained RNNs;
- compiled Tracr networks with known answers;
- the modular-arithmetic transformers our system already explains.

23 experiments (U1–U15 with their variants, P2–P5). Every decision rule was written down before its run.

**What holds up:**
1. **Function-level units exist and recur.** Single neurons computing "how far is this input below the running max"
   or "above the 2nd max" appear across sizes, across two tasks, and in all 5 independent training runs at 8 neurons
   (none in shuffled controls). This is the convergence Shreyas predicted, at small size. (F1, U8)
2. **They are approximate, and in these networks approximate is not enough.** When units replace all the neurons they
   describe, the network keeps only 23–82% of its decisions. A unit at R² 0.98 can cost a third of decisions by
   itself. (F2)
3. **The reason is measured: precision.** AlgZoo networks flip 5% of their decisions under noise of about 1% of a
   neuron's spread (13 networks, plus 15 more training runs). The modular-arithmetic networks tolerate about 80× more.
   About a factor of 2.5 comes from nearly tied inputs; the rest is the networks themselves. (F3, P5, U11)
4. **Two routes to an explanation:** exact structure, which works even for fragile networks (ARC's M4,3), or
   tolerance, which lets approximate accounts work (the modular nets). AlgZoo above 32 parameters offers neither: no
   exact structure in 7 searches. (F3, F4)
5. **Width matters for 2nd argmax.** With more neurons, ARC's networks become more accurate, about 2× more tolerant,
   and better described by units. At ARC's challenge size (10 inputs), 32-neuron networks are about half single
   recognisable units, against 2–22% at 16 neurons. This fits superposition theory: cramped networks blend their
   units. (U12, U13)

**What did not hold up:**
- Units do not converge across training runs at 16 neurons (3–4 of 5). (U8)
- Training with internal noise raised tolerance only about 3× (bar 5×), and accuracy fell beyond that. The causal
  version of point 3 is not established. (U10b)
- The width effect does not appear in argmedian, even with units found from data. (U14, U15)
- Transformers: no textbook attention units, no content lookup, no clean residual units. Training adds partial
  relational structure. (F6)
- Most of each argmedian network remains unexplained. One clean "soft rank" neuron was found. (F5, U9)

**What is new, after a prior-work check:**
- **Still a candidate, with no prior measurement found:** the precision requirement (ε95) as a measured, replicated
  predictor of when approximate accounts can be faithful. Nearest work: Arora et al. 2018, noise stability and
  compression.
- **Still new:** unit maps with per-part confidence for argmedian networks above 32 parameters. No public
  interpretation was found.
- **A known mechanism, newly observed here:** wider networks give units their own neurons. Superposition theory
  (Elhage et al. 2022) predicts it, and Kong et al., "Expand Neurons, Not Parameters" (ICML 2026), shows it.
- **A known question, with a new measurement:** units converging across training runs. Gurnee et al. 2024 found 1–5%
  of GPT-2 neurons universal. For ARC's 16-neuron recipe, a LessWrong commenter notes representations vary a lot by
  seed, which matches our 3–4 of 5. What we add is the 5-of-5 convergence at 8 neurons, with controls.

The rest replicates known work (comparators in M16,10: Keane, LessWrong; timing tables: LessWrong comments).

**What it means for Glass Networks:**
- Report ε95 with every certificate. A small ε95 means an approximate construction cannot be faithful.
- For fragile networks, offer the unit map (now in the catalog) as the honest partial product.
- Width and tolerance are levers: networks trained wider, or for tolerance, should be easier to explain.

## Findings, one by one

### F1. A comparator unit recurs across sizes and tasks (replication, plus an extension)

**What:**
- Single hidden units that compute relu(running statistic − current input), or the reverse.
- relu(max − x_t): 15 neuron-steps in M16,10, 3 in M8,5, 5 in M8,8, 1 in argmedian 8,5.
- relu(x_t − 2nd max): 7 in M16,10, 2 in M8,5.
- relu(x_t − min): in argmedian 8,5.

**Controls:** shuffled-weight networks reach history coverage 0.00–0.07, against up to 0.42 in the trained networks.

**Known answer:** the unit map labels M16,10 units 1, 6 and 7 as relu(max − x_t) and unit 8 as relu(2nd − x_t).
This matches ARC/Chen's N7 and N8 and the LessWrong group {N1, N6, N7, N8}.

**Status:**
- For M16,10 this is a **replication** of prior work (Martin Keane's comparators on LessWrong; candle-mi's
  "stoicheia" classifier does a similar labelling).
- The **extension** is that the same unit appears in a different task (argmedian), which no prior work covers.

**Across training runs (U8: ARC's 5 seeds per size, 15 networks):**
- At M8,5, both comparators appear in all 5 seeds and in none of the shuffled controls. Independent training runs
  converge on the same function-level units.
- At larger sizes the inventory becomes individual. In the zoo M16,10 and argmedian 16,5 seeds, the comparators
  appear in 3–4 of 5 runs, and partly in the controls too.
- So convergence holds at small size and weakens with size. This did not meet the pre-registered bar of recurring
  at all three sizes.
- ARC's challenge M16,10 is unusually comparator-rich next to its five zoo siblings (history coverage 0.37 vs
  0.02–0.22).

### F2. Approximate units can't stand in for the network (new as a measured result)

**What:** replace every explained neuron-step by its fitted unit inside the running network, then count how many
decisions survive.

| unit family | M16,10 | M8,5 | argmedian 16,5 | argmedian 8,5 |
|---|---|---|---|---|
| 2-term comparators (U3) | 23% | 76% | 82% | – |
| rank-based (U1b) | 19% | 36% | 29% | – |
| timing tables (P2') | 26% | 33% | 27% | 32% |

**Notes:**
- The timing-table units replicate the LessWrong fit for M16,10 (R² 0.97–0.999; they reported 0.992). They still
  keep only about a quarter to a third of decisions when used together.
- One unit alone at R² 0.98 can cost a third of decisions.

**Prior work:**
- A LessWrong post by Josh Snider (Oct 26, 2026) reports the same symptom from another angle: propagating summaries
  instead of running the network failed because the top two logits fall out of sync.
- ARC's comment thread notes that "almost all ablations hurt the accuracy dramatically".
- Neither measures it as we do.

### F3. The precision requirement ε95 (candidate general finding)

**What ε95 is:** add Gaussian noise at an intermediate site, scaled to each coordinate's own spread. ε95 is the
noise level at which 5% of decisions flip. A small ε95 means any account of that site must be very precise.

| network family | ε95 |
|---|---|
| AlgZoo RNNs (9 networks) | 0.001–0.016, except M2,2 at 0.16 |
| AlgZoo longest-cycle transformers (residual) | 0.008–0.05 |
| AlgZoo median (13 networks) | 0.010 |
| modadd_p113, modmul_p113 | 0.79 |
| Tracr compiled networks | 0.05–0.32 |
| Breakout policy MLP (trained, never explained) | 0.02 |

**Replication across training runs (U8):** ε95 ≤ 0.05 in 15 of 15 seed networks, and nearly constant within a
size: M8,5 0.006–0.013, zoo M16,10 0.003–0.005, argmedian 16,5 0.008–0.010. So it is a property of the task and size,
not of one training run.

**Control for near-tie inputs (U11):**
- 2nd argmax and argmedian have continuous inputs, so some inputs always sit next to a tie, where any perturbation
  flips the decision. That could make ε95 small for reasons that have nothing to do with the network.
- Measured on the clear half of the inputs (task margin above the median), ε95 rises by a factor of only 1.0–5.0
  (median 2.5) to 0.001–0.08 (median 0.016).
- That is still about 50× below the modular nets. Near-ties account for part of the effect, not most of it.

**The intervention (U10, U10b):** I trained M8,5 networks with hidden-state noise during training, to make them
tolerant. The first attempt (U10) was under-trained and inconclusive. The proper run (U10b, 12 networks, AlgZoo-level
accuracy for the standard ones):

| training noise σ | accuracy | ε95 | joint substitution |
|---|---|---|---|
| 0 | 0.76–0.88 | 0.006–0.008 | 42–51% |
| 0.03 | 0.80–0.88 | 0.016–0.020 | 68–73% |
| 0.1 | 0.73–0.74 | 0.020 | 69–73% |
| 0.3 | 0.55–0.62 | 0.020–0.025 | 75–78% |

- Tolerance and the faithfulness of approximate units rise together (Spearman 0.89), in the predicted direction.
- But the pre-registered bars were not met: 3.2× rather than 5×, and 78% rather than 90%. So F3 stays
  correlational.
- Tolerance saturates near 2.5%. Beyond that the networks lose accuracy instead.
- My reading after U10b: for this task at this size, precision is the price of accuracy.

**Capacity (U12, ARC's own 2nd-argmax networks at n = 5 with 8, 16 and 32 hidden units, 5 seeds each):**
- Wider networks are more accurate (median 0.87 → 0.99 → 0.996) AND more tolerant (ε95 0.8% → 2.0% → 1.6%).
- Approximate units also stand in for them much better (joint substitution 43% → 73% → 78%).
- So the trade-off binds only at low capacity: a capacity-starved network must be fragile to be accurate.
- Even at 32 units, these networks stay far more fragile than the modular nets, and under 90% joint substitution.
- By the pre-registered rule this was "mixed" (2× more tolerant, against a 3× bar).

**Width and units (U13, U14):**
- At n = 10, ARC's 2nd-argmax networks with 32 units are about half single-neuron library units: history coverage
  0.49–0.67, against 0.02–0.22 at 16 units, the size of ARC's challenge network.
- This is what superposition theory predicts (Elhage et al. 2022): a capacity-starved network blends its units across
  neurons, and a wider one gives them neurons of their own.
- The pre-registered joint-substitution bar was not met (+15 points against +20), partly because the narrow networks
  substitute almost nothing.
- **Not replicated on argmedian:** wider argmedian networks are far more accurate but no more library-like (coverage
  ratio 1.6 and 0.8, against a bar of 2).
- The same holds with units found from data rather than from the library (U15: total explained 0.63 vs 0.64 at
  n = 5, 0.34 vs 0.32 at n = 7).
- So the width effect is task-specific so far.
- **Replicated at another length (U16):** for 2nd argmax with n = 8, history+raw coverage goes 0.02 → 0.18 → 0.46 at
  8 → 16 → 32 units.
- Across all 2nd-argmax configurations, unit-likeness tracks hidden size relative to sequence length. At 16 units,
  coverage falls from about 0.37 (n = 5) to 0.18 (n = 8) to 0.06 (n = 10). At 32 units it stays at 0.40–0.60.

**Reading:**
- An account can be faithful in one of two ways. It captures EXACT structure, as with M4,3 (ε95 0.016, explained
  exactly by ARC) and Tracr count_greater (0.05). Or the network tolerates imprecision, as with the modular networks
  (0.79), where our approximate constructions certify at 97–100%.
- This is the measured reason the reader failed on AlgZoo above 32 parameters, and it predicts that failure.

**My own prediction was partly wrong (P3):** I expected M4,3 to be robust because it is explainable. It isn't. That
error is what led to the two-routes reading.

**Prior work:**
- Arora, Ge, Neyshabur and Zhang (ICML 2018) show that noise stability is what makes trained nets compressible.
- Morcos et al. (2018) link robustness to node noise with generalization.
- Our finding is the interpretability version: noise tolerance at intermediate sites predicts whether a short
  APPROXIMATE readable account can be faithful.
- I found no prior measurement of it as a predictor of explainability. One more search is needed before calling it
  new.

**Limits:**
- Only 2 networks are both explained approximately and tolerant.
- Architectures differ, and task margins differ (113 classes with large logit gaps).
- One site per network.

### F4. No exact compact structure beyond M4,3 (negative, across 7 routes)

| route | result |
|---|---|
| Single-neuron formulas (U1) | covers at most 42% of neuron-steps |
| Rank formulas (U1b) | median R² 0.59–0.81 |
| Linear task variables (U2) | the trained networks don't hold positions linearly. The argmedian networks hold the median value LESS than shuffled ones (0.02–0.09 vs 0.60–0.63) |
| Rotation clock (U6a, U6a') | in 3 of 9 networks beyond chance (M4,3 p 0.007, M8,5 0.04, argmedian 8,5 0.007), not general (Fisher p 1.2e-3 against a 1e-3 bar) |
| Linear modes (U6b) | up to 18 / 74 / 114 ReLU patterns per step for M8,5 / M16,10 / argmedian 16,5 (bar ≤ 8) |
| Exact signed pairs (P4) | M4,3 50% (its known structure), others 2–19% (bar 25%) |
| Timing tables (P2') | see F2 |

### F5. Unit maps of networks never interpreted before (partial)

**Coverage:**
- The argmedian RNNs above 32 parameters (8,5; 8,7; 16,5; 16,7) now have unit maps with per-unit confidence.
- History-unit coverage is 0.01–0.16 and raw/filter coverage 0.11–0.28.
- So most of each network is NOT explained by the library.

**The unexplained part (U9):**
- A data-driven search was run on the remainder. Each neuron got up to 3 terms from the inputs, their order
  statistics and every pairwise comparison relu(x_i − x_j).
- It explains only 2–3% of the remaining variance in three networks and 33% in 16_5 (shuffled controls 0–2%).
- One clean find, in 16_5: a neuron that is exactly a weighted count of how far earlier inputs exceed x4, a soft rank
  of x4. That is the quantity a median finder needs.

**Status:**
- This is a partial decomposition, honestly reported per unit. It is not a certified construction.
- As of Nov 18 I found no public explanation of these networks; xabier.me explains argmedian M4,3 only.

### F6. Transformers: no textbook units, but training adds relational structure

**Attention patterns (U7):**
- Only 8–12% of (layer, head, query) cells carry ≥ 90% of their mass on a standard unit (self, fixed position, same
  token, pointer, inverse pointer, uniform). Shuffled networks: 0–12%.
- Known answers pass on Tracr.

**Content lookup (U7b):**
- The attended token is determined by (position, query token) in 29–54% of hard queries, against 28–57% in shuffled
  networks.
- Hard attention is equally common in shuffled networks, so it comes from the architecture.

**Residual (U7c):**
- The decision-relevant part of the layer-0 write is explained at R² 0.71–0.88 by tokens-at-positions plus
  relational flags (fixed point, pointer read, 2-cycle, preimage count, same-token count).
- The relational part beats shuffled networks by +0.04, +0.26, +0.30 and +0.17.
- So training adds relational features, but not cleanly.

## What this means for the hypothesis

**Premise 1, "every network is an algorithm":**
- True trivially: the network itself is one.
- The real question is whether a SHORT algorithm made of units exists at the precision the network needs.

**Premise 2, "every algorithm decomposes into basic units":**
- These experiments support a qualified version.
- Units exist and recur at the level of function. That is the weak universality of the literature (Chughtai et al.;
  Maheswaranathan et al.): same functions, individual details.
- In hypersensitive networks the individual details carry the decisions.
- One-line version: **"approximate universality, exact individuality"**.

**Not shown:**
- That units don't exist at some other level we didn't test.
- That a decomposition method like SPD wouldn't find exact components.

## What this means for Glass Networks

1. **ε95 belongs in every certificate.** It says in advance whether an approximate construction can work. The unit
   map already reports it (CHANGES T4).
2. **For networks with small ε95, only exact structure or an explicit remainder can be faithful.** Pushing a reader
   to write shorter approximate accounts there cannot succeed. That is the measured reason behind rounds R2–R8.
3. **The unit map is the honest partial product** for such networks: which parts are explained, how well, and how
   much precision the network demands.

## Open items

- Put the unit maps in the catalog as per-part-confidence entries (next build step).
- A data-driven or SPD-style decomposition of the unexplained argmedian neurons.
- Replicate F3 on more tasks and on more networks that are explained approximately (seeds: done in U8).

## Sources

- [ARC, "AlgZoo: uninterpreted models with fewer than 1,500 parameters"](https://www.lesswrong.com/posts/x8BbjZqooS4LFXS8Z/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters) and its comment thread.
- [Josh Snider, "Brute-Force Progress on AlgZoo's 432-param Interpretability Challenge"](https://www.lesswrong.com/posts/sAhFp6gYoyw3xeCaF/brute-force-progress-on-algzoo-s-432-param-interpretability).
- [Arora et al., "Stronger generalization bounds for deep nets via a compression approach"](https://arxiv.org/abs/1802.05296).
- [xabier.me, "Taking apart a 32-number network"](https://forum.effectivealtruism.org/posts/boaXwxTaB2pYuyGmj/taking-apart-a-32-number-network-1).
- The rest of the literature is in `UNITS_PROGRAM.md` §2.
