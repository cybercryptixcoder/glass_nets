# What the units program found (consolidated, written Nov 18, 2026, 21:07 ET)

This is a summary of every experiment in the units program, for Shreyas. Every number comes from a file in
`results/glass/units/`. Every decision rule was written in `docs/glass/DIARY.md` before the run it judged. The plan
and the literature review are in `UNITS_PROGRAM.md`.

## The question

**Shreyas's hypothesis (Nov 18):**
- Networks build their behaviour from basic units, the same at the level of function (not weights), possibly spread
  over clusters of weights.
- Because training is similar, the same units may recur across networks.
- If so, "these units, wired like this" would be a readable account of a network, between one rule for everything
  and a copy of every weight.

**Tested on:**
- 11 AlgZoo RNNs: 2nd argmax and argmedian, 10 to 432 parameters.
- 4 AlgZoo longest-cycle transformers: 108 to 656 parameters.
- Compiled Tracr transformers (known answers).
- For comparison: the modular-arithmetic transformers, which the system already certifies.

## The short answer

1. **Units exist at the level of function, and they recur.** The same small comparator appears across network sizes
   and across two different tasks.
2. **But they are approximate.** In these networks no account built from approximate units stands in for the
   network's parts: decisions collapse when units replace the neurons they describe.
3. **The reason can be measured.** These networks need about 1%-precise accounts (ε95). The networks the system
   CAN explain approximately tolerate about 80× more imprecision.
4. **That gives two routes to an explanation.** Either find EXACT structure (that works at any tolerance), or the
   network must tolerate imprecision. AlgZoo above 32 parameters offers neither, as far as 7 different routes
   could find.
5. **Transformers in AlgZoo show no textbook units.** Neither the attention patterns nor the layer-0 residual
   decompose into a library of standard operations. Training does add partial relational structure, well above
   shuffled controls.

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
- Replicate F3 on more networks (more seeds, more tasks) before any public claim.

## Sources

- [ARC, "AlgZoo: uninterpreted models with fewer than 1,500 parameters"](https://www.lesswrong.com/posts/x8BbjZqooS4LFXS8Z/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters) and its comment thread.
- [Josh Snider, "Brute-Force Progress on AlgZoo's 432-param Interpretability Challenge"](https://www.lesswrong.com/posts/sAhFp6gYoyw3xeCaF/brute-force-progress-on-algzoo-s-432-param-interpretability).
- [Arora et al., "Stronger generalization bounds for deep nets via a compression approach"](https://arxiv.org/abs/1802.05296).
- [xabier.me, "Taking apart a 32-number network"](https://forum.effectivealtruism.org/posts/boaXwxTaB2pYuyGmj/taking-apart-a-32-number-network-1).
- The rest of the literature is in `UNITS_PROGRAM.md` §2.
