# The mechanism atlas: what it is, what it does, and what it has shown

*Started Nov 19, 05:52 ET; main result (A28) landed 08:34 ET (clock). Detailed record: `docs/glass/DIARY.md`; change ledger: `docs/glass/CHANGES.md`; plan and status: `docs/glass/ATLAS.md`.*

## What it is, in one paragraph

Give it a trained network. It splits every weight matrix exactly into small rank-1 pieces ("mechanisms"), using stochastic parameter decomposition (SPD; Bushnaq, Braun & Sharkey 2025). For each mechanism it learns a readable rule for **when** the network needs it (a "trigger", e.g. "ON when the max arrived in the last step and x_t is ≥ 0.64 below it"). It measures **what** the mechanism does (removing it where needed flips the answer from the 2nd largest to the max, say). Then it checks the whole account: it rebuilds the network from its mechanisms, switches each one on and off by its rule, and counts how many of the real network's decisions survive. The output is a certificate (markdown and JSON), a catalog page, and a playground where you step through one input and watch which mechanisms switch on.

## How to use it

```
# decompose at a few sparsity settings (minutes to an hour each); C = max(20, 1.25 x hidden width), e.g. 40 for 32-wide
python3 scripts/glass/atlas_spd_job.py OUT 30000 512 cpu --coef-min=1e-3 --C=40 --tag=a TARGET
python3 scripts/glass/atlas_spd_job.py OUT 30000 512 cpu --coef-min=3e-4 --C=40 --tag=b TARGET
python3 scripts/glass/atlas_spd_job.py OUT 30000 512 cpu --coef-min=1e-4 --C=40 --tag=c TARGET
# (for a weak network, put the sparser default --coef-min=3e-3 first)
# build the atlas from the sparsest one that is precise enough (training half), with absorption
python3 lab/atlas.py TARGET x 4 --ladder OUT/..._a.pt,OUT/..._b.pt,OUT/..._c.pt [--control-spd SHUF_a.pt,SHUF_b.pt,SHUF_c.pt]
python3 lab/server.py   # then open /catalog?t=TARGET
```

## What a certificate claims, and how each claim is checked

| claim | measured as | bar |
|---|---|---|
| (i') the mechanisms account for the network | decisions kept with all mechanisms on and nothing else | ≥ the task rule (how often "the task's own answer" matches the network) |
| (ii) the triggers say when each mechanism is needed | decisions kept when each trusted trigger that also passed a decision check on training data switches its mechanism (the rest stay on) | ≥ the task rule |
| (iii) most mechanisms have a trigger you can trust | share of switched mechanisms whose trigger is ≥ 90% right on held-out clear-cut cases | ≥ 50% |
| (iv) mechanisms matter where they are said to | output change from removing a mechanism where needed vs where not (median) | ≥ 10× |
| (v) the structure is learned, not an artefact | mechanisms in functional groups, trained vs shuffled-weights network | trained > shuffled |
| (vi) the triggers do real work | share of not-needed calls the triggers switch off | ≥ 50% |

All bars were fixed before the networks they judge were run. The task-rule bar means an account must predict the network at least as well as the one-line description of its task.

## The main result (A28, held out, pre-registered)

Six well-trained networks that were never used to design anything: seed 2 of six AlgZoo architectures, 95–99.4% accurate, up to 1,376 parameters and 32 hidden units. Every setting was fixed beforehand on other networks. **5 of the 6 pass every criterion** (the bar was 4), and the shuffled-weights control shows no functional structure (17 functional mechanisms vs 0).

| network (accuracy, parameters) | rebuilt by rules vs task rule | rules switch off (of not-needed activity) | trusted rules | removal matters where needed | result |
|---|---|---|---|---|---|
| 2nd argmax 16_5 (99.2%, 352) | 99.5% vs 99.2% | 86% | 76% | 1,853× | pass |
| 2nd argmax 16_8 (95.7%, 400) | 97.9% vs 95.7% | 86% | 57% | 142× | pass |
| 2nd argmax 32_5 (99.4%, 1,216) | 99.8% vs 99.4% | 95% | 70% | 1,854× | pass |
| argmedian 32_5 (99.3%, 1,216) | 99.1% vs 99.3% | 83% | 68% | 819× | fail (narrowly) |
| 2nd argmax 32_10 (99.1%, 1,376) | 99.6% vs 99.1% | 95% | 62% | 1,171× | pass |
| argmedian 32_7 (95.2%, 1,280) | 98.0% vs 95.2% | 98% | 53% | only where needed | pass |

**What made it work, in the order it was found:**
1. **A better gate (A15).** Switch a mechanism off only where its rule is nearly sure it is not needed. A wrong "off" costs decisions; a wrong "on" costs nothing.
2. **Scoring rules on clear-cut cases (A19).** Many mechanisms are only "half needed" at times.
3. **Precision for accurate networks (A26).** Weaker sparsity pressure, then folding the small leftover weight into the mechanisms.
4. **A decision check (A23).** A rule is used in the certificate only if switching by it keeps the decisions on the training data.
5. **Adaptivity and width (A28).**
   - Each network gets the sparsest decomposition that is precise enough on its training half, so weak networks keep few, readable mechanisms.
   - Wider networks get proportionally more components.

**The road there (all held out, all pre-registered):** default settings 0 of 6 (A18) → precise settings 2 of 6 (A27; 32-wide networks ran out of components, predicted before the verdict) → adaptive and width-scaled 5 of 6 (A28).

**Development networks with the final pipeline:** 6 of 7 RNNs pass every criterion (the seventh, zoo M16,10 seed 0, is being rerun with the ladder). They include:
- ARC's 2nd-argmax challenge network M16,10: rebuilt 96.8% vs 95.3%, with 8 functional mechanisms vs 0 in its shuffled control;
- argmedian 16_5: 98.9% vs 98.4%, 37 vs 0.

**Transformers (A33, held out, pre-registered):** the same pipeline, unchanged, passes on lc_8_6, the one longest-cycle transformer not used in design (8-dimensional, 2 attention layers): rebuilt 90.2% vs task rule 85.4%, triggers switch off 89% of not-needed activity, 64 of 106 mechanisms have trusted task-level explanations (token values, positions, fixed points). On development, after fixing a data-split bug (A32), lc_4_5 and lc_6_5 pass and lc_3_4 does not; it keeps its decisions only by switching off a third of its not-needed activity.

**Scaling to 227k parameters (Nov 19, all held out and pre-registered).** The grokked modular-arithmetic transformers (one attention layer and one MLP, 227k parameters, about 165× larger than anything certified above) have a known mechanism: a few Fourier frequencies, which differ from one training run to the next.
- **A35 (the first try): narrow fail.** 71% of trusted frequency rules used the key frequencies (bar 80%, chance 9%).
- **A37 (rule vocabulary of angles "k·v mod p"): PASS.** On two freshly trained adders never used for design, 84% and 86% of trusted frequency rules use that network's own key frequencies. With the old vocabulary the same decompositions give 72% and 84%.
- **A40 (multiplication, where the structure lives in discrete-log order; the atlas is offered both orderings): FAIL.** 56% (bar 80%, chance 5%). The basis is chosen correctly: only 16% of rules use those frequencies on the number line, and the MLP and unembedding are led by the right log-basis frequencies. The larger vocabulary gives the rule-fitter more ways to memorise token lists.
- **A41 (rule vs lookup).** Each mechanism is labelled "rule" (its explanation carries over to token values never seen) or "lookup" (it only describes the cases seen).
  - In the fresh adders, 82% and 88% of trusted MLP/unembedding mechanisms are rules. The embedding (34 of 35, 48 of 56) and attention-value mechanisms are mostly lookups, so the atlas no longer presents them as understood.
  - The memorization check on a network trained on random labels is pending.
- **Post hoc, to be tested as A42:** over rule-labelled mechanisms only, 90–98% of frequency rules use key frequencies in every network so far, multiplication included. The structural claims are nearly all right; the noise is memorised rules.
- **Certificates at this size:** everything passes except "mechanisms alone". The decomposition gives the attention query/key weights no mechanisms: they are spread over many components, and SPD's random partial masks never remove them all. A version that fixes this (A38d) breaks a planted known answer, so it is not used.

**Known answers** (networks built by hand, so the true mechanisms are known):
- A recurrent network with 5 planted features: all 15 planted mechanisms recovered.
- A transformer that copies the previous token: all 16 planted "copy token v" mechanisms recovered.
- Both rebuild 100% of decisions.

## What the atlases find (examples)

- **ARC's M16,10** (the 2nd-argmax challenge network): 35 mechanisms with the default settings; 4 in functional groups, against 0 in the shuffled control. W_h.6 switches on when the max arrived in the last step and the current value is well below it; removing it makes the network return the max instead of the 2nd largest.
- **argmedian 16_5**: the mechanisms hold the median against both neighbours. With the default settings, 12 stop the answer moving to the next-larger value and 9 to the next-smaller; the shuffled control has 1. With the precise settings: 25 vs 12.
- **argmedian 8_7** (49.5% accurate): most recurrent mechanisms switch on by time step ("after step 3"). The network stages its computation over time, not by data content.
- **Recurrence across training runs: not confirmed.** ARC's M8,5 against five independently trained M8,5s:
  - Seeds 0 and 3 share 5 and 7 mechanisms with it on role and trigger variables, against 2.0 and 2.6 by chance (p ≤ 0.003).
  - Seeds 1 and 2 are at chance, and seed 4 shares none.
  - The pre-registered confirmation needed 3 of seeds 1–4 to beat chance; 1 did. So recurrence depends on the training run, and no general claim is made.

## Limits, stated plainly

- Transformers: one held-out certificate pass (lc_8_6, attention only, d_model ≤ 8). At 227k parameters (MLP transformers) the known answer is recovered on held-out adders (A37) but not yet for multiplication (A40), and no certificate passes yet because of the attention query/key gap.
- Only one network family so far (AlgZoo ReLU RNNs on 2nd-argmax and argmedian). 1,376 parameters is larger than anything certified before in this project, but small in absolute terms.
- Explanations are often shallow: many trusted rules are time gates ("after step 2"). The certificate says when a mechanism acts and which decision it protects. What it computes is told only coarsely, as a profile of the unit types it reads and writes (A31). Example, M16,10 W_h.0: "reads relu(max − x_t) 29%, relu(2nd − x_t) 14%…". In 32-wide networks 20–50% of a profile sits on neurons the unit library cannot label.
- At the precise settings almost every component the decomposition offers is used (57 of 60 in argmedian 16_5), so mechanism counts partly reflect that budget.
- With absorption, criterion (i') becomes close to automatic whenever the alive mechanisms' input directions span the matrix's inputs: least squares can then make their sum exact on the data. Under A26 the informative criteria are (ii), the triggers keeping the decisions, and (vi), the triggers switching off most not-needed activity. Absorption does not make those easier: at minimality 1e-3 it made (ii) worse (0.876 → 0.723 on argmedian 16_5).
- Triggers for output-layer mechanisms are often not trusted: their importance is graded, not on/off.
- Findings are stated in the task's terms (ranks, steps, values), but many triggers still name other neurons. A unit-map gloss is shown where one exists.
