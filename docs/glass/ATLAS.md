# CURRENT WORK: the mechanism atlas (decompose → explain → certify)

*Started Nov 19, 2026, by plan change, approved by Shreyas at 03:11 ET. This file is the single source of truth for the
current build. Read it right after `CLAUDE.md`. Keep its "Status" and "Next" sections current after every step. They
must never be stale.*

## 1. Why the plan changed

**Shreyas's direction (Nov 19, 03:09 ET):**
- Build something that works. Experiments exist only to inform the build.
- Try many ways to build it and test each one.
- Come back with a working system, not a report.
- Target: real, discovery-grade interpretability on networks larger and harder than any we have explained so far.

**What the units program established** (details: `UNITS_FINDINGS.md`):
- On ARC's dense networks (AlgZoo above 32 parameters), readable APPROXIMATE accounts cannot be faithful.
  - The networks are fragile: 5% of decisions flip at about 1% noise.
  - Rules can replace at most 7% of a network before the account drops below the task rule (U17).
  - That is why the reader failed in rounds R2–R8.
- An EXACT decomposition is faithful.
  - Stochastic Parameter Decomposition (SPD; Bushnaq, Braun & Sharkey 2025), re-implemented here for recurrent
    networks, recovers planted mechanisms perfectly (U19(b')).
  - On ARC's M8,5 it splits the recurrent weights into 11 rank-1 mechanisms, of which about 5 are used on any input.
    Masking by importance keeps 90–93% of decisions, above the task rule's 88% (U19(c)).
- What is missing is the readable layer:
  - **when** each mechanism fires: its trigger;
  - **what** it does: its effect;
  - a certificate for each.

## 2. Intent (the question the build answers)

**Can the system take a trained network it has never seen and produce a mechanism atlas?** That is, a list of the
network's mechanisms, where for each:
1. **exact:** the mechanisms sum to the weights;
2. **sparse:** on each input, only a few are needed, and the importance-masked network still beats the task rule;
3. **readable trigger:** a rule in task terms says when it fires, scored on held-out inputs, and that score is its
   confidence;
4. **readable effect:** what removing it changes in the output, on the inputs where it fires;
5. **causally specific:** removing it changes decisions where its rule says it fires, much more than elsewhere;

and as a whole:

6. **rule-gated fidelity:** the network rebuilt from its mechanisms, each switched on by its readable rule,
   reproduces the network's decisions above the task rule.

- **"Works"** = items 1–6 hold, every number comes from held-out inputs, and the known-answer tests below pass.
- **"Discovery-grade"** = run on networks nobody has interpreted (the argmedian RNNs, the longest-cycle transformers,
  the zoo M16,10 seeds), it yields mechanism-level claims that are each tested and carry their own confidence.

**What this is not:** a short program replacing the network. The atlas keeps the exact weights, split into mechanisms.
Its readable content is when each mechanism fires and what it does, each claim tested. For fragile networks that is
what the evidence says is achievable. Approximate short accounts remain the route for tolerant networks (the modular
nets), where the existing reader and judge already work.

## 3. Components and tests

| # | component | file | known-answer test | status |
|---|---|---|---|---|
| 1 | SPD engine (any forward function; repeated calls for recurrent weights) | `scripts/glass/units/spd_core.py` → to `lab/atlas.py` | Planted RNN, 5 mechanisms per matrix (U19(b')): PASS. TMS 5-2: structure PASS on 2 seeds (U19(a), (a')) | built, tested |
| 2 | Trigger explainer (shallow trees over task features + neuron states; held-out accuracy) | `scripts/glass/units/explain_triggers.py` (prototype) | Planted RNN: triggers must be "feature j arrives / has arrived", accuracy about 1 | prototype; M8,5: 8/11 W_h mechanisms ≥ 0.9 with neuron states, 4/11 with task features only |
| 3 | Rule-gated construction + fidelity | `scripts/glass/units/rule_construction.py` (prototype) | Planted RNN: rule-gated fidelity about 1 | prototype; M8,5: 0.895 (depth 5, task + neuron features) vs task rule 0.879 |
| 4 | Effect + causal specificity per mechanism | to build | Planted RNN: removing mechanism j changes only feature j's output | not started |
| 5 | Atlas certificate (JSON + markdown) + catalog card | to build (`lab/atlas.py`, catalog) | End to end in headless Chrome, as T5/T6 | not started |
| 6 | Transformer support (attention QK/OV matrices as decomposed weights) | to build | Tracr compiled networks (known mechanisms) | not started |

**Rules for building:**
- `CLAUDE.md` applies: a CHANGES.md entry before code in `lab/`, a test that fails before and passes after,
  known-answer tests before real runs, no thresholds fitted to the results they judge.
- Experiments only when a specific build step needs an answer, and each says which step.
- Try alternatives quickly when a component falls short: different feature sets, tree depths, an LLM writing
  triggers, more SPD steps. Record each try in DIARY with its number.

## 4. Status (keep current)

- Nov 19, 03:13 ET (clock):
  - Components 1–3 exist as prototypes.
  - The M8,5 SPD model is saved (`results/glass/units/u19c_model_zoo_2nd_argmax_8_5_original_30000_512.pt`).
  - The shuffled-control SPD for U19(c) is running (`/tmp/u19c_shuf.log`).
  - Next: package components 1–3 into `lab/atlas.py` with a CHANGES entry and the planted-RNN known-answer test,
    then add component 4.

## 5. Next (in order)

1. **A1:** `lab/atlas.py`, packaging components 1–3.
   - CHANGES entry A1 first.
   - Known-answer test: on the planted RNN the atlas must recover 5 mechanisms per matrix, triggers with held-out
     accuracy ≥ 0.95, and rule-gated fidelity ≥ 0.99.
2. **A2:** component 4 (effects and specificity), with the same planted test.
3. **A3:** run the atlas on M8,5, argmedian 8_5 and 16_5, and zoo M16,10 seed 0.
   - Improve triggers by trying alternatives until rule-gated fidelity beats the task rule with most mechanisms at
     confidence ≥ 0.9, or record why not.
4. **A4:** certificate, catalog card and end-to-end browser test.
5. **A5:** transformers (Tracr known answers first, then the longest-cycle networks).

## 6. How to run what exists

- **SPD on an AlgZoo RNN:**
  `OMP_NUM_THREADS=1 python3 scripts/glass/units/u19c_m85.py <target> <original|shuffled> 30000 512`
  takes about 25 minutes on CPU and saves `results/glass/units/u19c_model_<target>_<variant>_30000_512.pt`.
- **Triggers:** `python3 scripts/glass/units/explain_triggers.py <target> original "" <depth>`.
- **Rule-gated construction:** `python3 scripts/glass/units/rule_construction.py <target> original <depth>`.
- **Planted known answer:** `python3 scripts/glass/units/u19b_rnn.py 20 40000 3e-3 1.0 1.0 1e-3 512`, about 35 min.
