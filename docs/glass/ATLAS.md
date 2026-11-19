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
| 1 | SPD engine (any forward function; repeated calls for recurrent weights) | `lab/spd.py` (A1) | Planted RNN, 5 mechanisms per matrix (U19(b')): PASS. TMS 5-2: structure PASS on 2 seeds (U19(a), (a')) | built, tested |
| 2 | Trigger explainer (shallow trees over task features + neuron states, simplified to 'ON when …' rules; held-out balanced accuracy = confidence) | `lab/atlas.py` (A1) | Planted RNN: PASS (exact planted triggers, confidence ≥ 0.98) | built; M8,5 depth 4: W_x 4/4, W_h 9/11, W_o 2/7 at ≥ 0.9 |
| 3 | Rule-gated construction + fidelity (trusted policy: a trigger switches its mechanism only at confidence ≥ 0.9, otherwise always on) | `lab/atlas.py` (A1) | Planted RNN: PASS (0.997) | built; M8,5 (design case) depth 4: 0.924 vs task rule 0.879 |
| 4 | Effect + causal specificity per mechanism (remove it at one call; output change where needed vs not; flip rate; which output it lowers/raises) | `lab/atlas.py` (A2) | Planted RNN: PASS | built; M8,5: W_h specificity median ×75 (min ×16) |
| 5 | Atlas certificate (JSON + markdown) + catalog card | `lab/atlas.py` render, `lab/catalog.py`, `catalog.html` (A4) | `scripts/glass/atlas_catalog_test.py` (headless Chrome): PASS | built |
| 6 | Transformer support (attention QK/OV matrices as decomposed weights) | to build | Tracr compiled networks (known mechanisms) | not started |

**Rules for building:**
- `CLAUDE.md` applies: a CHANGES.md entry before code in `lab/`, a test that fails before and passes after,
  known-answer tests before real runs, no thresholds fitted to the results they judge.
- Experiments only when a specific build step needs an answer, and each says which step.
- Try alternatives quickly when a component falls short: different feature sets, tree depths, an LLM writing
  triggers, more SPD steps. Record each try in DIARY with its number.

## 4. Status (keep current)

- Nov 19, 03:17 ET (clock):
  - **A1 done** (`lab/spd.py`, `lab/atlas.py`; test `scripts/glass/atlas_test.py` passes).
  - **On M8,5** (the design case; saved SPD model), the depth-4 atlas has:
    - 4 W_x, 11 W_h and 7 W_o mechanisms;
    - triggers at confidence ≥ 0.9 for 4/4 W_x, 9/11 W_h and 2/7 W_o;
    - rule-gated fidelity (trusted policy) 0.924, against the task rule's 0.879.
  - W_o triggers are weak (0.60–0.65); with the trusted policy those mechanisms stay always on.
  - Depth 4 was chosen on M8,5, so it is fixed for the held-out networks.
  - Running: SPD of argmedian 8_5 (held out, `/tmp/u19c_am85.log`) and the U19(c) shuffled control (`/tmp/u19c_shuf.log`).

## 5. Next (in order)

1. **A1: DONE.**
2. **A2: DONE.**
3. **A3:** run the atlas on M8,5, argmedian 8_5 and 16_5, and zoo M16,10 seed 0.
   - Improve triggers by trying alternatives until rule-gated fidelity beats the task rule with most mechanisms at
     confidence ≥ 0.9, or record why not.
4. **A4: DONE.**
5. **A5:** transformers (Tracr known answers first, then the longest-cycle networks).

## 6. How to run what exists

- **SPD on an AlgZoo RNN:**
  `OMP_NUM_THREADS=1 python3 scripts/glass/units/u19c_m85.py <target> <original|shuffled> 30000 512`
  takes about 25 minutes on CPU and saves `results/glass/units/u19c_model_<target>_<variant>_30000_512.pt`.
- **Triggers:** `python3 scripts/glass/units/explain_triggers.py <target> original "" <depth>`.
- **Rule-gated construction:** `python3 scripts/glass/units/rule_construction.py <target> original <depth>`.
- **Planted known answer:** `python3 scripts/glass/units/u19b_rnn.py 20 40000 3e-3 1.0 1.0 1e-3 512`, about 35 min.
