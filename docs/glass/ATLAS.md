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
   reproduces the network's decisions above the task rule;
7. **control:** the same atlas built for the network with shuffled weights.
   - A shuffled network also decomposes sparsely (U19(c) control: masked fidelity 0.97), so decomposability alone
     proves nothing.
   - Claims about learned structure rest on what differs from the control: task-related triggers, and effects that
     follow the task.

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
| 7 | Playground: one input, per-step mechanisms switched on by triggers vs needed by importance, network vs rebuilt answer | `atlas.play`, `POST /api/catalog/<t>/atlas_play`, catalog (A10) | `atlas_play_test.py` | built |
| 6 | Transformer support (W_E, per-layer Q/K/V/O, W_U; rows = (sample, position)) | `lab/atlas.py` `TransformerTarget` (A5) | Planted transformer that copies the previous token (`scripts/glass/atlas_tf_test.py`) | built; known-answer run in progress |

**Rules for building:**
- `CLAUDE.md` applies: a CHANGES.md entry before code in `lab/`, a test that fails before and passes after,
  known-answer tests before real runs, no thresholds fitted to the results they judge.
- Experiments only when a specific build step needs an answer, and each says which step.
- Try alternatives quickly when a component falls short: different feature sets, tree depths, an LLM writing
  triggers, more SPD steps. Record each try in DIARY with its number.

## 4. Status (keep current)

- Nov 19, 04:06 ET (clock):
  - **Built and tested:**
    - A1: SPD engine, triggers, rule-gated fidelity.
    - A2: effects and specificity.
    - A4: certificate and catalog.
    - A5: transformer support (code done; known-answer test pending its SPD).
    - A6: GPU/CPU jobs.
    - A7: task-relative effects.
    - A8: findings with a control comparison.
    - A9: float32 SPD (validation running).
    - A10: playground.
    - A11: atlas comparison.
    - Readable trigger summaries.
  - **M8,5 (design case):**
    - 22 exact mechanisms, 11 with trusted triggers.
    - Rebuilt network: 92.5% of decisions, against the task rule's 87.9%.
    - Two functional groups (13 mechanisms) against 0 in the shuffled control.
    - In the catalog, with a playground.
  - **Running:**
    - Colab "atlas":
      - CPU queue: argmedian 8_5, 8_7, 16_5, 16_7, M8,8, shuffled argmedian 16_5.
      - GPU queues: planted transformer → lc_3_4; ARC's M16,10 → its shuffled control; lc_6_5 → zoo M16,10 seed 0.
      - Outputs go to `/content/spd_out2`.
    - Locally: planted RNN in float32 (the A9 known answer) and lc_4_5 in float32.
  - **A3** (held-out networks) is pre-registered in DIARY. It is judged when those SPDs finish.

## 5. Next (in order)

1. As SPD outputs land:
   - download them to `results/glass/atlas/spd/`;
   - build atlases with controls (`python3 lab/atlas.py T SPD 4 --control-spd SHUF`);
   - judge A3 against its pre-registered criteria;
   - run the A5 known-answer test (`atlas_tf_test.py`) and the A9 test (planted RNN float32 through `atlas_test.py`).
2. If A3 falls short on a criterion, improve that component and try alternatives: features, depth, gating policy, SPD steps.
3. Run `compare_atlases` on ARC's M16,10 vs zoo M16,10 seed 0 (do mechanisms recur across training runs?).
4. Transformers: atlases for lc_3_4 … lc_8_6, with controls.
5. Larger: ARC's 2nd argmax M32,10 (1,344 params) on the GPU.
6. Write the atlas results up for Shreyas: what works, on which networks, and the findings.

## 6. How to run what exists

- **Decompose (SPD):**
  `OMP_NUM_THREADS=1 python3 scripts/glass/atlas_spd_job.py OUTDIR STEPS BATCH cpu|cuda target[:variant] ...`
  - Targets: AlgZoo RNN ids, `lc_*` transformers, `planted_rnn`, `planted_transformer`.
  - Writes `OUTDIR/spd_<target>_<variant>_<steps>_<batch>.pt`; float32 since A9.
  - Locally about 25–90 min per network on one core. On Colab, use the CPU cores (one process each); the GPU only helps
    for a single process.
- **Atlas, with its shuffled control:**
  `python3 lab/atlas.py TARGET SPD.pt 4 --control-spd SPD_shuffled.pt`
  - Writes `results/glass/atlas/TARGET_d4.{json,md}` and `TARGET_shuffled_d4.json`, and prints the findings.
  - The catalog picks up `TARGET_d4.json` automatically.
- **Tests:** `scripts/glass/atlas_test.py` (A1, A2), `atlas_role_test.py` (A7), `atlas_findings_test.py` (A8),
  `atlas_tf_test.py PLANTED_TF.pt` (A5), `atlas_catalog_test.py` (A4).
- **Colab:** session `atlas` (`colab sessions`).
  - Code: `/content/repo` (upload a tarball of `lab/` and `scripts/glass/`).
  - Outputs: `/content/spd_out2`; logs: `/content/atlas_logs2`.
  - Status: `colab exec -s atlas -f /tmp/colab_atlas_status2.py`. That script lists the logs and outputs; recreate
    it if missing.
