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

- Nov 19, 07:26 ET (clock):
  - **Built and tested:** A1–A29. Each has a CHANGES entry and a test that failed before and passes after. The latest:
    - A26: precise decomposition, minimality 1e-4 plus residual absorption;
    - A28: adaptive decomposition, a ladder of settings with the sparsest precise-enough one chosen on the training half;
    - A29: what each mechanism reads and writes, glossed with unit-map labels.
  - **Held-out verdicts so far:**
    - A18 (default settings): 0 of 6.
    - A27 (A26 settings): 2 of 6. Both 16-wide accurate networks pass every criterion; all 32-wide fail from component saturation.
    - A25 (recurrence across M8,5 training runs): not confirmed. 1 of 4 seeds beats chance; it varies by run.
  - **Development set with A28 (design check):** all 7 pass every criterion. Weak networks keep the sparse default; accurate ones get denser settings.
  - **Running now: A28's held-out test** (pre-registered 07:23; s2 seeds of the six architectures).
    - C = 20 for 16-wide and 40 for 32-wide; ladder 1e-3 / 3e-4 / 1e-4; certificate = decision-checked gate v4.
    - 15 runs on Colab CPU (32-wide plus the control) and 6 locally (16-wide).
  - **Transformers:** the known answer passes. The lc runs at 1e-4 were stopped to free Colab for A28 and need restarting.
  - **Write-up:** `docs/glass/ATLAS_RESULTS.md` (kept current).

## 5. Next (in order)

1. When A28's decompositions land, for each network:
   `python3 lab/atlas.py T x 4 --ladder SPD_1e-3,SPD_3e-4,SPD_1e-4 [--control-spd SHUF_1e-3,SHUF_3e-4,SHUF_1e-4]`.
   Then judge with `python3 scripts/glass/atlas_a3_eval.py --v4 <the six s2 ids>`.
2. If A28 passes: make the adaptive ladder the default pipeline and promote the adaptive atlases to the catalog (`T_d4.json`).
3. Transformers: rerun lc_4_5 / lc_6_5 at 1e-4 (C scaled to d_model), then build their atlases with absorption.
4. Deepen explanations (A29 is a start): unit maps for networks that lack them, so reads/writes carry task meaning.
5. Finish `ATLAS_RESULTS.md` for Shreyas.

## 6. How to run what exists

- **Decompose (SPD):**
  `OMP_NUM_THREADS=1 python3 scripts/glass/atlas_spd_job.py OUTDIR STEPS BATCH cpu|cuda [--coef-min=X] [--C=N] [--tag=T] target[:variant] ...`
  - Targets: AlgZoo RNN ids (seeded zoo networks end in `_s0` … `_s4`), `lc_*` transformers, `planted_rnn`, `planted_transformer`.
  - Writes `OUTDIR/spd_<target>_<variant>_<steps>_<batch>[_tag].pt`; float32 since A9.
  - Locally about 25–90 min per network on one core. On Colab: CPU cores at batch 512 (one process each), the GPU at batch 2048 (it slows down when shared by more than about 4 runs or when the CPU is saturated).
- **Atlas, with its shuffled control:**
  `python3 lab/atlas.py TARGET SPD.pt 4 [--control-spd SPD_shuffled.pt] [--narrate]`
  - Writes `results/glass/atlas/TARGET_d4.{json,md}`, `TARGET_d4_model.pkl` (trees, for the playground) and `TARGET_shuffled_d4.json`, and prints the findings. About a minute per RNN on a free core.
  - The catalog picks up `TARGET_d4.json` automatically (`python3 lab/server.py`, then `/catalog?t=TARGET`).
- **Judge:** `python3 scripts/glass/atlas_a3_eval.py --v3` (development set) or `--v3 --a18` (held-out set A18). `--v2`/no flag give the older versions.
- **Overview of every atlas:** `python3 scripts/glass/atlas_overview.py` → `results/glass/atlas/OVERVIEW.md`.
- **After a presentation-only change:** `python3 scripts/glass/atlas_rerender.py` (findings, glosses, markdown from the saved JSON).
- **Tests** (each fails before its change and passes after): `atlas_test.py` (A1, A2, A12), `atlas_role_test.py` (A7), `atlas_findings_test.py` (A8), `atlas_catalog_test.py` (A4), `atlas_play_test.py` (A10), `atlas_compare_test.py` (A11), `atlas_narrate_test.py` (A13), `atlas_tf_test.py PLANTED_TF.pt` (A5), `atlas_gate_test.py` (A15), `atlas_f3_test.py v3c` (A16), `atlas_resid_test.py` (A17), `atlas_clear_test.py` (A19), `atlas_gloss_test.py` (A20), `atlas_task_test.py` (A21), `atlas_v3_headline_test.py` (A22).
- **Colab:** session `atlas` (`colab sessions`; set `OAUTHLIB_RELAX_TOKEN_SCOPE=1`).
  - Code: `/content/repo` (upload a tarball of `lab/atlas.py lab/spd.py scripts/glass/atlas_spd_job.py` and untar; models a target needs must be in `/content/repo/lab/models`, upload them as a tarball if missing).
  - Start jobs with a shell script launched through `setsid` (`subprocess.Popen(["setsid", "bash", script], start_new_session=True)`), so `colab exec` returns at once.
  - Outputs: `/content/spd_out2`; logs: `/content/atlas_logs2`.
