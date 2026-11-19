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

- Nov 19, 16:08 ET (clock): A38d planted transformer keeps its Q mechanism at step 8,000 (A38c lost it). Waiting on the planted results (about 17:00), then the s1/s2 verdict for A37 (about 17:00), modmul (A40, about 18:00) and the A38d modadd run.
- Nov 19, 15:51 ET (clock): A38/A38c had a translation error (leaky negative masks let dead components rebuild Q; found with the planted transformer, which shows the same Q failure as modadd). Fix A38d: mask value = clamped importance, leaky gradient only. The planted transformer is now the fast test bed (about 50 min on CPU).
- Nov 19, 15:32 ET (clock): fresh adders s3/s4 trained and reserved (100% accurate). A38c's planted RNN shows 5/5/5 at step 32,000 (A38 gave 7 in W_h). GPU: s1/s2 about halfway.
- Nov 19, 15:14 ET (clock): A38c's modadd run keeps query/key mechanisms at step 2,000 (all earlier runs had none); its planted checks are running. A38 keeps the planted transformer passing but still breaks the planted RNN.
- Nov 19, 14:56 ET (clock): A38 (CI-masked term, all matrices at once) breaks the planted RNN known answer (7 W_h mechanisms vs 5): heading for a fail. A38c (one matrix at a time) is running on modadd (GPU) and on both planted known answers (local CPU).
- Nov 19, 14:38 ET (clock): GPU busy with 7 runs (s1/s2 for A37, modmul for A40, the A38 development run). A38 is likely to fail its development test (Q/K still dead at step 4,000); a cheap CPU test bed (planted transformer, C = 160) is running to try other designs quickly.
- Nov 19, 14:21 ET (clock): A39 (6.7× faster gated runs, numbers identical) done. A40 (modular multiplication: the atlas must pick the discrete-log basis; known answer {4, 41, 28, 31, 30} pre-registered) coded and tested. Waiting on GPU runs: s1/s2 for A37 (about 16:20), the A38 development run, then modmul.
- Nov 19, 14:04 ET (clock):
  - A37 (angle vocabulary for modular arithmetic) is built and tested. Held out: two fresh modadd networks (s1, s2; 100% accurate; key frequencies {43, 30, 18, 19, 38} and {4, 44, 10, 28, 9}) are being decomposed on the GPU (about 3 hours).
  - A38 diagnosed: (i') fails on modadd because SPD leaves the attention Q/K with no mechanism (spread over many components; random partial masks never remove them all). A CI-masked reconstruction term is entered and coded, off by default; a development run is next.
- Nov 19, 13:48 ET (clock). Since 11:59:
  - **A35 known answer FAILS narrowly** on modadd_p113 (227k parameters, about 10× the largest network certified before). 71% of trusted Fourier triggers use the network's key frequencies {8, 11, 22, 23, 25}; the pre-registered bar was 80%, chance 9%. By matrix: MLP output 100%, unembedding 93%, embedding 78%, attention values 36–60%.
  - Its certificate passes (ii) v4 1.000, (iii) 0.60, (iv) 33×, (vi) 0.88, and fails (i') mechanisms alone 0.597. 232 mechanisms; the attention Q/K matrices keep none alive.
  - Likely cause, seen after the verdict: the trigger vocabulary has no Fourier features of a and b separately (what the MLP input needs). A fix is a new version judged on a fresh network.
  - A36: feature cache (1.8× faster atlas builds). Model files no longer store the trees' training features (293 MB → 0.5 MB).
- Nov 19, 11:59 ET (clock). Since 09:47:
  - A33 PASS: held-out transformer lc_8_6 passes every criterion with the unchanged pipeline (after the A32 data-split fix).
  - ARC's M16,10 passes with its matched control; the development table under the final pipeline is 6/7.
  - A31 (read/write profiles by unit label) and A34 (always-on rows grouped) done.
  - A35: modadd_p113 (227k parameters, MLP transformer, known Fourier mechanism) adapter built and tested; decomposition running on the Colab GPU (3e-4 and 1e-4, C = 160, batch 1024; due about 13:30); its known-answer verdict is pre-registered.
- Nov 19, 09:47 ET (clock). Since 08:34: development atlases rebuilt with the adaptive pipeline in their main files (catalog/playground show the certified v4 versions); argmedian 16_5 passes every criterion including its matched control (37 vs 0); transformer ladders running on the Colab GPU (lc_3_4/4_5/6_5), lc_8_6 reserved as the held-out transformer.
- Nov 19, 08:34 ET (clock):
  - **MAIN RESULT: A28 passes its pre-registered held-out test.** 5 of 6 accurate networks (95–99.4%, up to 1,376 parameters, three of them 32-wide) pass every criterion, and the shuffled control shows no functional structure (17 vs 0). The write-up for Shreyas is `docs/glass/ATLAS_RESULTS.md`.
  - **The pipeline that passed:**
    - decomposition: an adaptive ladder (minimality 1e-3 / 3e-4 / 1e-4), components C = max(20, ⌈1.25·H⌉), the sparsest rung precise enough on the training half (mechanisms alone and v3 ≥ task rule + 2 SE), with absorption;
    - certificate: v4 (trusted rules that also pass a decision check on training data).
  - **Built and tested:** A1–A30 (each has a CHANGES entry and a test that failed before and passes after; the full atlas test suite passes).
  - **Held-out history:** A18 0/6 → A27 2/6 → A28 5/6. A25 (recurrence across training runs) is not confirmed.
  - **Not yet done:**
    - transformers with the adaptive pipeline (only the known answer passes so far);
    - promoting the adaptive atlases to the catalog defaults;
    - deeper "what it computes" explanations.

## 5. Next (in order)

0. A38c (one matrix at a time): development run on modadd_p113 plus both planted known answers; if all pass, a held-out test on fresh seeds. (A38 all-at-once breaks the planted RNN.)
1. (Built; verdict pending the s1/s2 decompositions) A37: add Fourier features of each input token (cos/sin of k·a and k·b at every position) to the modular-arithmetic vocabulary; judge it on a fresh network with a pre-registered known answer, never on modadd_p113 itself.
2. (Done: A36 + A39, 33 → 4.7 min.) Faster atlas builds.
2b. A40: decompose modmul_p113 (A35 pipeline) and score its pre-registered known answer.
3. More MLP transformers (tracr compiled programs need multi-position outputs in the judge and an adapter with biases).
4. Explanations: unit maps for the A28 networks, so reads/writes (A29) carry task meaning; consider a narrative (A13) for one certified network (ask Shreyas about cost first).

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
  - Local background jobs stop when the session pauses (the workspace stops leftover processes); Colab jobs keep running. Put anything longer than a few minutes on Colab.
  - Start jobs with a shell script launched through `setsid` (`subprocess.Popen(["setsid", "bash", script], start_new_session=True)`), so `colab exec` returns at once.
  - Outputs: `/content/spd_out2`; logs: `/content/atlas_logs2`.
