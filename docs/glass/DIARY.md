# Glass Networks: diary

Every step, decision, result and failure, with the reasoning. Times are ET, Nov 17, 2026 unless noted. The newest
entry is at the bottom. Its companion is `docs/glass/CHECKLIST.md` (what to do next).

## Background (how we got here; full detail in the conversation summaries and `docs/V2_CONSTRUCTOR_PLAN.md`)

- Literature deep dive (Pitfalls paper and its authors, HYVE, InterpAgent, MechEvalAgent, Arithmetic in the Wild,
  causal abstraction critiques, compact proofs, MIPS, AlgZoo, SPD, weight-sparse transformers, attribution graphs).
- Conclusions that drive the build:
  - The open seat: a runnable construction of a whole network, certified by an independent causal judge, published
    for others to use.
  - "Proven accurate" isn't possible beyond tiny nets. A measured certificate is: behaviour; inside-the-network
    interchange; the worst slice; the network's own errors; controls; the unexplained remainder.
  - Coverage of the whole model is what makes a construction useful (prediction, proofs, audits, edits).
    One-operation explanations unlock little.
  - Whole-model understanding is currently solved only up to tens of parameters (ARC, AlgZoo). So the path is a
    ladder of whole models.
- 19:51: branch `constructor-v2` cut from `constructor-attempt` at `d6b706ff`; plan pushed (`c1dd25a0`).

## Log

- **20:16** Shreyas: go. Name the project Glass Networks; implement everything first; compare with the Pitfalls
  system; document everything; run on Colab with automatic pushes; build the catalog/playground in parallel;
  updates every 10–15 minutes, check-ins every 9. Four Luminal keys were given.
- **20:17** Keys written to `lab/.env` (gitignored; `git check-ignore` confirms). All four answered a live call.
- **20:18** Branch renamed to `glass-networks` and pushed. Deleting the old `constructor-v2` branch failed twice
  over git (the remote hung up) and once over the GitHub API (the proxy forbids writes). Left in place; harmless.
- **20:19** Colab sign-in is still valid (`colab sessions`: no active sessions).
- **20:20** Tried to add a repo deploy key so Colab can push by itself. Blocked by the permission system. Decision:
  don't work around it. Asked Shreyas for a fine-grained token (this repo only, contents read/write). Meanwhile,
  relay: Colab writes to disk, the workspace pushes at check-ins.
- **20:22** AlgZoo facts checked by reading its code:
  - inputs are `torch.randn` (standard normal);
  - the RNN is `nn.RNN(1, h, relu, bias=False)` + `Linear(h, n, bias=False)` on the final state;
  - checkpoints download over plain https from `storage.googleapis.com/arc-ml-public/...`.
  Decision: convert the checkpoints to numpy once (`build_algzoo.py`) instead of adding torch to the lab, so the
  judge and reader keep their one numpy code path.
- **20:26** Shreyas asked for a checklist/instruction file plus this diary, kept current to within 10 minutes. Created
  both.
- **20:30** Item 1 built.
  - `nets.rnn_relu` (forward + sites h1..hT, one stage per step).
  - `lab/targets/build_algzoo.py`. No `alg_zoo` package is needed: the seed table is copied, and checkpoints are
    fetched over https.
  - Six targets: `zoo_2nd_argmax_{2_2, 4_3, 4_4, 8_5, 8_8, 16_10}`.
  - The top rung is ARC's exact challenge checkpoint (`one_layer_16_hidden_94_acc_2nd_argmax.pth`), not the zoo's
    16×10 entry. Reason: our result should be about the very network ARC reports on.
  - numpy vs torch: identical (max error 0.0).
  - Accuracy on 8,000 N(0,1) inputs:

    | Target | Accuracy |
    |---|---|
    | M2,2 | 0.9998 |
    | M4,3 | 0.981 |
    | M4,4 | 0.601 |
    | M8,5 | 0.879 |
    | M8,8 | 0.459 |
    | M16,10 | 0.953 (matches ARC's ~95%) |

  - Controls on M16,10: permuted 0.953 (same function, as it should be); shuffled 0.098 and untrained 0.110 (chance
    is 0.1).
  - Low-accuracy models (M4,4, M8,8) are kept. The job is explaining the network, and their many mistakes exercise
    the error test.
- **20:33** Item 0: `scripts/glass/baseline.py --write` started with the *pre-change* judge in memory (it imports
  the judge at start). It re-judges all 27 saved non-LM constructions. LM constructions are skipped: GPT-2 isn't
  cached here, and `lmjudge` only calls `verdict`/`render`, which keep working without the new fields.
- **20:40** Items 2–4 written in `lab/judge.py`:
  - `certificate()`: joint substitution; per-decision slices plus a near-tie slice; the network's own mistakes vs
    labels; the disagreement search (continuous inputs only; only inputs where the network's decision is clear, its
    margin above the 10th percentile, count).
  - `coverage()`: the coverage line and the unexplained list.
  - `verdict()` adds three construction-level requirements: joint ≥ 90% when 2+ sites are named; worst slice ≥ 80%;
    mistakes reproduced ≥ 50% when there are ≥ 30.
  - The search is informational only. Reason: an adversarial search will always find something; its job is to hand
    the reader counterexamples.
  - The constructor's prompt gained a paragraph "6." describing these tests.
- **20:44** Tests on hand-written constructions:
  - (a) An exact copy of M2,2's own computation: VERIFIED AND COMPLETE; joint 100%; nothing unexplained.
  - (b) The *task* algorithm (true 2nd argmax as one variable at h10) on M16,10: behaviour 95.5%, but the error test
    fails as designed. It reproduces 0% of the network's 182 mistakes and is right on 100% of them, so the judge says
    "it explains the task, not this network". The search found 52 clear disagreements from 200 starts (26%, against
    2% on the starting inputs).
- **20:43** **A judge flaw found and fixed.** The exact copy (a) first failed the edit test at h2 (65.9%). Cause:
  edited ReLU states are often all zero, so the outputs tie exactly, and the decision was set by ±1e-13 floating-point
  noise. Fix: `decide()` rounds outputs to 1e-9 before argmax. After: 100%. This can shift old numbers only where an
  exact tie was decided by noise; item 0 will show whether any did.
- **20:38** Smoke test: a 2-step constructor run on M2,2, live. The reader sees the weights, sites and inputs, and
  runs code. The pipeline works on `rnn_relu` targets unchanged. The run folder was deleted afterwards (a test, not a
  result).
- **20:41** **A structural flaw found: a construction can just copy the weights.** The smoke test's reader
  immediately printed all 10 weights of M2,2. My own exact-copy construction had already shown that writing the
  weights out and redoing the matrix products passes *every* judge test. On tiny networks that is the easiest route,
  and it explains nothing. The rule was only stated in the prompt ("not a copy of the network").
  - Fix: `copy_check()` in the judge, as test 7, "not a copy". It counts the numeric literals in the file and the
    share of the network's weights (|w| ≥ 0.05) that some literal reproduces within 1%. It's flagged when there are
    at least half as many literals as parameters *and* they reproduce at least half the weights. A flag fails the
    verdict.
  - Checked: the exact copy of M2,2 is flagged (15 literals, 100% of weights); the task construction on M16,10 isn't
    (4 literals, 1%).
  - Reasoning for these thresholds: an honest construction needs few constants, because the judge's affine alignment
    absorbs scales. So a high literal count combined with reproduced weights is the signature of a copy.
  - The prompt gained rule "7. Not a copy".
- **20:45** Colab runtime `glass1` created: CPU, high-RAM (8 CPUs, 50 GB, Python 3.13, numpy/scipy/sklearn/torch
  present). No GPU: the networks are tiny and the LLM is remote, so a GPU would only burn units.
- **20:46** Driver written:
  - `scripts/glass/run_ladder.py`: a queue of 18 jobs (6 networks × main/prior-only/shuffled), 4 workers with one
    key each, state in `results/glass/queue.json`, resume via `--resume <run id>`, up to 3 attempts per job.
  - `scripts/glass/results.py`: generates `results/glass/RESULTS.md`.
  - `scripts/glass/colab_job.sh`: the queue plus a 2-minute sync loop. It always writes a tar snapshot, and
    commits and pushes too when a token is configured.
  - `colab_setup.py`, `colab_start.py`, `colab_status.py`.
- **20:47** Baseline written (`docs/glass/baseline.json`) with the pre-change judge: 25 constructions (27 minus 2
  whose targets aren't buildable here). The check with the new judge is running.
- **20:58** Item 0 result: all 25 saved constructions re-judged with the new judge (joint, worst slice, errors,
  search, tie fix; the copy check came later). **24 unchanged in every recorded number; 1 changed:**
  `modmul_p113/20261113-190435-ec84` goes from VERIFIED AND COMPLETE to NOT VERIFIED.
  - Reason: the new worst-slice test. On the 58 test inputs where the network answers 112, the construction
    agrees on **0.0%**, and its 97.3% average hid that.
  - This is a real systematic hole in a construction the old judge called complete, and exactly the failure item 3
    was built to catch.
  - The tie fix changed nothing in the baseline. Output: `docs/glass/baseline_check_after_items_2-4.txt`.
- **20:58** Pitfalls comparison written: `docs/glass/COMPARISON_PITFALLS.md`.
- **21:00** Colab:
  - `colab exec` of the setup script hung twice (a known problem with long execs). Fix: launch the clone detached
    and poll. Clone OK (352 MB).
  - `lab/.env` uploaded; dry run on Colab lists 18 jobs; the judge imports there.
  - Queue order changed: ARC's challenge network M16,10 is now third. It's the most informative and the longest, so
    it shouldn't wait for the smaller ones to finish.
- **21:05** Colab job running (started 21:03 ET; Colab's clock reads 01:50 UTC).
  - First launch attempt: "RUNNING" was falsely reported, because `pgrep -f run_ladder.py` matched its own shell
    command. Fixed with a pid file (`/content/glass_job.pid`).
  - The sync loop's snapshot failed whenever `tar` saw a file change while reading it (exit 1, normal while runs
    write). Fixed: the sync loop is now a standalone `colab_sync.sh` that accepts exit 1. It replaced the old loop
    on the live runtime without stopping the runs.
  - The relay (`sync_from_colab.sh`) works: first results snapshot pushed at 21:08.
- **21:07** **First result: M2,2 (10 params) is VERIFIED AND COMPLETE**, blinded, in 12 steps and 2 submissions
  ($0.04).
  - All whole-model tests: 100%.
  - The weight-copy test passes (literals reproduce 20% of the weights).
  - The reader's account: step 1 splits x1 by sign into two ramps; step 2 compares x2 with a sign-dependent
    piecewise-linear "hurdle" (one rectifier fires per class); the readout is an almost antisymmetric pair, so the
    decision is `ev_above > 4.35·ev_below`.
  - This matches the case ARC reports as fully understood, which is the calibration rung we expected to pass.
- **21:10** An untracked `docs/GLASS_CHECKLIST.md` (dated 20:23) appeared in the workspace, a near-duplicate of the
  checklist. I didn't create it in this flow, and deleting it was blocked by the permission system. It's left
  uncommitted. **The canonical files are `docs/glass/CHECKLIST.md` and `docs/glass/DIARY.md`.** One idea from it is
  adopted: say "Glass Networks" in the README and START_HERE.
- **21:15** Catalog built (task D):
  - `lab/catalog.py` and `lab/static/catalog.html`; routes `/catalog`, `/api/catalog`,
    `/api/catalog/<target>/<run>/play`.
  - One card per network: the best run (original weights only, so controls are excluded), its description, the
    certificate, the "not explained" list and the failures.
  - Playground: one input through the network and the construction side by side, plus an intervention. Take one
    variable from another input; the construction predicts the network's decision, and the network shows what it
    does.
  - `scripts/glass/rejudge.py` re-judges saved constructions with the current judge (`report_current.*`; the
    original reports are kept), so catalog cards show today's certificate.
- **21:18** End-to-end browser test (headless Chrome via Playwright, `/tmp/e2e/test_catalog.py`): 15 entries
  load; a random input → same decision; an intervention on M2,2's `pos1` → prediction right; no JS errors.
  Screenshot reviewed.
- **21:19** **The copy check over-flagged hand-written nets.** The Boolean networks have weights 1, -1, 2, 0.5, and
  any honest account writes those numbers, so all four Boolean constructions were flagged as copies.
  - Fix: whole and half numbers are excluded from the weights being matched. Trained networks almost never have
    such exact values.
  - After: the Boolean constructions are complete again, and the exact copy of M2,2 is still flagged.
  - The change doesn't affect the AlgZoo runs on Colab (their weights aren't round numbers).
- **Correction on times.** Entries from "20:30" to "21:19" above carry times I estimated; they ran ahead of the clock.
  The real span was about 20:28–20:58 ET. From here on, times come from `date`. (The Colab job started at
  01:50 UTC = 20:50 ET, not 21:03.)
- **21:00 (real)** Check-in. Colab job alive. Queue: 2 done, 4 running (M4,3, M8,5, M8,8, M16,10). **M4,4 (36
  params) came back VERIFIED AND COMPLETE**: 100% on every whole-model test, including reproducing all 1,608 of the
  network's mistakes (it's a 60%-accurate network).
- **21:05** **But the M4,4 construction is a rescaled copy.**
  - Its own description says the state is updated as `s_t = relu(x_t*S + s_(t-1) @ A)` with "a dense recurrent
    gain matrix A", in "units of their own input gain". That's W_h with each unit rescaled.
  - The value-match copy check missed it: the literals reproduce only 23% of the raw weights, because rescaling
    changes every number.
  - Fix: a second copy rule. The largest array literal in the file (numbers in one list, nested or not) is compared
    with the network's largest weight tensor; a table with at least half as many entries is a copy up to rescaling.
    This rule is invariant to scaling and permutation.
  - After: M4,4's construction is rejected (complete=False). M2,2's (scalar constants that each mean something)
    and the Boolean and Tracr constructions still pass. The prompt's rule 7 now names both checks.
- **Decision:** don't restart the live queue for this (it would interrupt M16,10). The fixed judge reaches the
  running constructors at their next submission once Colab pulls it (each submission runs `judge.py` fresh). After
  the first pass, a second pass re-runs the main job on every network whose best construction isn't certified by
  the current judge (M4,4 at least).
- `results.py` now prefers `report_current.json` (the current judge's verdict) over the run's own report.
- **21:01** Pushing the judge fix to Colab: `git pull` there aborted, because the relay commits run folders that exist on Colab as untracked files. Nothing was lost. New `colab_update_code.py` checks out only code paths (`lab/*.py`, `lab/static`, `lab/targets`, `scripts/glass`). Colab now has the table-copy rule (c98ef8ee).
- **21:03** **M4,3 (32 params): VERIFIED AND COMPLETE.** 26 steps, 6 submissions, $0.13.
  - Behaviour 99.2%; all sites at once 99.1%; worst slice 98.1%; 79.7% of the network's 74 mistakes reproduced; the
    search found 4 clear disagreements in 200 starts.
  - The account: the 4 units form two signed channels (each a positive and a negative part). The cell is
    mirror-symmetric. The signed part rotates about 120° per step. The readout compares the ratio of the two channel
    magnitudes with two thresholds.
  - ARC lists M4,3 as "more or less" understood, so this is the second calibration rung, passed.
- The table-copy rule first flagged M4,3: its symmetric mixing table has 8 numbers against W_h's 16, and the bar was
  50%. That was wrong. Using the symmetry to halve the numbers *is* explanation. Bar raised to 90% (a table about as
  large as a whole weight tensor). After: M2,2 and M4,3 are complete, M4,4 is still rejected. Noted in the code:
  this is a heuristic and could be gamed by splitting a table; the reader isn't adversarial, and a semantic review
  (a person, or a model reading the description) is the stronger check later.
- **21:04** M16,10 mid-run (66 of 80 steps): submission 7 verified 8 of 10 stages; submission 8 tried the task's algorithm and the judge rejected it ("explains the task, not this network": right on 100% of the network's 182 mistakes). Decision: pass 2 *continues* this run to 160 steps (same run, notes kept) instead of a fresh 80-step run, because 80 steps is clearly short for the 432-parameter network. The other networks that fail get a fresh run.
- **21:09** Every saved non-LM construction re-judged with the current judge (`docs/glass/rejudge_all.txt`); `report_current.*` committed next to each (the originals are kept). Catalog cards now show today's certificates. Changes from the original verdicts: modmul ec84 is no longer complete (worst slice, as in item 0); M4,4 is a copy. Everything else keeps its verdict.
- **21:11** **M16,10, pass 1 ended (80 steps, 11 submissions, $0.38): not explained.**
  - The run delivered submission 7, which is an exact re-implementation. The reader labelled it "diagnostic exact
    re-implementation ... before writing a semantic construction": 439 literals, every weight. The copy check
    rejects it.
  - Its real attempts (submissions 3–6, 8, 10, 11) reach at most 1 of 10 stages. They match 95.5% of decisions, but
    they're the task's algorithm: right on the network's mistakes, not reproducing them.
  - **A constructor bug this exposed:** the best submission was chosen on stages and score alone, so a flagged copy
    could become the delivered construction. Fixed in `better()`: a flagged copy never beats a non-copy.
    Pass 2's continuation recomputes the best with this rule.
- **M2,2 prior-only control came back VERIFIED AND COMPLETE** (told the task, no weights, one submission).
  - So on M2,2 the network computes 2nd-argmax-of-two the "obvious" way, and the task's natural decomposition
    already aligns with its insides.
  - Consequence for how we report: M2,2's certification isn't evidence of reading the weights. It shows the judge
    and pipeline work end to end.
  - The M4,4 prior control (59.8% behaviour, 0/4 stages) behaves as a control should. M4,3's prior control is
    pending; it's the one that matters.
