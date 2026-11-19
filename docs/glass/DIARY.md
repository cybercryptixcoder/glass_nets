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
- **21:12** Check-in. **M4,3 prior-only control: NOT verified (1 of 3 stages; behaviour 98.2%).** The blinded main run explained all 3 stages, and the told-but-weightless reader explains 1. So on M4,3 the certification needs the network itself, not just knowing the task. This is the first rung where reading the network is clearly necessary. Running: M8,5 and M8,8 main (19 and 13 min), their prior controls.
- **21:23** Check-in. Pass-1 main runs all finished; the shuffled controls are running.
  - **M8,8 (136 params, a 46%-accurate network) came back VERIFIED AND COMPLETE, but it's a re-description, not an
    explanation.**
    - Its own account: "a linear reservoir with a clip, nothing more". It's positively homogeneous, with no
      comparisons or running maxima, a real observation about this network.
    - But the file holds 136 constants for 136 parameters: the weights in a per-unit scale, split into 8-number rows.
      That evades both earlier copy rules (value match: 46%; largest table: 8 of 64).
    - Fix, a third rule invariant to rescaling and splitting: **compression.** Constants = non-integer literals other
      than 0.5. A file with ≥ 0.9 constants per network parameter is flagged.
    - Calibration from this session: honest accounts used 0.56 (M4,3) and 0.80 (M2,2) constants per parameter;
      re-descriptions used exactly 1.00 (M8,8, and M16,10's diagnostic copy).
    - The ratio is now printed in every report (test 7), so the catalog shows how much each construction compresses.
  - **M8,5 (112):** not verified, 0/5 stages. Behaviour 88.9%, but it reproduces 0% of the network's mistakes, so it
    explains the task, not this network.
  - **M2,2 shuffled control: VERIFIED AND COMPLETE.** This is not a false positive. The shuffled network is a
    different 10-weight network, and the construction (5 constants) matches *that* network. On a network this
    small, any weights compute something simple enough to explain. The shuffled control can only discriminate on
    larger networks; its results there are pending.
  - **M16,10 prior-only control:** 1/10 stages, 95.5% behaviour, the same as the main run's real attempts. So in
    pass 1, reading the network added nothing on M16,10.
  - Re-judged with the current judge: M2,2 (main, prior, shuffled) complete; M4,3 complete; M4,4, M8,8 and the
    M16,10 delivered file flagged as copies; M8,5 not verified.
- **21:33** Pass 1 finished (18/18 jobs done; Colab job ended 02:29 UTC = 21:29 ET).
  - **Shuffled controls** (re-judged): M4,3 0/3; M8,5 0/5; M8,8 1/8 (partial); M16,10 0/10. These fail, as a
    control should.
  - **M4,4 shuffled came back COMPLETE** with 32 constants: a per-unit-rescaled re-description that sat just under
    the 0.9-constants-per-parameter bar. The bar's denominator was wrong. Rescaling a ReLU unit (and its outgoing
    weights inversely) leaves the function unchanged, so a re-description needs exactly *parameters minus hidden
    units* constants (36 − 4 = 32).
  - Fix: the compression rule now counts constants per **degree of freedom** (parameters minus one scale per
    hidden unit). It flags at 0.9.
  - Effect: the M2,2 *main* construction is now flagged. It has exactly 8 constants for 8 degrees of freedom: the
    weights rescaled, with meaningful names but no compression.
  - **The M2,2 prior-only construction (0 constants) remains complete.** So the compressed explanation of M2,2 is
    the task's own algorithm, and the network's insides align with it.
  - The catalog now admits a certified prior-only construction as an entry (it's checked against the network
    like any other) and says how each entry was written. Shuffled/untrained runs are still excluded (a different
    network).
  - Constants per degree of freedom across this session's constructions: honest/compressed 0.0–0.64; rescaled
    re-descriptions 0.875 (caught by the table rule), 1.0 and 1.06.
- **21:37** Pass 2 started by hand (code updated; the old sync loop was stopped so there's only one). Pass 2 decided its own jobs by re-judging each pass-1 main run on Colab with the current judge: fresh runs for M2,2, M4,4, M8,5 and M8,8; M16,10 continues its pass-1 run to 160 steps (5th in line).
- **21:43** Check-in. **M2,2 pass-2 run: VERIFIED AND COMPLETE in 7 steps, 1 submission ($0.03), and this time compressed.** 0.625 constants per degree of freedom: it states the readout as one ratio (class 1 iff 4.353·N2 > P2) instead of four numbers. The account: a two-channel leaky accumulator (positive and negative parts); each channel decays and is re-driven by the opposite one; the decision compares the channels with the negative one weighted 4.35×. Running: M4,4, M8,5, M8,8 (fresh) and the M16,10 continuation (89 steps so far).
- **21:52** Check-in. **M4,4 pass 2: complete under the old rule, but again a re-description.**
  - Its account (mixing "circulation", homogeneity) is in normalised coordinates: a 4×4 mixing matrix as 4 rows,
    and the readout written as 3 class-difference directions with one entry fixed to 1. That's 27 constants.
  - 27 is exactly the network's true degrees of freedom: 36 parameters − 4 unit scales − 4 readout shift directions
    − 1 overall scale. The decision (argmax) is also invariant to adding one vector to every class's readout column
    and to an overall positive scale, and my degrees-of-freedom count had left those out.
  - Fix: the degrees of freedom now subtract the readout's symmetries for class outputs. The prompt explains it.
  - **With the corrected count, constants per degree of freedom separate cleanly across all 17 AlgZoo
    constructions of this session:** every re-description sits at ≥ 1.0 (1.0–1.6); every compressed account at
    ≤ 0.78 (M4,3 main 0.78, its shuffled 0.70, priors 0.0–0.04, M8,5 0.08).
  - Consequences:
    - M2,2's two blinded runs (1.6 and 1.0) and its shuffled control (1.0) are now flagged. The only certified M2,2
      construction is the prior-only one, which has 0 constants: the network computes 2nd-argmax-of-two the plain
      way.
    - **M4,3 main (0.78) remains the one blinded-run certification**, and its prior control fails.
  - M2,2 note: its minimal exact description has 5 numbers (= its degrees of freedom). Writing them in meaningful
    coordinates is understanding of a kind, but not compression; the 0-constant account is.
- **22:02** Check-in. Pass-2 fresh runs finished. Only the M16,10 continuation is still running (148/160 steps).
  - **M8,8 (pass 2): not explained, and the reader argues no compression exists.**
    - Five of its 9 submissions were exact re-descriptions (8/8 stages, 99.5–100%, all flagged at 1.06–1.14
      constants per degree of freedom). Its compressed attempts reached 0/8 stages and 13–27% behaviour.
    - Its closing argument: the network is "a generic 8-dim clipped linear system with 136 − 8 − 8 = 120 genuinely
      free constants". That's a substantive claim about this 46%-accurate network, consistent with pass 1's account
      ("a linear reservoir with a clip, nothing more").
    - The delivered file is its best non-copy (0/8); the copies stay in the run folder as submissions.
  - **M8,5 (pass 2): not explained.**
    - Best compressed attempts: 85–89% behaviour, 0/5 stages, reproducing 23–49% of the network's mistakes. The
      ideal 2nd-largest rule gets 89.1%, the ceiling for a task-exact rule, since the network is 88.9% accurate.
    - The reader: the 11% residual is "the network's own lossy rectified-difference top-2 tracking", which it
      couldn't reduce to a compact closed form.
    - Submission 1 was again an exact re-description (5/5 stages, flagged).
  - **Pattern across the ladder:** with the copy rules in place, the reader either finds a compressed account (M4,3)
    or produces an exact re-description (M2,2, M4,4, M8,5, M8,8, M16,10). There's no middle ground yet: a
    compressed construction that verifies some stages.
- **22:13** **M16,10 continuation finished** (160 steps, 24 submissions, $0.78 in total): not explained. Best compressed attempts: 97.0% behaviour, stage 1 of 10 verified, 45–50% of the network's mistakes reproduced (the 50% bar is close). The reader's closing note: no compact rule reproduces the network's near-tie mistakes, since in the top-gap < 0.02 regime the network still picks the true second max 78% of the time. All runs done: 22 runs, $3.12. Final report written: `docs/glass/REPORT.md`. Colab runtime idle; asked Shreyas whether to stop it.
- **22:46** Shreyas: stop the runtime. `colab stop -s glass1` done; no sessions remain. All data was already on GitHub. Check-ins end; one already-scheduled check-in fires at 22:54 and finds nothing to do (cancelling it was blocked by the permission system).
- **23:04** Shreyas set round 2's rules: compare M4,3 with ARC's notes; find our real mistakes and learnings with
  confidence; no patchwork fixes; find a target we can claim with no asterisks; re-run the ARC ladder; plan
  everything first, research now, wait for approval.
- **23:10–23:30** Research and trace analysis (no building).
  - ARC already has M4,3 "more-or-less" understood: the same checkpoint, an analysis using approximate symmetries,
    in Drive notes by Robinson, Hilton and Chen. Our M4,3 is a rediscovery, not a discovery. Opening the Drive notes
    needed a permission that went unanswered.
  - Community work found: M16,10 (LessWrong comments: running max / second max / spiral; a population-fitted
    estimator post) and argmedian M4,3 (xabier.me). No public work on the longest-cycle or median families.
  - Longest-cycle transformers load and run: 2-layer, attention-only, 108–656 params, 81–95% accurate, with input
    sets small enough to test exhaustively. Chosen as the no-asterisk target.
  - Trace findings E1–E8 (with file evidence) are in PLAN_ROUND2.md. The key ones:
    - copies everywhere;
    - judge rules changed mid-run;
    - the degrees-of-freedom rule alone suffices;
    - on M16,10 the reader stopped testing internal stages after submission 16 and spent about 80 steps fitting
      mistake tables;
    - M8,5's 0.98/0.89/0.84/0.79 partial progress was invisible behind "0/5";
    - M4,3's success came from compressing a working copy.
- **23:38** Shreyas approved round 2. Step 1: opening ARC's Drive notes was again blocked by an unanswered permission prompt (not worked around); marked pending. Step 2: `docs/glass/LEARNINGS.md` frozen (E1–E8).
- **23:47** Step 3 fixes implemented.
  - **F1:** `copy_check` is now the single compression rule. Degrees of freedom now also cover transformers: per
    head QK and OV products (k² each) and the residual basis (d²).
  - **F2:** `JUDGE_VERSION = "glass-2"` is stamped on every report and every run's config.
  - **F3 and F4** (constructor, `GLASS_READER=v2`, the default; `v1` = round 1's reader, kept for the A/B):
    - the v2 prompt adds a stage-by-stage working order, "don't trade internal stages for output scores", and the
      copy-then-compress route;
    - v2 reports open with a STAGE TABLE and a "FIX FIRST" line;
    - v2 picks the best submission by stages verified from the first, then behaviour.
  - Rule 7 in the prompt, which both readers see, now states the single compression rule.
  - **Regression check:** every saved non-LM construction re-judged with glass-2
    (`docs/glass/rejudge_glass2.txt`). Every verdict is identical to the round-1 final verdicts. F1 removed rules
    without changing any answer, as E3 predicted.
- **23:49** Step 5a/b built: `lab/targets/build_longest_cycle.py`.
  - Four AlgZoo longest-cycle transformers are converted to the lab's existing `transformer` kind (bidirectional,
    attention-only, read at the last position). No new network code was needed.
  - Checked against ARC's own torch forward on **every possible input**: max difference ≤ 9e-13. Labels match
    ARC's `task_longest_cycle` exactly.

    | Target | Params | Accuracy (exhaustive) | Inputs |
    |---|---|---|---|
    | lc_3_4 | 108 | 90.2% | 256 |
    | lc_4_5 | 188 | 81.0% | 3,125 |
    | lc_6_5 | 378 | 94.7% | 3,125 |
    | lc_8_6 | 656 | 85.4% | 46,656 |

  - The judge runs on them (a task-algorithm construction on lc_3_4: 85.9% behaviour, rejected). A 2-step v2
    smoke run works and records reader v2 and judge glass-2 in its config (deleted afterwards).
- Round-2 queue (`run_ladder.py --round2`, state in `results/glass/round2/`): 40 jobs.
  - Longest-cycle: 2 blinded v2 runs per network (80 steps for the two small ones, 120 for the two large ones), plus
    prior-only and shuffled controls.
  - The ladder A/B: reader v1 vs v2, two runs each per network, same budgets (80; M16,10 160).
  - The longest-cycle runs are interleaved early.
  - Round 1's controls are reused (the frozen judge gives them the same verdicts).
- **23:52** Colab runtime `glass2` created (CPU, high-RAM); detached clone; `.env` uploaded; dry run lists 40 jobs; round-2 job started 04:51 UTC (23:51 ET) with 4 workers. Judge glass-2 is frozen from here until round 2 ends.
- **23:55** Step 1 done: the Drive connector became available and read all three ARC notes. M4,3 = independent rediscovery of Robinson's symmetric reduction (doc `M43_VS_ARC.md`, E9). New, checked by code: the reduced recurrence is a 120° rotation; the exact reduction loses no accuracy.
- **00:04** Check-in: 4 done, 4 running.
  - lc_3_4 maina: glass-2 COMPLETE, but it is a weight copy stored as integers × 1/10000, which gets past F1 (E10). Not a claim; the judge stays frozen; re-judge everything after the round with the fix.
  - M4,3 v2a: COMPLETE in 1 submission, 0.78 constants per dof, i.e. at the claim line. It is a re-description in a P,D,Q,E basis, not a compression; the account matches E9.
  - M2,2: v1 and v2 both COMPLETE.
- **00:15** Check-in: 6 done, 4 running (lc_3_4 mainb, lc_4_5 maina/b, lc_6_5 a/b). M4,3 v1a: COMPLETE, 5 submissions, 0.78 constants per dof, the same result as v2a, which took 1 submission. No new lc results.
- **00:26** Check-in: 8 done, 4 running.
  - lc_4_5 mainb and lc_6_5 maina are glass-2 COMPLETE, but both are E10 copies (integer mantissas over 1000, or times S/T scales).
  - lc_4_5 does carry a partial account: layer 0 is near-hard copying (86% of rows have max weight > 0.9), and layer 1 at the readout is a soft max/min selector over the keys through a class table G. That is a lead to follow, not a claim.
  - lc_3_4 mainb: not verified (0/3, 52%).
  - So far 3 of 3 lc COMPLETEs are E10 copies, all from the v2 reader. No claim.
- **01:03** HANDOFF. Shreyas (01:02): keep the Colab job running and hand off to a new session. Status: 12/40 done, 4 running (lc_6_5 mainb, M8,5 v2a, lc_8_6 a/b). Relayed and pushed. Wrote `docs/glass/HANDOFF.md`; START_HERE and CHECKLIST point to it. Option A (glass-2 frozen) is in force. No further check-ins are scheduled from this session.
- **01:25** Check-in (Shreyas asked): 14 of 40 done; 4 running (lc_8_6 a/b, M8,8 v1a/v2a). Relayed.
  - New glass-2 COMPLETEs, checked by hand:
    - lc_4_5 maina: 0.81 constants per dof, mostly a re-description (above the 0.78 claim line);
    - M4,4 v1a: 0.89, near-copy;
    - M8,5 v1a: 0 constants, an E10 copy;
    - M8,5 v2a: 0.87, near-copy.
  - Not verified: lc_6_5 mainb, M4,4 v2a. Still no claim.
- **01:43** Check-in (Shreyas asked): 17 of 40 done; 4 running (M8,8 v1a/v2a, M16,10 v2a, M2,2 v1b). Relayed and pushed.
  - New glass-2 COMPLETEs, checked by hand:
    - lc_8_6 maina: 0.88 constants per dof, a near-copy (288 decimals);
    - M16,10 v1a: 0 constants with a scale factor, an E10 copy.
  - lc_8_6 mainb: not verified (1/3, 92.4%).
  - All 4 lc targets have finished both main runs. No run meets the claim criterion.
- **03:59** Sync (Shreyas asked); status above relayed.
- **04:02** ROUND 2 FINISHED: 40/40; all 757 run files and 70 logs synced (counts match Colab). Wrote REPORT_ROUND2.md (A/B: v2 no better, so revert F3/F4; no lc claim; shuffled lc_3_4 passed, so the judge is too loose) and E11. Coding is left for the next agent.
- **09:51** Review with Shreyas. Retracted the 'v2 no better' reading and the proposed glass-3 rule (both patchwork). Wrote the corrections in REPORT_ROUND2, E12 (how it happened, the binding change rule), CHANGES.md (ledger) and the CHECKLIST rule.
- **10:07** Whole-project review written (`docs/PROJECT_REVIEW.md`) at Shreyas's request: history, parts, experiments, the core problem (no valid 'is it an explanation' measure), why we circle, an audit of earlier decisions, the build routine, and his decisions. Nothing run.
- **10:42** Diagnosed failures above 32 parameters from the transcripts (DIAGNOSIS_ABOVE_32.md). Researched ARC's evaluation (no harness; surprise accounting; MSE vs sampling) and wrote EVIDENCE_LADDER.md. Wrote INTENT.md (v2: tolerance-based certificate, information versus copy frontier, feedback on errors, reader v1 default). Added CLAUDE.md (binding routine) and pointers. Deeper Pitfalls analysis. No code changed; nothing run.
- **11:10 PRE-REGISTRATION: rebuild step 1 (proving the information measure).** Written before any measurement code.
  - **Measure.**
    - B = bits of the construction file, measured as the raw LZMA (preset 9e) size of its code after removing
      DESCRIPTION, ALIGN and comments (ast round-trip).
    - F = agreement with the network's decisions on the target's own inputs.
    - C(F) = bits of the cheapest *copy* reaching fidelity ≥ F. Copies are the weights quantised to k = 1..16 bits:
      (i) uniform per tensor, as integers over a scale; (ii) a 2^k-value codebook plus indices. Each is written as a
      file with a fixed generic forward pass and measured the same way.
    - Score S = B / C(F).
  - **Must-fail (each S ≥ 0.8):**
    - saved copies: lc_3_4 maina, lc_4_5 mainb, lc_6_5 maina, M8,5 v1a, M8,8 v1b, M16,10 v1a, M16,10 v2a;
    - saved near-copies: lc_4_5 maina, lc_8_6 maina, M4,4 v1a, M8,5 v2a, M16,10 v1b;
    - the shuffled-network account (lc_3_4 shuffled, judged against its shuffled network);
    - new disguises written now, applied to M8,5, lc_6_5 and M16,10:
      - D1 float32 bytes in base64;
      - D2 6-significant-digit decimals;
      - D3 16-bit integers over a scale;
      - D4 a 256-value codebook plus indices;
      - D5 weights sorted, plus a permutation;
      - D6 zlib-compressed float16 bytes in base64.
  - **Must-pass (each S ≤ 0.5):**
    - M2,2 prior-only account (round 1);
    - M4,3 round-1 account (c928);
    - the lc_8_6 run-b account (the census rule).
  - **Reported but not labelled:** the round-2 M4,3 files (0.78).
  - **Decision:**
    - If both conditions hold, the measure goes forward to step 2.
    - If either fails, the idea is wrong as stated: report it and return to the intent. Do NOT tune the 0.8/0.5
      lines.
- **11:35 RESULT: rebuild step 1. The pre-registered criterion is NOT met** (`docs/glass/info_measure_step1.txt`; script
  `scripts/glass/info_measure.py`). One translation error fixed along the way: `load_construction` returns a tuple.
  - **Worked:**
    - all 18 new disguises (D1–D6 on three networks) scored S = 0.91–1.72;
    - 12 of 13 saved copies and near-copies scored ≥ 0.8;
    - the shuffled-network account scored 1.09;
    - the lc_8_6 census rule scored 0.35, clearly separated.
  - **Failed:**
    - M2,2's account scored 0.66 and M4,3's scored 1.39 (must be ≤ 0.5);
    - the lc_4_5 mainb copy scored 0.77 (must be ≥ 0.8).
  - **Why** (diagnosed, no tuning):
    - On tiny networks a copy is mostly program scaffolding. The generic forward pass alone is 1,352 bits of
      M2,2's 2,240-bit copy, so file length measures how verbose the code is, not how much is understood.
    - More fundamentally, M4,3's real account isn't shorter than its 32 weights. It reorganises them (the symmetry)
      rather than discarding them.
  - **Conclusion:** "explanation = fewer bits than a copy" is wrong as a general definition. It works as a signal on
    bigger networks (copies grow, accounts don't) but not as a gate. ARC's framing explains why: the weights are
    *given*, and a copy is bad not because it is long but because it tells you nothing without being run. A copy
    only answers questions by sampling.
  - INTENT change 2 is reopened. The 0.8/0.5 lines stay unused and are not tuned.
- **11:00** Research on scaling the reader (ARC M16,10 analysis, community work, MIPS, SPD, metric-guided decompilation cleanup). Wrote the reader-scaling plan (DIAGNOSIS_ABOVE_32.md, end; R1–R4) and recorded Shreyas's micro-experiment development loop in INTENT.md. Waiting for a go on R1/R2.
- **11:55 R1 (free; 20 big-network transcripts).** Steps per run: 21–160. Python calls: 18–137, averaging 500–870
  characters of code each.
  - **Submissions rejected only for "not an explanation" (constant-shaving):** 22/25 on M16,10 v2b, 17/17 on lc_6_5 b,
    11/11 on M8,5 v2b, 8/8 on M8,8 v2a. Overall, most submissions on big networks.
  - **Key-insight words first appear at steps 1–33.** On the 2nd-argmax networks "running max" appears by step 1–7,
    but that may be the task guessed, not the network read.
  - **Answer:** the reader is not short of steps for insight. Most of its budget went into gaming the old copy rule.
    No sign yet that missing tools are the bottleneck (R4 deferred), or that the 6-step submit rule is.
  - **Decision:** R2 (progressive replacement with error-locating feedback) is the main lever.
- **11:45** Translation error found and fixed (CHANGES T1, judge glass-2.1): lc_8_6's 'network's own mistakes' numbers were computed against misaligned labels (random 20k subsample of inputs vs full-order labels). All round-2 lc_8_6 mistake numbers are void. No other target has more than 20,000 inputs.
- **11:48** R2 launched locally (lab/replacer.py, a new entry point; the old path is untouched). It warm-starts from an exact copy plus the reader's own notes and gives intent-v2 certificate feedback, with no verdict. Runs: lc_8_6 (notes from run d7df, the census rule) and M16,10 (notes from v2a), 40 steps each, DeepSeek V4.1 Flash. Checked first: both copy starts reproduce the network exactly (fidelity 100%), S = 1.21 / 1.43, and the judge accepts the sites.
- **11:56** R2 progress after ~10 min: both runs reached exact programs with all inside claims passing (lc_8_6: fidelity 100%, 7/7 sites, S 0.996; M16,10: 100%, 10/10, S 0.98). No rules substituted yet. Added R2 on lc_6_5 (notes from ca67, the pointer story) and M8,5 (notes from 293b).
- **12:35 R2 first results.**
  - **lc_8_6** (40 steps, $0.16): best S 0.90, fidelity 100%, 7/7 claims.
  - **M16,10** (stopped itself at 23 steps, $0.10): S 0.98, 10/10. Its account: step 1 is an exact two-sided sign rule; steps 2–3 live in 3- and 5-dimensional cones; steps 4–10 "genuinely fill the space" and stay copied. That is ARC's open problem.
  - **What we learned:**
    - Progressive replacement removes the old swing. Readers stay exact, with every claim passing.
    - But in 23–40 steps they make only small structural reductions, not rule-level replacements, on the hardest networks.
  - **Free check:** promoting the census rule's counts (fixed points, 2-cycles, preimages) to inside claims on lc_8_6.
    At resid1.mid: substitution 92.4%, interchange 71% (no-change control 43%, random 34%; bar 90%). At resid0.mid
    the preimage counts fail (substitution 37%). So it is a partial inside account; the network's layer-1 state is
    not linear in these counts.
- **12:07** Criterion tightened (stricter; logged in PLAN_SCALING): a passing inside claim is needed after the first step. The census rule meets the old letter but not this, so it doesn't count. R5 launched (top-down strategy; trading exactness for rules is allowed; 80 steps): lc_3_4 (notes 2b58), lc_4_5 (notes 93e7), lc_8_6 (notes d7df). M8,5 follows when its R2 run ends.
- **12:13** R2 final: lc_6_5 S 0.64 (canonical-basis re-description plus a pointer-chase description; 5/6 claims), M8,5 S 0.78 (4/5 claims; the description finds a rotating 'clock' in W_h). None meets the criterion. R5 added on M8,5 and lc_6_5 (80 steps). R6 (by hand) on lc_3_4: layer 0 is not a clean pointer (the most-attended position matches no simple relation better than 46%), and the residual is 3 wide. Paused in favour of reading R5.
- **13:25 T2 (a translation error in the information measure, CHANGES T2).** The copy frontier only quantised raw
  weights, but "the cheapest copy" includes the reduced form (QK/OV products; rescaled RNN units). It was exposed by
  lc_8_6 r5 submission 3, a products-form re-description at S 0.70. Fixed and re-run (`info_measure_T2.txt`):
  - every must-fail now scores ≥ 0.91 (lc_4_5 mainb 0.77 → 1.06);
  - the r5 file scores 0.99, i.e. a copy;
  - the census rule moves from 0.35 to 0.50, exactly on the line.
  - No threshold was changed.
  - R5 runs already in flight compute S against the OLD frontier: re-score them afterwards.
- **12:24** R5 interim (all 5 runs): fidelity stays about 100% and S about 0.7 under the old frontier (about 1 under T2). The cheap reader does not trade exactness for rules even when told to. **R7 launched: rule first.** Two lc_8_6 runs start from the census rule (fidelity 92.4%, S 0.50); the job is to find where its quantities live inside the network so that deep claims pass. 80 steps each.
- **12:26** R6 (by hand). lc_8_6 layer 0: the token at the position explains 37–79% of attn0.out per position; fixed-point, 2-cycle and preimage features add ≤ 0.03; the argmax key equals x_p 49% of the time. No crisp relational code. Symmetry test (/tmp/sym.py → ARC's U/V reduction): it holds for M4,3 (the reduced net reproduces 98.7%) and fails for M8,5 (25.5%), M8,8 (11.8%) and M16,10 (no pairing). **Reframing:** the obstacle is messiness, not size. The constructor-era readers verified constructions of much bigger CLEAN networks (mod-add about 226K parameters, mod-mul, Breakout, Tracr). Next: re-certify those under the v2 certificate (free), alongside R7.
- **13:55 Results.**
  - **Re-certification of constructor-era constructions** under the v2 certificate (`results/glass/scaling/recert_clean.json`):
    - tracr_count_greater (compiled, 1,161 params): fidelity 100%, S 0.12, 7/7 claims;
    - astra_breakout (hand-written, 20,486): 100%, S 0.50, 2/2;
    - **modmul_p113 (TRAINED, 226,048 params; blinded reader, Nov 13): fidelity 97.4%, S 0.009, 7/7 claims, 7 after the first step.**
    - modadd still running.
  - **R5 final:** lc_3_4 and lc_4_5 ended with fidelity about 100% and S 0.72–0.86 (old frontier). No run works.
  - **R7b (rule first) moved AWAY from the target:** S 0.50 → 0.85 to make inside claims pass. It copied layers back in.
  - **The finding of the day:** the system already works on big CLEAN trained networks (mod-mul, 226K). AlgZoo's small DENSE networks are the frontier, as ARC says. On dense networks, inside faithfulness and compression pull against each other, and the cheap reader resolves the conflict by copying.
  - **Criterion note:** networks without task labels (mod-mul) can't be "above the task rule". Their network-specificity comes from the prior-only control (0 stages on mod-mul, E10).
- **12:34** Criterion tightened again, before testing: inside claims must be at an INTERMEDIATE site (a claim at the last site can be the logits themselves, which is behaviour in disguise). replacer.py updated. **Rival test, census rule at layer 0** (/tmp/census_B.py: tokens plus per-position fixed-point, 2-cycle and preimage flags at resid0.mid, with the rule computed from them): substitution 72.3%, interchange 55.2% (no-change control 40.2%), partials 18–50%. **lc_8_6 does not compute the census rule internally:** the rule matches 92% of its behaviour by a different route. This belongs on the certificate as a rejected rival.
- **12:36** Built the AlgZoo argmedian targets (8_5: 112 params, acc 0.705 [corrected; I first wrote 0.98 without checking]; 8_7: 0.495; 16_5: 0.984; 16_7: 0.712; exact against torch). Mirror symmetry is absent beyond 4 units. replacer feedback now names intermediate claims explicitly. **R8 launched:** 8 runs (top-down, 100 steps) on M8,5, lc_3_4, lc_4_5 and argmedian 8_5. PLAN_SCALING findings section written.
- **12:36** Re-certification of modadd_p113 (TRAINED, 226,048 params, blinded reader, Nov 13): fidelity 100%, S 0.007, 5/5 claims pass (intermediate: mlp_post, mlp_out). Note: replacer's intermediate-site rule covers rnn_relu and transformer only; for onelayer_transformer and mlp the intermediate sites are named by hand in the report.
- **15:05 R8 result (8 runs, stopped at 55–83 of 100 steps):** the same two modes as before. Either copies (S 0.82–1.12, intermediate claims passing) or short accounts far below the task rule (lc_4_5: S 0.31 at fidelity 45% vs task 81%). No run meets the criterion. **Stopped on purpose:** more cheap-reader batches with reworded instructions would be going in circles. Scaling session total: 19 runs, $4.30.
- **15:05 STATE FOR SHREYAS.**
  - **Works and verifiable:** the v2 certificate (validated on 30+ copies and disguises); big CLEAN trained networks
    (mod-mul and mod-add, 226,048 params) pass the strict criterion.
  - **Does not work yet:** dense AlgZoo networks over 32 params.
  - **Levers not yet pulled:**
    - a stronger reader (needs an API key);
    - analysis tools: automatic decision-subspace discovery plus symbolic regression;
    - partial certificates in the catalog for dense networks.
- **19:07** Started the UNITS PROGRAM (Shreyas 19:06): his hypothesis that networks are compositions of basic units, the same at the level of function and spread over clusters of weights. Research done (universality, component reuse, universal neurons, group-operation universality, Clock/Pizza, modularity, Platonic representations, APD/SPD, weight-sparse transformers, RNN dynamical motifs, RNN universality), experiment tree U1–U7 and conditional build intent are all in docs/glass/UNITS_PROGRAM.md. CLAUDE.md and CHECKLIST point to it.
- **≈19:08 U1 PRE-REGISTRATION.** Question: do single hidden units of the AlgZoo RNNs compute simple functions of the input history?
  - **Method.**
    - Each hidden unit at each step is a neuron-step.
    - For each one, fit relu(a·p + b·q) for every pair (and single) from a library of 7 base quantities: x_t, x_{t-1},
      running max M, running 2nd max S, running min m, M_{t-1}, S_{t-1}. The fit is linear on the active set, then
      R² is scored on all inputs.
    - **Coverage** = the variance-weighted share of neuron-steps with R² ≥ 0.9.
    - Control: the same network with shuffled weights.
  - **Decision.**
    - Network coverage ≥ 2× the control AND ≥ 50% → units exist at the neuron level → go to U3 (causal check).
    - Within 1.25× the control → neuron-level units absent → go to U2 (distributed).
    - Otherwise mixed → U3 on the explained part, U2 on the rest.
  - Networks: 2nd argmax 2_2, 4_3, 4_4, 8_5, 8_8, 16_10; argmedian 8_5, 16_5.
- **≈19:09 U1 RESULT** (`results/glass/units/u1.json`; cleaned: steps after the first, raw-input units separated).
  - **History-unit coverage (vs shuffled):**
    - M16,10: 0.42 (0.00); M8,5: 0.24 (0.07); argmedian 16,5: 0.10 (0.02); argmedian 8,5: 0.04 (0.00); M8,8: 0.00.
    - M2,2 and M4,3 are fully raw-input units (relu of x_t and x_{t-1}).
  - **One unit type recurs across networks and tasks: a single-ReLU comparator of the current input against a running statistic.**
    - relu(max − x_t): 15× in M16,10, 3× in M8,5, 5× in M8,8 (scaled variants), 1× in argmedian 8,5.
    - relu(x_t − 2nd): 7× in M16,10, 2× in M8,5.
    - relu(x_t − min): argmedian 8,5.
  - **Decision (per the pre-registration):** M16,10 and M8,5 are mixed → U3 on the explained neurons, U2 on the rest.
- **≈19:09 U3 PRE-REGISTRATION.**
  - **Method.** In M16,10, M8,5 and argmedian 16,5, take every neuron-step that U1 explains (R² ≥ 0.9, steps after the
    first, any primitive). Replace those activations by their fitted formulas, computed from the true input history,
    inside the running network.
  - **Measures.**
    - (a) Substitution fidelity (decision agreement).
    - (b) The same neurons set to their mean (ablation).
  - **Decision.** Units are verified if (a) ≥ 95% and (b) drops agreement by ≥ 10 points below (a). If (a) holds but
    not (b), the units are real but unused. If (a) fails, the correlational fit doesn't capture what the network uses
    (go to U2).
- **≈19:10 U3 RESULT: FAILED the pre-registered bar.**
  - **All explained neurons substituted at once:**
    - M16,10: 62 neuron-steps → 23.2% of decisions kept (mean-ablation 9.2%);
    - M8,5: 12 → 75.9% (28.4%);
    - argmedian 16,5: 24 → 81.8% (26.5%).
  - **Diagnostic, one unit at a time:**
    - M16,10: median 97.6%, min 66.8%, 35% of units ≥ 99%. Late-step "x_t vs max" units fitted at R² 0.92–0.98 cost up to 33% of decisions alone.
    - M8,5: median 93.8%.
  - **Reading:** the comparator units are approximately right, but decisions depend on detail the 2-term formulas miss.
- **≈19:10 U1b PRE-REGISTRATION.** Question: is the missing detail structured?
  - **Method.** Richer unit family: relu(a·x_t + Σ_k c_k·x_(k)), where x_(k) is the k-th largest of x_1..x_t (at most
    t+1 coefficients). Applied to every neuron-step after the first, in M16,10, M8,5 and argmedian 16,5.
  - **Measures:** R², single-unit substitution and all-unit substitution.
  - **Decision.**
    - Joint substitution ≥ 95% → units are "rank-based comparators" (structured) → build the unit library around
      this family → U4 (composition: which few coefficients matter).
    - Otherwise → the residual is unstructured → U2 (distributed) and the explicit-remainder framing.
- **≈19:11 U1b RESULT: NOT structured in the rank family** (`results/glass/units/u1b.json`; fitted on 4,000 inputs, tested on another 4,000).
  - Median R² 0.71 / 0.81 / 0.59 for M16,10 / M8,5 / argmedian 16,5.
  - Joint substitution 18.6% / 35.7% / 28.7%; single-unit median 91% / 82% / 84%.
  - Reading: sorted values alone miss what units carry. These tasks output a POSITION, so units must also carry when things happened. Per the pre-registration, the residual is not explained by rank-based comparators. I am not adding more formula families: that would be circling.
  - **Finding so far (U1+U3+U1b):**
    - Comparator units recur at the function level across networks and tasks.
    - But in dense RNNs the network's decisions depend on detail that approximate units miss: one unit at R² 0.98 can cost a third of decisions.
    - Neuron-level units + wiring can't be exact here. A units account needs either a different level (dynamics) or an explicit remainder.
- **≈19:11 U6a PRE-REGISTRATION (the clock hypothesis).** M4,3's reduced recurrence is a 120° rotation with n = 3 (360°/n), and the M8,5 reader described a "rotating clock".
  - **Hypothesis:** 2nd-argmax and argmedian RNNs encode position with a rotation of about 360°/n.
  - **Measures.** Eigenvalue angles of W_h, and of the effective linear map in the most common ReLU on/off pattern (the active-set restriction of W_h), for every network with n ≥ 3. Control: shuffled weights.
  - **Decision.**
    - Positive if, for ≥ 3 of the 4 2nd-argmax networks with n ≥ 3 (M4,3, M8,5, M8,8, M16,10), a complex eigenvalue pair with |λ| ≥ 0.5 has angle within 15% of 360°/n (or a small integer multiple), and the shuffled control shows this no more often than chance.
    - Positive → the clock is a recurring dynamical unit → U6b (is it causal?).
    - Negative → drop it.
- **≈19:11 U6a RESULT: NOT SUPPORTED as pre-registered** (`results/glass/units/u6a.json`).
  - The originals hit 4/4 2nd-argmax networks, but the shuffled controls hit 2/4 (6/9 overall): with many eigenvalues and multiples allowed within 15%, chance hits are common.
  - The closest-angle errors of the originals are small (M4,3 0%, M4,4 5%, M8,5 1%, M8,8 6%, M16,10 11%, argmedian 8,5 0%, 8,7 11%, 16,5 0%, 16,7 2%), but reading that after the fact is cherry-picking.
- **≈19:11 U6a' PRE-REGISTRATION (the clock with a proper null).**
  - **Measure.** For each network: the minimum relative error between W_h's eigen-angles (|λ| ≥ 0.5) and k·360°/n. Null: the same statistic on 300 random shufflings of that network's weights.
  - **p-value** = the share of shuffles with error ≤ the original's.
  - **Decision.** The clock is supported if p ≤ 0.05 in ≥ 6 of 9 networks (Fisher-combined p ≤ 0.001). Otherwise drop it.
- **≈19:12 U6a' RESULT: the clock is NOT a general unit** (`results/glass/units/u6a2.json`).
  - p ≤ 0.05 in 3/9: M4,3 0.007, M8,5 0.040, argmedian 8,5 0.007. Fisher-combined p = 1.2e-3 (bar 1e-3). Dropped as a general unit.
  - Recorded fact: in those three small networks, the recurrence holds a rotation within 1% of k·360°/n, beyond chance.
- **≈19:12 U2 PRE-REGISTRATION (distributed units).** Question: are the task's natural quantities held as directions across neurons?
  - **Quantities:** running max M, running 2nd max S, running min m, x_t, the AGE (steps ago) of the max and of the 2nd max, and for argmedian the running median.
  - **Method.** Probe each from h_t with a linear map (fitted on 4,000 inputs, scored on 4,000 others). Control: shuffled weights.
  - **Decision.**
    - If at the last step the quantities the task needs (M2 networks: the position of the 2nd max via the ages; argmedian: the median and its position) have R² ≥ 0.9 and the control is clearly lower (≥ 0.3 below) → distributed units exist → U3' (judge test with those variables at the last site).
    - Otherwise the state encodes something else → U6b (dynamics view).
- **≈19:12 U2 RESULT** (`results/glass/units/u2.json`, last step, held-out R² of a linear probe).
  - The trained networks do NOT hold the task's quantities as linear directions. M16,10 holds max 0.99 and 2nd 0.85, but not positions; the others hold max ≤ 0.68.
  - The argmedian networks hold the median VALUE at 0.02 / 0.09, versus 0.60 / 0.63 in their shuffled versions: training removes it.
  - Limit of the measure, noted: positions are categories, so scalar "age" probes are weak evidence.
  - Decision per the pre-registration: → U6b.
- **≈19:12 U6b PRE-REGISTRATION (modes).** Question: how many ReLU on/off patterns does each network use at each step?
  - **Measure:** the number of distinct patterns needed to cover 90% of 8,000 inputs, per step, versus the shuffled control.
  - **Decision.**
    - ≤ 8 patterns at every step for M8,5, M16,10 and argmedian 16,5 → mode decomposition is viable (exact: the network is linear within a pattern) → U6c (are modes chosen by simple comparisons, and do they recur across networks?).
    - Otherwise modes are too many to read → record it and move to the explicit-remainder build.
- **≈19:13 U6b RESULT: modes are too many** (`results/glass/units/u6b.json`). Patterns covering 90%, per step:
  - M8,5 up to 18; M16,10 up to 74; argmedian 16,5 up to 114. The bar was ≤ 8.
  - Only small networks are compact: M4,3 ≤ 4, argmedian 8,5 ≤ 9, M8,8 ≤ 11.
  - Trained networks don't use fewer modes than shuffled ones.
- **≈19:13 FIRST-SWEEP CONCLUSION** (U1, U3, U1b, U2, U6a, U6a', U6b).
  - **Recurring units exist at the level of function** (single-ReLU comparators against running statistics, across sizes and tasks) **and they matter** (ablation). But in dense RNNs they are approximate, and the decisions depend on the detail they miss.
  - None of the exact routes gives a compact decomposition: neuron formulas, rank formulas, linear task variables, a rotation clock, linear modes.
  - This is a general finding (discovery type b, provisional; needs a prior-work check): **"approximate universality, exact individuality"**.
- **≈19:13 BUILD DECISION (from the evidence): a UNIT MAP**, not an exact units-plus-wiring construction. Intent and tests are in UNITS_PROGRAM §4b, written before any code.
- **19:15** UNIT MAP built (lab/unitmap.py). v1 failed K2 (M4,3 labelled as comparators: the library lacked a linear-filter class, and labels were degenerate at small t). That is a translation error, fixed as v2 (CHANGES T3). **v2 passes all known-answer tests:** K1, M16,10 units 1/6/7 = relu(max − x_t) and unit 8 = relu(2nd − x_t), matching ARC/Chen's N7 and N8 and the LessWrong {N1,N6,N7,N8} group; K2, M4,3 is fully raw/filter units (coverage 1.00); K3, controls ≤ 0.08. Running on the never-interpreted argmedian networks now.
- **19:23** Unit maps for 7 more networks. Argmedian (never interpreted): history coverage 0.01–0.16, raw/filter 0.11–0.28; comparators recur across tasks but most argmedian neurons are outside the library. M4,4 and M8,8: almost nothing. Findings and next steps P1–P4 are in UNITS_PROGRAM §8.

- **19:35 CORRECTION (times).** The units-program entries from "19:20" to "21:50" carried HAND-TYPED times that were wrong (the same error as E8). The real times, from commits and the system clock, are 19:07–19:34 ET. Corrected above with ≈ marks; the updates sent to Shreyas at "20:45" and "21:50" were really sent at about 19:12 and 19:24. Rule added to CLAUDE.md: times come only from `date`.
- **19:35 P1 PRIOR-WORK CHECK.**
  - **Comparator units in M16,10 are KNOWN.** Martin Keane (LessWrong) sorts its neurons into comparators, wave neurons, bridge neurons and special cases.
  - **A related tool EXISTS:** candle-mi "stoicheia" (Rust) classifies AlgZoo ReLU-RNN neurons by correlating them with reference signals such as cummax. That is the core of our labelling step.
  - **The residual we missed is modelled:** a LessWrong commenter fits M16,10's {N1,N6,N7,N8} as a(δ)·M_t + b(δ)·max(S_t,0), with δ = steps since the max, at R² 0.992.
  - **Argmedian above 32 params:** no public explanation found (xabier.me, Nov 2026, explains argmedian M4,3 and says argmedian hadn't been explained before).
  - **What remains ours:** causal per-unit confidence with controls and known-answer tests; unit maps of the argmedian networks above 32 params; the sufficiency finding (U3).
  - Claims are downgraded accordingly: "comparators in M16,10" is a replication, not a discovery.
- **19:35 P2' PRE-REGISTRATION (timing-aware units, from prior work).**
  - **Units:** h = relu(a(δ)·M + b(δ)·S + c(δ)·x_t), with coefficients fitted per δ = steps since the running max (and, as a variant, δ2 = steps since the running 2nd max).
  - **Known-answer check:** replicate R² ≥ 0.98 on M16,10 units {1,6,7,8}.
  - **Then:** joint substitution of ALL neuron-steps by their best timing-aware unit (held-out), on M16,10, M8,5, argmedian 8_5 and 16_5.
  - **Decision.**
    - Joint substitution ≥ 95% on a network → its units are exact enough → the build extends the unit map with timing tables, and that network gets a units-plus-timing construction.
    - < 95% → timing helps but isn't enough; record it.
- **19:35 P2' RESULT** (`results/glass/units/p2_timing.json`).
  - **Known answer replicated:** M16,10 units {1,6,7,8} reach R² 0.97–0.999 per step with δ-tables (LessWrong reported 0.992).
  - **Joint substitution still collapses:** M16,10 25.8%, M8,5 32.9%, argmedian 8_5 31.9%, 16_5 26.5% (median unit R² 0.93 / 0.96 / 0.72 / 0.82).
  - Per the pre-registration: timing helps but isn't enough.
- **19:35 P3 PRE-REGISTRATION (precision requirement).** Question: how precise must any account of each network's hidden state be?
  - **Measure ε95:** the relative noise level (Gaussian, std = ε × each neuron's activation std, added to every neuron at every step) at which decision agreement falls to 95%. Search ε over a log grid; 2,000 inputs.
  - **Hypothesis:** ε95 is small (< 0.05) for the networks where units failed (M16,10, M8,5, argmedian 8_5 and 16_5) and large (> 0.2) for M2,2 / M4,3, which ARC explains.
  - **Decision.** If confirmed, the certificate and the unit map report ε95 as the network's "precision requirement", and each unit's error relative to it. That turns "approximate units fail" into a predicted, measured quantity (a candidate general finding). Otherwise, record it.
- **19:36 P3 RESULT** (`results/glass/units/p3_precision.json`). ε95 (the relative noise at which 5% of decisions flip):
  - M2,2 0.16; M4,3 0.016; M4,4 0.005; M8,5 0.013; M8,8 0.001; M16,10 0.006; argmedian 8_5 0.004, 16_5 0.010, 16_7 0.005.
  - **The hypothesis is partly REFUTED:** M4,3 is NOT robust (predicted > 0.2, measured 0.016).
  - **Reading:** all but the tiniest network are hypersensitive, at about 0.1–1.6% of a neuron's spread. M4,3 is explainable because its account is EXACT (the symmetry reduction is a rewrite of the weights), not because it tolerates error.
  - **Consequence for the units program:** in these networks, units explain only when they capture EXACT structure. Approximate units (even at R² 0.99) cannot substitute. This accounts for U3, U1b and P2' in one measured quantity.
  - Candidate general finding (b): "AlgZoo RNNs require ≈1%-precise accounts". Prior-work check pending (robustness, Lipschitz, ARC's noise framing).
- **20:03 T4 built** (unit map v3: ε95 and per-unit relative error). It reproduces P3 exactly; K1–K3 unchanged. Substitution-grade neuron-steps: M4,3 8/12, M16,10 16/160.
- **20:03 P4 PRE-REGISTRATION (exact structure).**
  - **Signature 1 (signed pairs, as in M4,3):** for a pair (i,j) at step t, h_i − c·h_j is LINEAR in x_1..x_t with relative error ≤ ε95 (c fitted).
  - **Signature 2 (exact redundancy):** the numerical rank of the hidden state at tolerance ε95 × the spread is below the number of active neurons.
  - **Decision.**
    - If either covers ≥ 25% of neurons (steps after the first) in a network bigger than M4,3 → build an "exact reduction" stage (reduce, then map units on the reduced network).
    - Otherwise record: M4,3's exact structure does not scale.
- **20:04 P4 RESULT: exact structure does NOT scale** (`results/glass/units/p4_exact.json`).
  - Neuron-steps in exact signed pairs (at the ε95 tolerance): M4,3 50% (known-answer check: its signed channels), M8,5 12%, M16,10 2%, argmedian 8_5 6%, 16_5 19%.
  - Exact rank deficiency only at step 2, which is trivial (two inputs).
  - Per the pre-registration: below 25% everywhere bigger than M4,3 → no exact-reduction stage. M4,3's exact structure does not scale.
- **20:04 STATE OF THE UNITS PROGRAM.**
  - Recurring units exist at the level of function (comparators; replicated on M16,10 against prior work; present in the argmedian networks).
  - The networks are hypersensitive (ε95 0.1–1.6%).
  - No exact compact structure was found beyond M4,3 (7 routes).
  - The unit map v3 reports all of this honestly per unit.
  - Remaining in the plan: U7 (transformers) and the ε95 prior-work check.
- **20:31 (d) PRIOR-WORK CHECK on ε95.** No published quantitative noise tolerance for AlgZoo networks was found. The closest is qualitative: "almost all ablations hurt the accuracy dramatically - the model makes use of all neurons" (LessWrong comment on ARC's post). ARC frames explanations as mechanistic estimates and measures them by MSE versus compute or by surprise accounting, not by tolerance. The ε95 measurement is therefore new as a number, and consistent with what is known qualitatively. Status: a candidate general finding; it needs replication on more networks before any claim.
- **20:31 U7 SPECIFICATION written (UNITS_PROGRAM §8).** It runs at the next check-in.
- **20:58 U7 RUN** (spec in UNITS_PROGRAM §8, pre-registered at the last check-in). Attention-unit part first; known answers: tracr_reverse (fixed-position map N−1−i), tracr_hist (same-token).
- **20:58 U7 known-answer step.** tracr_reverse passes (queries 1–4 → fixed keys 4,3,2,1 = the reversal map, start token at 0). tracr_hist FAILS: best unit 'same token' at every query, but mass only 0.674, because compiled histogram heads also attend to the start token at position 0. That is a translation error in the library (it lacks unit ∪ first position). Fix: add a '+ first position' variant of every unit; rerun everything.
- **20:59 U7 RESULT (attention units)** (`results/glass/units/u7.json`).
  - Known answers pass after the fix: tracr_reverse 100% labelled (the correct reversal map), tracr_hist 100% (same token + first position).
  - Trained lc networks: 8–12% labelled (controls 0–12%). This is far below the 50% bar → NEGATIVE: their attention does not use the library's textbook units.
  - One exception: lc_4_5 has one pointer (j = x_i) head-query, matching the round-2 reader's "copies one key".
- **20:59 U7b PRE-REGISTRATION (content lookup).** Layer-0 attention in lc_8_6 is hard (max weight about 0.91) but goes to unlisted keys.
  - **Question:** for hard queries (max weight ≥ 0.8), is the attended key's TOKEN determined by (query position i, query token x_i), i.e. a lookup "find the token g(i, x_i)"?
  - **Measure:** the share of inputs where the attended token equals the modal attended token for that (i, x_i).
  - **Decision.** ≥ 90% on the hard queries of ≥ 3 of the 4 lc networks → "content lookup" is a real unit → add it to the library and rerun U7. Otherwise record the negative.
- **20:59 U7b RESULT: NEGATIVE** (`results/glass/units/u7b.json`).
  - Lookup determinism (the attended token fixed by (i, x_i)) is 0.29–0.54 in the trained lc networks vs 0.28–0.57 in shuffled controls.
  - Hard attention is just as common in shuffled networks (hard share 0.61–0.96), so it is an architectural property, not a learned unit.
  - **Transformers:** neither the textbook attention units nor content lookup describe the trained longest-cycle networks.
- **21:03 U7c PRE-REGISTRATION (residual units; the second part of the U7 spec).**
  - **Question:** after layer-0 attention, does the trained longest-cycle network's residual carry recognisable relational features ("residual units"), even though its attention patterns are not textbook?
  - **Site:** the write of each attention layer, `attn{l}.out`, at each position i. Layer 0 is the only intermediate site in these 2-layer networks.
  - **Decision-relevant part:** the smallest number k of top principal components of `attn{l}.out` at position i that keeps ≥ 95% of decisions when the other components are set to their mean. Positions with k = 0 do not matter and are skipped.
  - **Features** (categorical, one-hot, per position i):
    - **Baseline (first order):** the token at every position.
    - **Relational:** fixed point (x_i = i); pointer read x_{x_i}; 2-cycle (x_{x_i} = i); preimage count #{j: x_j = i}; same-token count #{j: x_j = x_i}.
  - **Score:** variance-weighted held-out R² of the k PC scores (2-fold, least squares). Base R², full R², and relational share = (full − base)/(1 − base).
  - **Known answers first** (the measure must pass these before the lc runs count):
    - tracr_hist: full R² ≥ 0.95 at its attention write (same-token count).
    - tracr_reverse: base R² ≥ 0.95 at the attention layer that moves the tokens (token at a fixed position).
  - **Decision.** "Relational residual units are real" if, in ≥ 3 of 4 lc networks, the layer-0 decision-relevant residual has full R² ≥ 0.9 (the unit map's labelling bar), AND the relational share beats the shuffled control by ≥ 0.2. Otherwise record the negative.
  - **ε95 on the residual stream** (descriptive): noise on `resid0.mid` (as a share of each coordinate's spread) at which decision agreement falls to 95%. Expectation stated now: compiled Tracr networks tolerate far more noise than trained lc networks.
- **21:05 U7c known answers PASS.** tracr_hist: attention write explained at full R² 0.99 (base −0.16, so the same-token count does the work). tracr_reverse: the token-moving layer (attn3) at base R² 1.00.
  - Reporting fix in the experiment script, before the lc runs: the relational share is undefined when the base R² is about 1 (tracr_reverse gave a division by about 0). It is now reported as None when 1 − base < 0.01. This does not affect any lc number.
- **21:05 U7c RESULT: NEGATIVE by the pre-registered rule** (`results/glass/units/u7c.json`, read by hand).
  - **Layer-0 full R²** (bar 0.9): lc_3_4 0.88, lc_4_5 0.71, lc_6_5 0.87, lc_8_6 0.73. Below the bar in all four.
  - **Relational share, original vs shuffled:**

    | network | original | shuffled | gap |
    |---|---|---|---|
    | lc_3_4 | 0.05 | 0.01 | +0.04 |
    | lc_4_5 | 0.29 | 0.03 | +0.26 |
    | lc_6_5 | 0.41 | 0.11 | +0.30 |
    | lc_8_6 | 0.18 | 0.01 | +0.17 |

    Only 2 of 4 clear the +0.2 gap.
  - **Layer 1 at the output position** (descriptive; the last site before the readout): full R² 0.40–0.52, relational share 0.14–0.41 vs controls 0.03–0.17.
  - **What layer 0 writes:**
    - Mostly a first-order mix of "token at position j" (base R² 0.60–0.87).
    - Plus a partial relational component that training put there (well above shuffled in three networks).
    - No feature in the library captures it cleanly.
  - **ε95 on the residual** (`resid0.mid`): trained lc 0.008, 0.025, 0.050, 0.050; Tracr 0.063 and 0.100.
    - My stated expectation ("compiled networks tolerate far more noise") holds only weakly: about one noise-grid step for the bigger lc networks.
    - Shuffled controls tolerate more noise (0.06–0.50). Their decisions depend on fewer positions (decision-relevant k is often 1), so this is not a fair comparison of precision.
  - **Decision:** no residual-unit stage for transformers. The U7 line is closed: in these trained transformers neither the attention patterns nor the layer-0 residual decompose into the library's units. Training adds partial relational structure, but not clean units.
- **21:05 P5 PRE-REGISTRATION (does ε95 separate the networks we can explain from those we can't?).** This replicates the ε95 finding outside AlgZoo before any claim, as required at 20:31.
  - **Question:** the system certifies the modular-arithmetic transformers with approximate constructions (fidelity 97–100%, S < 0.01) but fails on AlgZoo. Is that because the modular networks tolerate imprecision and AlgZoo networks don't?
  - **Measure:** ε95, the same definition as P3: Gaussian noise of ε × each coordinate's spread, added at one intermediate site; the ε at which decision agreement falls below 95%. Same log grid, 3 seeds, at most 4,000 inputs.
    - Transformers: the residual after the first attention layer (`resid_mid` / `resid0.mid`).
    - MLPs: the first hidden layer `h1`.
  - **AlgZoo reference** (P3 and U7c, already measured): median 0.010 over 13 networks.
  - **Hypothesis:** modadd_p113 and modmul_p113 both have ε95 ≥ 0.1 (10× the AlgZoo median).
  - **Known confound, stated in advance:** these networks have 384 noised coordinates against 16–160 for the RNNs. More noised coordinates makes a network look LESS tolerant, so the confound works against the hypothesis.
  - **Decision.**
    - Yes → the findings doc reports ε95 as a candidate predictor of when approximate accounts can work, with its limits.
    - No → hypersensitivity is common and does not explain the AlgZoo failure; the findings doc says so.
  - **Descriptive only (no decision):** the compiled Tracr networks count_greater and first_match, the tiny boolean MLPs, and the Breakout policy MLP.
- **21:07 P5 RESULT: hypothesis CONFIRMED** (`results/glass/units/p5_eps95_cross.json`, read by hand).
  - **ε95 at the first intermediate site:**
    - modadd_p113 0.79 and modmul_p113 0.79: 80× the AlgZoo median of 0.010 (bar 0.1).
    - Tracr: count_greater 0.05, first_match 0.32 (hist 0.063 and reverse 0.10 from U7c).
    - Boolean MLPs: 0.25–1.0.
    - Breakout policy MLP (trained, never explained): 0.02.
  - **Correction to my pre-registered confound:** `resid_mid` here is the last position only, so 128 noised coordinates, not 384. That is still in the same range as the RNNs (16–160 neuron-steps), so the comparison stands.
  - **Reading: there are two routes to an explanation.**
    - **Exact structure** works whatever the tolerance: M4,3 at 0.016, Tracr count_greater at 0.05.
    - **Approximate accounts** work only where the network tolerates imprecision: the modular networks at 0.79, where our constructions certify at 97–100%.
  - **AlgZoo above 32 params has neither:** no exact structure (P4, 7 routes) and ε95 about 1%. This is the measured reason the reader failed there.
  - **Prior work:**
    - Arora, Ge, Neyshabur and Zhang (ICML 2018) show that noise stability of trained nets is what makes them compressible.
    - Morcos et al. (2018) tie robustness to noise injected at nodes to generalization.
    - Our result is the interpretability version: noise tolerance at intermediate sites predicts whether a short APPROXIMATE readable account can be faithful. As far as I found, no one has measured it as a predictor of explainability; I'll check this again before calling it new.
  - **Decision (per the pre-registration):** the findings doc reports ε95 as a candidate predictor, with its limits:
    - n = 2 explained-approximate networks;
    - architectures differ;
    - task margins differ: modular arithmetic has 113 classes with large logit gaps.
- **21:08 (f) DONE:** `docs/glass/UNITS_FINDINGS.md` consolidates every units-program result, with numbers, controls, prior work (including Josh Snider's Oct 26 LessWrong post and Arora et al. 2018) and limits. Next: (g) the catalog integration.
- **21:09 (g) started: catalog integration.** First, regenerating the unit maps that were still v2 (eight networks) with the current v3, so every map carries ε95 and per-unit relative error. This is not a code change, just a rerun of `lab/unitmap.py`. M16,10 and M4,3 are already v3.
- **21:12 Unit maps regenerated with v3** (`/tmp/unitmap_v3.log`, maps in `results/glass/units/maps/`). ε95 reproduces P3 for every network: M2,2 0.158, M4,4 0.005, M8,5 0.013, M8,8 0.001, argmedian 8_5 0.004, 16_5 0.010, 16_7 0.005; new: argmedian 8_7 0.006.
  - Substitution-grade neuron-steps: M2,2 3/4, M4,4 4/16, M8,5 8/40, M8,8 8/64, argmedian 8_5 8/40, 8_7 8/56, 16_5 22/80, 16_7 16/112.
- **21:12 (g) DONE: the catalog shows unit maps** (CHANGES T5).
  - `scripts/glass/units/t5_catalog_test.py` failed before the change (13 FAILs) and passes after (ALL PASS):
    - the designed case argmedian 16_5;
    - held-out cases argmedian 16_7 and M8,8;
    - M16,10;
    - existing cards unchanged;
    - no JavaScript errors.
  - Screenshots of argmedian 16_5 and M16,10 checked by eye.
  - **Noticed, not changed (T5 must not change verdicts):** the AlgZoo cards still carry round-1 "complete" badges from the old judge. Round 2 showed such constructions can be copies (J3, open). The Parts section now sits next to them and shows how little of each network is actually explained.
- **21:14 U8 PRE-REGISTRATION: do the same units appear across independent training runs?** This is Shreyas's convergence question asked directly, and it also replicates F3 (ε95).
  - **Networks:** ARC publishes all 5 training seeds per size (checked: HTTP 200 for seeds 0–4). Three sizes × 5 seeds = 15 networks, 13 of them new:
    - 2nd argmax M8,5 (our existing one is seed 3);
    - the zoo's 2nd argmax M16,10 (a different checkpoint from ARC's challenge model);
    - argmedian 16,5 (ours is seed 0).
  - **Measure:** unit map v3 (`lab/unitmap.py`, unchanged) per network, with its shuffled control. Each neuron's label class is one of: comparator relu(max − x_t), relu(x_t − 2nd), relu(2nd − x_t), relu(x_t − max), relu(x_t − min), other history unit, filter, raw input, or unexplained. Also history coverage and ε95.
  - **Q1 (convergence).**
    - A comparator subtype RECURS at a size if it labels at least one neuron in ≥ 4 of 5 seeds, and in ≤ 1 of the 5 shuffled controls.
    - "Function-level units converge across training runs" if at least one subtype recurs at each of the three sizes.
    - Otherwise the units are individual to the run.
  - **Q2 (ε95 replication).** Hypersensitivity replicates if ε95 ≤ 0.05 in ≥ 13 of the 15 networks.
  - **Descriptive:** how much the unit inventory (counts per class) varies across seeds.
  - **Infrastructure, not a code change:** seed models are built by a new experiment script (`scripts/glass/units/u8_seeds.py`) under new ids (`..._s<seed>`) with the same conversion and torch check as `build_algzoo.py`. Their maps go to `results/glass/units/seeds/`, not the catalog.
- **21:15 U8 RESULT** (`results/glass/units/seeds/*.json`, `/tmp/u8.log`, read by hand).
  - **Known answers:** the existing networks reappear exactly (M8,5 seed 3: ε95 0.0126, history 0.23, control 0.08; argmedian 16,5 seed 0: ε95 0.0100, history 0.16).
  - **Q1 (convergence): MIXED, so NOT supported as pre-registered** (a subtype must recur at each of the 3 sizes; it does at 1).

    | network | relu(max − x_t), seeds / controls | relu(x_t − 2nd), seeds / controls | recurs? |
    |---|---|---|---|
    | M8,5 | 5/5 / 0/5 | 5/5 / 0/5 | both recur |
    | zoo M16,10 | 3/5 / 1/5 | 4/5 / 2/5 | neither passes |
    | argmedian 16,5 | – | 3/5 / 0/5 | does not pass |

    - In the M16,10 seeds, 5–11 of 16 neurons are unexplained by the library, so "absent" may mean "not single-neuron" rather than "not there".
    - ARC's challenge M16,10 (history coverage 0.37) is unusually comparator-rich next to its five zoo siblings (0.02–0.22).
  - **Q2 (ε95): REPLICATES, 15 of 15 ≤ 0.05.**
    - M8,5: 0.006–0.013.
    - Zoo M16,10: 0.003–0.005.
    - Argmedian 16,5: 0.008–0.010.
    - ε95 is nearly constant within a size: a property of the task and size, not of the training run. Within 2nd argmax, bigger is more fragile.
  - **Reading:**
    - At a small size, training runs converge on the same function-level units (both comparators in all 5 seeds, none in the controls).
    - At larger sizes the inventory becomes individual: the comparators appear in some runs and not others.
    - The precision requirement replicates in every run. This updates F1 and F3 of UNITS_FINDINGS.
