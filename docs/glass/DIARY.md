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
- **21:16 Check-in bookkeeping.** I tried to replace the pending 21:25 check-in (old text, which lists the finished items e–g) with one carrying the new NEXT items. Deleting it was denied by the permission system, and I did not work around that.
  - Two check-ins are now pending: trig_011J9d1s1fmaeghJeosZAqbk (old text, ~21:25) and trig_01CBLY4uSWbktvN1g1F6tXJs (new text, ~21:42).
  - **Rule:** when the OLD one fires, do the work in §8 NEXT but do NOT schedule a successor, so there is only one chain.
- **21:17 U9 PRE-REGISTRATION ((i): what do the unexplained argmedian neurons compute?).** A data-driven search, not a guessed library.
  - **Targets:** every argmedian neuron-step (step ≥ 2) whose best library unit in the v3 map has held-out R² < 0.9. Networks 8_5, 8_7, 16_5 and 16_7.
  - **Search:** h ≈ relu(Σ c_f f) with at most 3 features, chosen greedily (forward selection) on the first half of the inputs and scored by held-out R² on the second half. The fit is the unit map's own active-set least squares, with no bias.
  - **Feature pool at step t:**
    - the inputs x_1..x_t (position-specific);
    - the order statistics of x_1..x_t (k-th largest);
    - every pairwise comparison relu(x_i − x_j), i ≠ j ≤ t.
  - **Known answer (must pass before the argmedian run counts):** on M16,10's library-explained neuron-steps of units 1, 6, 7 and 8 (the comparators), the search must reach R² ≥ 0.9 in ≥ 90% of them.
  - **Control:** the same search on the same networks with shuffled weights (all neuron-steps of step ≥ 2 that their library leaves unexplained).
  - **Decision.**
    - "Sparse comparison units explain the rest" if, in ≥ 3 of 4 argmedian networks, ≥ 50% of the variance of the previously unexplained neuron-steps reaches held-out R² ≥ 0.9, AND the control's explained share is at most half of the original's.
    - Yes → a unit-map v4 with this class (CHANGES entry first), then joint substitution.
    - No → record that the argmedian remainder is not sparse in comparisons.
- **21:17 U9 known answer: PASSES, at the bar.** On M16,10's 21 library-explained comparator neuron-steps (units 1, 6, 7, 8), the search reaches R² ≥ 0.9 in 90% (bar 90%). The greedy search sometimes picks a pairwise comparison first instead of the clean "max − x_t" form, but still fits. This is a marginal pass, noted as such.
- **21:17 U9 RESULT: NEGATIVE** (`results/glass/units/u9.json`, read by hand).
  - **Share of the previously unexplained variance now at R² ≥ 0.9** (bar 0.5 in ≥ 3 of 4):

    | network | original | shuffled control |
    |---|---|---|
    | argmedian 8_5 | 0.02 | 0.00 |
    | argmedian 8_7 | 0.03 | 0.00 |
    | argmedian 16_5 | 0.33 | 0.02 |
    | argmedian 16_7 | 0.02 | 0.00 |

  - Median held-out R² of the best 3-feature unit: 0.66, 0.52, 0.85, 0.69.
  - **Observed in 16_5 (descriptive, not a claim):** neuron 10 at step 5 is exactly relu(17.3·relu(x3−x4) + 10.2·relu(x1−x4) + 7.1·relu(x2−x4)) (R² 1.00): a weighted count of how far the earlier inputs exceed x4, i.e. a soft RANK of x4. That is the quantity argmedian needs. Several other 16_5 neurons use pairs like relu(x1−x2) and relu(x2−x1) together, i.e. distance-like terms.
  - **Decision:** no unit-map v4. The argmedian remainder is not sparse in pairwise comparisons, except a partial soft-rank structure in 16_5.
- **21:19 U10 PRE-REGISTRATION ((j), made causal): does making a network tolerant make approximate unit accounts faithful?**
  - **Why this instead of a plain ε95 prediction:** F3 is correlational. The modular nets differ from AlgZoo in architecture and task as well as tolerance. An intervention isolates tolerance: train the same architecture on the same task with and without hidden-state noise, then see whether approximate units now stand in for the network.
  - **Design:**
    - 2nd argmax M8,5 (ReLU RNN, no biases, as AlgZoo).
    - 3 seeds × {standard, noise}. Noise = Gaussian of 0.1 × each unit's batch std, added to the hidden state at every step during training only.
    - Same optimizer, steps and batches for both.
    - Trained nets are saved as lab targets `u10_m85_{std,noise}_s{seed}` (8,000 standard-normal inputs, labels).
  - **Pipeline known answer:** the standard nets must look like AlgZoo, with accuracy ≥ 0.75 and ε95 ≤ 0.03. Otherwise the test is inconclusive.
  - **Measures** (all with existing code): accuracy; ε95 (P3); unit map v3 coverage; JOINT substitution, i.e. every library-explained neuron-step (held-out R² ≥ 0.9, step ≥ 2) replaced by its unit at once, on the held-out half, as the share of decisions kept.
  - **Predictions:**
    - (a) Noise-trained median ε95 ≥ 5× the standard median.
    - (b) Joint substitution ≥ 90% in ≥ 2 of 3 noise-trained nets, and < 90% in ≥ 2 of 3 standard nets.
    - (b) counts a net only if the substituted neuron-steps hold ≥ 25% of the step-≥2 variance. This guards against a trivial pass from substituting almost nothing.
  - **Decision.**
    - (a) and (b) → interventional support for F3: tolerance is what lets approximate unit accounts be faithful. This also gives a design lever: train for tolerance to get explainable networks.
    - (a) without (b) → tolerance is not sufficient.
    - Not (a) → inconclusive.
- **21:23 U10 RESULT: INCONCLUSIVE by the pre-registered rule** (`results/glass/units/u10_8000.json`, read by hand).
  - **Pipeline known answer FAILED:** standard-trained accuracy 0.744, 0.789, 0.726 (bar ≥ 0.75 for all), because 8,000 steps is far less training than AlgZoo's. ε95 ≤ 0.03 held.
  - **Prediction (a) FAILED anyway:** noise training did not raise ε95 (noise 0.0158 / 0.0126 / 0.0200 vs standard 0.0158 / 0.0158 / 0.0100; median ratio 1.0, bar 5×). Noise-trained nets are less accurate (0.64–0.69) and have low coverage, so (b) is moot.
  - **What the failure points to (a new suspicion, to test next).** Training at 10% hidden noise should make a network robust to 10% noise unless something else sets ε95. The candidate is the TASK: 2nd argmax and argmedian have continuous inputs, so some inputs always sit next to a tie (2nd and 3rd largest almost equal). Any perturbation, however small, flips those decisions. A 5% flip budget may therefore be used up by near-tie inputs, whatever the network does inside.
  - **Rough check:** M2,2's neurons are about relu(x1 − x2). A tie-only model predicts ε95 ≈ 0.15 for it; measured 0.16.
  - **Consequence:** the modular-arithmetic nets have discrete inputs with no near-ties, so their 0.79 may say more about their task than about tolerance. F3's "two routes" reading may be confounded. It is not to be relied on until U11 settles this.
- **21:23 U11 PRE-REGISTRATION: is ε95 set by near-tie inputs?**
  - **Task margin of an input** (input units):
    - 2nd argmax: min(largest − 2nd largest, 2nd largest − 3rd largest).
    - argmedian: min gap between the median and its sorted neighbours.
  - **Measures** for the 9 P3 RNNs (P3's procedure, grid and seeds):
    - **(i) Clear-half ε95:** ε95 computed only on inputs whose task margin is above the median. The near-tie half is excluded, and the 5% flip budget applies to the clear half alone.
    - **(ii) Near-tie share of flips:** at ε = the overall ε95, the share of flipped inputs that lie in the lowest-10% task-margin decile.
  - **Decision** (ratio = clear-half ε95 / overall ε95):
    - Ratio ≥ 10 in ≥ 6 of 9 → ε95 is dominated by near-ties, and F3 is restated: the precision requirement is a property of the task's decision boundary on near-tie inputs, not internal fragility. The "two routes" reading must then be re-tested on clear inputs.
    - Ratio < 3 in ≥ 6 of 9 → internal fragility is real, and F3 stands.
    - Otherwise mixed; report.
- **21:24 U11 RESULT: internal fragility is REAL; F3 stands, with a correction** (`results/glass/units/u11_ties.json`, read by hand).
  - Translation fix before the run: M2,2 has only two inputs, so its margin is the single gap; the script first crashed on a third order statistic.
  - **Ratio clear-half ε95 / overall ε95:** M2,2 3.2, M4,3 5.0, M4,4 1.3, M8,5 2.5, M8,8 1.0, M16,10 2.5, argmedian 8_5 2.0, 16_5 4.0, 16_7 2.0.
  - Ratio < 3 in 6 of 9, and ≥ 10 in none → by the pre-registered rule, F3 stands.
  - **Correction to F3:** near-tie inputs do contribute, by a factor of about 2.5 (median ratio).
    - Clear-half ε95 is 0.001–0.08 (median 0.016), against 0.79 for the modular nets. That is still about 50× apart.
    - Near-tie inputs are over-represented among flips in the small networks (58–66% of flips in the lowest-margin decile for M2,2 and M4,3; chance is 10%), but hardly in M4,4 and M8,8 (10–13%).
  - **U10's failure is therefore not explained by ties.** Why 10% training noise did not raise ε95 stays open. The run was under-trained, so it is inconclusive, not negative.
- **21:26 Check-in (the OLD duplicate, trig_011J…, text lists e–g, all done).** Per the 21:16 rule, I do §8 NEXT (k) and do not schedule a successor. The new chain (trig_01CB…, ~21:42) continues.
- **21:28 Pipeline check for (k):** with 60,000 steps (cosine schedule, 65 s) a standard M8,5 reaches accuracy 0.870, inside ARC's range for its seeds (0.81–0.88). The training pipeline is now AlgZoo-like. A σ = 0.1 net is training for an accuracy check only (no ε95 or substitution looked at).
- **21:28 U10b PRE-REGISTRATION (the causal test of F3, done properly).**
  - **Design:** M8,5 2nd argmax, 60,000 steps, seeds 0–2, training noise σ ∈ {0, 0.03, 0.1, 0.3} (× each unit's batch std, hidden state, every step, training only). 12 networks, ids `u10b_m85_sig<σ>_s<seed>`. Measures as in U10.
  - **Known answer:** σ = 0 nets have accuracy ≥ 0.75 and ε95 ≤ 0.03.
  - **Predictions** (σ* = the largest σ whose 3 nets all keep accuracy ≥ 0.70):
    - (a) Median ε95 at σ* ≥ 5× the σ = 0 median.
    - (b) Joint substitution ≥ 90% in ≥ 2 of 3 nets at σ* (counting a net only if the substituted neuron-steps hold ≥ 25% of the step-≥2 variance), and < 90% in ≥ 2 of 3 σ = 0 nets.
  - **Decision.**
    - (a) and (b) → interventional support for F3 (tolerance makes approximate unit accounts faithful) and a design lever.
    - (a) without (b) → tolerance is not sufficient.
    - Not (a) → noise training does not produce tolerance at these levels; F3 stays correlational.
- **21:53 Check-in (new chain, trig_01CB…).** Continuing (k). This chain schedules the next check-in.
  - Process note: the first parallel launch of U10b thrashed (two processes × two torch threads on two cores; 6 CPU-minutes without finishing a net). I killed it and relaunched with one thread each. No results were read from the killed runs.
- **21:53 U10b RESULT: NOT SUPPORTED as pre-registered, but the trend is in the predicted direction** (`results/glass/units/u10b_m85_60000_*.json`, read by hand).
  - **Known answer passes:** σ = 0 accuracy 0.867 / 0.884 / 0.757, ε95 0.0063 / 0.0079 / 0.0063 (AlgZoo-like).
  - σ* = 0.1: σ = 0.3 nets fall to accuracy 0.55–0.62, below the 0.70 bar.

    | training noise σ | accuracy | ε95 | joint substitution |
    |---|---|---|---|
    | 0 | 0.76–0.88 | 0.006–0.008 | 42–51% |
    | 0.03 | 0.80–0.88 | 0.016–0.020 | 68–73% |
    | 0.1 | 0.73–0.74 | 0.020 | 69–73% |
    | 0.3 | 0.55–0.62 | 0.020–0.025 | 75–78% |

  - **(a) fails:** median ε95 at σ* is 3.2× the σ = 0 median (bar 5×).
  - **(b) fails:** no net reaches 90% joint substitution.
  - Decision per the rule: noise training does not produce enough tolerance at these levels, so F3 stays correlational.
  - **Descriptive, within the intervention:** across the 12 nets, ε95 and joint substitution rise together (Spearman 0.89, p = 1e-4), and ε95 and accuracy move against each other (Spearman −0.70, p = 0.01). Tolerance saturates near 2–2.5%. Pushing harder (σ = 0.3) buys accuracy loss, not tolerance.
  - **Reading (an interpretation, to test):** for this task at this size, precision appears to be the price of accuracy. An 8-unit ReLU RNN that tolerates imprecision is a worse 2nd-argmax solver. If that holds, approximate unit accounts of GOOD solvers of such tasks cannot be faithful, by necessity rather than by accident.
- **21:53 Error noted (hand-typed time, the third time):** the check-in I just scheduled (trig_01EouLH7BRh3PiE3jHtTRNny, fires ~22:19) says "as of 21:59 ET". I typed that by hand; the clock read 21:53. It only labels which NEXT item was current, so nothing depends on it, but it breaks the CLAUDE.md rule. From now on, scheduled-message text carries no times at all.
- **22:20 Check-in (trig_01Eou…).** Doing (k2).
- **22:20 U12 PRE-REGISTRATION ((k2): is precision the price of accuracy at a given capacity?).**
  - **Change of plan, with the reason:** ARC publishes fully trained 2nd-argmax RNNs at n = 5 with hidden 16 and 32 (5 seeds each; checked HTTP 200). They are trained far longer than my 60,000-step runs and by ARC's own recipe, so they remove my training as a confound. I use them instead of training wider nets myself.
  - **Networks:** 2nd argmax n = 5, hidden h ∈ {8, 16, 32}, seeds 0–4 (h = 8 = the U8 seed models). 15 networks.
  - **Known answer:** the h = 8 numbers reproduce U8 (ε95 0.006–0.013).
  - **Measures:** accuracy; ε95 (P3); unit map v3 coverage; joint substitution of all library-explained neuron-steps (U10's procedure).
  - **Confound stated in advance:** wider nets have more noised coordinates. Independent noise can average out across redundant neurons, so ε95 may rise for that reason alone. Joint substitution (systematic unit errors, not independent noise) is therefore the decisive explainability measure.
  - **Decision.**
    - "Capacity buys tolerance" if the h = 32 median ε95 ≥ 3× the h = 8 median, with median accuracy ≥ h = 8's.
    - "And wider is more explainable by approximate units" if, in addition, the h = 32 median joint substitution ≥ h = 8's + 20 points.
    - ε95 rising < 1.5× → precision is not relaxed by capacity at this task.
    - Otherwise mixed.
- **22:22 U12 RESULT: MIXED by the pre-registered rule; capacity raises accuracy, tolerance and substitutability together** (`results/glass/units/u12_width_*.json`, read by hand).
  - **Known answer:** h = 8 reproduces U8 exactly (ε95 0.0063, 0.0079, 0.0100, 0.0126, 0.0063).

    | hidden h | accuracy (median) | ε95 (median, range) | joint substitution (median, range) |
    |---|---|---|---|
    | 8 | 0.865 | 0.0079 (0.006–0.013) | 43% (38–73%) |
    | 16 | 0.991 | 0.0200 (0.010–0.025) | 73% (62–75%) |
    | 32 | 0.996 | 0.0158 (0.013–0.020) | 78% (75–83%) |

  - **Rule:** the h = 32 / h = 8 ε95 ratio is 2.0, between 1.5 and 3, so "mixed". The joint-substitution condition only counts if the first holds; descriptively it rose by 35 points.
  - **Reading:**
    - "Precision is the price of accuracy" holds only at low capacity.
    - With more neurons, ARC's networks become more accurate (0.87 → 0.996), about 2× more tolerant, and much better described by approximate units (43% → 78% of decisions kept).
    - They stay hypersensitive by the modular-net standard (ε95 1–2% vs 0.79), and below 90% joint substitution.
    - The noise-averaging confound (more neurons) could explain part of the ε95 rise, but not the joint-substitution rise.
  - **Next question this raises:** is ARC's challenge network (M16,10) hard partly because it is capacity-starved? ARC publishes 2nd argmax h = 32, n = 10.
- **22:23 U13 PRE-REGISTRATION ((k3): is the n = 10 task's difficulty partly capacity?).**
  - **Networks:** ARC's 2nd argmax n = 10, h = 16 (zoo seeds 0–4, the U8 models) vs h = 32 (seeds 0–4, available). ARC's challenge M16,10 is an h = 16, n = 10 network.
  - **Known answer:** h = 16 ε95 reproduces U8 (0.003–0.005).
  - **Measures:** as U12 (accuracy, ε95, unit-map coverage, joint substitution).
  - **Decision.**
    - "The n = 10 networks are capacity-starved in the U12 sense" if the h = 32 median joint substitution ≥ the h = 16 median + 20 points, with median accuracy ≥ h = 16's.
    - Otherwise, width does not make the n = 10 task's networks more explainable.
  - ε95 is reported descriptively, because of the noise-averaging confound.
- **22:25 U13 RESULT: NOT MET by the pre-registered rule, but the structure changes sharply with width** (`results/glass/units/u12_width_{16,32}_n10.json`, read by hand).
  - **Known answer:** h = 16 ε95 reproduces U8 (0.0032 ×4, 0.0050).

    | n = 10, hidden | accuracy (median) | ε95 (median) | history-unit coverage (median) | joint substitution (median) | variance substituted (median) |
    |---|---|---|---|---|---|
    | 16 | 0.914 | 0.0032 | 0.06 | 37% | 7% |
    | 32 | 0.993 | 0.0079 | 0.51 | 52% | 60% |

  - **Rule:** +15 points of joint substitution against a +20 bar → not met.
  - **Caveat:** the comparison favours h = 16, which substitutes almost nothing (2–29% of variance) and so keeps decisions cheaply. h = 32 substitutes 54–70% of its variance and still keeps more. My pre-registered rule did not adjust for that, and I am not changing it after the fact.
  - **Descriptive, and striking:** at h = 32, about HALF of all neuron-step variance is single library units (0.49–0.67), against 0.02–0.22 at h = 16.
    - The two h = 32 seeds I inspected both contain the comparators (max − x_t, x_t − 2nd, 2nd − x_t) plus running-max holders.
    - The capacity-starved h = 16 networks blend their units; the wider ones dedicate neurons to them.
  - **Prior work:** this matches superposition theory (Elhage et al., "Toy Models of Superposition", 2022): when a network needs more features than it has neurons, it blends them across neurons, and extra neurons reduce the blending. Here it shows up as units, in a recurrent algorithmic network.
  - **Bearing on ARC's challenge:** M16,10 is in the capacity-starved regime. Its siblings with twice the width are about half single-neuron units.
- **22:25 U14 PRE-REGISTRATION (replicate "wider → more unit-like" on a second task).** Argmedian n = 5 and n = 7 at h = 16 (seeds 0–4; n = 5 are the U8 models) vs h = 32 (ARC, seeds 0–4; available).
  - **Measures:** as U12.
  - **Decision:** "wider networks are more unit-like" replicates if the h = 32 median history+raw coverage ≥ 2× the h = 16 median at BOTH n = 5 and n = 7. Otherwise it does not replicate on argmedian.
- **22:28 U14 RESULT: does NOT replicate on argmedian** (`results/glass/units/u12_width_argmedian_16-32*.json`, read by hand).
  - **Known answer:** argmedian 16_5 seeds reproduce U8 (ε95 0.0100 ×4, 0.0079; history coverage 0.16 / 0.02 / 0.05 / 0.03 / 0.01).

    | argmedian | h | accuracy (median) | history+raw coverage (median) | ε95 (median) | joint substitution (median) |
    |---|---|---|---|---|---|
    | n = 5 | 16 | 0.972 | 0.37 | 0.010 | 65% |
    | n = 5 | 32 | 0.995 | 0.60 | 0.016 | 75% |
    | n = 7 | 16 | 0.700 | 0.24 | 0.005 | 52% |
    | n = 7 | 32 | 0.968 | 0.20 | 0.005 | 58% |

  - Coverage ratio 1.6 (n = 5) and 0.8 (n = 7), against a bar of ≥ 2 at both → not replicated.
  - Accuracy rises a lot with width (n = 7: 0.70 → 0.97), but the library does not describe the wider argmedian networks any better.
  - **Possible reason, not tested:** the library is built from 2nd-argmax quantities (max, 2nd, min). Argmedian needs rank and median quantities; U9 found one soft-rank neuron. So "wider → more unit-like" is shown for 2nd argmax with this library, and is open for argmedian.
- **22:55 Check-in (trig_01WDV…).** First open NEXT item: (m).
- **22:55 U15 PRE-REGISTRATION ((m): with units found from data, are wider argmedian networks more unit-like?).**
  - **Networks:** argmedian n = 5 and n = 7, h = 16 vs h = 32, seeds 0–2 (12 networks). Three seeds rather than five, to keep this to minutes.
  - **Method:** U9's search, unchanged (`scripts/glass/units/u9_search.py`), on every neuron-step (step ≥ 2) that the v3 library leaves below R² 0.9. Library fits are recomputed with `unit_map(confidence=False)`.
  - **Known answer:** argmedian 16_5 seed 0 must reproduce U9's explained share for zoo_argmedian_16_5 (0.33).
  - **Measure:** explained share = the variance-weighted share of the library-unexplained neuron-steps that reach held-out R² ≥ 0.9.
  - **Decision.**
    - "Wider → more unit-like" holds on argmedian with data-found units if the h = 32 median explained share ≥ 2× the h = 16 median at BOTH n = 5 and n = 7.
    - Otherwise it does not, and U14's negative is not just about the library.
  - **Descriptive:** total explained variance (library units plus searched units) per network.
- **22:57 U15 RESULT: NEGATIVE** (`results/glass/units/u15_{5,7}.json`, read by hand).
  - **Known answer:** argmedian 16_5 seed 0 reproduces U9 exactly (search explains 0.33 of the rest).
  - **Search's explained share of the library-unexplained variance** (median of 3 seeds):
    - n = 5: h = 16 0.33, h = 32 0.41 (ratio 1.2);
    - n = 7: h = 16 0.02, h = 32 0.06 (ratio 3.0).
  - The bar (≥ 2× at both n) is not met.
  - **Total explained (library plus search), medians:** n = 5 0.63 vs 0.64; n = 7 0.34 vs 0.32. No change with width.
  - **Decision:** "wider → more unit-like" does not hold on argmedian, even with units found from data (sparse, ≤ 3 terms). U14's negative is not just the library. So the U13 effect is task-specific so far (2nd argmax at n = 10).
  - **Descriptive:** argmedian at n = 5 is about two-thirds sparse units at both widths; at n = 7, about a third.
- **22:58 (l) DONE:** one-page summary at the top of `docs/glass/UNITS_FINDINGS.md` covering U1–U15 and P2–P5 (holds up / did not hold up / new / meaning for Glass Networks). Next: (n) prior-work check of the four 'new' items; (o) replicate the width effect at n = 8.
- **23:26 Check-in (trig_01RbF…).** (n) prior-work check of the four "possibly new" items. Searches: AlgZoo argmedian; width and monosemanticity; noise tolerance as a predictor of interpretability; universality across seeds and AlgZoo seeds.
  1. **ε95 as a predictor of when approximate accounts can be faithful.**
     - Nothing found that measures this.
     - Nearest: Arora et al. 2018 (noise stability makes nets compressible); Ghorbani et al. 2017 (fragility of saliency explanations, a different notion); Ladner and Althoff 2026 (formal faithfulness of replacement networks under input perturbations, a different question).
     - **Stays a candidate.**
  2. **Argmedian unit maps above 32 parameters.** No public interpretation found. xabier.me covers a 32-parameter network and notes most work focuses on 2nd argmax. **Stays.**
  3. **Width → more single-neuron units.** **DOWNGRADED:** the mechanism is known.
     - Superposition theory (Elhage et al. 2022) predicts it.
     - "Expand Neurons, Not Parameters" (Kong et al., ICML 2026) shows that more neurons at a fixed parameter count reduce polysemanticity and interference.
     - Ours is a new observation of a known mechanism, on ARC's networks.
  4. **Convergence of units across training runs.** **DOWNGRADED:** the question is established.
     - Gurnee et al. 2024: 1–5% of GPT-2 neurons are universal across seeds.
     - For the M16,10 recipe, a commenter on Josh Snider's LessWrong post says the representations "vary a lot depending on the seeds". That agrees with our 16-neuron result (3–4 of 5).
     - Snider's estimator counts neuron roles (comparator, max detector, write pointer) across about 8,000 trained models, but does not report how often each role appears.
     - What remains ours: the measured 5/5 convergence at 8 neurons, with controls.
- **23:26 U16 PRE-REGISTRATION ((o): is U13's width effect general across lengths, or specific to n = 10?).**
  - **Networks:** ARC's 2nd argmax n = 8 at h = 8, 16 and 32, seeds 0–4 (available).
  - **Known answer:** h = 8 seed 0 reproduces our zoo_2nd_argmax_8_8 (ε95 0.0010, coverage about 0).
  - **Measure:** history+raw coverage (the U14 measure), plus accuracy, ε95 and joint substitution as in U12.
  - **Decision.** The width effect generalizes to n = 8 if the h = 32 median history+raw coverage ≥ 2× the h = 16 median. Otherwise it is specific to n = 10 so far.
- **23:28 U16 RESULT: the width effect GENERALIZES to n = 8** (`results/glass/units/u12_width_*_n8.json`, read by hand).
  - **Known answer:** h = 8 seed 0 reproduces zoo_2nd_argmax_8_8 (ε95 0.0010, coverage 0.00).

    | n = 8, hidden | accuracy (median) | history+raw coverage (median) | ε95 (median) | joint substitution (median) |
    |---|---|---|---|---|
    | 8 | 0.43 | 0.02 | 0.0032 | 73% (on 0–4% of variance) |
    | 16 | 0.96 | 0.18 | 0.0079 | 48% |
    | 32 | 0.99 | 0.46 | 0.0100 | 73% |

  - h = 32 / h = 16 coverage ratio 2.6 (bar 2) → generalizes.
  - **Descriptive pattern across all 2nd-argmax configurations measured:** unit-likeness tracks hidden size relative to sequence length.
    - At h = 16, coverage falls as n grows: n = 5 about 0.37, n = 8 0.18, n = 10 0.06.
    - At h = 32 it stays high: 0.40–0.60.
    - The most cramped networks (h = 8, n = 8) are almost fully blended (coverage ≈ 0) and inaccurate (0.43).
- **23:57 Check-in (trig_01D2m…).** (q), the reader payoff test. I am **not running it as written**, for two reasons found while preparing the pre-registration.
  1. **Confound in the criterion.** The strict criterion requires fidelity above the TASK RULE, i.e. above the network's own accuracy. ARC's wide M32,10 is 99.4% accurate, so an account must agree with it above about 99.5%. The cramped zoo M16,10 seed 0 needs only about 91%. A reader comparison between them measures the bar, not unit-likeness.
  2. **Circling risk.** R8 showed that cheap-reader runs fall into the same two modes on every dense network (copies, or short accounts below the task rule). Two runs per network cannot tell the networks apart, and another reader batch is the circling CLAUDE.md warns against.
  - **Replacement:** a reader-free test that asks the same question without either problem.
- **23:57 U17 PRE-REGISTRATION (how much of a network can rules replace while still beating the task rule?).**
  - **Replacement curve:**
    - Take the network's library-explained neuron-steps (unit map v3, held-out R² ≥ 0.9, step ≥ 2).
    - Replace them by their units one at a time, best fit first.
    - After each, measure decision fidelity on the held-out half.
    - **R\*** = the largest share of step-≥2 variance replaced while fidelity stays ≥ the network's accuracy (the task-rule bar).
  - **Decisive pair (matched accuracy, same architecture and length):** ARC's challenge M16,10 (accuracy 0.953, history coverage 0.37) vs zoo M16,10 seed 2 (accuracy 0.949, coverage 0.10).
  - **Group comparison** (bars differ, reported with that caveat): zoo n = 10 h = 16 vs h = 32, seeds 0–4.
  - **Predictions:**
    - (i) From unit-likeness: challenge R\* > seed-2 R\*.
    - (ii) From F3: both R\* < 0.3, since ε95 is about 0.3–0.6%.
  - **Decision.**
    - (i) and (ii) → unit-richness lets more of a network be replaced by rules, but precision caps it.
    - (i) fails → unit-richness does not translate into replaceability.
    - (ii) fails (R\* ≥ 0.3) → F3 overstates the cap.
- **23:59 U17 RESULT: both predictions hold, and the cap is severe** (`results/glass/units/u17_*.json`, read by hand).
  - **Decisive pair:** ARC's challenge M16,10 R\* 0.05 vs zoo seed 2 R\* 0.00 → (i) holds, by a small margin. Both are < 0.3 → (ii) holds.
  - **Group:** zoo n = 10 at h = 16, R\* 0.00–0.03; at h = 32, 0.02–0.07, even though h = 32 must clear a 99%+ bar.
  - **Cap:**
    - In every network, rules can replace at most 7% of the step-≥2 variance before the account falls below the task rule.
    - That holds even where 50–70% of the variance IS recognisable units at R² ≥ 0.9.
    - Replacing all of them drops fidelity to 23–73%.
  - **Decision:** unit-richness helps a little; precision caps replacement at a few percent. A units-plus-wiring construction cannot meet the strict criterion on these networks. Only partial accounts with a small replaced share can.
- **23:59 U18 PRE-REGISTRATION (for the next check-in): is ε95 the right admission rule for replaceable parts?**
  - **Question:** if only SUBSTITUTION-GRADE units (relative error ≤ ε95, unit map v3) are replaced, all at once, does fidelity stay ≥ the task rule?
  - **Networks:** all 11 v3 maps in `results/glass/units/maps/` plus the 10 n = 10 seed networks.
  - **Decision.**
    - Fidelity ≥ the task rule in ≥ 80% of networks → ε95 is a valid admission rule. Build step (CHANGES entry first): the catalog marks substitution-grade parts as "replaceable", and states the joint fidelity of replacing them all.
    - Otherwise per-unit admission by ε95 is not enough, because errors add up; record the gap.
- **00:26 Check-in (trig_012aW…).** Running U18 exactly as pre-registered.
  - Count correction: `results/glass/units/maps/` holds 10 v3 maps (6 2nd argmax, 4 argmedian), not 11. With the 10 n = 10 seed networks that makes 20; the 80% bar means ≥ 16 of 20.
  - Clarification (no change to the rule): "all substitution-grade units" includes step 1, where they are exact raw-input units.
- **00:28 U18 RESULT: PASSES by the rule (19 of 20), but the admitted parts are mostly trivial** (`results/glass/units/u18_*.json`, read by hand).
  - Joint fidelity when every substitution-grade unit is replaced at once is ≥ the task rule in 19 of 20 networks.
  - The one failure is M2,2: fidelity 0.974 against a bar of 1.000, since it is 100% accurate on these inputs and any error fails.
  - **Honest reading:**
    - Above M4,3, the admitted parts are almost all step-1 raw-input units, which are exact.
    - After step 1 they cover 0% of the variance in 15 of 20 networks, and at most 6% (argmedian 16_5 5%, M32,10 seeds 1–6%).
    - Only M2,2 (54%) and M4,3 (51%) admit substantial non-trivial parts.
  - So ε95 is a SAFE admission rule (joint replacement stays above the task rule), but it admits little. This matches U17's cap of a few percent.
  - My pre-registration did not exclude step 1, which made the test weaker than it looks. Noted, and I am not changing the rule after the fact.
  - **Decision (per the rule):** build the catalog's "replaceable parts" line. It must show the share AFTER step 1, so it cannot overstate. CHANGES T6 first.
- **00:32 T6 DONE: replaceable parts in the unit map (v4) and the catalog.**
  - `t6_test.py` failed before (5 FAILs) and passes after; held-out networks match U18 exactly; all v3 fields are unchanged in the 10 regenerated maps; the T5 test still passes.
  - Page text read by hand:
    - argmedian 16_5: "5% of the network's activity after step 1 … keeps 99.6% of decisions; the task rule alone keeps 98.4%".
    - M4,3: "51% … keeps 99.3%; task rule 98.1%".
  - Next: (s), a synthesis step with no new runs.
- **01:00 Check-in (trig_01US4…). (s) DONE (synthesis, no runs):** added 'Where the units program leaves Glass Networks' to UNITS_FINDINGS. It sets the plan (§3–§4) against the outcomes, says what to build next (choose targets by ε95; exact-structure route via SPD; unit map as the partial product), and lists decisions D1–D5 for Shreyas. Numbers cited there were checked against U17, U8 and P5 results. Next: (t) SPD feasibility.
- **01:27 Check-in (trig_01YLT…). (t) step 1, SPD feasibility and prior work.**
  - **Tool:** Goodfire's library (github.com/goodfire-ai/spd; paper Bushnaq, Braun and Sharkey 2025) clones fine here. The maintained trainer is JAX/multi-GPU and LM-only. The compact reference (`nano_param_decomp/run.py`, 1,212 lines, torch) is written for token-sequence LMs: wandb, a transformer causal-importance function, KL on logits. It cannot run on our RNNs as is.
  - **Prior work:** I found no SPD or APD decomposition of AlgZoo or of any ReLU RNN.
  - **Feasibility:** not minutes. A faithful SPD for our RNNs means re-implementing the core method for a recurrent network: rank-1 subcomponents of W_x, W_h and W_o; a causal-importance function per subcomponent; stochastic masks; faithfulness, stochastic-reconstruction and importance-minimality losses. About 150–250 lines.
  - **Known-answer problem:** M4,3's correct decomposition into PARAMETER components is not defined by ARC's account (signed channels are an activation-level description), so it cannot serve as the known answer. The known answer has to be planted.
  - **Decision:** a staged build (U19), each stage gated by its own known-answer test:
    - (a) core SPD on a single linear layer, tested on a planted toy whose ground-truth components are known by construction (one component per sparse feature, as in the SPD paper's toy models);
    - (b) extension to a ReLU RNN, tested on a hand-built RNN with two planted mechanisms;
    - (c) only then M8,5.
  - This is the one plan item never run (U2's decomposition) and costs no money. If Shreyas chooses a different direction under D4, it stops.
- **01:29 Process correction (Shreyas asked why there were 25-minute breaks).**
  - Nothing was running in the background. I had turned "use scheduled check-ins to survive compaction" into a cadence: work a chunk, end the turn, idle until the next check-in. That wasted most of every half hour.
  - **Fix:** work continuously, one experiment straight into the next. Compaction happens on its own. The scheduled check-in is only a safety net that restarts work if the session has stopped. Updates to Shreyas go out as results arrive, not on a timer.
- **01:29 U19(a) PRE-REGISTRATION (core SPD on a planted known answer).**
  - **Planted toy:** Anthropic's toy model of superposition, TMS 5-2 (the SPD paper's own known answer). 5 sparse features (each on with probability 0.05, value U[0,1]), hidden size 2, x̂ = relu(W2 W1 x + b) with W2 = W1ᵀ trained tied. Trained here first.
    - Ground-truth mechanisms: per feature j, W1[:, j] e_jᵀ in W1, and e_j W1[:, j]ᵀ in W2.
  - **Method:** SPD after Bushnaq, Braun and Sharkey (2025).
    - Each matrix is written as Σ_c U_c V_cᵀ with C = 20.
    - A causal-importance MLP per subcomponent reads V_c · (the layer's input).
    - Stochastic masks m = g + (1 − g)·r.
    - Losses: faithfulness, stochastic reconstruction (MSE of outputs) and importance minimality Σ|g|^p.
  - **Known answer (must pass before stage b):**
    - In each matrix, exactly 5 subcomponents are alive (mean causal importance > 0.01 over sparse inputs).
    - They match the 5 planted mechanisms one-to-one, with cosine ≥ 0.9 between flattened rank-1 matrices.
    - Each alive component's importance is high (> 0.5) on ≥ 90% of inputs where its feature is on, and low (< 0.1) on ≥ 90% of inputs where all features but another one are off.
  - Hyperparameters may be tuned on this planted case; the pass criterion above is fixed now.
- **01:52 U19(a) progress (tuning on the planted case, allowed by the pre-registration).**
  - **First configuration:** my own initialisation, faithfulness 1e3, lr 2e-3. It got stuck: reconstruction about 5e-3 while the ideal decomposition gives 2.5e-4 (computed directly), and 13–19 components alive.
  - **Fix:** the SPD code release's initialisation (unit-norm random U and V, U scaled by its overlap with W), with the paper's faithfulness 1.0 and lr 1e-3. A 12k-step run then improved but split features across two components each (FAIL).
  - **A 30k-step run** reached 5 alive per matrix with reconstruction 2.6e-4 by step 15,000; it is still running. A p = 0.9 variant is running in parallel, since splitting a feature costs nothing under p = 1.
- **01:52 U19(b) PRE-REGISTRATION (written now, run only after (a) passes).**
  - **Planted RNN:** h_t = relu(W_x x_t + W_h h_{t−1}), out = W_o h_6, with W_x = W_h = W_o = I (5×5). The input is 5 sparse features per step (each on with probability 0.03, value U[0,1]); the network sums each feature over 6 steps.
  - **Planted mechanisms:** e_j e_jᵀ in each matrix. Each is needed exactly where its input coordinate is non-zero:
    - W_x: at the steps where feature j arrives;
    - W_h: at the steps after feature j has arrived;
    - W_o: if feature j arrived at all.
  - **SPD:** the same code with W_h's importances computed per step (`spd_core` handles repeated calls), using the configuration that passes (a).
  - **Pass:** per matrix, exactly 5 alive, a one-to-one match at cosine ≥ 0.9, and importance > 0.5 on ≥ 90% of the calls where the mechanism is needed and < 0.1 on ≥ 90% of the calls where only others are needed.
  - **Script:** `scripts/glass/units/u19b_rnn.py`.
- **02:03 U19(a) RESULT: FAIL by the pre-registered criterion** (30k steps, p = 1; `results/glass/units/u19a_tms_20_30000_3e-3_1.0_1.0_1e-3_2048.json`).
  - **What passed:**
    - 5 alive per matrix, matched one-to-one to the 5 features.
    - W2: all 5 components match their planted mechanisms (cosine 0.993–0.996).
    - W1: 3 of 5 match perfectly (cosine 1.00).
    - Importance is selective: low on 80–100% of the inputs where only another feature is on.
  - **What failed:**
    - Two W1 components are not clean (cosine 0.19 and 0.69).
    - Importance is > 0.5 on only 53–72% of the inputs where the feature is on (bar 90%).
  - **Diagnosis, to be checked and not assumed:**
    - (1) **The importance criterion conflates "feature on" with "mechanism needed".** TMS has a bias of about −0.25, so a feature with value below about 0.2 produces output 0 whether or not its mechanism is present. SPD is then right to give it low importance. That is an error in my test's intent, not in the code.
    - (2) **Dead components can absorb weight** (faithfulness constrains only the sum of all 20). An alive component may then carry off-feature entries that a dead one cancels, which lowers the raw cosine without changing the function.
  - **Next:**
    - Rerun with model saving, so I can check (1) directly: importance against x_j and against the target output.
    - Write the revised criterion BEFORE that rerun, and judge it on a fresh seed (seed 1), not on the run that suggested it.
  - The p = 0.9 variant ended with 4 alive in W2 (a feature merged or dropped), so p = 1 stays.
- **02:03 U19(a') PRE-REGISTRATION (revised known-answer criterion; fresh seed 1).**
  - **Configuration:** C = 20, p = 1, minimality 3e-3, faithfulness 1, lr 1e-3, 30k steps.
  - **(i) Structure:**
    - 5 alive per matrix, matched one-to-one.
    - The alive components alone reconstruct W (relative error ≤ 0.1).
    - Each alive component's input direction (V_c for W1) or output direction (U_c for W2) aligns with its feature's planted direction (|cosine| ≥ 0.9).
  - **(ii) Importance:**
    - > 0.5 on ≥ 90% of the inputs where its feature is on AND the target output for that feature is > 0.05 (the mechanism is needed);
    - < 0.1 on ≥ 90% of the inputs where only another feature is on.
  - Pass both → stage (b) results count.
- **02:06 U19(b) compute note:** at batch 2048 the RNN decomposition ran about 500 steps per 3 minutes (about 3 hours for 30k). I restarted it with batch 512 and 20k steps (about 30–40 minutes), before looking at any of its results. Only the training budget changed; the method and the pass criterion are as pre-registered.
- **02:08 U19(c) PRE-REGISTRATION (SPD on M8,5; runs only after (a') and (b) pass).** Script: `scripts/glass/units/u19c_m85.py`.
  - **Target:** zoo_2nd_argmax_8_5 (ARC's seed 3), with the shuffled-weights version as control.
  - **Setup:** W_x, W_h and W_o are decomposed (C = 20 each), with the U19(a') configuration, 30k steps. Logits are divided by their spread, so the reconstruction loss is relative.
  - **Measures:**
    - alive components per matrix;
    - mean number of components in use per call (L0, importance > 0.1);
    - decision fidelity with the importance-masked model (masks = importances) and with alive components only;
    - for each alive W_h component, the best balanced accuracy at predicting its importance > 0.5 from one task event at that step: x_t is a new max; x_t is a new 2nd; x_t is below the 2nd; the max was at t−1; x_t > 0; or the negation of any of these.
  - **Decision.** "SPD yields a units decomposition of M8,5" if all of these hold:
    - (i) the importance-masked fidelity is ≥ the task rule (0.879);
    - (ii) W_h's mean L0 per call is ≤ half its alive count (sparse use);
    - (iii) ≥ 50% of alive W_h components are event-labelled (balanced accuracy ≥ 0.9), against ≤ 20% in the shuffled control.
  - Otherwise SPD does not give a readable exact decomposition of M8,5; record which part failed.
- **02:23 U19(a') RESULT (seed 1): FAIL by the revised criterion, by one number** (`results/glass/units/u19a2_tms_seed1.json`).
  - **W1 PASSES fully:**
    - 5 alive, one-to-one;
    - input directions match the features at cosine 1.00;
    - alive-only reconstruction error 0.004;
    - importance high when needed on 92–94% of inputs, low when only another feature is on on 100%.
  - **W2:**
    - Structure passes: directions 0.991–0.996, reconstruction error 0.024, high when needed 90–93%.
    - Selectivity fails: one component (feature 1) is low on only 88.8% of the inputs where only another feature is on (bar 90%).
  - **Diagnosis, checked directly on the saved model (200k fresh inputs):**
    - For three W2 components, EVERY case where importance ≥ 0.1 while only another feature is on is a case where the target's output for that feature is non-zero (TMS interference: a strong neighbouring feature leaks into output j through the 72° overlap).
    - In those cases the mechanism IS causally needed, and SPD is right.
    - Where importance is < 0.1, the target output is non-zero in only 6% of cases.
    - So "only another feature is on" does not mean "not needed" for W2. That is a second error in how I defined the TMS test, not in the code.
  - **Decision:**
    - I will NOT revise the TMS criterion a third time; that would be moving the goalposts.
    - What the TMS runs establish: SPD recovers the planted structure in both matrices, on two seeds, and its importances track causal need.
    - The clean gate is (b), the planted RNN with no superposition, so no interference: "needed" is exact there. (b)'s pre-registered criterion now gates (c) alone.
    - This is a plan change made after seeing (a'), recorded here with its reason.
- **02:27 U19(b) RESULT (20k steps, batch 512): FAIL** (`results/glass/units/u19b_rnn_20_20000_3e-3_1.0_1.0_1e-3_512.json`).
  - W_o: structure perfect (5 alive, cosine 1.00), importance high when needed on 80–82% (bar 90%).
  - W_x: 6 alive, feature 2 split into two components.
  - W_h: 7 alive, features 2 and 4 split. Importance high on 84–91%.
  - **Reading:**
    - (1) Splitting means it is not converged. In TMS, splitting cleared between 12k and 15k steps at batch 2048; this run had a quarter of the batch.
    - (2) The "high when needed" shortfall has the same form as in TMS. The criterion counts any non-zero input as "needed", but SPD's importance is causal, so a mechanism carrying a tiny value (U[0,1] inputs) may legitimately get low importance.
- **02:27 U19(b') PRE-REGISTRATION (before the long run's results exist).** I stopped the 40k run that had just started (no results read) and restarted it with model saving and this criterion, fixed now:
  - **"needed"** = the mechanism's input coordinate is ≥ 0.2;
  - **"only others needed"** = its coordinate is exactly 0 and some other coordinate is ≥ 0.2;
  - **pass** = exactly 5 alive per matrix, one-to-one, cosine ≥ 0.9, importance > 0.5 on ≥ 90% of needed calls and < 0.1 on ≥ 90% of only-others calls.
  - Run: 40k steps, batch 512. If it fails, I check the saved model for the cause before any further change.
- **02:28 U19(c) launched early on the free core (original M8,5, 30k steps, batch 512 for compute; the pre-registered method is otherwise unchanged). Its result counts only if (b') passes. The shuffled control runs after it.**
- **03:09 U19(b') RESULT: PASS, cleanly** (`results/glass/units/u19b_rnn_20_40000_3e-3_1.0_1.0_1e-3_512.json`, read by hand).
  - In all three matrices (W_x, W_h, W_o): exactly 5 alive, one-to-one, cosine 1.00 with the planted mechanisms.
  - Importance > 0.5 on 100% of needed calls and < 0.1 on 100% of only-others calls.
  - So the SPD implementation, including per-step importances for the recurrent W_h, recovers planted recurrent mechanisms exactly. The gate for (c) is passed.
- **03:09 U19(c) RESULT (M8,5, original): NOT a readable decomposition by the pre-registered rule; (iii) fails** (`results/glass/units/u19c_zoo_2nd_argmax_8_5_original_30000_512.json`).
  - **(i) passes:** decision fidelity with the importance-masked model is 0.901, above the task rule's 0.879. Alive components alone give 0.979.
  - **(ii) passes:** W_h has 11 alive rank-1 components, and about 4.8 of them are in use per step (half would be 5.5). W_x has 4 alive; W_o has 7 alive, about 4.8 in use.
  - **(iii) fails:** none of the 11 W_h components' use is predicted by a single task event. The best balanced accuracies are 0.71–0.84, e.g. "not (max was at t−1)" 0.84 and "not (x_t is a new max)" 0.82, against a bar of 0.9.
  - **Reading:**
    - SPD finds an EXACT, sparsely used decomposition of M8,5: about 5 of 11 recurrent mechanisms per step, and faithful above the task rule. The unit map could not do that (U17: at most 7% replaceable).
    - But the mechanisms' usage does not line up with single task events. Event-like structure is partly there (0.84), but not clean.
    - The shuffled control is running; it decides whether even (i) and (ii) are special to the trained network.
- **03:10 Direction from Shreyas: build, not more reports.** He wants a working system that does real, discovery-grade interpretability on networks larger than we have managed so far, reached by trying many ways to build it and testing each, not by more experiments for their own sake.
- **03:10 BUILD DECISION, from the evidence: a "decompose → explain → certify" pipeline**, the new constructor route for fragile networks.
  - **Why this and not more reader work:** U17 and P3 show approximate accounts cannot be faithful there. U19(c) shows an exact, sparsely used decomposition IS faithful on M8,5 (0.90 against a task rule of 0.88). The missing piece is a readable trigger for each mechanism.
  - **Components:**
    - (1) SPD engine: `spd_core`, validated on the planted RNN.
    - (2) Trigger explainer: for each mechanism, a readable rule for when its importance is high. Learned (shallow rules over interpretable features) and/or written by the reader, scored on held-out inputs.
    - (3) Construction plus certificate: the network as Σ mechanisms, with importance functions REPLACED by the readable rules. Measured: fidelity against the task rule; per-mechanism confidence (rule accuracy); an ablation specificity check (removing mechanism c changes decisions mainly where its rule says it is on).
    - (4) Catalog integration.
  - Each component gets a known-answer test before use (the planted RNN first).
- **03:13 PLAN CHANGE, approved by Shreyas at 03:11 ET:** from the units program (experiments) to building the **mechanism atlas** (decompose → explain → certify).
  - The new current-work document is `docs/glass/ATLAS.md`: why, intent, components with tests and status, ordered next steps, how to run.
  - Routine updated in `CLAUDE.md`:
    - read ATLAS first;
    - building is the goal, and experiments only answer a named build step's question;
    - when something falls short, try another way;
    - work continuously; the check-in is only a safety net; update Shreyas when results land.
  - CHECKLIST NOW, the HANDOFF header and the UNITS_PROGRAM header now point to ATLAS.md.
  - Rule-gated construction prototype on M8,5: fidelity 0.895 (depth 5, task + neuron features) against the task rule's 0.879. Importance-masked fidelity is 0.918–0.927, and all alive mechanisms on gives 0.981 (`results/glass/units/rule_construction_*`).
- **03:13 INTENT.md addendum** (approved plan change): the library gets a second kind of entry, the mechanism atlas, for fragile networks. Constructions remain the route for tolerant ones. I also corrected two hand-typed times I had just written (ATLAS status and CHECKLIST NOW said 03:30; the clock said 03:13). Times are now set from `date`.
- **03:17 A1 DONE: the mechanism atlas v1** (`lab/spd.py`, `lab/atlas.py`; CHANGES A1).
  - **Known answer:** `scripts/glass/atlas_test.py` failed before (no module) and passes after. On the planted RNN it gives 5 mechanisms per matrix, with readable triggers that are exactly the planted ones ("ON when feature 4 arrives now", "ON when neuron 1 active in the state before this step"), confidence ≥ 0.98 and rule-gated fidelity 0.997.
  - **M8,5 (design case):**
    - W_h triggers at ≥ 0.9: 8/11 at depth 3, 9/11 at depth 4. W_o triggers are weak (0.60–0.65).
    - Gating every mechanism by its rule gives 0.71–0.74 (the weak W_o triggers hurt).
    - **Trusted policy:** a trigger is used only at confidence ≥ 0.9, the labelling bar used throughout; otherwise the mechanism stays always on. That gives 0.896 / 0.924 / 0.918 at depths 3 / 4 / 5, all above the task rule (0.879).
    - Depth 4 is fixed from here on, for the held-out networks.
- **03:18 A2 DONE: effects and causal specificity** (CHANGES A2; `atlas_test.py` failed before, passes after).
  - **Planted RNN:** every mechanism's removal lowers its own feature's output, specificity ≥ 10.
  - **M8,5:**
    - Removing a W_h mechanism where its importance says it is needed changes the output 16–266× more than where it is not (median 75×). It flips 6–41% of decisions.
    - Each mechanism gets a readable effect, e.g. W_h.4 "fires when neuron 7 was active and the max is at most 1 step old; removing it lowers position 4 and raises position 1" (confidence 0.99, specificity 266×).
- **03:23 A4 DONE: atlas certificate and catalog** (CHANGES A4).
  - `lab/atlas.py` writes `results/glass/atlas/<target>_d4.md`.
  - The catalog card shows a "Mechanism atlas" section: rule-gated fidelity against the task rule, then one row per mechanism with its trigger, confidence, effect and specificity.
  - `atlas_catalog_test.py` failed before and passes after (headless Chrome, 22/22 rows for M8,5); T5 and T6 still pass.
  - Rules are now simplified: same-feature conditions merge to the tightest bounds, e.g. W_x.2 "ON when x_t <= -0.22, or x_t > 0.20".
- **03:38 U19(c) shuffled control finished** (`results/glass/units/u19c_zoo_2nd_argmax_8_5_shuffled_30000_512.json`; see the numbers there). This completes U19(c). Its decision stands: the trained M8,5 is not event-readable by single events. That question has since moved to the atlas's multi-condition triggers.
- **03:38 A6 in progress:** a Colab L4 session "atlas" (658 compute units available) decomposes 11 targets with SPD at 40k steps and batch 2048, 8 processes in parallel. The targets are the planted RNN (GPU known answer); argmedian 8_5, 8_7, 16_5, 16_7; ARC's challenge M16,10; zoo M16,10 seed 0; M8,8; M8,5; and shuffled controls for argmedian 16_5 and M16,10. Each takes about 30–60 min. Outputs go to `/content/spd_out`, to be downloaded here.
- **03:39 U19(c) control, read by hand:** shuffled M8,5 also decomposes sparsely. On its own decisions: masked fidelity 0.973, alive-only 0.980, 9 W_h mechanisms with about 4.3 in use.
  - **So a sparse exact decomposition is NOT by itself evidence of learned structure.**
  - Consequence for the atlas: every atlas of a trained network must be compared with the atlas of its shuffled control. The discovery-grade claims are the triggers and effects that are task-related in the trained network and not in the control.
  - The GPU batch includes shuffled controls for argmedian 16_5 and M16,10 for this comparison. ATLAS.md §2 gains this as a requirement.
- **03:45 A6 finding: the GPU is the wrong tool for these tiny networks.**
  - Eight SPD processes sharing one L4 reached only 2–4k of 40k steps in 25 min, because the GPU saturates on kernel overhead.
  - One process alone does about 28 steps/s, but eight together manage about 13 steps/s in total.
  - Switched to CPU on the same Colab VM (12 cores): 12 single-thread processes at 30k steps, batch 512, the configuration of the M8,5 atlas.
  - Targets on Colab: argmedian 8_5, 8_7, 16_5, 16_7; ARC's M16,10; zoo M16,10 seed 0; M8,8; shuffled argmedian 16_5 and M16,10; planted_transformer; lc_3_4; lc_6_5.
  - Locally: lc_4_5 and lc_8_6.
- **03:45 A5 built (code; known-answer test pending its SPD run).** `TransformerTarget` (forward matches `nets.transformer` to 7e-13 on lc_6_5), functional-graph features, `planted_transformer` (copies the previous token; 100% correct by construction), and `build_atlas` made generic. RNN atlas test still passes.
  - **Translation fix before any lc result:** lc logits are large (spread 39 for lc_3_4), so the reconstruction loss started at 5e7. Transformers now divide logits by their spread, as RNNs already did; the two lc jobs were restarted.
- **03:46 A7 DONE: task-relative effects** (CHANGES A7).
  - Effects now say what a mechanism's removal does to the answer in task terms.
  - **M8,5 examples:**
    - W_h.4 (fires when neuron 7 was active and the max is at most 1 step old): removing it flips the answer from the 2nd largest to the max in 46% of flips, and to the 3rd in 34%.
    - W_h.0: rank 2 → rank 1 in 41% of flips.
  - So these mechanisms keep the network from answering the max: a readable functional role, tested by ablation.
- **03:46 Standing process:** a 15-minute reminder (Shreyas's request) re-schedules itself. Each time: pull Colab outputs, add a dated diary entry, refresh ATLAS §4/§5, commit, push, send Shreyas a short update, then continue. It is not an interruption.
- **03:51 A8 DONE: atlas findings** (CHANGES A8). The atlas now states its conclusions with numbers. **For M8,5:**
  1. 22 exact mechanisms, 12 with a trusted trigger, 3 always needed.
  2. All 12 trusted triggers use task variables. The most used are x_t (7), the step (6), x_t minus the previous max (6), x_t minus the previous 2nd (5) and steps since the max (4).
  3. Two functional groups:
     - 10 mechanisms whose removal flips the answer from rank 2 to rank 3;
     - 3 mechanisms (W_h.0, W_h.4, W_h.18) whose removal flips it from rank 2 to the max (rank 1).
  4. Rebuilt with trusted triggers: 92.5% of decisions, against the task rule's 87.9%.
  - Control comparisons are added when the shuffled atlases exist (Colab batch).
- **03:54 Atlas CLI with control:** `python3 lab/atlas.py TARGET SPD.pt 4 --control-spd SHUFFLED.pt`. Findings are now in the JSON, and the catalog section lists them.
  - **M8,5 against its shuffled control:**
    - Trained: 22 mechanisms, 12 trusted triggers (12 use task variables), 13 mechanisms in rank-based functional groups.
    - Control: 20 mechanisms, 11 trusted (9 using task variables), **0** in functional groups.
  - **Reading:** trigger vocabulary is not distinctive, since shuffled networks also gate on x_t and ranks. Functional roles in task terms are what separate a trained network from its control. Findings must lean on that.
- **03:57 Feature set frozen for A3 (feature set v2).** Two running-median features were added for the argmedian networks: x_t minus the running median, and the distance of x_t's rank from the middle. M8,5 is unchanged in substance (trusted fidelity 0.925; 11 trusted triggers; 13 vs 0 in functional groups against the control). All atlas tests pass.
- **03:57 A3 PRE-REGISTRATION (held-out networks; feature set v2, depth 4, trusted bar 0.9, SPD 30k steps at batch 512).**
  - **Networks:** argmedian 8_5, 8_7, 16_5, 16_7; ARC's M16,10; zoo M16,10 seed 0; M8,8. Controls exist for argmedian 16_5 and M16,10.
  - **Per network:**
    - (i) exact: alive-only relative error ≤ 0.05 in every matrix;
    - (ii) rule-gated (trusted) fidelity ≥ the task rule;
    - (iii) ≥ 50% of the non-always mechanisms have a trusted trigger;
    - (iv) median specificity ≥ 10;
    - (v) where a control exists: more mechanisms in functional groups than the control.
  - **Decision:** "the atlas works on networks it was not designed on" if (i)–(iv) hold in ≥ 4 of the 7, and (v) holds in both controlled pairs. If not, the failing criterion says which component to improve next, and I try alternatives there.
- **04:00 A10 DONE: atlas playground** (CHANGES A10). In the catalog, "Random input" runs one input through the network and through its rebuilt-from-mechanisms version. A per-step table shows which mechanisms the triggers switch on, next to which the network's importances say are needed, plus both answers. Screenshot checked by eye.
  - **Noted for improvement:** depth-4 triggers with many paths are long to read (e.g. W_h.0 has 6 paths). Next readability step: show the main path, the one covering most ON cases, with "+ N rarer cases" folded away.
- **04:02 Readability:** each trigger now shows its main paths, those covering ≥ 80% of its ON cases, each with its share, and folds the rest ("+ N rarer cases, x%"). The full rule is kept in `rule_full`. Example: W_h.4 "ON when neuron 7 active in the state before this step AND steps since the max <= 1.5 (96% of its ON cases)". Tests pass; fidelity unchanged.
- **04:05 Colab rebalanced.**
  - **Problem:** at 12 CPU processes, M16,10 reached only 4k of 30k steps in about 40 min (about 4.5 h to finish). The GPU process was also starved of CPU: the planted transformer ran at only 5.5 steps/s.
  - **Fix:**
    - Removed the M16,10 and lc CPU jobs.
    - Six CPU jobs remain (argmedian 8_5, 8_7, 16_5, 16_7, M8,8, shuffled argmedian 16_5; float64, batch 512).
    - Three GPU queues: (1) planted transformer then lc_3_4; (2) M16,10 then shuffled M16,10; (3) lc_6_5 then zoo M16,10 seed 0. Queues 2 and 3 run float32 at batch 2048.
  - **Config note, recorded before results:** the M16,10 and lc decompositions use batch 2048 (the GPU's cheap batch) instead of the pre-registered 512. Same steps (30k); only the batch differs. Larger batches only lower gradient noise.
- **04:05 A11 DONE: comparing atlases** (`compare_atlases`). Mechanisms are matched across networks by functional role and trigger variables. This answers Shreyas's original recurrence question at the mechanism level. It will run on M16,10 (ARC's) vs zoo M16,10 seed 0 when both atlases exist. Self comparison 13/13; M8,5 vs shuffled 0.
- **04:08 A12 DONE: exact residual.** Each matrix keeps an always-on residual (the dead subcomponents' weight), so the rebuilt network with every mechanism on IS the network (fidelity 1.000; it was 0.981). This matters for accurate networks, where the task-rule bar is close to 100%. M8,5 rule-gated trusted 0.923 (was 0.925). All tests pass.
- **04:15 A9 check, planted RNN in float32: PASS.** A 40k-step float32 decomposition (`results/glass/atlas/spd/spd_planted_rnn_original_40000_512.pt`, 22 min on one local core) passes every check in `atlas_test.py` (39 PASS: mechanisms, triggers, effects, the exact residual, rule-gated fidelity). The float32 planted-transformer check is running locally (`/tmp/spd_planted_tf32.log`).
- **04:17 A13 DONE: guarded plain-English narrative** (`--narrate`). The reader model writes one paragraph from the atlas's fact sheet; any number or mechanism id not in the facts is rejected. The M8,5 paragraph is honest about its limits ("The facts do not explain how the network combines these mechanisms into a single answer …"). It appears in the markdown and the catalog.
- **04:29 A5 known answer (planted transformer, P = 5, V = 4, GPU float64): FAIL** (6 checks failed).
  - **What passed:** rule-gated fidelity 1.000; W_V triggers 1.00; 4 mechanisms in each of W_V, W_O and W_U; W_O and W_U one per token.
  - **What failed:**
    - W_V input directions are not token-aligned (cosine about 0.45); alive-only error 1.26, so dead components carry large weight.
    - W_O and W_U triggers are at chance (0.62, 0.50).
    - Q and K have no alive mechanisms: everything is in the residual.
  - **Diagnosis (two flaws in the planted design, not in the code):**
    - (1) **Slack dimensions.** With P = 5 > V = 4 the head size is 5, so one value dimension is unused by W_O. That, plus the position dims in W_V's input, lets dead components park weight where it has no effect. The cosine check assumes the one canonical solution.
    - (2) **Missing feature.** The mechanisms needed at the last position read the token at i − 1, but the planted feature library had only the token at i. So W_O and W_U triggers could not be expressed.
    - Q/K having no alive mechanisms is a property of SPD's importance: random scaling of Q/K barely changes the output here. It is not a failure of the decomposition; it is recorded in the atlas as residual.
  - **Fix of the known answer** (criteria unchanged): P = V = 4, which removes the slack; added features "token at i−1 is v". A fresh run is going locally in float32, and it also serves as the A9 transformer check.
- **04:42 A3 first results (held-out, never-interpreted networks)** (`python3 scripts/glass/atlas_a3_eval.py`).
  - **argmedian 8_5: PASSES (i)–(iv).**
    - Residual 0.023; rebuilt with trusted triggers 0.814 vs task rule 0.705; 69% of non-always mechanisms trusted; median specificity 852×.
    - 39 mechanisms. Triggers mostly use the step, x_t, and x_t against the previous min, max and the running median.
  - **argmedian 16_5: FAILS (i) and (ii).**
    - The W_h residual is 0.28, and the rebuilt network keeps 0.732 against a task rule of 0.984. Even SPD's own importance masks keep only 0.744.
    - But its findings are strong: 21 mechanisms in two functional groups, against 1 in the shuffled control (v holds). Twelve keep the answer from moving to rank 2, nine from moving to rank 4: they hold the median against both neighbours.
  - **Reading:** for an accurate (98.4%), fragile network, the decomposition itself is not precise enough at the decision level; the MSE loss and the minimality coefficient trade away what decisions need.
  - **Alternatives being tried (A14),** on Colab CPUs: KL reconstruction at minimality 3e-3; MSE at 1e-3; KL at 1e-3. Plus the planted RNN with KL as A14's known answer.
- **04:46 A3: ARC's challenge network M16,10 FAILS (ii) and (iii)** (GPU float32 SPD, batch 2048; `results/glass/atlas/zoo_2nd_argmax_16_10_d4.{json,md}`; no control yet).
  - (i) residual 0.037 ✓. (ii) rebuilt with trusted triggers 0.700 vs task rule 0.953 ✗. (iii) 37% of non-always mechanisms trusted ✗ (W_x 5/5, W_h 8/20, W_o 0/10). (iv) median specificity 26× ✓.
  - All-alive fidelity 1.000, importance-masked 0.795.
  - Findings: 35 mechanisms, 13 trusted. Functional groups: W_h.5, W_h.6, W_o.19 keep the answer from moving from the 2nd largest to the max (rank 2 → 1); W_h.3 from moving to rank 3.
- **04:49 Why (ii) fails on M16,10: the gate switches mechanisms off too readily** (`scripts/glass/atlas_gate_sweep.py`, exploratory, design case only).
  - A trusted trigger switches its mechanism off wherever its tree says P(needed) < 0.5. A wrong "off" loses decisions. A wrong "on" costs nothing, because all-alive is exact (A12). So the gate should switch off only when it is nearly sure.
  - Moving only that bar: < 0.5 → 0.700; < 0.3 → 0.844; < 0.1 → 0.915; < 0.03 → 0.954; < 0.01 → 0.955 (task rule 0.953).
  - **Pre-registration (A15, before any other network is evaluated this way):** gate v2 uses τ = 0.05, fixed now as a conventional "nearly certain" level, not one of the swept values. M16,10 is the design case and does not count as held out.
  - **A gate that keeps everything on would pass (ii) trivially,** so v2 must also do work. New criterion (vi): of the calls where a trusted mechanism is not needed (importance < 0.1), the gate switches off ≥ 50% (`off_recall`).
  - **A3 under v2:** same criteria (i)–(v), with (ii) using v2, plus (vi). Pass if (i)–(iv) and (vi) hold in ≥ 4 of the 7 networks and (v) holds in both controlled pairs. The v1 verdicts stay recorded as they are.
- **04:50 Colab:** argmedian 8_7 SPD done (atlas building locally). 16_7 at 28k, M8,8 at 26k; A14 variants at 8k; shuffled M16,10, M16,10 s0, lc_6_5, lc_3_4 running on GPU. Local: planted transformer P = 4 (float32) at 22k of 30k with exactly 4 alive mechanisms in each of W_E, W_V, W_O, W_U, as planted; lc_4_5 at 26k.
- **04:53 A15 DONE (gate v2), held-out results.** `atlas_gate_test.py` failed before (8 FAILs) and passes after: on the planted RNN v2 keeps fidelity 1.000 and still switches off 99% of not-needed calls; on M16,10 (design case) v2 gives 0.951 vs v1 0.700 (task rule 0.953) with off_recall 0.85. All v1 numbers unchanged.
  - **Held-out networks, rebuilt with v2 (τ = 0.05 fixed before they were run):**

    | network | v1 fidelity | v2 fidelity | task rule | v2 off_recall |
    |---|---|---|---|---|
    | argmedian 8_5 | 0.814 | 0.904 | 0.705 | 0.91 |
    | argmedian 8_7 | 0.884 | 0.956 | 0.495 | 0.91 |
    | argmedian 16_5 | 0.732 | 0.839 | 0.984 | 0.84 |

  - **Reading:** in every network, switching off only when nearly sure recovers most lost decisions, while the triggers still switch off 84–91% of the calls where a mechanism is not needed. argmedian 16_5 stays below its bar: there the decomposition itself is the limit (importance-masked 0.744, residual 0.28), which A14 addresses.
  - **A3 under v2 so far:** argmedian 8_5 PASS; argmedian 8_7 fails (i) residual 0.060 (bar 0.05) and (iii) trusted share 0.39; argmedian 16_5 fails (i) and (ii). Four networks still to build.
- **04:55 A16 known answer: the first feature design FAILS, a variant passes.**
  - v3 (neurons as values instead of on/off) on the planted RNN: triggers read "neuron 1 > 0.10" and rule-gated fidelity drops to 0.925 (bar 0.99). On/off plus values (v3b): the trees still prefer the value split, same 0.925.
  - **Why:** SPD's importance labels a mechanism "not needed" when its neuron is only slightly active (below about 0.1), but removing it there still changes decisions. The on/off feature switches it on for any activity, which matches what the network needs. So a threshold learned on importance labels is not the right gate at small activations.
  - **Decision:** intent (a) of A16 is withdrawn (recorded in CHANGES). v3c keeps neurons on/off and adds only the task features (3rd largest, gaps); it passes the planted known answer (fidelity 1.000, all triggers ≥ 0.95, same rules as before). Running on M16,10 (design case), then on the held-out networks.
- **04:59 A5 known answer (planted transformer, P = V = 4, float32): PASS** (`scripts/glass/atlas_tf_test.py results/glass/atlas/spd/spd_planted_transformer_original_30000_512.pt`, 15/15). This also closes A9 (float32) for transformers.
  - The atlas recovers the planted mechanism exactly: 4 mechanisms in each of W_E, W_V, W_O and W_U, one per token ("copy token v"). Triggers ≥ 0.997, using "token at i is v" (W_V) and "token at i−1 is v" (W_O, W_U). Rule-gated fidelity 1.000.
  - **The test's alignment measure had to change** (a translation error in the test, recorded before the rerun's verdict was read in full). Raw-weight cosines (0.40–0.94) are not identifiable here:
    - The planted inputs have directions where weight does nothing on the data (the token one-hots and the position one-hots each sum to 1; the slot dims are empty). Dead components park large weight there (alive-only errors 0.5–2.3).
    - W_V acts at every position, but only its value at the attended position (P − 2) reaches the output.
  - **New measure:** on the data, where the weights act: |correlation| between the mechanism's read (V_c · a) and "the copied token is v" (at the attended position, for W_V); for W_O, the cosine of its centred logit effect with token v. All 12 mechanisms: 1.000.
  - **Consequence for A3 criterion (i):** weight-space residuals overstate what is unaccounted for whenever inputs have such directions. Recurrent states can have them too (never-active neurons, fixed relations), so argmedian 16_5's W_h residual of 0.28 may be partly inert. Next: report the residual on the data, ‖(W − Σ alive) a‖ / ‖W a‖ over the matrix's real inputs, as a new named measure next to the old one (A3's recorded verdicts stay as they are).
- **05:01 A17 DONE: the residual is judged by what it does.** The first design (residual on each matrix's real inputs) failed its own known answer: on the planted transformer it is still 1.20 (W_V) and 0.48 (W_O), because those matrices write outputs nothing reads. The adopted measure: decisions kept with every alive mechanism on and the residual removed ("mechanisms alone"). Planted transformer 1.000 (weight-space errors 0.49–1.16), planted RNN 0.997, and M8,5 reproduces A12's recorded "before" number (0.9805) exactly. A3 criterion (i') = mechanisms alone ≥ the task rule.
- **05:03 A16 v3c: no gain on the design case, not adopted.** M16,10 trusted triggers stay 13/35; W_o stays 0/10. Exploratory (`/tmp/wo_feats.py`): per-position features for W_o (rank and value at each position, "position j holds rank r") also leave its triggers at 0.62–0.88, at depth 4 or 6.
  - **Why W_o resists:** its importances are graded, not on/off. 38% of (input, W_o mechanism) importances in M16,10 sit between 0.1 and 0.9, against 12% for W_h (M8,5: 26% vs 11%). A trigger predicts a yes/no "needed" label, which is noisy where the importance itself is half-way.
- **05:06 All A3 atlases rebuilt with A15 + A17** (two local queues; also lc_6_5 and lc_4_5). `python3 scripts/glass/atlas_a3_eval.py --v2`:

  | network | (i') alone ≥ task rule | (ii) v2 vs task rule | (iii) trusted ≥ 0.5 | (iv) spec. | (v) control | (vi) off_recall | result |
  |---|---|---|---|---|---|---|---|
  | argmedian 8_5 | 0.963 ✓ | 0.904 vs 0.705 ✓ | 0.69 ✓ | 852× ✓ | - | 0.91 ✓ | PASS |
  | argmedian 8_7 | 0.900 ✓ | 0.956 vs 0.495 ✓ | 0.39 ✗ | 56× ✓ | - | 0.91 ✓ | fail |
  | argmedian 16_5 | 0.900 ✗ | 0.839 vs 0.984 ✗ | 0.50 ✓ | 60× ✓ | 21 vs 1 ✓ | 0.84 ✓ | fail |
  | argmedian 16_7 | 0.871 ✓ | 0.824 vs 0.712 ✓ | 0.29 ✗ | 34× ✓ | - | 0.85 ✓ | fail |
  | M16,10 (design case for A15) | 0.963 ✓ | 0.951 vs 0.953 ✗ | 0.37 ✗ | 26× ✓ | (control running) | 0.85 ✓ | fail |
  | M8,8 | 0.861 ✓ | 0.850 vs 0.459 ✓ | 0.41 ✗ | 46× ✓ | - | 0.86 ✓ | fail |
  | M16,10 seed 0 | (SPD running) | | | | | | |

  - **Reading:** with gate v2, (ii) holds wherever the decomposition is good enough. The binding criterion is now (iii), the share of mechanisms whose trigger is trusted (0.29–0.41 in four networks).
  - Transformers (reported, not judged in A3): lc_6_5: 83 mechanisms, 44 trusted; v2 0.775 vs task rule 0.947, mechanisms alone 0.806. lc_4_5: v2 0.748 vs 0.810, mechanisms alone 0.887. For both, the decomposition (mechanisms alone below the task rule) is the first limit.
- **05:07 Fresh held-out set A18, fixed now, before any of it is decomposed:** zoo_2nd_argmax_16_5_s0, zoo_2nd_argmax_16_8_s0, zoo_2nd_argmax_32_5_s0, zoo_argmedian_32_5_s0 (Colab CPU), zoo_2nd_argmax_32_10_s0 (1,376 parameters, the largest RNN yet) with its shuffled control, and zoo_argmedian_32_7_s0 (GPU queue 4). From now on the A3 networks are the development set; design changes for (iii) are chosen on them and judged on A18 with A3's criteria (v2 form). (The earlier GPU-4 launch failed: the ids need the `_s0` suffix.)
- **05:08 Exploring (iii) on the development set** (`scripts/glass/atlas_trigger_sweep.py`): tree depth 4/5/6; confidence on crisp labels only (importance > 0.9 vs < 0.1); trees trained on crisp labels.
- **05:09 (iii) exploration result (development set only)** and **A19 pre-registration.**
  - Trusted share at depth 4 / 5 / 6, and at depth 4 scored on clear-cut calls (importance > 0.9 vs < 0.1):

    | network | d4 | d5 | d6 | d4 clear-cut |
    |---|---|---|---|---|
    | argmedian 8_5 | 0.69 | 0.69 | 0.69 | 0.79 |
    | argmedian 8_7 | 0.39 | 0.39 | 0.56 | 0.67 |
    | argmedian 16_5 | 0.50 | 0.55 | 0.55 | 0.70 |
    | argmedian 16_7 | 0.29 | 0.38 | 0.38 | 0.58 |
    | M16,10 | 0.37 | 0.43 | 0.43 | 0.51 |
    | M8,8 | 0.41 | 0.48 | 0.59 | 0.62 |
    | M8,5 | 0.58 | 0.63 | 0.63 | 0.74 |

  - Training on clear-cut labels too is mixed (better on M16,10 and M8,5, worse on argmedian 8_5 and 8_7), so it is not adopted.
  - **A19:** a trigger is trusted when its held-out accuracy on clear-cut calls is ≥ 0.9 (bar unchanged). Each trigger also reports its graded share, so a reader sees how often "needed" is a matter of degree. Gate v2's τ is kept; fidelity (ii) is re-measured under the new trust rule (more triggers are used, so it can drop).
  - **A18 decision rule (pre-registered):** A3's criteria in this form: (i') mechanisms alone ≥ task rule; (ii) v3 fidelity ≥ task rule; (iii) clear-cut trusted share ≥ 0.5; (iv) median specificity ≥ 10; (vi) v3 off_recall ≥ 0.5; plus (v) functional groups beat the shuffled control for zoo_2nd_argmax_32_10_s0. **Pass: (i')–(iv) and (vi) in ≥ 4 of the 6 A18 networks, and (v).**
- **05:13 A19 test passes; development set under v3** (`python3 scripts/glass/atlas_a3_eval.py --v3`, all dev atlases rebuilt with A19).

  | network | (i') alone | (ii) v3 vs task rule | (iii) clear-cut trusted | (iv) | (v) | (vi) | result |
  |---|---|---|---|---|---|---|---|
  | argmedian 8_5 | 0.963 ✓ | 0.904 vs 0.705 ✓ | 0.79 ✓ | 852× ✓ | - | 0.90 ✓ | PASS |
  | argmedian 8_7 | 0.900 ✓ | 0.942 vs 0.495 ✓ | 0.67 ✓ | 56× ✓ | - | 0.84 ✓ | PASS |
  | argmedian 16_5 | 0.900 ✗ | 0.818 vs 0.984 ✗ | 0.70 ✓ | 60× ✓ | 21 vs 1 ✓ | 0.81 ✓ | fail |
  | argmedian 16_7 | 0.871 ✓ | 0.825 vs 0.712 ✓ | 0.58 ✓ | 34× ✓ | - | 0.76 ✓ | PASS |
  | M16,10 | 0.963 ✓ | 0.939 vs 0.953 ✗ | 0.51 ✓ | 26× ✓ | (control running) | 0.75 ✓ | fail |
  | M8,8 | 0.861 ✓ | 0.835 vs 0.459 ✓ | 0.62 ✓ | 46× ✓ | - | 0.82 ✓ | PASS |
  | M16,10 seed 0 | (SPD running) | | | | | | |

  - 4 of 7 development networks pass (the A3 bar, but these networks shaped A15–A19, so this is not a held-out claim; A18 is).
  - Trade-off seen: trusting more triggers costs some fidelity (M16,10: v2 0.951 → v3 0.939; argmedian 16_5: 0.839 → 0.818).
  - The remaining failures are both "decomposition not precise enough for a very accurate network" (16_5: mechanisms alone 0.900 vs 0.984; M16,10: v3 just under 0.953).
- **05:13 A14 variants for argmedian 16_5 downloaded** (KL at minimality 3e-3 and 1e-3; MSE at 1e-3). Evaluating (i') and (ii).
- **05:15 A14 variants on argmedian 16_5 (task rule 0.984), judged on (i') and (ii):**

  | SPD variant | mechanisms alone (i') | v2 | v3 | mechanisms | W_h weight residual |
  |---|---|---|---|---|---|
  | MSE, minimality 3e-3 (default) | 0.900 | 0.839 | 0.818 | 22 | 0.28 |
  | MSE, minimality 1e-3 | **0.953** | 0.888 | 0.876 | 31 | 0.13 |
  | KL, minimality 3e-3 | 0.579 | 0.882 | 0.759 | 60 | 0.98 |
  | KL, minimality 1e-3 | 0.652 | 0.917 | 0.853 | 60 | 0.98 |

  - **KL reconstruction fails:** W_h's mechanisms come out as large pieces that cancel each other, so the alive mechanisms alone keep only 58–65% of decisions. Dropped.
  - **A lower minimality coefficient helps** (0.900 → 0.953 for mechanisms alone; 22 → 31 mechanisms), but is not yet at 0.984.
  - **Next variants, on Colab CPU:** MSE at minimality 3e-4 and 1e-4 (30k steps), and 1e-3 at 60k steps.
- **05:18 Reading the argmedian 16_7 certificate by hand.** The numbers pass, but the account is thin: most W_h triggers are stated in other neurons ("ON when neuron 7 active … AND neuron 12 active …"), and the findings name one functional mechanism. That is not yet a discovery a person can use. Two steps follow from this:
  - **A20 (done, text only): glosses from the unit maps.** Each neuron in a trigger now carries its unit-map label, e.g. M16,10 W_h.0: "ON when neuron 1 [≈ relu(max − x_t)] active … AND neuron 13 [≈ relu(+4.11·x_prev)] active". `atlas_gloss_test.py` failed before and passes after; no number changes.
  - **Task-level triggers (exploration, development set, clear-cut confidence ≥ 0.9 share):**

    | network | task + neurons | task only | neurons only |
    |---|---|---|---|
    | argmedian 8_5 | 0.79 | 0.76 | 0.48 |
    | argmedian 8_7 | 0.67 | 0.56 | 0.28 |
    | argmedian 16_5 | 0.70 | 0.45 | 0.40 |
    | argmedian 16_7 | 0.58 | 0.38 | 0.38 |
    | M16,10 | 0.51 | 0.46 | 0.26 |
    | M8,8 | 0.62 | 0.59 | 0.48 |
    | M8,5 | 0.74 | 0.58 | 0.58 |

    Triggers stated purely in task variables are often almost as good. Planned (A21): each mechanism also gets a task-only trigger, shown as its explanation whenever it is trusted. The gate is unchanged, so no fidelity changes. The certificate then reports how many mechanisms have a task-level explanation.
- **05:16 Started:** M16,10 at minimality 1e-3 and lc_4_5 at 1e-3 (local CPU); argmedian 16_5 at 3e-4, 1e-4, and 1e-3 for 60k steps (Colab CPU).
- **05:23 A21 and A22 DONE (presentation; no number changes).**
  - **A21:** each mechanism also gets a trigger in task variables only, shown as its explanation when trusted on clear-cut cases. Known answer: planted W_x get exactly their planted rule; planted W_h and W_o do not get one, because their truth ("feature j arrived earlier") is not in the task vocabulary. That is the must-fail case.
  - **A22:** the certificate, findings, comparison, narrative facts, catalog and playground now all use one definition: trusted = clear-cut confidence ≥ 0.9, gate v2 τ = 0.05 (fidelity v3). The certificate lists v1 and v2 for the record and adds "mechanisms alone".
  - Saved atlases were re-rendered from their JSON (`scripts/glass/atlas_rerender.py`). The catalog, playground, findings, comparison and gloss tests pass.
  - **Read by hand (argmedian 8_7):** 10 of its 18 switched mechanisms now have a task-level explanation. Most W_h mechanisms switch on by time step ("ON when step t > 3.5"): this 49.5%-accurate network stages its recurrent computation over time rather than by data content. Its W_o mechanisms have no trusted trigger (graded importances).
- **05:26 Control for M16,10 done (criterion v holds in both controlled pairs):** M16,10 has 4 mechanisms in functional groups, its shuffled-weights control 0 (argmedian 16_5: 21 vs 1). lc_3_4's atlas: 78 trusted triggers, all in task variables (mostly position i and token f(i)), but the rebuilt network keeps only 53.1% vs the task rule's 90.2%. For transformers the decomposition is the limit.
- **05:26 More decomposition variants started** for accurate networks: 40 components instead of 20 (argmedian 16_5, minimality 1e-3 and 3e-3; Colab CPU). `load_decomposition` now reads the number of components from the saved file; `atlas_spd_job.py` takes `--C`.
- **05:29 Overview table** (`scripts/glass/atlas_overview.py` → `results/glass/atlas/OVERVIEW.md`): one row per atlas (size, mechanisms, trusted and task-level explanations, certificate vs task rule, mechanisms alone, groups vs control). It will carry the A18 rows as they land.
- **05:29 Colab rebalanced for A18:** the GPU was shared by too many runs and A18's queue was sequential (the 32_10_s0 control and argmedian 32_7_s0 would have started only after 32_10_s0). I stopped the lc_4_5/lc_6_5 minimality-1e-3 runs and started the control and argmedian 32_7_s0 in parallel. GPU now: M16,10 seed 0, 32_10_s0, its control, argmedian 32_7_s0. CPU: the four 16/32-hidden A18 networks (22k of 30k) and five argmedian 16_5 variants (minimality 3e-4, 1e-4, 1e-3 at 60k steps, 40 components at 1e-3 and 3e-3). Locally: M16,10 at 1e-3 and zoo M8,5 seed 0.
- **05:32 Diagnosis: why M16,10's certificate stays just under its bar (93.9% vs 95.3%)** (`scripts/glass/atlas_gate_cost.py`, training half, design case).
  - Gating one trusted mechanism at a time: the most costly are W_x mechanisms whose triggers are perfect (confidence 1.00, e.g. "ON when x_t > 0.20"). Each costs 2–4% of decisions on its own (W_x.0 4.2%, W_x.10 4.1%, W_x.6 3.2%).
  - **The cause is SPD's importance label, not the trigger.** Importance says a mechanism is "not needed" when its input is small (|x_t| below about 0.2), but removing it there still flips near-tie decisions in this accurate network. The trigger faithfully predicts a label that is wrong for decisions at small magnitudes. Same pattern as the planted RNN with neuron values (DIARY 04:55, A16): a learned threshold misses small but decision-relevant activity.
  - **Candidate fix (A23, not yet pre-registered):** check each trusted trigger against decisions, not only importances. If gating it alone loses decisions on the training half, keep the mechanism on in the certificate's gate; it keeps its trigger as an explanation of when it is important. The bar must come from a control, not from these results. Option: the decision loss of gating the same mechanism with a random switch at the same on-rate, or the network's own near-tie fragility (U11's control).
- **05:32 A18 progress:** 16_5_s0 at 26k, 32_5_s0 and argmedian 32_5_s0 at 24k, 16_8_s0 at 16k (Colab CPU). On the GPU: 32_10_s0 at 14k; its control and argmedian 32_7_s0 just started.
- **05:34 A23 pre-registration (gate v4, decision-checked), fixed before any A18 atlas exists.** On the training half, trusted mechanisms are added to the switched set in order of their single gating cost while the switched network keeps ≥ task rule + 2 standard errors (a sampling margin); the rest stay on. Judged on the held-out half.
  - **A18 is judged twice, both pre-registered:** v3 (05:09) as the primary verdict; v4 with the same criteria, where (ii) uses v4 fidelity and (vi) uses v4 off_recall over all trusted mechanisms (so a gate that drops most mechanisms fails (vi)). Pass: ≥ 4 of 6 and (v).
- **05:35 A23 (gate v4) built and tested.** The test fails on the pre-change code and passes after. Development results: argmedian 16_5's rebuilt network now keeps 98.6% (task rule 98.4%), but only by switching few mechanisms (off_recall 0.32, fails (vi)): in this network few trigger claims survive a decision check. M16,10 keeps 96.3% (task rule 95.3%) with off_recall 0.71.
- **05:36 A18, first held-out network: zoo_2nd_argmax_16_5_s0 (99.1% accurate) FAILS under v3 and v4.**
  - v3: mechanisms alone 0.935 (bar 0.991) ✗; rebuilt 0.872 ✗; trusted share 0.77 ✓; specificity 22× ✓; off_recall 0.88 ✓.
  - v4: rebuilt 0.997 ✓, but only by switching few mechanisms: off_recall 0.17 ✗. Mechanisms alone still ✗.
  - Findings (reported): 6 mechanisms keep the answer from moving from rank 2 to rank 3.
  - **Reading:** a 99%-accurate network is described almost perfectly by its task ("it returns the 2nd largest"), so a mechanism account must be more precise than that to pass. The decomposition loses 6.5% of decisions with its residual removed. Making the decomposition more precise for accurate networks (A14 variants running) is the lever.
- **05:38 Exploration (development only): absorb the residual into the alive mechanisms** (`scripts/glass/atlas_absorb_explore.py`). Each alive mechanism keeps its input direction, so its importance and trigger are unchanged; its output direction is refit by least squares on the matrix's real inputs.

  | argmedian 16_5 (task rule 0.984) | mechanisms alone | v3 | v4 (off_recall) |
  |---|---|---|---|
  | minimality 3e-3 | 0.900 | 0.818 | 0.986 (0.32) |
  | minimality 3e-3, absorbed | 0.916 | 0.896 | 0.988 (0.28) |
  | minimality 1e-3 | 0.953 | 0.876 | 0.981 (0.38) |
  | minimality 1e-3, absorbed | **0.991** | 0.723 | 0.984 (0.20) |

  - Lower minimality plus absorption makes the mechanisms alone more faithful than the task rule, criterion (i'), for the first time on an accurate network.
  - But then switching mechanisms off by their triggers costs more. Absorbed output directions are larger, so a wrong "off" removes more.
  - **Reading:** in a 98%-accurate network, faithfulness of the parts and safe switching pull against each other. The decision-checked gate keeps the rebuilt network above the bar but can switch off only 20–38% of the not-needed calls.
- **05:39 A18, three of six held-out networks in: all fail, the same way** (v3 / v4):

  | network (accuracy) | (i') alone | (ii) v3 | (ii) v4 | (iii) | (iv) | (vi) v3 | (vi) v4 |
  |---|---|---|---|---|---|---|---|
  | 2nd argmax 16_5_s0 (99.1%) | 0.935 ✗ | 0.872 ✗ | 0.997 ✓ | 0.77 ✓ | 22× ✓ | 0.88 ✓ | 0.17 ✗ |
  | 2nd argmax 32_5_s0 (99.6%) | 0.944 ✗ | 0.920 ✗ | 0.999 ✓ | 0.84 ✓ | 47× ✓ | 0.85 ✓ | 0.23 ✗ |
  | argmedian 32_5_s0 (99.5%) | 0.949 ✗ | 0.893 ✗ | 0.996 ✓ | 0.82 ✓ | 103× ✓ | 0.84 ✓ | 0.17 ✗ |

  - **What works on them:** the decomposition, triggers (77–84% trusted), specificity and findings. Example: argmedian 32_5_s0 has 13 mechanisms that keep the answer from moving from the median to the next-larger value; 2nd argmax 32_5_s0 has 11 that stop it from returning the max.
  - **What fails:** precision against a 99%+ bar. The mechanisms alone keep 93.5–94.9%; rule switching either loses decisions (v3) or can switch off only about a fifth of the not-needed activity (v4).
  - **Honest note on the development set:** its passing networks are weak at their task (task rules 46–71%), where the bar is easy. Every accurate network so far (16_5 at 98.4%, M16,10 at 95.3%, all A18) fails on precision. This is the problem to solve.
- **05:39 Next, aimed at precision:** argmedian 16_5 at minimality 3e-4 and 1e-4 (finishing on Colab). A weaker minimality pressure should also make SPD's "not needed" labels more conservative, which is the gating problem. If it helps on the development network, it is tested on a NEW held-out set: seed s1 of the same six A18 architectures (A18 itself has now been seen).
- **05:43 Do mechanisms recur across training runs? ARC's M16,10 vs zoo M16,10 seed 0** (`compare_atlases`; seed 0's atlas built from its GPU decomposition).
  - Seed 0 (91.4% accurate) fails A3 as a development network: v3 0.835 vs 0.914, trusted share 0.42; under v4 its rebuilt network keeps 0.936 with off_recall 0.53.
  - The first comparison reported one recurring mechanism with similarity 1.0. Read by hand, it matched on role alone: both stop the network from returning the max, but neither side's trusted main rule uses a task variable, and A11 scores two empty sets as 1.0. Their task-level triggers differ: ARC's W_h.6 "the max arrived in the last step and x_t is ≥ 0.64 below it" (trusted) vs seed 0's W_h.16 "after step 2, the max arrived ≥ 2 steps ago" (0.83, not trusted).
  - **A24 (done):** comparisons use the task-level trigger's variables when trusted, and every pair names its basis. Result: 1 recurring mechanism, on role only (ARC W_h.5 ↔ seed 0 W_h.16). **No mechanism recurs on role and trigger variables** between these two training runs.
- **05:45 Exploratory result: mechanisms recur between ARC's M8,5 and zoo M8,5 seed 0, beyond chance.** Seed 0 (83.7% accurate) has its own atlas now (89.0% rebuilt vs 83.7% task rule; 10 functional mechanisms). The comparison finds 6 recurring mechanisms, 5 of them on role and trigger variables. Example: ARC W_h.4 ↔ seed 0 W_h.7 (similarity 0.83), both stopping the network from returning the max and triggered by the same variables (steps since the max and the 2nd, x_t against the previous max and min).
  - **Chance control:** shuffle seed 0's triggers among its own mechanisms, keeping roles: 1.97 matches on average, 95th percentile 4, P(≥ 5) = 0.003. ARC's M16,10 vs its seed 0: 0 matches (null 0.06).
  - **A25 pre-registration (confirmatory):** build atlases for zoo M8,5 seeds 1–4 (same SPD settings: 30k steps, batch 512, minimality 3e-3). Compare each with ARC's M8,5 (A25's `recurrence_control`, n = 300). **"Mechanisms recur across training runs of M8,5" holds if the real count exceeds the null's 95th percentile in ≥ 3 of the 4 seeds.** Otherwise seed 0's result is treated as chance.
- **05:47 Precision breakthrough on the development network argmedian 16_5 (98.4% accurate): minimality 1e-4 plus residual absorption passes every criterion.**

  | argmedian 16_5 | alone (i') | v3 (ii) | v4 (ii) | trusted (iii) | spec. (iv) | off_recall v3 / v4 (vi) | mechanisms |
  |---|---|---|---|---|---|---|---|
  | minimality 3e-3 (default) | 0.900 | 0.818 | 0.986 | 0.70 | 60× | 0.81 / 0.32 | 22 |
  | minimality 1e-4 | 0.992 | 0.947 | 0.982 | 0.64 | 561× | 0.93 / 0.84 | 57 |
  | minimality 1e-4, absorbed | **1.000** | **0.989** | **0.989** | 0.64 | 546× | **0.93 / 0.93** | 57 |

  - A weaker minimality pressure gives more, finer mechanisms (57), and SPD's own importance masks become far more faithful (0.744 → 0.952). That is the label problem from 05:32, solved at the source.
  - Absorption then removes the leftover residual. The rebuilt network beats the task rule while its triggers switch off 93% of the calls where a mechanism is not needed. 25 mechanisms have task-level explanations.
  - **A26 pre-registration** (this design used only argmedian 16_5). Held-out set **A27 = seed s1 of the six A18 architectures**, nothing decomposed yet. SPD at minimality 1e-4 (30k steps; CPU batch 512), atlas with absorption.
    - Criteria as A18: (i') mechanisms alone ≥ task rule; (ii) v3 ≥ task rule; (iii) clear-cut trusted share ≥ 0.5; (iv) median specificity ≥ 10; (vi) v3 off_recall ≥ 0.5; (v) groups beat the shuffled control for zoo_2nd_argmax_32_10_s1.
    - **Pass: (i')–(iv) and (vi) in ≥ 4 of 6, and (v).** v4 is reported alongside.
- **05:48 A26 built and tested** (`build_atlas(..., absorb=True)`, CLI `--absorb`; `--tag` writes a variant next to the main atlas). The test fails before and passes after; without absorb nothing changes. **A27 decompositions running on Colab CPU**: seven runs at minimality 1e-4 (six s1 networks plus the shuffled control of 32_10_s1). The superseded variants (40 components; 1e-3 at 60k steps) were stopped to free cores.
- **05:50 Second development network agrees: ARC's M16,10 (95.3%) passes every criterion at minimality 1e-3 with absorption** (local run; A26 itself uses 1e-4).

  | M16,10 | alone | v3 | v4 | trusted | spec. | off_recall v3 / v4 | mechanisms |
  |---|---|---|---|---|---|---|---|
  | minimality 3e-3 (GPU, batch 2048) | 0.963 | 0.939 | 0.963 | 0.51 | 26× | 0.75 / 0.71 | 35 |
  | minimality 1e-3 | 0.984 | 0.935 | 0.956 | 0.62 | 109× | 0.81 / 0.68 | 47 |
  | minimality 1e-3, absorbed | **0.997** | **0.968** | **0.968** | 0.62 | 113× | **0.81 / 0.81** | 47 |
- **05:51 A18 fourth network, zoo_2nd_argmax_16_8_s0 (97.6%): fails like the others** (default decomposition). Mechanisms alone 0.821; v3 0.636; v4 0.984 ✓ but off_recall 0.06 ✗; trusted 0.64 ✓; specificity 18× ✓. Six mechanisms stop the network from returning the max.
- **05:51 Minimality sweep on argmedian 16_5, completed (with absorption):** 3e-3 → v3 0.896; 1e-3 → 0.723; 3e-4 → 0.953; 1e-4 → 0.989 (bar 0.984). Mechanisms alone with absorption: 0.916 / 0.991 / 0.995 / 1.000. The 1e-3 dip in v3 aside, lower is better, and 1e-4 (A26's setting) is the first to clear the bar.
- **05:53 Caveat on A26, recorded before A27's verdict:** with absorption, (i') is close to automatic once the alive mechanisms' input directions span a matrix's inputs (20 alive components vs 16-dimensional states). Least squares can then make their sum exact on the data. So under A26, the informative criteria are (ii), triggers keep the decisions, and (vi), triggers switch off most not-needed activity. Absorption does not make those easier (at 1e-3 it lowered v3 from 0.876 to 0.723). The A27 verdict stands as pre-registered, and the write-up will say which criteria carry the weight.
- **05:55 A26 on the transformer known answer:** with absorption the planted transformer still passes all 15 checks (rebuilt 1.000, one mechanism per token, aligned 1.000). lc_4_5 and lc_6_5 at minimality 1e-4 are queued on the GPU behind A18's last runs. A27 is slow on the saturated Colab CPU (load about 12.7 on 12 cores): 16_5_s1 at 6k of 30k, the 32-hidden runs at 2–4k. They stay on CPU at batch 512, as pre-registered.
- **06:03 A25 (recurrence across training runs of M8,5) FAILS its confirmatory test.** Seeds 1 and 2 have 3 role-and-variable matches each with ARC's M8,5. The chance control (n = 300) gives mean 1.73 / 1.87, 95th percentile 3, P = 0.20 / 0.24. Neither beats the 95th percentile, so at most 2 of 4 seeds can; the bar was ≥ 3. **Seed 0's p = 0.003 is treated as chance, as pre-registered.** For the record, all three seeds sit above the null mean (5, 3, 3 vs about 1.8–2.0): a weak trend, not a finding. Seeds 3 and 4 will be added for completeness.
- **06:03 A18, fifth network: zoo_2nd_argmax_32_10_s0 (99.4%, 1,376 parameters; GPU, default settings) fails.** Only 32 mechanisms; mechanisms alone 0.453; v3 0.332; v4 0.998 with off_recall 0.01; trusted 0.91; specificity 11×. 17 mechanisms in functional groups (12 stop it returning the max). Its control is still decomposing.
- **06:04 A concern for A27, recorded before its verdict:** at minimality 1e-4 every one of the 20 components per matrix stays alive. In a 32-hidden network, 20 input directions cannot span a 32-dimensional state, so absorption cannot make W_h exact and (i') may fail for the 32-hidden A27 networks for that reason alone. A27 stays as pre-registered (C = 20). Design case for the number of components, outside every test set: zoo_2nd_argmax_32_8_s0 at 1e-4 with C = 20 and C = 40 (Colab GPU). Locally: the development networks at 1e-4 (M16,10 and M8,5 running; argmedian 16_7 and M8,8 queued) so that A26 can become the default if A27 passes.
- **06:07 A18 VERDICT (pre-registered 05:07/05:09/05:34): FAIL, 0 of 6 under v3 and 0 of 6 under v4.** Sixth network: argmedian 32_7_s0 (93.9%): mechanisms alone 0.844, v3 0.789, v4 0.952 (off_recall 0.17).

  | network | (i') alone | (ii) v3 | (ii) v4 | (iii) | (iv) | (vi) v3 | (vi) v4 |
  |---|---|---|---|---|---|---|---|
  | 2nd argmax 16_5_s0 | 0.935 ✗ | 0.872 ✗ | 0.997 ✓ | 0.77 ✓ | 22× ✓ | 0.88 ✓ | 0.17 ✗ |
  | 2nd argmax 16_8_s0 | 0.821 ✗ | 0.636 ✗ | 0.984 ✓ | 0.64 ✓ | 18× ✓ | 0.88 ✓ | 0.06 ✗ |
  | 2nd argmax 32_5_s0 | 0.944 ✗ | 0.920 ✗ | 0.999 ✓ | 0.84 ✓ | 47× ✓ | 0.85 ✓ | 0.23 ✗ |
  | argmedian 32_5_s0 | 0.949 ✗ | 0.893 ✗ | 0.996 ✓ | 0.82 ✓ | 103× ✓ | 0.84 ✓ | 0.17 ✗ |
  | 2nd argmax 32_10_s0 | 0.453 ✗ | 0.332 ✗ | 0.998 ✓ | 0.91 ✓ | 11× ✓ | 0.73 ✓ | 0.01 ✗ |
  | argmedian 32_7_s0 | 0.844 ✗ | 0.789 ✗ | 0.952 ✓ | 0.64 ✓ | 25× ✓ | 0.77 ✓ | 0.17 ✗ |

  - **Diagnosis, consistent across all six:** triggers and specificity are fine everywhere. The default decomposition is not precise enough for accurate networks, so either the switched network loses decisions (v3) or almost nothing can be switched safely (v4).
  - This is exactly what A26 targets; its held-out test is A27 (running).
- **06:21 A25 complete: 1 of 4 seeds beats chance (bar ≥ 3). Recurrence claim not made.**

  | seed | matches on role and trigger variables | chance mean / 95th pct | P |
  |---|---|---|---|
  | 0 (exploratory) | 5 | 1.97 / 4 | 0.003 |
  | 1 | 3 | 1.73 / 3 | 0.200 |
  | 2 | 3 | 1.87 / 3 | 0.237 |
  | 3 | 7 | 2.63 / 5 | 0.000 |
  | 4 | 0 | 0.57 / 2 | 1.000 |

  - **Reading (descriptive, not a claim):** recurrence varies by training run. Two of five seeds share several mechanisms with ARC's M8,5 well beyond chance, two are at chance, and one shares nothing. A natural next question: do the seeds that recur also behave alike in other ways? Not pursued now.
  - Side result: every M8,5 seed's atlas beats its task rule with the default settings (rebuilt 89–96% vs 81–88%). At roughly 85% accuracy the default decomposition is precise enough.
- **06:23 A27, first two held-out networks (minimality 1e-4 + absorption):**
  - **zoo_2nd_argmax_16_5_s1 (99.2% accurate): PASSES every criterion**, under v3 and v4. Mechanisms alone 1.000; rebuilt v3 0.994 vs 0.992; trusted share 0.63; specificity 948×; off_recall 0.94 (v4: 0.995, 0.82). **The first accurate network never used in design to pass.**
  - **zoo_argmedian_32_5_s1 (99.5%): fails (i') 0.953 and (iii) 0.29.** As predicted at 06:04, all 20 components are alive in every matrix, and W_h's 20 directions cannot cover a 32-dimensional state (W_h residual 0.049 after absorption). 32 of its 60 mechanisms are always on, so the rebuilt network (1.000) and off_recall (1.00, over only 8 trusted mechanisms) say little here. The "1,000,000×" specificity is a division by a zero "not needed" effect, i.e. "only where needed". This decomposition is saturated: there are not enough components to separate mechanisms in a 32-wide network.
- **06:24 A27, third: zoo_2nd_argmax_32_5_s1 (99.5%) fails (i') 0.970 and (iii) 0.44**, the same saturation as argmedian 32_5_s1. (ii) passes: 0.996 vs 0.995, off_recall 0.96. Pattern so far: the 16-wide network passes; 32-wide networks run out of components at C = 20.
- **06:24 Development: ARC's M8,5 at minimality 1e-4.** Rebuilt 99.3% (99.5% absorbed) vs task rule 87.9% (default: 95.0%); off_recall 0.93; trusted 0.74. But it now has 54 mechanisms for a 113-parameter network, half of them always on. W_x is rank 1, yet it is split into 20 pieces that each cover a range of x_t ("ON when x_t > 0.2"). Weaker sparsity buys precision at the cost of more, smaller pieces. Readability needs watching.
- **06:28 Development: ARC's M16,10 at minimality 1e-4** (vs 1e-3 + absorption at 05:50). Rebuilt 99.6% vs 95.3%, off_recall 0.88, but trusted share 0.48: just under the 0.5 bar. All 60 components alive, 27 always on. At 1e-3 + absorption it passed with trusted share 0.62.
  - **So the best minimality depends on the network:** 1e-4 for argmedian 16_5, 1e-3 for M16,10. "Every component alive" is the warning sign that the decomposition has run out of room.
  - **Idea for the next version (A28, to be designed on development and design-case networks only):** an adaptive decomposition. Run a short ladder of minimality values with the number of components scaled to the width. Pick the sparsest one whose atlas is precise enough on the TRAINING half ((i') and (ii) there), then judge on the held-out half. Weak networks keep sparse, readable decompositions; accurate ones get the precision they need. The C = 20 vs 40 design case (zoo_2nd_argmax_32_8_s0) is running.
- **06:44 A27 VERDICT (pre-registered 05:47): FAIL, 2 of 6 (bar ≥ 4); (v) holds (13 functional mechanisms vs 0 in the shuffled control).** v4 gives the same split.

  | network (accuracy) | (i') alone | (ii) v3 | (iii) | (iv) | (vi) | result |
  |---|---|---|---|---|---|---|
  | 2nd argmax 16_5_s1 (99.2%) | 1.000 ✓ | 0.994 vs 0.992 ✓ | 0.63 ✓ | 948× ✓ | 0.94 ✓ | **PASS** |
  | 2nd argmax 16_8_s1 (97.7%) | 1.000 ✓ | 0.980 vs 0.977 ✓ | 0.65 ✓ | only where needed ✓ | 0.85 ✓ | **PASS** |
  | 2nd argmax 32_5_s1 (99.5%) | 0.970 ✗ | 0.996 ✓ | 0.44 ✗ | ✓ | 0.96 ✓ | fail |
  | argmedian 32_5_s1 (99.5%) | 0.953 ✗ | 1.000 ✓ | 0.29 ✗ | ✓ | 1.00 ✓ | fail |
  | 2nd argmax 32_10_s1 (99.3%) | 0.964 ✗ | 0.998 ✓ | 0.38 ✗ | ✓ | 0.97 ✓ | fail |
  | argmedian 32_7_s1 (97.3%) | 0.878 ✗ | 0.915 ✗ | 0.60 ✓ | 723× ✓ | 0.95 ✓ | fail |

  - **What held out:** A26 works on held-out 16-wide accurate networks (97.7–99.2%). Every criterion passes, where the default decomposition failed every 16-wide A18 network.
  - **What did not:** every 32-wide network fails (i'). With 20 components per matrix the decomposition is saturated: all components alive, many always on, few trusted triggers. This was predicted at 06:04, before the verdict.
  - **Next (A28):** components scaled to the width, designed on zoo_2nd_argmax_32_8_s0 (outside every test set), then tested on a fresh held-out set.
- **06:58 A28 design case, zoo_2nd_argmax_32_8_s0 (99.3%, outside every test set), minimality 1e-4 + absorption:**
  - **C = 20:** alone 0.974 ✗, v3 0.998, trusted 0.46 ✗, off_recall 0.97. The same failure as A27's 32-wide networks.
  - **C = 40:** alone 1.000 ✓, v3 0.975 ✗ (v4 0.997 ✓, off_recall 0.85), trusted 0.54 ✓. But all 120 components are alive again, so it is still saturated.
  - Reading: more components fix (i') and (iii); at 1e-4 the decomposition fills whatever room it gets. Next on the design case: C = 40 and 64 at minimality 1e-3 and 3e-4. M16,10 at 3e-4 completes its ladder.
- **06:58 Colab incident:** several `colab exec` launches hung, the CLI pruned and reattached the session, and every running job was killed, including lc_4_5/lc_6_5 at 1e-4 (12k and 8k of 30k). Nothing finished was lost (outputs are files). Everything was relaunched: the design-case ladder (four runs), M16,10 at 3e-4, and lc_4_5/lc_6_5 at 1e-4 on the GPU.
- **07:00 Development networks at minimality 1e-4 (absorbed), against their default-setting atlases:**

  | network (task rule) | default: v3 / trusted | 1e-4: v3 / trusted | mechanisms (switched) at 1e-4 |
  |---|---|---|---|
  | argmedian 16_5 (0.984) | 0.818 ✗ / 0.70 | 0.989 ✓ / 0.64 ✓ | 57 (44) |
  | argmedian 16_7 (0.712) | 0.825 ✓ / 0.58 | 0.971 ✓ / 0.73 ✓ | 59 (33) |
  | M8,5 (0.879) | 0.950 ✓ / 0.74 | 0.995 ✓ / 0.74 ✓ | 54 (27) |
  | M16,10 (0.953) | 0.939 ✗ / 0.51 | 0.996 ✓ / 0.48 ✗ | 60 (33) |
  | M8,8 (0.459) | 0.835 ✓ / 0.62 | 0.985 ✓ / 0.35 ✗ | 53 (17) |

  - 1e-4 always helps precision but can cost trust: in M8,8 and M16,10 most extra mechanisms are always on, and the switched ones are harder to predict.
  - So no single minimality wins. This supports the adaptive design for A28: per network, take the sparsest setting that is precise enough on the training half.
- **07:03 A28 part 2 built: adaptive decomposition** (`build_adaptive`; CLI `--ladder`). Each network gets a ladder of decompositions, sparsest first. The first whose training-half mechanisms-alone and rebuilt (v3) fidelities clear the task rule + 2 standard errors is used. The test fails before and passes after, including a check that wrecking the held-out half does not change the choice. On argmedian 16_5 (development) it picks 1e-4: training half 1.000 / 0.994 vs bar 0.988; the sparser rungs fail on the rebuilt network (0.905, 0.960).
  - **Still open before pre-registration:** the number of components for 32-wide networks. On the design case at 1e-4, C = 40 passed (i') and (iii) but not v3 (0.975 vs 0.993). Running there: C = 40 at 1e-3 and 3e-4 (CPU and GPU), C = 64 at 3e-4 (GPU), and M16,10 at 3e-4 for its ladder. The A28 test set (s2 seeds) will be fixed once a setting passes the design case.
- **07:06 A28 adaptive decomposition on the development set (design check): all 7 pass every criterion** (v3; (v) not run).

  | network (task rule) | rung chosen | alone | rebuilt v3 | trusted | off_recall | mechanisms |
  |---|---|---|---|---|---|---|
  | argmedian 8_5 (0.705) | default (3e-3) | 1.000 | 0.918 | 0.79 | 0.90 | 39 |
  | argmedian 8_7 (0.495) | default | 0.924 | 0.865 | 0.67 | 0.84 | 20 |
  | argmedian 16_5 (0.984) | 1e-4 | 1.000 | 0.989 | 0.64 | 0.93 | 57 |
  | argmedian 16_7 (0.712) | default | 0.968 | 0.831 | 0.58 | 0.76 | 26 |
  | M16,10 (0.953) | 1e-3 | 0.997 | 0.968 | 0.62 | 0.81 | 47 |
  | M8,8 (0.459) | default | 1.000 | 0.834 | 0.62 | 0.82 | 31 |
  | M8,5 (0.879) | default | 0.998 | 0.915 | 0.74 | 0.86 | 22 |

  - The ladder does what it was built for: weak networks keep the sparse, readable default (20–39 mechanisms); accurate ones get the precision they need. These networks shaped the method, so this is a design check, not evidence. Evidence comes from the A28 test set, once the wide-network setting passes its design case.
- **07:12 Reading the passing held-out certificate (zoo_2nd_argmax_16_5_s1) by hand.** The numbers are strong, but the explanations are shallow:
  - Many task-level triggers are time gates ("ON when step t > 2.5", "ON once a 2nd largest exists").
  - The main finding is that 25 mechanisms keep the network from returning the max.
  - True and checked, but it says *when* mechanisms act and *what decision* they protect, not *what they compute*.
  - **Next presentation step (A29 idea):** describe each mechanism's input and output directions with the unit-map labels of the neurons it mostly reads and writes, e.g. "reads neuron 4 [≈ relu(max − x_t)], writes neuron 9 [≈ …]". Unit maps exist for the development networks; `lab/unitmap.py` can make them for the others.
- **07:23 A28 design decisions, fixed before the test set is decomposed.**
  - **Wide networks (design case zoo_2nd_argmax_32_8_s0, 99.3%):**

    | setting | (i') alone | v3 | v4 | trusted | v4 off_recall |
    |---|---|---|---|---|---|
    | C = 20, 1e-4, 30k | 0.974 ✗ | 0.998 | 0.998 | 0.46 ✗ | 0.97 |
    | **C = 40, 1e-4, 30k** | 1.000 ✓ | 0.975 ✗ | **0.997 ✓** | 0.54 ✓ | 0.85 ✓ |
    | C = 40, 3e-4, 10k (quick) | 1.000 | 0.945 | 0.991 ✗ | 0.64 | 0.53 |
    | C = 40, 1e-3, 10k (quick) | 0.636 (not absorbed) | 0.545 | 0.995 | 0.70 | 0.04 |

    Only C = 40 at 1e-4 passes, and only with the decision-checked gate (v4).
  - **Selection rule:** I tried selecting on v4's training numbers. It overfits: v4 reaches the bar on the training half by construction. On argmedian 16_5 it picked 3e-4, which then falls short held out (v4 0.979 vs 0.984). So selection stays on v3's training numbers (every trusted trigger used), which picked 1e-4 there.
  - **A28 pre-registration.**
    - **Test set: seed s2** of the six A18/A27 architectures: zoo_2nd_argmax_16_5_s2, 16_8_s2, 32_5_s2, zoo_argmedian_32_5_s2, zoo_2nd_argmax_32_10_s2 (+ shuffled control), zoo_argmedian_32_7_s2.
    - **Decomposition:** components C = max(20, ⌈1.25·H⌉), i.e. 20 for 16-wide and 40 for 32-wide; ladder of minimality 1e-3, 3e-4, 1e-4. The 3e-3 default is left out because every ≥ 94%-accurate network so far needed more precision. 30k steps, CPU batch 512, absorption.
    - **Selection:** the first rung whose training-half mechanisms alone and v3 reach task rule + 2 SE; else the densest.
    - **Certificate (primary): v4.** (i') alone ≥ task rule; (ii) v4 ≥ task rule; (iii) clear-cut trusted ≥ 0.5; (iv) median specificity ≥ 10; (vi) v4 off_recall ≥ 0.5; (v) for 32_10_s2 the control, on the same rung, has fewer functional mechanisms. v3 reported alongside.
    - **Pass: (i')–(iv) and (vi) in ≥ 4 of 6, and (v).**
- **07:32 A28 protocol amendment (compute only; no A28 result had been seen).** Fifteen 32-wide CPU runs on 12 Colab cores would take about 3 hours (2k steps in 7 min; the 10-step networks had not started). So zoo_2nd_argmax_32_10_s2 (+ control) and zoo_argmedian_32_7_s2 moved to the GPU at batch 2048, the setting under which the design case passed. The two 32_5 networks stay on Colab CPU at batch 512; the 16-wide networks run locally at batch 512. Each network's ladder and its control use one device throughout.
- **07:53 A28, first held-out network: zoo_2nd_argmax_16_5_s2 (99.2%) PASSES under v4 (primary).** The ladder found no rung precise enough on the training half (v3 0.942 / 0.987 / 0.980 vs bar 0.995), so the densest (1e-4) was used. Held out:
  - v4: mechanisms alone 1.000, rebuilt 0.995 vs 0.992, trusted 0.76, specificity 1,853×, off_recall 0.86. PASS.
  - v3 for the record: rebuilt 0.980, fails (ii).
- **08:14 A28, second held-out network: zoo_2nd_argmax_16_8_s2 (95.7%) PASSES** (v4 and v3). The ladder stopped at its sparsest rung: 1e-3 was precise enough on the training half (v3 0.980 vs bar 0.963), so the 1e-4 run was stopped. Held out: alone 1.000, rebuilt 0.979 vs 0.957, trusted 0.57, specificity 142×, off_recall 0.86.
- **08:14 A30 done** (certificate and playground lead with v4, the decision-checked gate, when present; v3 listed for the record). The 32-wide A28 runs are at 20–26k of 30k on Colab.
