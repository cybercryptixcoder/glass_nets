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
