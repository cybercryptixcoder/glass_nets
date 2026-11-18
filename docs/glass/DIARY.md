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
