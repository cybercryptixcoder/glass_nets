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
