# Glass Networks: first report (Nov 17, 2026, 20:16–22:15 ET)

The day's whole record is in `docs/glass/DIARY.md`. The full results table is `results/glass/RESULTS.md`. Every run
folder (trace, every submission, every judge report) is in `lab/constructions/zoo_2nd_argmax_*/`.

## What was built (branch `glass-networks`)

| Piece | Where | What it's for |
|---|---|---|
| AlgZoo targets | `lab/targets/build_algzoo.py`, `nets.rnn_relu` | Six whole-model targets from ARC's AlgZoo (2nd argmax RNNs, 10–432 params). Includes ARC's exact challenge network. Our forward pass is identical to theirs |
| Joint substitution | `judge.certificate` | Every named site replaced at once, so sites that pass one at a time must also work together |
| Worst slice | `judge.certificate` | The average can hide a hole. It caught an old "complete" modular-multiplication construction that is 0% right whenever the answer is 112 |
| The network's own mistakes | `judge.certificate` (needs task labels) | A construction of the *network* must reproduce its errors. This is what rejects "the task's algorithm" passed off as the network's |
| Disagreement search | `judge.disagreement_search` | Finds clear-decision inputs where construction and network disagree, and hands them to the reader as counterexamples |
| Coverage line + "not explained" list | `judge.coverage` | What each catalog entry has to state |
| Not a copy (three rules) | `judge.copy_check` | (1) literals reproduce the weights; (2) a table about as large as a weight tensor; (3) **constants per degree of freedom ≥ 0.9**, where degrees of freedom = parameters minus the transformations that leave the decisions unchanged |
| Tie fix | `judge.decide` | Exact output ties were being decided by floating-point noise |
| Best-submission rule | `constructor.better` | A flagged copy is never the delivered construction |
| Run queue | `scripts/glass/run_ladder.py`, `colab_job.sh`, `colab_sync.sh` | 4 workers, one key each. Resumable. A second pass re-runs whatever the current judge doesn't certify |
| Catalog + playground | `/catalog` (`lab/catalog.py`, `lab/static/catalog.html`) | One card per network with its certificate. Runs the network and the construction side by side, plus interventions. Tested in a real browser |
| Comparison with the Pitfalls system | `docs/glass/COMPARISON_PITFALLS.md` | Their agent vs our constructor and judge |

## Results on AlgZoo (2nd argmax RNNs), under the final judge

| Network | Params | Net accuracy | Best blinded result | Prior-only control (told the task, no network) | Verdict |
|---|---|---|---|---|---|
| M2,2 | 10 | 99.98% | Exact re-description (5 numbers = its 5 degrees of freedom) | **Complete with 0 constants** | Explained by the plain rule; the network's insides align with it |
| M4,3 | 32 | 98.1% | **Complete**: two mirror-image signed channels, rotating ~120° per step; decision = ratio of channel magnitudes vs two thresholds (0.78 constants per degree of freedom) | 1 of 3 stages | **Explained, and reading the network was necessary** |
| M4,4 | 36 | 60.1% | Exact re-description (twice) | 0 of 4 | Not explained (no compression found) |
| M8,5 | 112 | 87.9% | 85–89% behaviour, 0/5 stages, ≤ 49% of mistakes | — (the one-shot submission errored) | Not explained |
| M8,8 | 136 | 45.9% | Exact re-description; the reader argues none shorter exists ("a generic clipped linear system with 120 free constants") | 0 of 8 | Not explained |
| M16,10 (ARC's challenge) | 432 | 95.3% | 160 steps: 97.0% behaviour, 1/10 stages, 45% of mistakes (the bar is 50%) | 1 of 10, 95.5% | Not explained |

Shuffled-weights controls: all fail under the final judge. 22 runs in total, $3.12 of Luminal calls.

## What we learned

1. **The judge needed a definition of "explanation", not just "faithful".** On tiny networks the reader's easiest
   faithful construction is the weights themselves: copied, then rescaled, then split into rows, then normalised.
   Each form passed every causal test. What separates explanation from re-description is **compression**.
   - Counted against the network's true degrees of freedom, the line is clean on all 17 AlgZoo constructions:
     re-descriptions sit at ≥ 1.0 constants per degree of freedom, compressed accounts at ≤ 0.78.
   - This matches ARC's view that understanding is compression.
2. **Controls change the story.** M2,2 looked like a success of reading, but the prior-only control explains it
   too, and with fewer constants. M4,3 is the case where reading the network clearly mattered.
3. **The reader is bimodal.** It either finds a compact rule (M4,3) or writes an exact re-description. It never
   produced a *partial* compressed account that verifies some internal steps of a larger network. On M16,10 it
   stayed at the task's algorithm plus small corrections (97% behaviour, 1/10 steps).
4. **The mistakes test is the most informative single test.** It separates "the task" from "this network", and it
   measures how close M16,10's best attempt came (45% of the network's mistakes reproduced, against a 50% bar).

## What's next (each tied to the failure it addresses)

- **Partial compressed accounts.** Teach the reader to explain the network step by step (stage 1, then stage 2, ...)
  with compact variables, instead of all at once. M16,10's stage 1 already verifies.
- **Tools for structure, not weights.** Symmetry finding, low-rank structure of W_h, clustering units by role
  (M4,3's success came from spotting a symmetry). Parameter decomposition (Goodfire's SPD) is the obvious
  candidate.
- **ARC's own standard on the networks we explain.** Can the M4,3 construction predict the network's accuracy
  faster than sampling?
- **GitHub token for Colab** (optional), so Colab pushes every 2 minutes instead of relaying at check-ins.
