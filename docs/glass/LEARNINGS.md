# Glass Networks: learnings from round 1 (frozen Nov 17, 23:40 ET)

What round 1 showed, each with the run files that show it and a confidence level. Round 2's fixes come only from
here. Step 1 (comparing M4,3 with ARC's notes) is pending, because the Drive notes need a permission that wasn't
granted when this was written; its result will be added as E9.

## The findings

Every claim below points to a file in `lab/constructions/zoo_2nd_argmax_*`.

| # | What went wrong | Evidence | Confidence |
|---|---|---|---|
| E1 | **Re-description is the reader's default.** It writes the weights (copied, rescaled, regrouped or normalised) because that verifies every stage | Copies in runs on every network: M2,2, M4,3 (subs 4–5), M4,4 (both runs), M8,5 (pass 2, sub 1), M8,8 (5 of 9 subs), M16,10 (sub 7: "diagnostic exact re-implementation") | Certain |
| E2 | **The judge's rules changed while the runs ran.** Pass 1 started with no copy rule; three versions came in mid-run; pass 2 began before the final degrees-of-freedom count. So the runs weren't all judged by the same rules, and readers didn't know all the rules from the start | DIARY, 20:41–21:57 | Certain |
| E3 | **One rule is enough to tell copies from explanations.** Constants per degree of freedom (parameters minus unit rescalings and readout symmetries) separates all 17 constructions: copies ≥ 1.0, compressed accounts ≤ 0.78. The value-match and table rules catch nothing it misses | `docs/glass/rejudge_all.txt`; DIARY 21:57 | Certain (on these 17) |
| E4 | **When internal stages fail, the reader abandons them and curve-fits the scoreboard.** On M16,10, submissions 11–24 (about 80 of 160 steps) aligned only h1. They added 10×10 tables of measured thresholds to raise the mistakes score from 0% to 50%, and never tried a stage again after submission 16 (h2 0.83, h3 0.75) | M16,10 `submission_11..24.json`, `submission_22.py` (TAU/SIG tables) | Certain (what happened); the cause is inferred |
| E5 | **The binary verdict hides real partial progress.** M8,5's compressed account (pass 2, sub 7) replaces the internal states at 98% / 89% / 84% / 79% and matches 86% of behaviour, yet it reads as "0/5 stages". The reader got no signal about which stage to fix first | M8,5 `20261118-023529-46c8/submission_07.json` | Certain |
| E6 | **The one success came by compressing a working copy.** M4,3: subs 4–5 were exact re-implementations (stages verified); sub 6 rewrote the same mechanism in its symmetric form with fewer numbers, and stayed verified | M4,3 `submission_04..06` | Certain that it happened; one case |
| E7 | **The reader writes the task's algorithm first** and then patches toward the network. On M16,10 the blinded run never beat its own prior-only control on stages (1/10 vs 1/10) | RESULTS.md (prior rows) | Certain |
| E8 | My own process errors: clock times estimated wrongly, judge changed under live runs, pgrep self-match, tar exit code | DIARY | Certain; process fixes only |

**What we are not confident about, and won't "fix":** whether the 90% substitution bar is too strict for approximate
mechanisms. E5 says partial progress is invisible. It doesn't say the bar is wrong. Lowering it would move the
goalposts, so we leave it.


## E9. M4,3 was a rediscovery (confidence: high; source `docs/glass/M43_VS_ARC.md`)

- ARC's notes, by Robinson and Hilton, already contain our M4,3 account: signed channels with positive and negative parts, the U/V symmetry, a magnitude-only readout, and reduction to a 2-neuron absolute-value network. The reader found it blind, so this is a validation of the system, not a discovery.
- New relative to those notes:
  - the reduced recurrence is a 120° rotation (eigenvalue angle ±119.96°, modulus 0.995);
  - zeroing the asymmetric coupling costs no accuracy (97.86% → 97.96%);
  - the result passes causal tests.
- Our account is missing the link between thresholds and orderings that Hilton gives.
- Consequence: every claim needs a prior-work comparison before it is made. The catalog and README must call M4,3 a "blind rediscovery of ARC's account".
