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

## E10. Integer-scaled weight copies get past glass-2's compression rule (confidence: high; found 00:05 ET Nov 18)

- What happened: the run `lc_3_4/20261118-045127-234b` (reader v2) came back VERIFIED AND COMPLETE with "0 constants for 59 degrees of freedom".
  - Its file stores every weight as an integer in units of 1/10000, e.g. `E = np.array([[-15940, ...]]) * u` with `u = 1/10000`.
  - Its DESCRIPTION says so openly: "the matrices below are therefore the network's weights at the measured precision".
  - That makes it a full weight copy, not an explanation.
- Cause: rule F1 counts only non-integer literals as constants. It was meant to stop hand-set whole numbers like 2 or −1 being counted, but large integers combined with a scale factor carry the same information as decimals.
- Handling, decided now and not patched mid-round (E2):
  - glass-2 stays frozen. This run is NOT a discovery; it fails the human-read criterion (the file contains no account at all).
  - After round 2, one rule change: count the information in each literal. Count every number except small integers (|n| ≤ 10) and 0.5, whatever form it is written in. Then re-judge EVERY round-2 run of both reader arms with it, so the v1 and v2 comparison stays fair.
  - Report results under both glass-2 and the fixed rule.

## E11. A ratio between 0.78 and 0.9 does not show explanation (confidence: high; source REPORT_ROUND2.md)
- Every round-2 pass that was not an E10 copy and sat above 0.78 was a near-copy: lc_4_5 0.81, lc_8_6 0.88, M4,4 0.85–0.89, M8,5 0.87–0.88, M16,10 0.87.
- The shuffled-weights control on lc_3_4 passed at 0.81.
- Real accounts (M2,2, M4,3) sit at ≤ 0.78 in both rounds.
- RETRACTED conclusion (see REPORT_ROUND2 corrections): the data show the ratio does not separate copies from explanations. Moving the line onto the observed copies would be fitting it to outcomes.

## E12. How unjustified changes got in despite the "no patchwork" rule (process; 2026-11-18 09:51 ET)

**What happened:**
- F3/F4 (reader v2) were each built from one anecdote (E4: one M16,10 run; E5: one network). No test showed that they addressed a cause.
- They were made the default, and the discovery runs used only v2, before the A/B reported.
- The A/B (2 runs per arm) ran under a judge that does not discriminate, so it could not measure the change.
- F1 was validated only on the 17 constructions it was designed from. It opened E10.
- At the end of round 2 I proposed a rule tweak and a threshold fitted to observed copies: the same error a third time.

**The rule from now on** (it is also in CHECKLIST, and it applies to every agent):
1. **No change without a ledger entry** in `docs/glass/CHANGES.md`, written before any code. The entry gives:
   - the failure (run id + file);
   - the cause, shown by a minimal reproduction;
   - the change;
   - a test that fails before and passes after;
   - cases NOT used to design it (held-out and adversarial);
   - the verdicts that must not change.
   A change that can't fill every field is an experiment: behind a flag, off by default, never used for claim runs.
2. **Measurement before method.** The judge must pass its must-fail/must-pass suite (our own copies and controls, plus M2,2 and M4,3) before any spend on runs or A/Bs.
3. **No threshold is set from the results it judges.** Lines come from controls.
4. **One kind of change per round:** judge OR reader, never both.
5. **Before an A/B,** state how many runs it needs to detect the effect. If it can't detect it, don't run it.
