# Glass Networks, round 2: plan (awaiting Shreyas's approval; written Nov 17, 23:30 ET)

## Shreyas's instructions (Nov 17, 23:04 ET), in his terms

1. Compare our M4,3 account with ARC's notes, and understand what we need to understand.
2. Work out what our mistakes were, and be confident that they are the mistakes (or the learnings).
3. **No patchwork, no going round in circles.** Only fix what the results and traces let us say with confidence.
   Each fix must trace back to something we saw go wrong in a run, and we must be able to show that it improves
   things.
4. Pick a target we can claim with **no asterisks** as an independent discovery: a real network that hasn't been
   interpreted before, not a variant of a solved problem. Run on it and check that it works.
5. Re-run the ARC 2nd-argmax ladder to show we do better than before.
6. Plan everything first, do the research now, then wait for approval before building.

## What the traces say (evidence, not guesses)

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

## The steps, in order (each has a gate)

### Step 1. Compare with ARC's notes on M4,3 (needs your OK to open them)
- The analysis is in three Google Drive PDFs linked from ARC's post (Robinson, "Symmetric RNNs"; Hilton, "Heuristic
  explanations for 2nd argmax models"; Zihao Chen, "RNNs for the 2nd argmax"), plus a notebook. **Opening them
  needed a permission I couldn't get tonight.**
- Output: `docs/glass/M43_VS_ARC.md`. For each element of ARC's account and ours (neuron roles, the symmetry, the
  rotation, the decision rule), say whether it matches, where they differ, and what ARC gets that we don't (their
  accuracy estimate).
- Gate: if our account contradicts theirs, that's a finding about our judge (it certified something wrong), and it
  goes to the top of step 3.

### Step 2. Freeze the learnings
- `docs/glass/LEARNINGS.md`: the E-table above, finalised with step 1's result. Each line has the evidence and a
  confidence level.

### Step 3. Fixes. Only those the evidence supports, each with a test that shows it helps
| Fix | From | What changes | How we know it helped |
|---|---|---|---|
| F1. **One copy rule** | E3 | Keep only constants per degree of freedom (flag at ≥ 0.9); drop the value-match and table rules | Re-judge all saved constructions: the same verdicts as now (a regression check). It's simpler, not new |
| F2. **Frozen judge, rules known from step 0** | E2 | Version-stamp the judge. Every run records the version, and the version is not changed during a benchmark. The reader's prompt states the compression rule from the start | Every report carries the same version |
| F3. **Stage-first reporting** | E4, E5 | The report opens with a per-stage table of graded numbers (substitution, interchange) and names the earliest failing stage as "fix this first". The pass bars stay the same. Best-submission choice prefers more verified stages counted from the first step, then behaviour | A/B in step 4: stages verified per run, old vs new reader, same networks and seeds |
| F4. **Copy-then-compress, as a stated route** | E6, E1 | Prompt: an exact re-implementation may be submitted as a diagnostic (it's flagged and never delivered). Then replace one stage at a time with compact variables, keeping that stage verified | Same A/B: how many constructions are verified and compressed |
| F5. Process | E8 | Times from `date` only; no judge edits during runs; liveness by pid file | — |

F1 and F2 are certain improvements: simpler, and the same answers. F3 and F4 come straight from the traces, but
whether they *raise results* is a hypothesis. **Step 4 is how we become confident instead of assuming.** If the A/B
shows no gain, they're reverted, not patched.

### Step 4. Re-run the ARC 2nd-argmax ladder (benchmark, frozen judge)
- Same six networks. **Two blinded runs each with the old reader (today's prompt) and two with the new (F3, F4)**,
  same budgets (80 steps; M16,10 160), plus the prior-only and shuffled controls once each.
- Compare: networks certified (compressed), stages verified, mistakes reproduced, cost.
- Cost: roughly 30 runs, about $10–15 of Luminal, CPU Colab for about 3–4 hours.

### Step 5. A target we can claim with no asterisks: AlgZoo's longest-cycle transformers
- **What they are.** The input is a function on {0, …, n−1}, written as a list of n numbers (position i holds
  f(i)). The output is the length of the longest cycle you reach by following the pointers.
  - Example: [1, 2, 0, 3] has the cycle 0→1→2→0 (length 3) and 3→3 (length 1), so the answer is 3.
  - The models are **2-layer, attention-only transformers**. Solving the task needs following pointers over several
    hops: a real algorithm, not a single operation, on a different architecture from the RNNs.
- **Checked tonight:** the four checkpoints load and run (2 layers each).

  | Hidden size | Length n | Params | Accuracy |
  |---|---|---|---|
  | 3 | 4 | 108 | 90.7% |
  | 4 | 5 | 188 | 81.0% |
  | 6 | 5 | 378 | 94.7% |
  | 8 | 6 | 656 | 85.4% |

- **Why no asterisks:**
  - ARC's post analyses only the 2nd-argmax family.
  - Community work found tonight covers 2nd argmax (M16,10) and argmedian (one 32-parameter RNN, xabier.me).
  - No public interpretation of any longest-cycle model was found.
- **Strong certificates:** the input set is small enough to test exhaustively (4⁴ = 256, 5⁵ = 3,125, 6⁶ = 46,656
  inputs). So "behaviour" and every other test run on *every* possible input, not a sample.
- **Work needed:**
  - (a) a lab target kind for a bidirectional, attention-only transformer, checked against torch to 1e-6;
  - (b) a builder with exhaustive inputs and labels;
  - (c) runs: 2 blinded runs per model, prior-only and shuffled controls;
  - (d) before any claim, a second, wider literature check.
- **Claim criterion, fixed in advance:** VERIFIED AND COMPLETE under the frozen judge; < 0.78 constants per degree of
  freedom; prior-only control fails; no prior interpretation found. Then a human read of the construction (you and
  me) to confirm the account says something real, written up in `docs/glass/DISCOVERY_LONGEST_CYCLE.md`.
- Fallback if it fails: the honest result is "not yet", with the per-stage numbers. The next candidate is the median
  family (scalar RNNs), also uninterpreted publicly.
- Cost: about 12 runs, about $5–10, a few CPU hours.

## Order and gates
1 → 2 → 3 → 4 → 5. Step 5's lab support (5a, 5b) can be built during step 4's runs. Nothing is built until you
approve. Step 1 needs one approval: opening the Google Drive notes.
