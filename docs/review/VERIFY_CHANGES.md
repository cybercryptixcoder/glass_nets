# Fact-check of the critique edits (docs/review, 1c91c590..HEAD)

Checked at HEAD **290c3301**. The branch moved on from c3e5e5f4 while I worked. The later X0 rewrite (02a52f59) already fixed the M2,2 tie and the below-one-forward-pass convention, and aeb002bf fixed the run C citation, so those are not listed. Line numbers are at 290c3301.

## Problems, most serious first

1. **X1 treats countdown as a target with a known account, but no account exists.** PREREG 101 lists "GPT-2 countdown" under "Known account, cheap reader failed". The same claim is at PREREG 85–87, NEXT_BUILD 1631 ("Targets where a short, verified account exists… GPT-2 countdown") and BRIEF 123.
   Source: RESULTS_LM.md 43 gives countdown "reader 1/5; hand-made 2/5", and lines 53–54 say the reader did not complete "month rollover or countdown". Nothing, not even the hand-built account, passes all 5 stages.
   Effect: if X2 has not run, "Yes" ("at least two" targets, every stage passing) needs a countdown account nobody has built. A frontier reader that recovers month alone fits none of the branches.
   Fix: move countdown out of the known-account list, and state the decision for month alone.

2. **The known-negative arm of X1 cites U17 for networks U17 never measured.** PREREG 94–96 (and NEXT_BUILD 1631, REVIEW_LOG 235) says M8,5 and lc_6_5 are "known to admit no short faithful account (rules replace at most 7% of either…, U17…)", so "'both fail' was decided in advance".
   Source: glass/DIARY 1002–1003 and 1013–1015 show U17 ran on challenge M16,10, zoo M16,10 seed 2, and zoo n = 10 networks at h = 16 and 32. It did not include M8,5 or lc_6_5. ATLAS_RESULTS 63 says "lc_4_5 and lc_6_5 pass" (development). DIARY 1165 says SPD found "an EXACT, sparsely used decomposition of M8,5 … faithful above the task rule". F4's routes are routes for RNNs.
   Fix: attribute the 7% cap to M16,10-class networks, and describe these two as stalled for the reader but decomposed by SPD and the atlas, not as known negatives.

3. **The reader arm's budget does not stop the reader from running the student, so "opaque programs of any size" does not follow.** PREREG 64–67 sets the budget "below what hand-executing the copy on one input takes" and says the student "cannot be executed within the budget either". NEXT_BUILD 424–427 and BRIEF 47–50 claim the reader version catches "opaque programs of any size".
   The student is small by design (PREREG 58), so it costs less to run by hand than the copy, and a budget sized from the copy can leave room to run it. This matters because the main arm hands exactly the tie between the account and an equal-cost student to this arm (PREREG 58–60).
   Fix: set the budget, before any call, from the student's hand-execution cost, or narrow the claim.

4. **The main X0 pass line is set only from the must-fail side. Arithmetic alone makes it fail samplers and ties.** PREREG 48–49 allows "Running the account itself as a cheap sampler". PREREG 55–57 requires, on all five quantities, an error "below the 5th percentile of the reference's error".
   (a) A sampling estimator gets one draw, and that draw must land below about 0.063σ/√n (σ is the spread of a single sample's score, n the Monte Carlo sample count). An unbiased sampler that is 100× cheaper per sample manages this with probability ≈0.47 per quantity, or ≈0.02 for all five. An account with fidelity below 1 (the census rule is at 0.922) also has a bias that Monte Carlo does not. M4,3's account "isn't shorter than its 32 weights" (DIARY 425), so as a sampler it is no cheaper than a copy.
   (b) On a sub-population where the network is almost always right, Monte Carlo's error is 0 in nearly every repeat, and no error can be strictly below 0. This is the M2,2 problem again, now possible on any blind-chosen sub-population.
   (c) "3 in 10 million" assumes the errors on the five quantities are independent, but the quantities are nested.
   Fix: score randomized estimators by mean squared error over repeated runs, against Monte Carlo's at matched compute. This is ARC's form (EVIDENCE_LADDER 37: "averaging the squared error"). Treat ties as "no worse than".

5. **The X4/X6 definition language brings back a constant budget, and the must-fail set tests only programs the language already forbids.** PREREG 251–253 allows "at most 5 constants fitted on the fit set". PREREG 351–352 runs fits from outside the language "to confirm that the language and the surrogate line exclude them".
   Source: NEXT_BUILD 393 (unchanged) says "Five literal-counting rules, each beaten (… tables plus indices)". A search tunes hard-coded numbers by selection. A literal piece-square table is allowed in the language and counts as zero fitted constants. Boosted trees or a linear map of earlier channels would beat the 5-coefficient surrogate (PREREG 265), so only the syntax rule excludes them.
   Fix: count every numeric literal and table entry against the budget. Add to the must-fail set a program that is legal in the language and carries a literal table tuned on the fit set.

6. **X4 step 0 has a foregone answer and reports no share.** PREREG 234–240 measures the partial correlation of the residual with outcome, controlling for the surrogate. Since the residual is LD2's value minus the surrogate, this asks whether a superhuman network's value predicts outcomes beyond a fit to 61 concepts, which is nearly certain. It also covers the whole residual, not the quiet-position part that channel 22 tracks. NEXT_BUILD 1634 promises "the share that is knowledge stated".
   Fix: run step 0 on quiet positions and on channel 22's component, and report the share.

7. **Minor issues:**
   - The 12% in X7 (PREREG 374–375, NEXT_BUILD 1637) is the real-lie direction M2 (liar/DIARY 243–246). The mode's own direction moved the detector "only 1–5%" (liar/DISCOVERY 74).
   - "the readers that work are closed" (NEXT_BUILD 1460–1461; "the strong readers are closed", 1638): K1 lines 11–12 record open-weight Qwen 3.8 27B and Kimi at "about 3/4 games", and Kimi at 4/8.
   - Run C's summary as "what moved the search furthest" (NEXT_BUILD 1112–1113): RECORDS 121 says "One of two seeds". Run D had the same summary and ended at 0.941244, below run B (no outside input, 0.941442; README 34–36).
   - "X0 … at no cost" (NEXT_BUILD 126–127 and 1625, BRIEF 132) conflicts with the reader arm, which needs "Flash calls, so Shreyas's go" (PREREG 62, 76).
   - PREREG 25 says X6's must-fail set is "scored with" X0, but X6's scorer never uses it.
   - The month figure "32% of decisions" (NEXT_BUILD 145, 266) is cited near PROJECT_REVIEW §8, but it comes from gpt2_month_rollover/DISCOVERY.md 119.

## Checked and correct
- T2 numbers, the 31 must-fail cases at S ≥ 0.91, "CRITERION MET: False"; the DIARY 11:35 quotes are verbatim.
- Rung Y and rung 5 wording; Wu et al. (about 3×, 55%, loss bounds fail).
- E7 (90.8/82.3, 57.0/13.9).
- U8 (5/5 at 8 neurons, 3–4/5 at 16, bar not met) and the units verdict quote. Atlas recurrence: 1 of 4.
- Decompiler counts (2/5, 0/5, 3/5, 2/8 then 4/8).
- INDEX counts.
- Helix: runs B, C and D; 19/19 restarts; RESEARCH_CASE quotes; p2563 is 516 lines; the C–D gap of 0.007.
- FIND (0.82–0.90 vs 0.38; 0.88/0.80/0.75; the +0.10 holds for v3 on numeric only).
- Singh (0.17/0.11; 0.34/0.34; oracle 0.84). K1 shapes scores.
- Kepler: LD2 is 128×10; channel 22 at 32% (K4); K9; the queen rule is t1's (−0.02); the surrogate in residual.py is boosted trees.
- Fluid ≈0.51 GFLOP vs ≈11 MFLOP per frame (≈47×), recounted from model.py and sim.py.
- Goal 2 item 4 and JOURNAL. REPORT_ROUND2's 0.81 and condition (c). PLAN_SCALING says Flash only.

## Nothing wrong found
- The numbers in §0, apart from item 7.
- The glass-2 paragraph in §3.12.
- The X3 known-answer framing.
- X8's protocol and pass line.
