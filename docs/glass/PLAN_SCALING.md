# Plan: constructions that work on bigger networks (started Nov 18, 2026, 11:50 ET)

**Mandate (Shreyas, 11:43).** Go past R4 if there is real progress. Use the API freely; time matters more than
money. Run micro experiments (minutes each), learn, continue. Come back when something **works on the larger
networks and is verifiable**. Document every step (DIARY) and keep this plan current. Rules: `CLAUDE.md`.

## What "works" means (fixed now, before any experiment)

A construction of a network with **more than 100 parameters** counts as working when all of these hold:
1. **Truth.**
   - Fidelity to the network's decisions is reported with a 95% confidence interval on held-out inputs.
   - Fidelity is **above what the task rule gives** (the prior-only control), with non-overlapping intervals.
     Matching the task rule only would mean we explained the task, not the network.
2. **Inside claims.** At least one inside variable must be claimed (amended 12:05, before any R2 result: with zero claims rule 2 would pass trivially), and coverage (which stages have one) is reported. **Tightened 12:45, after seeing the census rule meet the letter of this rule with token-identity claims at the embedding only:** at least one passing claim must sit at a site AFTER the first computation step (not `embed`, not `h1`). The census rule therefore does NOT count. **Tightened again 14:00, before testing:** the claim must be at an INTERMEDIATE site, not the input/embedding site and not the last site before the readout. A claim at the last site can be the predicted scores themselves, which pass interchange almost trivially, so it is behaviour in disguise. Transformers (2 layers): attn0.*, resid0.mid, attn1.pattern. RNNs: h2..h(T-1). The change is stricter, so it can't favour a result. Every variable the construction places in the network passes the calibrated judge's
   substitution and interchange tests (rung 2) at that site.
3. **Not a copy.**
   - Information score S ≤ 0.5. The measure separated every copy (S ≥ 0.77) from the one real big-network account
     (0.35) on networks above 100 parameters. It failed on tiny networks, which aren't the target here.
   - Because the pre-registered test failed, this is a signal, and it is backed by the next point.
4. **Readable account.** Each rule is stated in words in the DESCRIPTION. I read it, and every claim in it is checked
   by a test in the certificate.
5. **Reproducible.** One script regenerates the certificate from the saved file.

**Target:** at least two networks above 100 parameters, from different families (RNN, transformer).

## Micro experiments (each logged in DIARY with its question, decision and result)

| # | Question | Decision it informs | Status |
|---|---|---|---|
| R1 (done) | Where do the readers' steps go on big networks, and when did their key insights appear? | Understanding vs tools vs steps; whether R4 is needed | Insight comes early; most of the budget went into gaming the old copy rule |
| R2 | With its notes plus an exact copy to start from, and progressive replacement, can the reader reach a working construction? | Whether progressive replacement becomes the construction procedure | Keeps readers exact, but no rules get substituted (S stays copy-like) |
| R3 | Same with a stronger reader | Whether capability is the limit | pending |
| R4 | Standard analysis tools in the workbench | Only if R1 or R2 show the reader rebuilding them | pending |
| R5 | Top-down strategy: find a few low-dimensional quantities at the last site before the readout that pass inside tests and that a stated rule turns into the decision. Then explain those quantities from the previous site. Fidelity may drop, as long as it stays above the task rule. 80 steps | Can the cheap reader reach the strict criterion with this method? | running |
| R6 | I act as a strong reader on the smallest networks above 100 parameters (lc_3_4, M8,5) | Is the strict criterion reachable at all, and by what method (to feed back into the reader)? | planned |


## Findings so far (14:20)

1. **Validation got stronger.**
   - The v2 certificate: fidelity with a confidence interval; the task rule; inside claims at INTERMEDIATE sites;
     information S against the reduced-form copy frontier (T2).
   - S classifies every one of 30+ copies and disguises at ≥ 0.91 (`info_measure_T2.txt`).
   - Two translation errors were found and fixed: T1 (lc_8_6 labels) and T2 (frontier form).
2. **The system works on big CLEAN trained networks.** modmul_p113 (226,048 params; blinded reader, Nov 13)
   re-certified: fidelity 97.4%, S 0.009, 7/7 claims at intermediate sites.
3. **On small DENSE networks (AlgZoo > 32 params) nothing meets the criterion yet.** The evidence on why:
   - no mirror symmetry beyond M4,3 (`/tmp/sym.py`: M8,5 25%, M8,8 12%, argmedian 8,5 and 16,5 25%);
   - no crisp relational code in lc_8_6 layer 0;
   - the best behavioural rule (census, 92.4% vs task 85.2%) is NOT the network's internal mechanism (rival test at
     layer 0: interchange 55%);
   - readers buy inside faithfulness by copying layers.
4. **New targets built:** AlgZoo argmedian RNNs 8_5 (112 params), 8_7, 16_5, 16_7 (`lab/targets/build_argmedian.py`;
   exact against torch).

| R7 | Rule first (start from the census rule) | Can rule-level accounts gain inside claims? | lc_8_6: S rose 0.50 → 0.85 (copying layers); in progress |
| R8 | Wide batch, top-down, intermediate-site claims, 100 steps × 2 runs on M8,5, lc_3_4, lc_4_5, argmedian 8,5 | Does any cheap-reader run reach the strict criterion on a dense network? | running (14:20) |
