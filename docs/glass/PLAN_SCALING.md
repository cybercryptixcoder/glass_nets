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
2. **Inside claims.** At least one inside variable must be claimed (amended 12:05, before any R2 result: with zero claims rule 2 would pass trivially), and coverage (which stages have one) is reported. Every variable the construction places in the network passes the calibrated judge's
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
| R1 | Where do the readers' steps go on big networks, and when did their key insights appear? | Understanding vs tools vs steps; whether R4 is needed | pending |
| R2 | With its notes plus an exact copy to start from, and progressive replacement, can the reader reach a working construction? | Whether progressive replacement becomes the construction procedure | pending |
| R3 | Same with a stronger reader | Whether capability is the limit | pending |
| R4 | Standard analysis tools in the workbench | Only if R1 or R2 show the reader rebuilding them | pending |
| R5+ | Planned after R1–R2 results | | |
