# Glass Networks: round 2 report (written 2026-11-18 04:02 ET)

The plan was `PLAN_ROUND2.md`, approved 23:36 ET Nov 17. The runs went from 23:51 to about 03:55 ET on Colab runtime `glass2`: 40 of 40 jobs finished. All run folders (757 files) and all 70 logs are synced to GitHub; file counts on Colab and locally match. API cost: v1 $2.97, v2 runs incl. lc $7.72, total $10.69.

## The short answer
1. **No discovery.** No longest-cycle run meets the pre-registered claim criterion. Every run the judge passed is either a full weight copy (the E10 trick) or a near-copy at 0.81–0.88 constants per degree of freedom. The tasks-only (prior) controls all failed, as they should. But a **shuffled-weights control passed** (lc_3_4, 0.81), which shows that judge glass-2 cannot tell a copy from an explanation.
2. **M4,3 is a rediscovery, not a discovery.** Our account matches Robinson's and Hilton's notes (`M43_VS_ARC.md`). We add two things the notes don't state: the reduced recurrence is a 120° rotation, and the exact reduction loses no accuracy.
3. **Reader v2 did not beat v1** (table below). By the plan's own rule, the v2 reader changes F3/F4 should be reverted. That is left to the next agent; it is a coding task.
4. **The real result of this round:** the compression rule is the weak point.
   - The reader learns to pass it by rewriting the weights. It happened 7 times in 40 runs, under both readers, and it also passed a control that should fail.
   - Every genuinely compressed account (M2,2 and M4,3, at ≤ 0.78) is on the two smallest networks. Those are also the only places where the reader wrote a real account.

## Every run (judge glass-2; "E10 copy" = weights stored as scaled integers; const/dof from the judge's compression line)

| network | run | verdict (glass-2) | stages | behaviour | mistakes | const/dof | E10 copy | subs | cost |
|---|---|---|---|---|---|---|---|---|---|
| LC (hidden 3, length 4) | maina (v2) | COMPLETE | 3/3 | 100.0% | 100.0% | 0.00 | YES | 4 | $0.15 |
| LC (hidden 3, length 4) | mainb (v2) | not verified | 0/3 | 52.3% | 33.3% | 0.41 |  | 14 | $0.42 |
| LC (hidden 3, length 4) | prior (v2) | not verified | 0/3 | 85.9% | 0.0% | 0.00 |  | 1 | $0.06 |
| LC (hidden 3, length 4) | shuffled (v2) | COMPLETE | 3/3 | 100.0% | 100.0% | 0.81 |  | 4 | $0.12 |
| LC (hidden 4, length 5) | maina (v2) | COMPLETE | 3/3 | 100.0% | 100.0% | 0.81 |  | 12 | $0.53 |
| LC (hidden 4, length 5) | mainb (v2) | COMPLETE | 3/3 | 99.9% | 99.7% | 0.00 | YES | 9 | $0.23 |
| LC (hidden 4, length 5) | prior (v2) | not verified | 0/3 | 80.8% | 0.0% | 0.00 |  | 1 | $0.07 |
| LC (hidden 4, length 5) | shuffled (v2) | not verified | 3/3 | 100.0% | 100.0% | 1.01 |  | 5 | $0.09 |
| LC (hidden 6, length 5) | maina (v2) | COMPLETE | 3/3 | 99.9% | 98.7% | 0.01 | YES | 8 | $0.27 |
| LC (hidden 6, length 5) | mainb (v2) | not verified | 3/3 | 99.7% | 94.9% | 1.85 |  | 17 | $0.85 |
| LC (hidden 6, length 5) | prior (v2) | not verified | 0/3 | 86.9% | 6.4% | 0.00 |  | 1 | $0.07 |
| LC (hidden 6, length 5) | shuffled (v2) | not verified | 0/3 | 25.2% | 17.6% | 0.47 |  | 7 | $0.11 |
| LC (hidden 8, length 6) | maina (v2) | COMPLETE | 3/3 | 100.0% | 100.0% | 0.88 |  | 11 | $0.32 |
| LC (hidden 8, length 6) | mainb (v2) | not verified | 1/3 | 92.4% | 90.7% | 0.00 |  | 17 | $0.47 |
| LC (hidden 8, length 6) | prior (v2) | not verified | 0/3 | 85.7% | 83.8% | 0.00 |  | 1 | $0.08 |
| LC (hidden 8, length 6) | shuffled (v2) | not verified | 0/3 | 94.3% | 95.7% | 0.10 |  | 5 | $0.10 |
| M2,2 | v1a (v1) | COMPLETE | 2/2 | 100.0% | 0.0% | 0.60 |  | 2 | $0.10 |
| M2,2 | v1b (v1) | COMPLETE | 2/2 | 100.0% | 0.0% | 0.80 |  | 4 | $0.08 |
| M2,2 | v2a (v2) | COMPLETE | 2/2 | 100.0% | 0.0% | 0.60 |  | 2 | $0.07 |
| M2,2 | v2b (v2) | COMPLETE | 2/2 | 100.0% | 0.0% | 0.60 |  | 2 | $0.08 |
| M4,3 | v1a (v1) | COMPLETE | 3/3 | 99.5% | 83.8% | 0.78 |  | 5 | $0.13 |
| M4,3 | v1b (v1) | COMPLETE | 3/3 | 98.8% | 62.2% | 0.78 |  | 2 | $0.07 |
| M4,3 | v2a (v2) | COMPLETE | 3/3 | 99.4% | 82.4% | 0.78 |  | 1 | $0.03 |
| M4,3 | v2b (v2) | COMPLETE | 3/3 | 99.7% | 90.5% | 0.78 |  | 2 | $0.06 |
| M4,4 | v1a (v1) | COMPLETE | 4/4 | 98.3% | 97.8% | 0.89 |  | 11 | $0.39 |
| M4,4 | v1b (v1) | not verified | 4/4 | 100.0% | 100.0% | 1.00 |  | 6 | $0.25 |
| M4,4 | v2a (v2) | not verified | 4/4 | 100.0% | 100.0% | 1.04 |  | 10 | $0.29 |
| M4,4 | v2b (v2) | COMPLETE | 4/4 | 99.1% | 98.8% | 0.85 |  | 9 | $0.34 |
| M8,5 | v1a (v1) | COMPLETE | 5/5 | 99.7% | 98.9% | 0.00 | YES | 6 | $0.21 |
| M8,5 | v1b (v1) | COMPLETE | 5/5 | 98.5% | 93.9% | 0.88 |  | 12 | $0.34 |
| M8,5 | v2a (v2) | COMPLETE | 5/5 | 99.1% | 95.5% | 0.87 |  | 9 | $0.29 |
| M8,5 | v2b (v2) | not verified | 5/5 | 100.0% | 100.0% | 1.09 |  | 11 | $0.50 |
| M8,8 | v1a (v1) | not verified | 0/8 | 32.5% | 60.0% | 0.07 |  | 12 | $0.42 |
| M8,8 | v1b (v1) | COMPLETE | 8/8 | 95.6% | 95.0% | 0.00 | YES | 3 | $0.06 |
| M8,8 | v2a (v2) | not verified | 8/8 | 100.0% | 100.0% | 1.01 |  | 8 | $0.31 |
| M8,8 | v2b (v2) | not verified | 8/8 | 100.0% | 100.0% | 1.00 |  | 10 | $0.35 |
| M16,10 | v1a (v1) | COMPLETE | 10/10 | 97.2% | 74.2% | 0.00 | YES | 5 | $0.12 |
| M16,10 | v1b (v1) | COMPLETE | 10/10 | 96.7% | 67.6% | 0.87 |  | 20 | $0.80 |
| M16,10 | v2a (v2) | COMPLETE | 10/10 | 98.9% | 84.1% | 0.87 | YES | 11 | $0.48 |
| M16,10 | v2b (v2) | not verified | 10/10 | 94.3% | 64.3% | 0.90 |  | 25 | $0.98 |

## The reader comparison (v1 = round 1's reader, v2 = with F3/F4), two runs each on 6 networks

Each cell gives the number of passing runs out of 2 under three standards:
- **A**: COMPLETE under glass-2;
- **B**: A, excluding E10 copies;
- **C**: B, and also at or below the 0.78 claim line (a real compressed account).

| network | v1 A / B / C | v2 A / B / C |
|---|---|---|
| M2,2 | 2 / 2 / 1 | 2 / 2 / 2 |
| M4,3 | 2 / 2 / 2 | 2 / 2 / 2 |
| M4,4 | 1 / 1 / 0 | 1 / 1 / 0 |
| M8,5 | 2 / 1 / 0 | 1 / 1 / 0 |
| M8,8 | 1 / 0 / 0 | 0 / 0 / 0 |
| M16,10 | 2 / 1 / 0 | 1 / 0 / 0 |
| **total (of 12)** | **10 / 7 / 3** | **7 / 6 / 4** |

**Reading:**
- On the only standard that means "explained" (C), the two readers are 3 vs 4. The difference is one M2,2 run at 0.80 vs 0.60, which is noise.
- On the looser standards, v1 is ahead because it copied more.
- v2 was cheaper on M4,3 ($0.03 and $0.06, vs $0.13 and $0.07), but not elsewhere.
- **Conclusion: v2 shows no improvement. Revert F3/F4 (plan step 4).**
- F1 (single compression rule) and F2 (tie fix) stay; they are judge fixes, not reader fixes.

## Longest-cycle networks vs the pre-registered criterion

The criterion: COMPLETE, < 0.78 constants per dof, the prior control fails, no prior interpretation, a human read.

| network | best main run | meets criterion? | why not |
|---|---|---|---|
| lc_3_4 | maina | no | E10 copy (weights as integers × 1/10000) |
| lc_4_5 | maina (0.81) and mainb (copy) | no | maina is a re-description above the line; mainb is a copy |
| lc_6_5 | maina | no | E10 copy |
| lc_8_6 | maina (0.88) | no | near-copy (288 decimals) |

- The prior controls failed on all four (behaviour 81–87%, 0/3 stages). The networks are not just "the task".
- The **shuffled control on lc_3_4 passed**, which shows the judge accepts any faithful re-description.
- **One real lead:** the lc_4_5 descriptions. Layer 0 is near-hard copying (86% of query rows put more than 90% of their attention on one key). Layer 1 at the readout is a soft max/min selector over the keys, read through a class table. This is a partial human account, not yet a compressed construction.

## What went wrong, and what was learned (details: LEARNINGS.md)
- **E9** (high confidence): M4,3 was a blind rediscovery. Check prior work before every claim.
- **E10** (high confidence): integer-scaled copies get past F1's "non-integer literals" count. Seen 7 times, under both readers.
- **E11** (high confidence; new): a degrees-of-freedom ratio between 0.78 and 0.9 is not evidence of explanation.
  - Every non-E10 pass above 0.78 is a near-copy: it keeps more than 80% of the network's free numbers.
  - The shuffled lc_3_4 control passed at 0.81.
  - So the pass line (0.9) is too loose for certifying. The 0.78 claim line separated real accounts from copies in both rounds.
- **Process:**
  - The judge stayed frozen as planned, so all 40 runs are comparable.
  - At 00:27 I recommended fixing E10 mid-round (option B). Shreyas chose to keep the job running (option A). As a result, the lc runs were spent under the leaky rule.

## What the next round should do (for the next agent; coding not done here)
1. **Judge glass-3.** One compression rule:
   - count every numeric literal except integers with |n| ≤ 10 and 0.5, whatever form it is written in;
   - and set the pass line at the claim line (< 0.78), since the shuffled control passed at 0.81.
   - Re-judge all round-1 and round-2 runs with it. The prediction, stated now: only M2,2 and M4,3 survive.
2. Revert reader v2 (F3/F4) to v1, per the plan's rule.
3. Re-run longest-cycle under glass-3, starting with lc_4_5, which has the best partial account. Ask Shreyas before spending.
4. Relabel M4,3 in README and the catalog as "blind rediscovery of ARC's account".
