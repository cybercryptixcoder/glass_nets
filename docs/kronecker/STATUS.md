# Kronecker sprint: status

Branch `kronecker`. Ledger row KR1 in docs/glass/CHANGES.md (the pre-registration). Sprint 1 of two run in parallel on Nov 25 (sprint 2: branch `liar`). Final report due about 12:40 ET.

## Goal
An explicit, checkable rule for when the Kronecker coefficient g(lam, mu, nu) of S_n is zero, as accurate as the black boxes (about 98%), verified exhaustively on n = 15..18, which are never used to find it.

## Done
- Exact tables for n = 3..18 (scripts/kronecker/kron_tables.py), checked by a third prime and the dimension identity. Data lives outside the repo in ~/kdata (kron_n<n>.npz: parts, g, dims); regenerate with `python3 scripts/kronecker/kron_tables.py ~/kdata 3 4 ... 18` (about 30 s in total).
- Nonzero fraction by n: 0.61 (12), 0.72 (16), 0.74 (17), 0.76 (18).

## Next
1. Kill test: B1 (known necessary conditions) and B2 (LightGBM) per n, on identical triples.
2. Rule search on n <= 14 only; freeze; score n = 15..18.

## Colab
None yet for this sprint.

## Update, Nov 25 03:44 ET (clock)
- **KR1b (A) met.** P2 beats per-n LightGBM and is above 99.0% at every held-out n = 15–18 (results/kronecker/KR1b_A.json). It has 0 false zeros everywhere; LightGBM has 2.7k–7.4k on its test triples.
- **Known-math lines P3r (KR1c) and P4 (KR1d)** are running locally on n = 15–18: `scripts/kronecker/p34_heldout.py`, log in ~/kdata/p34_heldout.log, results in results/kronecker/heldout_p3r_n*.json and heldout_p4_n*.json.
- **The Colab `kron` session** (CPU, High-RAM) finished all held-out jobs. It can be stopped or reused.
- **KR1b (B):** the residue after P4 lies in families with known exact formulas (two-row, hooks). The open seat is the Catalan family. See DIARY 03:44.

## Update, Nov 25 04:26 ET (clock): state for the next session

**Held-out results, all with 0 false zeros:**

| n | P2 | P3r | P4 | LightGBM (test triples) |
|---|---|---|---|---|
| 15 | 99.82% | 99.86% | 99.88% | 99.25% |
| 16 | 99.74% | 99.79% | 99.82% | 99.07% |
| 17 | 99.81% | 99.83% | 99.85% | 99.14% |
| 18 | 99.74% | 99.77% | 99.80% | 99.07% |

- **KR1b (A) met** (results/kronecker/KR1b_A.json).
- **KR1b (B):**
  - The rule search (KR1e) found nothing near the line on dev n.
  - The best rule (9 atoms) was frozen at 04:01:50 (results/kronecker/FROZEN_RULE.json).
  - Held-out scoring is in scripts/kronecker/rule_heldout.py, results/kronecker/rule_heldout.json: n = 15 caught 0%, n = 16 2.5%, n = 17 0%, n = 18 running. So (B) is **not met**.
- **KR1f, n = 19 and 20,** on Colab `kron`:
  - Tables are built and pass every check: n = 19 has 490 partitions, max g 2,274,958; n = 20 has 627 partitions, max g 6,408,361.
  - `p2_heldout.py` (19, 20) and `p34_heldout.py p4` (19, then 20) are running.
  - Logs are in /content/kdata/heldout_19.log, heldout_20.log, p4_19.log, p4_20.log, and the marker /content/kdata/HELDOUT_19_20_DONE.
  - Results go to /content/kron/results/kronecker/heldout_n19.json, heldout_n20.json, heldout_p4_n19.json, heldout_p4_n20.json. Download them, check the sha1, commit.
- **Exact-table checks** pass for every n ≤ 20: symmetry, conjugation, trivial and standard characters, the full dimension identity (results/kronecker/table_checks.json; n = 19, 20 in /content/kron/results/kronecker/table_checks.json).
- **Next:**
  1. Run `python3 scripts/kronecker/resid_atlas.py 15 16 17 18` locally after rule_heldout finishes. It shows where P4's leftover zeros live on held-out n; this is allowed now that the rule is frozen.
  2. Finish docs/kronecker/DISCOVERY.md: the (B) outcome, n = 19 and 20, the atlas.

## Update, Nov 25 06:15 ET (clock): Kronecker sprint complete

- **KR1b (A) met at n = 15–18; KR1f met at n = 19 and 20.** P2 is above LightGBM by 0.55–0.68 points at every n, with 0 false zeros over 465.2M triples.
- **KR1b (B) not met.** The frozen rule catches 0–2.5% of the leftover zeros.
- **All results are committed** under results/kronecker/; the summary is docs/kronecker/DISCOVERY.md.
- **The Colab `kron` session was stopped after every result had been downloaded and committed.** The n = 19 and 20 tables can be rebuilt with scripts/kronecker/kron_tables.py in about 5 minutes.
- **Nothing is left running for this sprint.**
