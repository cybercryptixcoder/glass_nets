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
