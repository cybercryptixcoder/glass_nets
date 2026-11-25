# Kronecker sprint: diary

## Nov 25, 02:05 ET (clock): exact tables and the first kill test

- **Exact tables, n = 3 to 18.** These were computed in under a minute on the workspace and checked by a third prime and by the dimension identity.
- **Known necessary conditions alone (B1)** already decide whether g is zero for **97.2–97.9% of all ordered triples at n = 12–18** (balanced accuracy 95.5–96.2%). B1 is:
  - Dvir's bound ν₁ ≤ |λ ∩ μ| in every ordering;
  - its conjugate forms;
  - the length bounds ℓ(ν) ≤ ℓ(λ)ℓ(μ) and their conjugate forms.
- **B1 has no false negatives at any n**, as a set of necessary conditions must. Every error is a zero that the conditions miss, a "hard zero": 68,200 at n = 14 and 1,216,263 at n = 18. They are 2.1–2.8% of all triples, and about 8% of all zeros.
- **This changes the picture in the target report.** The published "best interpretable model, about 83%" was never compared with this 1993 theory, which reaches 97–98% on our triples. The black boxes (98–99%) can therefore only add about 0.5–2 points over known math, and all of it must come from the hard zeros.
- **Pre-registered criterion KR1(c), "beat B1 by ≥ 5 points", is impossible:** B1 is at 97.6–97.9% on the held-out n. The change is declared in the ledger (KR1b) before any rule search.
