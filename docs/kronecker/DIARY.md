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

## Nov 25, 02:11 ET (clock): where the hard zeros are (dev n ≤ 14 only)

- **B2, LightGBM on raw parts** (orbit-level 80/20 split, same triples):
  - n = 12: 97.35% against B1's 96.50%; it catches 56% of the hard zeros.
  - n = 14: 98.80% against B1's 97.25%; it catches 71% of the hard zeros, with 0.6% false zeros.
  - So black boxes do learn structure beyond Dvir.
- **Almost all hard zeros sit on the Dvir face**, where one of the bounds is tight (minimum slack 0).
  - That is 89% of them at n = 12 and 94% at n = 14.
  - About 20% of the triples on the face are zero, against 1% at slack 1 and 0.0–0.2% beyond.
  - So the question becomes: **on the Dvir face, when does g vanish?**
- **Small cases** (scripts/kronecker/dev_list_hard.py):
  - At n = 6 the hard zeros fall into 8 symmetry classes, all involving the rectangles 33 or 222 (for example 33|33|33 and 51|33|33).
  - At n = 7 there are 4 classes.
- **Next:** test Dvir's own description of the face. If his theorem gives g on the face as a smaller Kronecker problem built from skew characters (λ/α and μ/α with α = λ∩μ), the face reduces exactly. A recursive Dvir test would then be known math explaining the black-box edge, and would itself be the explicit procedure to benchmark.

## Nov 25, 02:14 ET (clock): known math, applied recursively, matches the black boxes (dev n ≤ 12)

- **Dvir's face formula holds exactly** on the face ν₁ = |λ∩μ|, with α = λ∩μ: g(λ,μ,ν) = ⟨χ^{λ/α} χ^{μ/α}, χ^{ν̄}⟩, where ν̄ is ν without its first row. Checked on every face triple for n = 4–10, with 0 mismatches (scripts/kronecker/face_check.py).
- **P2:** apply the Dvir bounds to the smaller problem on every tight face, recursively (scripts/kronecker/face_reduce.py). It never calls a nonzero triple zero.

  | n | P2 catches the hard zeros | P2 accuracy | LightGBM | transformer (published) |
  |---|---|---|---|---|
  | 10 | 77% | 99.09% | | |
  | 11 | 97% | 99.92% | | |
  | 12 | 82% | 99.36% | 97.35% | 99.42% |

- **Odd n are almost fully explained; even n keep a residue.** Examples of the residue are rectangles such as 33|33|33. That residue is the remaining open seat.
- **The ledger now carries KR1b, a declared change made before any rule search:**
  - (A) does known math match the black boxes on held-out n = 15–18?
  - (B) is there an explicit rule for the residue?
