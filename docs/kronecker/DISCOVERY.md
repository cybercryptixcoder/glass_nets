# Known theorems, not learning, explain the machine-learning "edge" on Kronecker vanishing

*Sprint 1 of Nov 25, 2026. Branch `kronecker` of this repo. Plain-language summary for readers outside the project.*

## The result in one sentence

Dvir's 1993 theorem, applied recursively by a short procedure with no learning at all, decides whether a Kronecker coefficient is zero for 99.7–99.8% of all ordered triples at n = 15–20 (more than 465 million triples), with no false zeros. That beats a gradient-boosted classifier trained separately at each n (99.1–99.3%) and the best published neural results (transformers, 99.1–99.7% at n = 12–14). The machine-learning papers on this problem never compared against these classical theorems.

## What the question is

- **What a Kronecker coefficient is.** For three partitions λ, μ, ν of the same number n, g(λ, μ, ν) counts how often one irreducible representation of the symmetric group S_n appears inside the tensor product of two others.
- **Why vanishing matters.** Whether g is zero is a central open problem:
  - there has been no combinatorial rule since Murnaghan asked for one in 1938;
  - deciding positivity is NP-hard;
  - it matters for geometric complexity theory.
- **The machine-learning record:**
  - Lee (2023): LightGBM and CNNs predict vanishing with about 98% accuracy.
  - Butbaia, Lee and Ruehle (arXiv 2502.11774; Adv. Theor. Math. Phys. 30, 2026): transformers reach 99.42 / 99.65 / 99.06% at n = 12 / 13 / 14, and "explicit formulas" reach about 83%.
- **The gap in that record.** Neither paper measures what the known theorems already decide.

## What we did

1. **Exact tables.** Every coefficient for n = 3–18, about 100 million ordered triples at n = 15–18. They were computed from characters and summed modulo three primes, with each value confirmed by the third prime.
   - Checks: symmetry in all three arguments, the conjugation rule, the trivial and standard characters, and Σ_ν g(λ,μ,ν) dim ν = dim λ · dim μ for every pair (λ, μ).
   - Code: scripts/kronecker/kron_tables.py; checks in scripts/kronecker/table_checks.py, results/kronecker/table_checks.json.
2. **Known necessary conditions (B1):** Dvir's bound ν₁ ≤ |λ ∩ μ| and the length bounds, in every ordering and with conjugates. Alone they reach 97.2–97.9%.
3. **P2, the recursive step.** On the boundary where Dvir's bound is tight, Dvir's theorem rewrites g exactly as a smaller Kronecker problem: g(λ,μ,ν) = ⟨χ^{λ/α} χ^{μ/α}, χ^{ν̄}⟩, with α = λ∩μ and ν̄ = ν without its first row.
   - We checked this identity against the exact tables, with 0 mismatches for n ≤ 10.
   - P2 applies B1 again to every piece of the smaller problem, recursively. It can only say "zero" when a theorem forces it.
   - Code: scripts/kronecker/face_reduce.py, frozen before any held-out test.
4. **The test.** We held n = 15–18 out from the start. Only n ≤ 14 was used to build anything (pre-registration: ledger rows KR1, KR1b in docs/glass/CHANGES.md).
   - On held-out n, P2 was scored on every ordered triple.
   - LightGBM was trained per n on 80% of the symmetry orbits, so no symmetric copy of a test triple is in training. Both were scored on the same test triples.

## The numbers

| n | P2 (all triples) | P2 (test triples) | LightGBM (test triples) | P2 false zeros | LightGBM false zeros |
|---|---|---|---|---|---|
| 15 | 99.82% | 99.80% | 99.25% | 0 | 2,697 |
| 16 | 99.74% | 99.73% | 99.07% | 0 | 6,560 |
| 17 | 99.81% | 99.82% | 99.14% | 0 | 6,893 |
| 18 | 99.74% | 99.75% | 99.07% | 0 | 7,444 |
| 19 (declared later, KR1f) | 99.77% | 99.78% | 99.11% | 0 | 7,405 |
| 20 (declared later, KR1f) | 99.74% | 99.74% | 99.06% | 0 | 7,850 |

- **Pre-registered criterion KR1b (A):** P2 at least as accurate as LightGBM and at least 99.0% at every held-out n. **Met at n = 15–18, and again at n = 19 and 20** (declared before those tables existed, ledger KR1f).
- **Two more known theorems.** Each was declared before it was run on held-out n:
  - P3r adds Garsia–Wallach–Xin–Zabrocki 2012 (two-row rectangles squared);
  - P4 adds the Berele–Regev 1987 hook theorem.
- **Held-out results for those two:**

  | n | P3r | P4 |
  |---|---|---|
  | 15 | 99.86% | 99.88% |
  | 16 | 99.79% | 99.82% |
  | 17 | 99.83% | 99.85% |
  | 18 | 99.77% | 99.80% |
  | 19 | — | 99.81% |
  | 20 | — | 99.77% |

  None of them has any false zeros.
- **The same procedures on the published n:**

  | n | P2 | P4 | transformer (published) |
  |---|---|---|---|
  | 12 | 99.36% | 99.64% | 99.42% |
  | 13 | 99.87% | 99.95% | 99.65% |
  | 14 | 99.69% | — | 99.06% |

  P2 was built while looking at n ≤ 12, so these are not held-out numbers.

## What this does and does not show

- **It shows** that the accuracy of the learned classifiers, which those papers present as unexplained structure found by machine learning, is already exceeded by classical theorems applied systematically.
- **It shows** that, unlike the classifiers, the theorems never call a nonzero coefficient zero.
- **It does not give a new theorem.** Every step is known mathematics; the new part is applying it recursively and benchmarking it exhaustively.
- **What is left over** after P4 is 0.12–0.20% of triples at n = 15–18 (results/kronecker/resid_atlas.json):
  - triples with a hook: 34–52%;
  - triples with a two-row shape: 19–43% (the Catalan shape (k, k) alone is 13–26% at even n);
  - triples with none of these special shapes: 19–37%, growing with n. That part is the genuinely open territory.
- **The second pre-registered question, KR1b (B), was not met.** It asked for a short new rule catching at least 50% of that remainder at every held-out n, with at most 0.1% false zeros.
  - A declared search over 85 symmetric shape features found nothing near the line even on the development sizes (KR1e; results/kronecker/rule_search.json).
  - Its best rule was frozen before scoring (results/kronecker/FROZEN_RULE.json). It catches 0%, 2.5%, 0% and 0.9% of the remainder at n = 15, 16, 17, 18.
  - The shuffled-label control found even less (0.6% mean coverage on the development sizes).
  - LightGBM catches about half of the same remainder, but only by also calling thousands of nonzero coefficients zero.
- **Caveats:**
  - The transformer numbers come from the authors' own random test splits, not from our exhaustive scoring.
  - Our LightGBM is a stronger baseline than the published one (99.1–99.3% against about 98%).

## How to check it yourself (no model needed)

- **Build the tables:** `python3 scripts/kronecker/kron_tables.py`. This writes kron_n*.npz to $KDATA; the sha1 of each g table is recorded in results/kronecker/heldout_n*.json.
- **Held-out scoring:**
  - `python3 scripts/kronecker/p2_heldout.py 18` scores P2 and LightGBM;
  - `python3 scripts/kronecker/p34_heldout.py p4 18` scores P4.
- **Second question, KR1b (B):** `python3 scripts/kronecker/rule_search.py` (development sizes) and `python3 scripts/kronecker/rule_heldout.py 15 16 17 18` (frozen rule).
- **Table checks:** `python3 scripts/kronecker/table_checks.py 3 ... 20`.
- **Technical summary with every verdict:** docs/kronecker/FINDINGS.md.
