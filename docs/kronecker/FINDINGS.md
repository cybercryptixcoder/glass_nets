# Kronecker sprint: findings (technical)

*Nov 25, 2026. Branch `kronecker`. Plain-language version: docs/kronecker/DISCOVERY.md. Story: DIARY.md. Ledger: docs/glass/CHANGES.md rows KR1 to KR1f.*

## Verdicts by ledger row

| row | question | verdict | evidence |
|---|---|---|---|
| KR1 | explicit rule ≤ 10 terms, ≥ 98%, within 1 point of B2, ≥ 5 points above B1 | amended before any rule search (criterion c impossible: B1 alone 97.2–97.9%) | results/kronecker/kill_b1.json |
| KR1b (A) | P2 (recursive Dvir, frozen) ≥ LightGBM and ≥ 99.0% at every held-out n = 15–18 | **met** | results/kronecker/KR1b_A.json, heldout_n15..18.json |
| KR1f | the same criteria at n = 19, 20 (declared before their tables existed) | **met** | results/kronecker/KR1f.json, heldout_n19.json, heldout_n20.json |
| KR1b (B), via KR1c–KR1e | a new rule (≤ 10 atoms) catching ≥ 50% of the known-math residue at every held-out n, false zeros ≤ 0.1% | **not met** (frozen rule: 0%, 2.5%, 0%, 0.9% at n = 15–18) | results/kronecker/FROZEN_RULE.json, rule_heldout.json, rule_search*.json |

## Numbers

| n | triples | P2 all | P2 test | LightGBM test | P2 false zeros | LightGBM false zeros (test) | P3r | P4 |
|---|---|---|---|---|---|---|---|---|
| 15 | 5.45M | 99.82% | 99.80% | 99.25% | 0 | 2,697 | 99.86% | 99.88% |
| 16 | 12.3M | 99.74% | 99.73% | 99.07% | 0 | 6,560 | 99.79% | 99.82% |
| 17 | 26.2M | 99.81% | 99.82% | 99.14% | 0 | 6,893 | 99.83% | 99.85% |
| 18 | 57.1M | 99.74% | 99.75% | 99.07% | 0 | 7,444 | 99.77% | 99.80% |
| 19 | 117.6M | 99.77% | 99.78% | 99.11% | 0 | 7,405 | — | 99.81% |
| 20 | 246.5M | 99.74% | 99.74% | 99.05% | 0 | 7,850 | — | 99.77% |

- **Procedures:**
  - P2: Dvir bounds plus Dvir's face formula applied recursively (scripts/kronecker/face_reduce.py).
  - P3r: P2 plus the GWXZ theorem (scripts/kronecker/gwxz.py).
  - P4: P3r plus the Berele–Regev hook bound (scripts/kronecker/fathook.py).
  - P3r and P4 were declared before they ran on held-out n (KR1c, KR1d).
- **LightGBM** (600 trees, 255 leaves) was trained per n on 80% of the symmetry orbits. It is scored on up to 2M triples of the held-out orbits.
- **Exact tables** for n = 3–20 pass every check in results/kronecker/table_checks.json:
  - symmetry in the three arguments;
  - conjugating two arguments;
  - the trivial and standard characters;
  - Σ_ν g·dim ν = dim λ·dim μ for every pair.

## Where the black box's errors fall (KR1g, descriptive; results/kronecker/lgbm_errors_n15.json, n16.json)

- LightGBM re-trained exactly reproduces its recorded test accuracy.
- Of its test errors, P2 is right on 86.9% (n = 15) and 83.5% (n = 16); P4 on 91.3% and 88.7%.
- LightGBM is right on 51.3% and 43.5% of P2's errors.

## Residue after P4 (held-out n = 15–18; results/kronecker/resid_atlas.json)

- **Size:** 0.12–0.20% of ordered triples.
- **By family:**
  - hooks: 34–52%;
  - two-row shapes: 19–43%, including the Catalan shape (k,k) at 13–26% of the even-n residue;
  - none of these special shapes: 19–37%, rising with n.
- **What these families have:**
  - Hooks have a positive combinatorial rule (Blasiak 2012).
  - Two-row shapes have alternating Littlewood–Richardson formulas. Ballantine–Orellana's positive rule needs μ₁ ≥ 2p − 1, which fails for the Catalan family.

## Novelty

- **Butbaia–Lee–Ruehle** (arXiv 2502.11774; Adv. Theor. Math. Phys. 30, 2026) and **Lee** (2023) report:
  - LightGBM / CNN at about 98%;
  - transformers at 99.42 / 99.65 / 99.06% for n = 12 / 13 / 14;
  - "explicit formulas" at about 83%.
- They do not benchmark against Dvir's theorem or any known vanishing criterion. No other benchmark of classical theorems against these models was found.
- The theorems themselves are known: Dvir 1993; Garsia–Wallach–Xin–Zabrocki 2012; Berele–Regev 1987. The new part is applying them recursively and scoring them exhaustively.

## Caveats

- No new mathematics.
- P2's recursion was built while looking at n ≤ 12, then frozen before any held-out run.
- The transformer numbers use the authors' random test splits.
- Our LightGBM is stronger than the published one: 99.1–99.3% against about 98%.
