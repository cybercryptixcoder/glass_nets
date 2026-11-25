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

## Nov 25, 03:44 ET (clock): KR1b (A) met on held-out n = 15–18

- **Known math beats the black box at every held-out n.** P2 (recursive Dvir, frozen at 41549693) against LightGBM trained per n, on the same 2M test triples (results/kronecker/heldout_n15..18.json, summary in KR1b_A.json):

  | n | P2, all ordered triples | P2, test triples | LightGBM, test triples | LightGBM false zeros (test) | P2 false zeros |
  |---|---|---|---|---|---|
  | 15 | 99.82% | 99.80% | 99.25% | 2,697 | 0 |
  | 16 | 99.74% | 99.73% | 99.07% | 6,560 | 0 |
  | 17 | 99.81% | 99.82% | 99.14% | 6,893 | 0 |
  | 18 | 99.74% | 99.75% | 99.07% | 7,444 | 0 |

- The Colab tables match the local ones (sha1 of g equal for n = 15–18).
- **Two more known theorems were added as declared known-math lines** (ledger KR1c, KR1d), before any held-out run of them:
  - P3r adds GWXZ (two-row rectangles squared);
  - P4 adds the Berele–Regev fat-hook bound.
  - Held-out so far: n = 15 P3r 99.86%, P4 99.88%; n = 16 P3r 99.79%, P4 99.82%. No false zeros.
- **Novelty check for (A).** Butbaia–Lee–Ruehle (arXiv 2502.11774, ATMP 30, 2026) report transformers at 99.42 / 99.65 / 99.06% (n = 12 / 13 / 14) and ~83% for explicit formulas. They do not benchmark against Dvir's theorem or any known vanishing criterion; neither does Lee 2023 (LightGBM ~98%).
- **The residue after P4 (dev n ≤ 14) sits in families with known exact formulas.**
  - Two-row shapes: 51–64% of the residue at even n is the Catalan family (k,k).
  - Hooks: most of the residue at odd n.
  - So KR1b (B), a *new* rule catching ≥ 50% at every held-out n, now looks out of reach. A rule for the hook part would be Blasiak's (2012).
  - The one open seat is the Catalan family g((k,k), μ, ν). Known positive rules (Ballantine–Orellana) need μ₁ ≥ 2p − 1, which fails here.

## Nov 25, 04:02 ET (clock): KR1b (B) rule search on dev n: no short rule reaches the line

- **The search** is in scripts/kronecker/rule_search.py, method declared in ledger row KR1e.
  - Universe: symmetry orbits that P4 calls nonzero at n = 8–14, about 208k orbits but only 4,497 distinct feature vectors.
  - Features: 13 symmetric features, 85 atoms. A rule is an OR of up to 3 conjunctions of up to 3 atoms.
  - Constraint: a false-zero budget of 0.1% of the nonzero triples at every dev n.
- **First run.** It died from memory: masks for about 100k conjunctions over all orbits. Rebuilt over the distinct feature vectors.
- **Second run.** It returned an empty rule. No conjunction has positive coverage at every dev n, so all candidates tie at 0.
  - Declared amendment, dev only: ties are broken by mean coverage.
- **Best rule found**, an OR of three conjunctions with 9 atoms:
  - (one Catalan shape, all three two-row, Dvir slack 1);
  - or (one Catalan shape, two rectangles, Dvir slack 1);
  - or (one two-row shape, two rectangles, fat-hook slack 2).
  - It catches 0% of the residue at odd n (9, 11, 13) and 1–26% at even n, mean 7.8%.
  - It was frozen at 04:01:50 (results/kronecker/FROZEN_RULE.json) and is being scored on n = 15–18 as pre-registered (scripts/kronecker/rule_heldout.py).
- **Shuffle control** (labels shuffled within n): mean coverage 0.6%. That is far below the real 7.8%, so the search does find some structure, but nowhere near the 50% line.
- **Reading.**
  - Within this feature space the known-math residue has no short description that also stays inside the false-zero budget.
  - That matches the families it lives in: two-row shapes (Catalan) and hooks.
  - Their known exact formulas are alternating sums of Littlewood–Richardson products, not conditions on simple shape features.

## Nov 25, 04:39 ET (clock): held-out n = 19; the frozen rule fails; where the leftover zeros live

- **KR1f, n = 19** (Colab `kron`; tables pass every check):
  - P2 reaches 99.77% on all 117.6M ordered triples and 99.78% on the test triples, with 0 false zeros.
  - LightGBM on the same test triples: 99.11%.
  - So the KR1b (A) criteria also hold at n = 19. n = 20 is still running.
- **KR1b (B): the frozen rule fails as expected.** On held-out n it catches 0% (n = 15), 2.5% (16) and 0% (17) of P4's leftover zeros. n = 18 is being re-run: two local jobs at n = 18 together ran out of memory, so they now run one after the other.
- **Where P4's leftover zeros live on held-out n** (scripts/kronecker/resid_atlas.py, run after the rule was frozen):

  | n | hook | two-row (Catalan (k,k) or (k+1,k)) | two-row, other | rectangle squared | other |
  |---|---|---|---|---|---|
  | 15 | 52% | 14% | 13% | 2% | 19% |
  | 16 | 36% | 26% | 18% | 0.2% | 21% |
  | 17 | 46% | 6% | 13% | 0% | 35% |

  - The 'other' part, with no two-row shape, hook or squared rectangle, grows with n. That part is the genuinely open territory.

## Nov 25, 05:27 ET (clock): n = 20 P2 job died; rerun queued alone

- On Colab `kron`, the n = 20 P2 + LightGBM job (`p2_heldout.py 20`) left an empty log and is no longer running.
  - It ran in parallel with the n = 19 jobs. The most likely cause is memory: n = 20 has 246M ordered triples and the orbit computation holds several arrays of that size.
- The P4 job for n = 20 is running alone.
- A solo rerun of `p2_heldout.py 20` is queued behind it: /content/p2_20.sh, log /content/kdata/heldout_20b.log, marker /content/kdata/P2_20_DONE. Same frozen code, no change.
