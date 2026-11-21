# The decompiler (D1): plan and pre-registered decision rules

*Written Nov 21, 2026, before any discovery-set network was decompiled. Changes after this point are logged at the bottom with their reason.*

## Intent

The question: can a program be recovered from a network's **weights alone**, blind (no task name, no token meaning,
no supplied vocabulary)? The program should:
- replace the network;
- have intermediate variables that are causally real inside the network;
- be readable as an algorithm.

- **If yes:** point it at networks whose algorithm is unknown and report what they compute.
- **If no:** find out which stage breaks and rebuild that stage.

**What counts as "from the weights":**
- The program's tables and coefficients are read from the weight matrices.
- The network's input→output behaviour is used **only to verify**, never to fit.
- A program fitted to the network's outputs would just be a lookup table and proves nothing.

## Measures (frozen for D1)

| measure | definition | pass line |
|---|---|---|
| M1 replacement | agreement of the program's output with the network's argmax, over all n² inputs | ≥ 0.90, and ≥ 0.5 above the best control |
| M2 causal | patch only the recovered coordinates of input x (the projection onto the recovered planes) with those of x′, in the real network's residual stream; the network's output should become program(x′, y) | ≥ 0.80 of patches; random planes of the same dimension must score below 0.2 |
| M3 secret recovery (secret_* tasks only) | recovered coordinates vs. the true group element, up to a group automorphism | ≥ 0.95 of symbols |

## Known cases that must come out as stated before discovery runs count

| case | must |
|---|---|
| modadd_p113_s1, s2 | pass M1, M2; read as addition on a circle of 113 points |
| modmul_p113_s1 | pass M1, M2; circle of 112 points; symbol 0 off the circle |
| modrand_p113 (memorizer) | FAIL M1 (no program) |
| a modadd network with its weights shuffled within each matrix | FAIL M1 |

## Discovery set (scripts/decomp/tasks.py)

| network | what would be new |
|---|---|
| secret_add | the secret relabelling recovered from the weights |
| xyxy, a+b+ab | the hidden map t → t+1 then the discrete log, found unprompted |
| secret_z8z14 | a torus (two circles of different periods), with no labels |
| secret_s4 | non-abelian: needs matrix (irrep) coordinates, not circles; the stretch case |
| pow, a^b | unknown mechanism |
| sqsum, a²+b² | unknown mechanism; squaring is 2-to-1 |

**Prior work:** checked before any claim of novelty (Nanda 2023, Chughtai 2023, Zhong 2023, Doshi 2024 on modular polynomials, and others found in the search).

## Change log
- **Nov 21, 15:25 (clock): pre-registration for the pow held-out primes.** This was written before `scripts/decomp/pow_mechanism.py` ran on `dc_pow_p97_s1` (96 = 32·3), `dc_pow_p101_s1` (100 = 4·25) and `dc_pow_p109_s1` (108 = 4·27). The mechanism counts as general across primes only if, for each network, all of the following hold:
  - **(a) T2:** the CRT split along the prime powers of p−1 explains ≥ 0.90 of each side's representation, and more than every other split.
  - **(b) T4:** swapping one CRT part (each side, each part) succeeds ≥ 0.80, with the random control ≤ 0.20.
  - **(c) T5:** swapping one unit-group factor succeeds ≥ 0.80.

  Seeds at p = 113 (s2, s3) are reported against the same lines. The pass lines are the D1 lines (0.8 causal, 0.2 control), not tuned to these runs.
- **Nov 21, 16:05 (clock): M2 v2.** M2 v1 patched the span of the recovered circle planes. That span depends on which plane-finding variant won, and for a 2-to-1 symbol map (sum of squares) it missed part of what the embedding carries about the coordinate. As a result, a correct program (M1 0.991) failed a causal test that patching the embedding's top PCA subspace passes at 1.0.

  **M2 v2** is an interchange intervention on the recovered variable. One input is moved onto another symbol only inside the subspace through which the representation depends on the recovered coordinate (PCA of its class means grouped by element, 95% of their variance, at most 24 dimensions). It is run in the embedding, and in the side representation when that is where the coordinates came from; the better of the two is reported, with both listed. The random control is a random subspace of the same dimension.

  Pass lines are unchanged (≥ 0.80, control < 0.20). Every network is re-run on v2 (`results/decomp/`); no v1 result is used for claims.
