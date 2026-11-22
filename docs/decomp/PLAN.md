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
- **Nov 21, 16:36 (clock): non-abelian mode (orbit), added after exploration on `dc_secret_s4_s2`.** This entry is written after looking at that network, so the S4 result is a development case, not a held-out one. The held-out non-abelian case is `dc_secret_s4_s3`, being trained now, with a different seed and the same secret.
  - **Method.** In the whitened representation, at the smallest rank where the symbols sit on a sphere, find every orthogonal map that permutes the points; the regular subgroups of order n give the Cayley table.
  - **Disclosed selection.** Which symbol is the identity, and left versus right, is not fixed by the orbit. The decompiler picks it by agreement with the network (M1) among the candidates, and reports their number.
  - **M2.** The interchange test, inside the sphere subspace.
  - **Pass lines.** Unchanged.
- **Nov 21, 16:40 (clock): pre-registration, multiplication mod a composite n.** The networks are `dc_mulmod120_s1`, `dc_mulmod120_s2` and `dc_mulmod105_s1`, all trained, none examined yet.
  - **Hypothesis.** By analogy with a^b, the network multiplies in Z_n by splitting the base ring along the prime powers of n: 120 = 8·3·5 and 105 = 3·5·7. Inside each part it uses that part's multiplicative structure.
  - **Test.** `scripts/decomp/pow_mechanism.py --mulmod n`, with the same criteria as the pow held-out primes: (a) T2 ≥ 0.90 and the best split; (b) T4 ≥ 0.80 with control ≤ 0.20; (c) T5 ≥ 0.80.
  - **Also run.** The group decompiler on `dc_addmod120_s1`; the prediction is Z_120 (or an isomorphic product) with M1 and M2 passing. And on `dc_cubesum_s1`: since gcd(3, 112) = 1, cubing is a bijection mod 113, so the prediction is G = Z_113 with the name k·t³. Naming does not test cubes yet; the prediction is only about the group and M1/M2.
- **Nov 21, 16:55 (clock): pre-registration, blind challenge set.** Ten random finite abelian groups (orders 72–120, invariant factors drawn by seed from a fixed list in `tasks.py`), each under a random secret labelling: `dc_randab{1,2,4,5,6,9,11,13,19,21}`. The seeds were picked to cover different groups by looking only at the group lists, never at networks. Every network is decompiled blind.
  - **Per network, a pass needs:** (1) the recovered group isomorphic to the true one, i.e. equal invariant factors; (2) M1 ≥ 0.90; (3) M2 ≥ 0.80 with the control < 0.20; (4) answer-key isomorphism ≥ 0.95 (`score_known.py`).
  - **Reporting.** The count of passes is reported as it comes out. Networks that fail to generalise (test accuracy < 0.95) are excluded and reported as excluded.
- **Nov 21, 22:30 (clock): factor mode (`lab/decomp/factorize.py`), and its pre-registered predictions.** The tool was developed on `dc_pow_s1`: it splits into "multiplication in Z_7" and "multiplication in Z_16". It is now run unchanged on networks it was not developed on.
  - **SPLITS** means argmax agreement ≥ 0.98 and logit share ≥ 0.90.
  - **Predictions, written before running:**

    | network | prediction |
    |---|---|
    | `dc_mulmod120_s1`, `dc_mulmod120_s2` (never examined) | split into parts matching multiplication in Z_8, Z_3 and Z_5 |
    | `dc_mulmod105_s1` (never examined) | Z_3, Z_5 and Z_7 |
    | `dc_pow_p97_s1` | Z_32 and Z_3 |
    | `dc_pow_p101_s1` | Z_4 and Z_25 |
    | `dc_pow_p109_s1` | Z_4 and Z_27 |
    | `dc_pow_s2`, `dc_pow_s3` (these use mixed characters) | may not split cleanly; whatever comes out is reported |
    | `modadd_p113_s1`, `dc_secret_add_s1` (113 is prime, nothing to split) | no split |
    | `modrand_p113` (memorizer) | no split |
    | `dc_secret_z8z14_s1` (Z_8 × Z_14 ≅ Z_2 × Z_7 × Z_8) | may split along its prime parts, as addition tables |
    | `dc_addmod120_s1` | may split as addition in Z_8, Z_3 and Z_5 |

  - **A miss is reported as a miss.** For a part whose number of values matches a prime power, a "match" means the same row and column profiles as Z_m's multiplication or addition table.
- **Nov 21, 23:19 (clock): challenge set v1 result (pre-registered at 16:55), and the v2 plan.**
  - **v1 result: 3 of 10 pass** (randab1 Z7×Z14, randab5 Z3×Z36, randab19 Z4×Z28). This is the pre-registered outcome and it stands.
  - **v1 failures:**
    - randab2 found Z2×Z18 instead of Z6×Z18.
    - randab11 found the right group, Z2×Z50, but its program scored 0.02.
    - randab6 and randab13 each missed a Z2.
    - randab4 and randab9 found no group.
    - randab21 found Z140 instead of Z5×Z20.
  - **Causes found:**
    1. The two inputs' coordinates were taken from different representations (F for a, G for b). They were aligned only for cyclic groups, which caused randab11 and randab2.
    2. The group vote picked a group missing a factor, or one made up by a false relation.
  - **Decompiler v3:**
    - one set of symbol coordinates for both inputs, since they share an embedding;
    - the result kept is the one under which the MLP depends most on g(a) + g(b) (diagonal share), which uses activations only.
  - **v3 is developed on the v1 failures,** so v1 cannot test it. **Challenge set v2:** 12 new networks, one for each group in the list: randab25, 38, 46, 35, 28, 30, 41, 33, 37, 27, 40, 26. They are trained after this entry and decompiled once with v3. The pass criteria are the same as v1.
- **Nov 22, 00:05 (clock): pre-registration, factor-mode challenge over primes.** Networks to be trained on a^b mod p for p = 89, 103, 107, 127, 131, 137, 139 and 149, never seen before; p − 1 = 8·11, 2·3·17, 2·53, 2·9·7, 2·5·13, 8·17, 2·3·23 and 4·37.
  - **Prediction.** Factor mode SPLITS (argmax ≥ 0.98) into parts whose tables are multiplication in Z_m. The m's multiply to p − 1, and each m is a prime-power factor of p − 1 or a product of several, since merged parts like Z_15 are allowed.
  - **Pass** means all of that. Anything else is reported as a miss.
  - The block-count ceiling was raised from 40 to 60 before training, because p = 107 needs a 53-block part.
