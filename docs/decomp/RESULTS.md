# Blind decompilation of trained transformers: results on one page

*Branch `decompiler`, Nov 21–22, 2026. Code: `lab/decomp/`.*
- *The rules for every test below were written in `docs/decomp/PLAN.md` before the networks they test were trained, and the commit times record that.*
- *Only held-out, pre-registered results count here. Development results are listed separately and are not evidence.*

## What it does

You give it the weight file of a trained transformer and nothing else: no task name, no meaning for the input symbols, no candidate features. It returns three things:

1. **The structure the network uses.** For example:
   - "a group isomorphic to Z5 × Z20";
   - "a non-abelian group of order 60 with A5's element orders";
   - "two independent multiplications, in Z4 and in Z27".
2. **A short runnable Python program** that computes what the network computes. Every table in it is read from the weights and internal activations; none is fitted to the outputs.
3. **Two checks of that program:**
   - **M1:** it agrees with the network on every input.
   - **M2:** moving only the recovered coordinates inside the real network changes the answer exactly as the program predicts, against a random-subspace control.

The networks are trained with their symbols relabelled by a secret random permutation, so there is no number line or familiar ordering to find. The answer keys are used only afterwards, by `scripts/decomp/score_known.py`. A pass needs all of these:
- the right group;
- M1 ≥ 0.9;
- M2 ≥ 0.8, with the control below 0.2;
- the recovered structure equal to the secret one on ≥ 95% of pairs.

```
python3 lab/decomp/decompile.py <model_id>     # group mode, abelian then orbit mode: report + program
python3 lab/decomp/factorize.py <model_id>     # factor mode: does the network split into independent parts?
python3 lab/decomp/explain.py <model_id>       # both, summarised in plain words
python3 scripts/decomp/score_known.py <id>     # afterwards only: compare with the hidden answer key
```

## Held-out, pre-registered results

Versions:
- **v3.3** is the decompiler frozen at 00:32 on Nov 22.
- **v3.5** = v3.3 + orbit mode for groups of any rank, frozen at 01:53. It was pre-registered as a second run on the sets that were still training then.
- **Factor v2** is factor mode with absorbing symbols detected up to the network's own errors, frozen at 02:40.

| test (pre-registration time) | networks | result |
|---|---|---|
| **Random abelian groups, order 72–120, challenge v2** (Nov 21 23:19; v3.3) | 12 | **8 / 12 recovered exactly.** In each pass the program equals the secret operation on every pair (law vs truth 1.000), with M1 1.000 and M2 1.000 against 0.000. Three of the 4 failures have three or more Z2 factors |
| **Random abelian groups, challenge v3** (00:45; v3.3) | 10 trained (2 lost with a GPU session) | **3 / 5 completed recovered exactly** (Z7×Z14, Z2×Z2×Z30, Z4×Z28). The 2 failures again have several Z2 factors. 5 more were still running at wrap-up |
| **Held-out S4** (00:06) | 2 generalized | **2 / 2 recovered exactly.** The 24 × 24 table equals the secret one on all 576 pairs; M1 1.000; M2 1.000 and 0.986 against 0.000 |
| **Blind non-abelian groups never seen by the tool** (00:53): A5, S3×S3, A4×Z3, SL(2,3), D12, Z7⋊Z3, Z5⋊Z4, D10 | 9 trained, 6 generalized | **v3.5: 2 / 5 completed recovered exactly:** **A5** (the 60 × 60 table equals the secret one) and **S3×S3**. Each has law vs truth 1.000, M1 1.000, and M2 1.000 against 0.000. v3.3, the version pre-registered first: 0 / 5. A5 seed 2, S3×S3 seed 2 and A4×Z3 fail; SL(2,3) was still running |
| **Factor mode, a^b mod 109; a·b mod 120 and mod 105** (Nov 21 22:30) | 3 | **3 / 3 split as predicted.** a^b mod 109 is multiplication in Z4 and Z27. Mod 120 and mod 105 are Z8 × Z15 and Z7 × Z15: the network keeps its 3- and 5-parts together, which the logit test decides |
| Factor mode, a^b mod 97 and mod 101 (22:30) | 2 | 0 / 2 (97 partial at 0.949; 101 not split) |
| **Factor mode prime challenge** (00:02; v1): p = 89, 103, 107, 127, 131, 137, 139, 149 | 8 | **2 / 8 by the strict rule.** p127 gives Z7 and Z18; p131 gives Z5 and Z26. Four more split cleanly, but their part tables were polluted by near-absorbing symbols, which led to factor v2 |
| **Factor mode v2, prime set v2** (01:56; frozen 02:40): p = 61, 67, 71, 73, 79, 83, 151, 157 | 8 | **4 / 8 pass:** p79 gives Z6, Z13; p83 gives Z2, Z41; p151 gives Z3, Z50; p157 gives Z4, Z39. Each is the factorisation of p − 1, read from the weights. p61, p67 and p71 split cleanly with unnamed parts; p73 does not split |
| Challenge v1 (Nov 21 16:55; first version) | 10 | 3 / 10. It stands as the first result. Its failures drove v3, which is why v1 is not counted again |

## What the passes look like

- **Abelian:** "Z_5 × Z_20; program `c = CLASS_OF[(g(a) + g(b)) mod (5, 20)]` with a 100-entry symbol table". The program equals the secret operation on all 10,000 pairs.
- **Non-abelian:** a 60 × 60 Cayley table that is exactly the secret A5 table. The network stores the 60 symbols as an orbit on a 6-dimensional sphere (two 3-D icosahedral representations), and the tool reads the group off the rotations that permute that orbit. For S4, the symbols are the rotations of a cube in 3-D.
- **Factor mode:** "the network computes a^b mod 107 as two independent multiplications, in Z_2 and in Z_53". That is the factorisation 106 = 2 · 53, read from the weights. On p = 107 this came from factor v2 run afterwards, so it is development only.

## The finding the tool led to: how a transformer computes a^b mod p

Group mode correctly reports that a^b is not a group law. Its tools then exposed a mechanism we have not seen reported, **nested discrete logarithms** (`POW_MECHANISM.md`):
1. The network takes x = log_g(a).
2. It splits the exponent ring Z_{p−1} into its prime-power parts by the Chinese remainder theorem.
3. Inside each part it takes a second logarithm. At p = 113 these are:
   - mod 7: the Legendre and cubic characters;
   - mod 16: the 2-adic valuation and the factors ⟨−1⟩ × ⟨5⟩ of the unit part.

Swapping a single part causally moves the answer exactly as predicted: 1.000 on every swap at p = 113 and at p = 109. Factor mode, built afterwards, finds the split blind. It predicted the parts in advance at p = 109, 127 and 131.

## Development results (used to build the tool; not evidence)

| network | result |
|---|---|
| a+b mod 113, plain and under a secret labelling (2 seeds) | Z_113; the secret is recovered exactly |
| a·b, a+b+ab, a²+b² mod 113 | Z_112 plus an absorbing zero, named k·log_3(t); the same with k·log_3(t + 1), found unprompted; Z_113 with k·t², "the input is squared" |
| Z8 × Z14 under a secret labelling | Z_2 × Z_56 (isomorphic) |
| challenge v1 networks, re-run with v3.3 | 9 / 10 |
| v2 failures, re-run with v3.5 | Z2×Z2×Z30 now exact; Z2×Z2×Z2×Z12 gets the exact table, but its M2 is 0.63 |
| factor v2 on the first prime challenge | p89 → Z8, Z11; p107 → Z2, Z53 |
| memorizer (random labels) | no structure, correctly |

## Limits, plainly

- **One architecture:** a one-layer, four-head transformer without LayerNorm. The method reads the embedding, the MLP's input and output, and the unembedding.
- **Binary operations on a finite symbol set only.**
- **Groups with several Z2 factors often fail.** Their order-2 characters are 1-D axes. When two characters multiply to an order-2 character, a family of mixed planes is radius-uniform, and the plane finder can land in it.
- **Orbit mode needs the symbols on a sphere.** For non-abelian groups and as a fallback, the symbols must be an orbit of an orthogonal action, and the network must store them faithfully there. The identity symbol and left versus right action are chosen by agreement with the network among the candidates; every report says how many candidates there were.
- **Factor mode is less robust than group mode.** It fails when a part is weakly represented or over-split.
- **Compute was flaky.** Three GPU sessions were reclaimed mid-run. Networks lost that way were never trained and are listed as such, not counted.

## Closest prior work, and what is different

- **Nanda et al. 2023; Chughtai et al. 2023; Stander et al. 2024.** They reverse-engineered modular addition, group composition (including S4, A5 and others) and permutations. They knew the task and the group and used them to read the network. Here the group is secret, and the tool finds it and then checks itself causally.
- **MIPS (Michaud et al. 2024).** It distils RNNs on algorithmic tasks into Python via an integer autoencoder and symbolic regression, solving 32 of 62 tasks. It is the closest automated work. Differences:
  - it works on RNN hidden states, not transformers;
  - it uses no group or ring structure;
  - it was not tested under secret relabellings or with causal checks.
- **Learning Transformer Programs (Friedman et al. 2023) and Tracr (Lindner et al. 2023).** These build transformers that are decompilable by design, or compile programs into them. Here the transformers are ordinary.
- **Mechanisms for multiplication:**
  - the discrete-log clock (2026);
  - "Multiplication Beyond Groups" (July 2026): local, stratified Fourier regions for products modulo square-free composites;
  - the approximate-CRT account of addition (2025).

  None of these reports the nested-logarithm mechanism for a^b, as far as a web search shows. That search is not a full literature review.
