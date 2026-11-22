# Blind decompilation of trained transformers: results on one page

*Branch `decompiler`, Nov 21–22, 2026. Code: `lab/decomp/`. Rules for every test below were written in `docs/decomp/PLAN.md` before the networks they test were trained. Only held-out, pre-registered results count here; development results are listed separately and are not evidence.*

## What it does

You give it the weight file of a trained transformer and nothing else: no task name, no meaning for the input symbols, no candidate features. It returns:

1. **The structure the network uses.** For example, "a group isomorphic to Z5 × Z20", "a non-abelian group of order 24 with S4's element orders", or "two independent multiplications, in Z4 and in Z27".
2. **A short runnable Python program** that computes what the network computes. Every table in it is read from the weights and internal activations, never fitted to the outputs.
3. **Two checks of that program.**
   - **M1:** it agrees with the network on every input.
   - **M2:** moving only the recovered coordinates inside the real network changes the answer exactly as the program predicts, against a random-subspace control.

The networks are trained with their symbols relabelled by a secret random permutation, so no "number line" or familiar ordering exists for the tool to find. The answer keys are used only afterwards, by `scripts/decomp/score_known.py`.

```
python3 lab/decomp/decompile.py <model_id>     # group mode, abelian then non-abelian: report + program
python3 lab/decomp/factorize.py <model_id>     # factor mode: does the network split into independent parts?
python3 lab/decomp/explain.py <model_id>       # both, summarised in plain words
python3 scripts/decomp/score_known.py <id>     # afterwards only: compare with the hidden answer key
```

## Held-out, pre-registered results

| test (pre-registered) | networks | result |
|---|---|---|
| **Challenge v2: random abelian groups of order 72–120 under secret labellings** (23:19, version frozen 00:32) | 12, one per group in a fixed list | **8 / 12 pass**. In each pass the recovered program equals the secret operation on every pair (law vs truth 1.000), with M1 1.000 and M2 1.000 against 0.000 for the random control. Three of the 4 failures are groups with three or more Z2 factors |
| **Held-out non-abelian: S4 under a secret labelling** (00:06) | 2 that generalized (seeds 7, 9) | **2 / 2 pass**. The recovered 24 × 24 table equals the secret one on all 576 pairs; M1 1.000; M2 1.000 and 0.986 against 0.000. Both networks hold S4 as rotations on a 3-D sphere |
| Challenge v3: 12 more abelian groups, same frozen version (00:45) | 12 | *running* |
| Blind non-abelian challenge: D10, Z5⋊Z4, Z7⋊Z3, SL(2,3), D12, A4×Z3, S3×S3, A5 (00:53) | up to 16 | *training* |
| **Factor mode, a^b mod 109** (Nov 21 22:30) | 1 | **pass**. The network splits into multiplication in Z4 and in Z27, as predicted from 108 = 4 · 27; it agrees with the network on 1.000 of inputs |
| **Factor mode, a·b mod 120 and mod 105** (22:30) | 2 | **CRT split found in both**: Z8 × Z15 and Z7 × Z15. The 3 and 5 parts are kept together, a fact about these networks that the logit test decides. Agreement 0.999 and 1.000 |
| Factor mode, a^b mod 97 and mod 101 (22:30) | 2 | **miss**: 97 partial (0.949), 101 not split (0.866) |
| Factor mode prime challenge: a^b mod p for p = 89, 103, 107, 127, 131, 137, 139, 149 (Nov 22 00:02) | 8 | *running* |
| Challenge v1 (16:55, first version) | 10 | **3 / 10**. It stands as the first result. Its failures drove v3, which is why v1 is not counted again |

## The finding the tool led to: how a transformer computes a^b mod p

Group mode correctly reports that a^b is not a group law. Its tools then exposed a mechanism we have not seen reported: **nested discrete logarithms** (`POW_MECHANISM.md`).

1. The network takes x = log_g(a).
2. It splits the exponent ring Z_{p−1} into its prime-power parts by the Chinese remainder theorem.
3. Inside each part it takes a second logarithm: the Legendre and cubic characters mod 7, and the 2-adic valuation and ⟨−1⟩ × ⟨5⟩ mod 16.

Single-part causal swaps move the answer exactly as predicted: 1.000 on every swap at p = 113 and at p = 109. Factor mode, built afterwards, finds the split blind from the weights.

## Development results (used to build the tool; not evidence)

| network | result |
|---|---|
| a+b mod 113; a+b mod 113 under a secret labelling (s1, s2) | Z_113; secret recovered exactly |
| a·b mod 113; a+b+ab mod 113; a²+b² mod 113 | Z_112 plus an absorbing zero, named k·log_3(t); the same with k·log_3(t+1), found unprompted; Z_113 with k·t², "the input is squared" |
| Z8 × Z14 under a secret labelling | Z_2 × Z_56 (isomorphic) |
| challenge v1 networks, re-run with v3.3 | 9 / 10 (randab9, Z2 × Z6 × Z6, fails) |
| S4 seed 2 | pass |
| memorizer (random labels) | no structure, correctly |

## Limits, plainly

- **One architecture:** a one-layer transformer with four heads, no LayerNorm. The method reads the embedding, the MLP's input and output and the unembedding.
- **Binary operations on a finite symbol set only.**
- **2-groups of rank ≥ 3 fail often.** Several order-2 characters are 1-D axes, and the plane finder can pair two axes into a fake 4-point circle.
- **Non-abelian mode** reads the group from the symmetries of a sphere orbit. The identity symbol and left versus right action are chosen by agreement with the network among the candidates; that is disclosed in every report.
- **Factor mode is less robust than group mode.** It fails when a part is weakly represented (p = 101) or over-split (p = 97).

## Closest prior work, and what is different

- **Nanda et al. 2023; Chughtai et al. 2023; Stander et al. 2024.** They reverse-engineered modular addition, group composition (S4, S5, A5 and others) and permutations. They knew the task and the group and used it to read the network. Here the group is secret and the tool finds it, then checks itself causally.
- **MIPS (Michaud et al. 2024).** It distils RNNs on algorithmic tasks into Python via an integer autoencoder and symbolic regression, solving 32 of 62 tasks. It is the closest automated work. Differences:
  - it works on RNN hidden states, not transformers;
  - it uses no group or ring structure;
  - it was not tested under secret relabellings or with causal checks.
- **"Learning Transformer Programs" (Friedman et al. 2023) and Tracr (Lindner et al. 2023).** These build transformers that are decompilable by design, or compile programs into them. Here the transformers are ordinary.
- **Mechanisms for multiplication.**
  - The discrete-log clock (2026).
  - "Multiplication Beyond Groups" (Chen et al., July 2026): local, stratified Fourier regions for products modulo square-free composites.
  - The approximate-CRT account of addition (2025).

  None of these reports the nested-logarithm mechanism for a^b, as far as a web search shows. That search is not a full literature review.
