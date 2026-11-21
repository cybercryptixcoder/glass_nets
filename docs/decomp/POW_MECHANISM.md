# How a one-layer transformer computes a^b mod 113: nested discrete logarithms

*Found Nov 21, 2026 (D1). Network: `lab/models/dc_pow_s1`, the repo's one-layer transformer, trained on half of all (a, b) pairs; it answers 99.99% of all 12,769 pairs correctly. Every number below can be re-run with `python3 scripts/decomp/pow_mechanism.py dc_pow_s1`.*

## The algorithm, in one paragraph

Write g = 3, a generator of the multiplicative group mod 113, and x = log_g(a), so that x lies in Z_112. Then a^b = g^(x·b), so the network has to multiply x by b in the ring Z_112. It never does this multiplication directly.

1. **It splits the ring.** By the Chinese remainder theorem, Z_112 = Z_7 × Z_16. Both x and b reach the MLP as one feature set for their residue mod 7 plus another for their residue mod 16.
2. **It takes a second logarithm in each part.**
   - **Mod 7:** a nonzero residue is stored by its discrete log inside Z_7* ≅ Z_6 = Z_2 × Z_3. Concretely that is its quadratic character (the Legendre symbol) and its cubic character. Zero has its own feature.
   - **Mod 16:** a residue is stored by its 2-adic valuation, plus the two factors of its unit part in Z_16* = ⟨−1⟩ × ⟨5⟩.
3. **It multiplies by adding.** Multiplying residues means adding these second-level logarithms, and characters turn that addition into products that the MLP can form.
4. **It reads out characters of the result.** The unembedding uses additive characters of log_g(c) of order 7, 2, 4, 8 and 16. That is, it reads each CRT part of the output exponent separately.

So the network computes a power by taking a logarithm, splitting the exponent ring into its prime-power parts, and taking a logarithm again inside each part.

## Evidence

| claim | test | result |
|---|---|---|
| The MLP's input is a sum of an a-part and a b-part | additive decomposition of the last position's residual after attention, x1 = F(a) + G(b) | 99.69% of variance. Heads 0 and 2 read a, heads 1 and 3 read b |
| Each part is a function of the CRT residues | least squares of F on one-hot(log a mod 7) + one-hot(log a mod 16); of G on b mod 7 + b mod 16 | 99.7% (F) and 99.6% (G). Mod 7 alone: 47%; mod 16 alone: 52% |
| The mod-7 features are multiplicative, not additive | spectrum of the mod-7 class means | additive characters: flat (about 0.16 each over k = 1..6). Multiplicative characters of Z_7*: orders 2 and 3 carry 95% (χ3 0.34, χ2 0.31, χ4 0.31); order 6 carries about 0 |
| The mod-16 features follow the multiplicative monoid | valuation alone; character spectrum on units | the 2-adic valuation alone explains 58% (F) and 57% (G). On units, 4 of the 8 characters of Z_16* carry the units' variance |
| The readout is CRT-split | character transform of the logits over log_g(c) | top output characters have orders 2, 8, 4, 7 and 16. Each is low-rank (5–6) as a function of (a, b), and its factors are functions of (log a mod m, b mod m), explaining up to 99.6% |
| **Causal: CRT parts** | Swap only the mod-7 (or only the mod-16) features of a, or of b, with those of another input, inside the network. Prediction: log_g(c) = CRT(b·x′ mod 7, b·x mod 16), and so on | **1.000** for each of the 4 swaps (3,000 inputs each). Random subspaces of the same size: 0.27 and 0.19, which is the rate at which the prediction happens to equal the unpatched answer |
| **Causal: second log, mod 7** | Swap only the Legendre-symbol part of b mod 7 (keep the cubic part), or only the cubic part. Prediction: the second-level log of b takes its parity (or its residue mod 3) from b′ | **1.000** (926 cases) and **1.000** (1,250 cases); the unpatched answer survives in 0.000 |
| **Causal: second log, mod 16** | For odd b, swap only the ⟨−1⟩ factor, or only the ⟨5⟩ factor, of b's unit part mod 16 | **1.000** (382 cases) and **1.000** (542 cases) |

## What is new, as far as we can tell

- Nanda et al. (2023): addition uses one clock.
- The discrete-log clock (Chughtai et al. 2023; Nguyen, ICML 2026): multiplication uses a logarithm, then a clock.
- Africa et al. (NeurIPS 2025): modular exponentiation with a variable modulus, in a 4-layer encoder-decoder. They localise the computation to attention heads and do not report a number-theoretic decomposition.
- We found no report of a network computing powers through CRT plus a second discrete logarithm.

This search was web-only and is not exhaustive. It needs a proper literature pass before any paper claim.

## Open, and planned

- **Universality:**
  - more seeds at p = 113;
  - other primes, where the prediction is that the split follows the factorisation of p − 1: p = 97 (96 = 32·3), p = 101 (100 = 4·25), p = 109 (108 = 4·27). Training is in progress (Colab round 2).
- **Zero-divisor handling:** what happens when b or x is divisible by 7 or by 2, and how the valuation features combine.
- **Gauss sums:** how the MLP turns products of multiplicative characters into the additive characters the readout needs.
- **Automation:** the decompiler should find this from the weights without being told about log_g, 7 or 16 (see `PLAN.md`).
