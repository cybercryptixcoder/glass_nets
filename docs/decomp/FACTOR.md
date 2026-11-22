# Factor mode: finding a network's hidden split into independent sub-computations, blind

*Built Nov 21, 2026 (22:00–23:30), on branch `decompiler`. Code: `lab/decomp/factor.py` and `lab/decomp/factorize.py`. Pre-registered predictions: `PLAN.md`, 22:30 entry.*

## What it does

The tool reads a trained network's weights and activations. It is told nothing about the task, the symbols or any number theory. It answers two questions:

1. Does the network secretly compute its answer as several small, independent computations?
2. If so, what is each one?

```
python3 lab/decomp/factorize.py <model_id>        # -> results/decomp/<id>_factor.json, also shown in the catalogue
```

## How it works

1. **Find repeated differences (the parallelogram signature).** Suppose each input is stored as a sum of independent parts, x(a) = u₁(a₁) + u₂(a₂), where each part takes only a few values. Take two symbols that differ only in part 1. Their difference vector is the same no matter what part 2 is, so that vector recurs once for every value of part 2. Repeated difference vectors therefore expose each part's subspace.
2. **Read off the blocks.** Project the symbols onto one part's subspace and they collapse into a few blocks. The blocks of different parts form a full product grid: every combination occurs, once each. The number of parts, their subspaces and their block counts are all chosen by how well that grid comes out.
3. **Candidates.** Candidate splits come from:
   - the two input sides at the MLP's input;
   - the shared embedding.
4. **The logit test, which decides.** The network's logits are fitted as a sum of one small table per part, each indexed by the pair of blocks. The split counts only if this sum gives the network's own answer on (almost) every input; that is the falsifiable claim. A network that does not split cannot be reproduced this way: the memorizer and the addition networks give "no split".
5. **Name each part.** Each part's table is read from the network's answers given the blocks. Symbols whose answer never depends on the other input, like 0 in a product, are set aside and reported. The table is then compared with multiplication and addition in Z_m by row and column profiles, and its invertible core (a Latin square) is identified as a group from its element orders.

## Results

"Pred." marks predictions written before the run (PLAN.md 22:30). Agreement means the same answer as the network, over every input.

| network | development or held-out | result | agreement | logit share |
|---|---|---|---|---|
| a^b mod 113, s1 | development | **two parts: multiplication in Z_7 and multiplication in Z_16**. Invertible cores: cyclic Z_6, and a non-cyclic group of order 8 (Z_16* = Z_2 × Z_4) | 1.000 (305 table entries for 12,543 inputs) | 0.993 |
| a^b mod 109 | held-out; pred. Z_4 and Z_27 | **multiplication in Z_4 and multiplication in Z_27**, exactly as predicted | 1.000 | 0.996 |
| a·b mod 120, s1 | held-out, never examined before; pred. Z_8, Z_3, Z_5 | **multiplication in Z_8 and multiplication in Z_15**. The network keeps the mod-3 and mod-5 parts together: the 3-part split explains less (see below) | 0.999 | 0.990 |
| a·b mod 105 | held-out, never examined before; pred. Z_3, Z_5, Z_7 | **multiplication in Z_7 and multiplication in Z_15**. The 3-way split [3, 5, 7] is found but explains only 0.918 of the answers, against 1.000 for [7, 15] | 1.000 | 0.995 |
| a^b mod 97 | held-out; pred. Z_32 and Z_3 | partial: the 3-part is found, but the 32-part is over-split (36–38 blocks) | 0.949 | 0.975 |
| a^b mod 101 | held-out; pred. Z_4 and Z_25 | miss: the mod-4 part is weakly represented (two of its residues lie 1.07 apart, against about 4 for the others) | 0.866 | 0.920 |
| a^b mod 113, s2 and s3 | held-out | miss: s2 no split; s3 does not split cleanly. By hand, their inputs are only 0.76 and 0.95 CRT-additive, against 0.997 for s1 | 0.756 (s3) | 0.906 (s3) |
| a·b mod 120, s2 | held-out | miss: right block counts (8 and 15), but boundary symbols are misplaced | 0.830 | 0.954 |
| add mod 113, secret add, memorizer | controls; pred. no split | **no split**, correctly, for all three | — | — |

### Prime challenge (pre-registered Nov 22 00:02, 8 primes never seen, run with the frozen commit 98081655): 2 of 8 pass

| p (p − 1) | result | agreement |
|---|---|---|
| 127 (2·9·7) | **pass**: multiplication in Z_7 and in Z_18 | 1.000 |
| 131 (2·5·13) | **pass**: multiplication in Z_5 and in Z_26 | 1.000 |
| 89 (8·11) | split, parts not named | 1.000 |
| 107 (2·53) | split into 2 × 53 blocks, parts not named | 1.000 |
| 139 (2·3·23) | split, parts not named | 0.999 |
| 103 (2·3·17) | split, parts not named (network only 0.986 accurate) | 0.987 |
| 149 (4·37) | no clean split | 0.948 |
| 137 (8·17) | no split | — |

**Why four splits went unnamed.** The absorbing symbols (a = 0 and 1; b = 0 and p − 1) answer the same on only 97–99% of partners, because the network makes a few errors on them. The exact-constant test missed them, they entered the parts, and they added stray values to the part tables. Factor mode v2 (D10) treats a symbol as absorbing when 95% of its partners give the same answer. Run on these same networks afterwards (development, so the 2/8 stands):
- p89 becomes **Z_8 and Z_11**;
- p107 becomes **Z_2 and Z_53**, the factorisation of p − 1 exactly;
- p139 and p103 are still unnamed.

A fresh prime set (61, 67, 71, 73, 79, 83, 151, 157) is pre-registered for v2 (PLAN, 01:56).

## What this shows, and what it doesn't

- **Shown.** From activations alone, the tool discovers that a trained transformer computing a^b mod p is really two independent multiplications in the prime-power parts of the exponent ring. That is the Chinese remainder theorem plus a second logarithm, and it says which ring each part is. On the held-out prime 109 it says exactly what the number theory predicts.
- **Also shown.** On networks trained on multiplication modulo 120 and 105, which nobody had examined, it finds a CRT split in the base ring. It also finds that these networks merge the mod-3 and mod-5 parts into one Z_15 part rather than splitting them three ways. That is a fact about these networks, decided by the logit test, not an assumption.
- **Not shown.**
  - It is not universal: p = 101 is missed and p = 97 is partial, because a part is weakly represented.
  - It needs the answer to be close to a sum over parts at the logit level.
  - Naming is by row and column profile plus the group of the invertible core: strong evidence of the ring, but not a full isomorphism proof for the non-invertible rows.
