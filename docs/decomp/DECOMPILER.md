# The decompiler: from weights to a verified program, blind (D1)

*Built Nov 21, 2026, on branch `decompiler`. Plan and pre-registered rules: `PLAN.md`. The new mechanism it led to: `POW_MECHANISM.md`.*

## What it does

The decompiler is given a trained network's weight file and nothing else:
- no task name;
- no meaning for any input symbol;
- no list of candidate features.

It returns a short program that computes the same function. Every table in that program is read from the weights, and two checks run on it:

| check | what it measures |
|---|---|
| **M1 replacement** | how often the program gives the network's answer, over every possible input |
| **M2 causal** | Inside the real network, move only one input's recovered coordinates onto another symbol's. Does the network's answer change exactly as the program says it should? Compared against moving a random subspace of the same size. |

The network's outputs are used only by these two checks, never to build the program. A program fitted to the outputs would just be a lookup table.

```
python3 lab/decomp/decompile.py <model_id>          # writes results/decomp/<id>.json (report) and <id>.py (program)
python3 scripts/decomp/score_known.py               # afterwards: compare with the hidden answer keys (never read by the decompiler)
python3 lab/server.py                               # then open http://localhost:8765/catalog?t=<id>
```

## How it works

### Stage 1: what structure do the input symbols have?

1. **Where to look.** It looks in the embedding. When the MLP's input splits into an a-part plus a b-part (x1 = F(a) + G(b), checked and reported), it also looks in F and G separately.
2. **Find circles.** It searches for 2-D projections in which every symbol sits at the same radius (circle planes), and for 1-D directions on which every symbol sits at +r or −r (order-2 characters).
3. **Find relations.** Integer combinations of plane angles that are constant across all symbols are relations among the characters.
4. **Identify the group.** The relation lattice, put into Smith normal form, gives the finite abelian group the planes describe. It can be cyclic (Z_113) or a product (Z_2 × Z_56).
5. **Place each symbol.** Each symbol is decoded to its group element using all planes at once. One plane alone is too noisy: about 2.6° of jitter against 3.2° spacing between lattice points.
6. **Robustness.**
   - Three plane-finding variants are run on each representation.
   - A relation is kept only while the decode stays consistent.
   - The group found most often across runs wins (a false relation gives a different group in one run).
   - Symbols that sit at the centre of most planes are flagged as absorbing elements, such as 0 under multiplication.

### Stage 1b: orbit mode, for non-abelian groups and as a fallback (`orbit.py`)

If the network embeds group element g as ρ(g)u, for an orthogonal representation ρ of any group, then after whitening the symbols lie on a sphere and the group acts on that point set by rotations. Orbit mode works in four steps:
1. Find the smallest whitened rank at which the symbols sit on a sphere.
2. Find every orthogonal map that permutes the points.
3. Find the subgroups that act regularly, one element per symbol, by closure from two generators. Since v3.5 it also builds them greedily from any number of generators, with and without requiring the generators to commute.
4. Read the Cayley table off each such subgroup.

Which symbol is the identity, and left versus right action, are not fixed by the orbit. They are chosen by M1 among the candidates, and the report says among how many. If the table is abelian, it is named by its invariant factors. Orbit mode runs when the abelian stage finds no group, or finds one whose program fails its checks.

### Stage 2: the law and the program

1. **Is it a group law?** The MLP's output is written as characters of G × G in the recovered coordinates. Its "diagonal share" says how far it depends on g(a) + g(b) alone.
2. **Output classes.** Each output class is read from the unembedding along what the MLP writes for each character, then classes are assigned to elements one-to-one. A class that no element claims, together with an off-circle input symbol, gives the absorbing-element rule.
3. **The program:** `c = CLASS_OF[ g(a) + g(b) ]`.
4. **Naming.** Simple closed forms are tried for the symbol maps: k·t, k·t², k·log(t + s). This step is naming only and is never used in verification.

### Stage 3: verification

M1 and M2, with the pass lines from `PLAN.md` (M1 ≥ 0.90; M2 ≥ 0.80 with the random control < 0.20).

## Results

All networks share one architecture: the repo's one-layer, four-head transformer with 226k parameters. Under "found":
- G is the recovered group;
- "name" is the closed form found for the symbol map, where one exists.

M2 is v2 (see the `PLAN.md` change log). The final column is the answer-key check, scored after the decompiler ran.

| network (task hidden from the decompiler) | found | M1 | M2 (random) | vs. hidden answer key |
|---|---|---|---|---|
| a+b mod 113 (s1) | G = Z_113; name k·t | 1.000 | 1.000 (0.000) | 1.000 |
| a+b mod 113 under a **secret random relabelling** (s1) | G = Z_113; no closed form, so a table | 1.000 | 0.996 (0.000) | 1.000; the secret is recovered exactly, up to the group's own symmetries |
| group Z8×Z14 under a **secret labelling** (s1) | G = Z_2 × Z_56 (isomorphic) | 1.000 | 1.000 (0.000) | 1.000; isomorphism 1.000 |
| a·b mod 113 | G = Z_112 plus an absorbing zero; name k·log_3(t) | 1.000 | 0.95 (0.01) | 1.000 |
| a+b+ab mod 113 (s1) | G = Z_112 plus an absorbing symbol; name **k·log_3(t + 1)**, found unprompted | 1.000 | 0.92 (0.01) | 1.000 |
| a²+b² mod 113 (s1) | G = Z_113; name **k·t² ("the input is squared")**, a 2-to-1 symbol map | 0.991 | 0.990 (0.010) | 0.991 |
| memorizer (random labels) | **no structure**, correctly | — | — | — |

*These are development networks. The pre-registered, held-out tests (challenge sets v1 and v2, held-out S4, the blind non-abelian set) are on one page in `RESULTS.md`. The headline there: 8 of 12 fresh secret abelian groups and 2 of 2 held-out S4 networks recovered exactly.*

## The discovery: a^b mod p

The group decompiler correctly reports that a^b mod p is not a group law. Its tools (side representations, character spectra, causal swaps) then led to the mechanism in `POW_MECHANISM.md`: nested discrete logarithms. The network:
1. takes x = log_g(a);
2. splits the exponent ring Z_{p−1} into its prime-power parts by the Chinese remainder theorem;
3. stores each residue by its multiplicative structure (a second logarithm).

Causal swaps of single parts move the answer exactly as predicted:
- on the seed-1 network at p = 113: 1.000 for every swap tested;
- at p = 109: 1.000 everywhere;
- at p = 97 and 101: 0.87–1.00, with some weaker swaps in the 2-power parts (pre-registered line, reported as failed where it failed).

## Limits, plainly

- **Groups with several Z2 factors are the weak spot.** Their order-2 characters are 1-D axes. When two characters multiply to an order-2 character, a whole family of mixed planes is radius-uniform, and the plane finder can land inside it. Orbit mode (v3.5, D9) rescues some of these cases.
- **Non-abelian groups go through orbit mode.** The symbols must lie on a sphere, as an orbit of an orthogonal action. The identity symbol and left versus right action are chosen by agreement with the network among the candidates; every report says how many candidates there were.
- **One architecture.** Everything here is the repo's one-layer transformer. The method uses only the embedding, the MLP's input, the MLP's output and the unembedding, so a deeper network needs these read at the right layer.
- **Naming covers a few forms only.** The closed-form check tries k·t, k·t² and k·log(t + s). It is a convenience; the program does not depend on it.
