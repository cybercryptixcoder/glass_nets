# M4,3: our certified construction vs ARC's notes (round 2, step 1)

Sources, read in full through the Google Drive connector on 2026-11-17:
- George Robinson, *Symmetric RNNs* (Dec 2024).
- Jacob Hilton, *Heuristic explanations for 2nd argmax models* (Dec 21, 2024). This note covers M4,3 directly.
- Zihao Chen, *RNNs for the 2nd argmax* (Apr 2024). This note covers M16,10, not M4,3.

Our construction: `lab/constructions/zoo_2nd_argmax_4_3/20261118-015037-c928/construction.py`. It was a blinded round-1 run, rated COMPLETE under glass-2. The reader had no access to these notes.

## Verdict: an independent rediscovery, not a new discovery

The core of our account is the structure Robinson describes.

| Our construction says | ARC's notes say |
|---|---|
| The 4 units are two signed channels, each split into a positive part and a negative part. The cell is mirror-symmetric under (1↔4, 2↔3). | Robinson: W_hh ≈ Uᵀ W′_hh U and W_hi ≈ Uᵀ W′_hi, where U = [[1,0,0,−1],[0,1,−1,0]]. That is the same split and the same pairing. Hilton confirms it holds for M4,3. |
| The recurrence acts on the signed values s = (u1−u4, u2−u3). | Robinson: U·ReLU(Uᵀv) = v, so the recurrence passes the 2-d signed state through exactly. |
| The readout uses only the magnitudes m = (u1+u4, u2+u3). | Robinson: W_oh ≈ W′_oh V, and V·ReLU(Uᵀv) = \|v\|. The readout sees only absolute values. |
| There is a small rank-one "magnitude coupling". | Hilton: this is the error term ε_hh, the deviation from exact symmetry. |
| Net effect: logits are linear in (m1, m2). | Robinson: the network is equivalent to a 3→2→3 network with absolute-value activation. |

So the reduction "M4,3 = a 2-neuron absolute-value network" is ARC's. We cannot claim it as a discovery, with or without asterisks.

## What we found that the notes don't state

Each of these was checked by code in this repo on the 8000 judge inputs.

1. **The reduced recurrence is a 120° rotation.**
   - The 2×2 signed recurrence is [[1.6167, −2.4174], [2.1552, −2.6103]].
   - Its eigenvalues have angle ±119.96° and modulus 0.995. So it is a rotation by one third of a turn, with almost no shrinking.
   - In plain terms: the three inputs are written into the plane along three directions 120° apart (x0 is rotated twice, x1 once, x2 not at all) and added up. The readout then takes absolute values of the two coordinates and checks where their ratio falls.
   - Neither note gives W′_hh's values or this geometric reading. It may be known informally at ARC, so we record it only as "not in the notes we read".

2. **The exact symmetric reduction loses nothing.**
   - We set the magnitude coupling to zero, which gives exactly Robinson's reduced network.
   - Task accuracy went from 97.86% to 97.96%. The pruned version agrees with our full construction on 98.5% of inputs.
   - So Hilton's ε_hh term is not needed for accuracy.

3. **The readout as two ratio thresholds.** With three logits that are linear in (m1, m2) ≥ 0, the decision depends only on where m1/m2 falls (thresholds T_k ≈ 0.73, 1.04, 1.17). This follows from Robinson's form, but he doesn't spell it out for 2nd argmax.

4. **Causal verification.** ARC's notes check their explanations by estimating accuracy: Hilton reports about 3% and 5% RMS error on 256 noisy copies of the model. Ours passed interchange, substitution and necessity tests at every stage. Our construction also reproduces 79.7% of the network's own mistakes. These are different standards of evidence, and they complement each other.

## Where ARC's notes go further

- Hilton splits input space into regions and orderings (14 likely and 4 unlikely subregions). That explains why the thresholds compute the 2nd argmax, ordering by ordering. Our DESCRIPTION stops at "the decision is a ratio threshold" and never links the thresholds to the orderings.
- Both notes turn the explanation into an accuracy estimate with a measured error. We don't.
- Robinson covers the argmedian and argoutlier models too. We only ran 2nd argmax.

## What this means for round 2

- The M4,3 result shows the system can **recover a known, peer-checked mechanism blind**. That is a validation result, not a discovery. Describe it that way everywhere: README, REPORT and the catalog label.
- The gap ("why these thresholds compute the answer") is a known weakness: the reader stops at a correct but opaque final step. This links to E7 (the reader writes the task algorithm first) and to the claim criterion's "human read" requirement.
- Discovery claims stay with the longest-cycle transformers, as planned. Before any claim, run the same check: search for prior notes, then compare against them.
