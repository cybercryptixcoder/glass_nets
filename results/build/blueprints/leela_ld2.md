# Blueprint: Leela LD2, its evaluation at block 10 (a real network)

**Verdict: partial.** The top level is certified by interventions; its four parts are located and causal, but none
is yet defined in the vocabulary, so all four sit in the unresolved ledger.

## Top level

LD2's value (win probability minus loss probability) is a linear readout of four variables of its block-10
dictionary, each taken as its board-wide total (an invented operation: four weights and an offset, the five-constant
budget):

`value = -0.0343 * total(f434) + +0.0233 * total(f193) + +0.1762 * total(f304) + +0.0195 * total(f493) -0.2700`

- Held-out R2 against LD2's own value: **0.806** (training positions chose the four; scored on the
  held-out split). Control line: the same search over 16 variables drawn from outside the 64 most important,
  best of three draws: 0.445.

## Parts

| part | where | definition from the board | intervention test |
|---|---|---|---|
| `f434` | block 10, dictionary variable 434 (its decoder direction on every square) | unresolved; best try `(gt (max control) (count own_R))` (held-out correlation 0.054, line 0.461) | pass: swapping its board map between positions changes LD2's value as the top level predicts, correlation 0.811 against line 0.191 |
| `f193` | block 10, dictionary variable 193 (its decoder direction on every square) | unresolved; best try `(or (max opp_R) (max opp_B))` (held-out correlation 0.427, line 0.461) | pass: swapping its board map between positions changes LD2's value as the top level predicts, correlation 0.813 against line 0.191 |
| `f304` | block 10, dictionary variable 304 (its decoder direction on every square) | unresolved; best try `(add (min see_them) (count control))` (held-out correlation 0.198, line 0.461) | pass: swapping its board map between positions changes LD2's value as the top level predicts, correlation 0.752 against line 0.191 |
| `f493` | block 10, dictionary variable 493 (its decoder direction on every square) | unresolved; best try `(sum (and passed_pawn (king_any own_N)))` (held-out correlation 0.229, line 0.461) | pass: swapping its board map between positions changes LD2's value as the top level predicts, correlation 0.802 against line 0.191 |

## Unresolved ledger

All four parts: each is a real variable of the network (located, and causal for the evaluation as the top level
says), but no expression of the frozen vocabulary within L defines it from the board. Earlier work (Rosetta,
section 9 of its RESULTS) found these variables track win, loss and drawishness and are only partly restated by
textbook concepts. Defining them needs ingredients from earlier blocks: the next level down of the hierarchy.

## How it was made

`python3 -m box.run leela` (box/leela_box.py). Map: the block-10 sparse dictionary, ranked by attribution to the
value. Define: the vocabulary search, 60 probe positions, 600 held-out. Certify: 236 position pairs; controls swap
a variable outside the top 64 instead (20 repetitions, best of K, 99th percentile).
