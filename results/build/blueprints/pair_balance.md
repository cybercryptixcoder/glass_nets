# Blueprint: pair_balance

**Verdict: certified.** run alone: agreement 1.000 against line 1.000
Family-wise control line for the intervention tests: 0.000.

## Top level

The network's output is part `v6`: `(sequence_map minus @v5 @v4)`.

## Parts, in the order the network computes them

| part | where | kind | definition | reads | intervention test |
|---|---|---|---|---|---|
| `v0` | resid_0, 5 direction(s) | categorical | `indices` | the input | pass: the network's input (indices), at the embedding |
| `v1` | resid_0, 3 direction(s) | categorical | `tokens` | the input | pass: the network's input (tokens), at the embedding |
| `v2` | resid_2, 1 direction(s) | numerical | `(map (eqc ")") @v1)` | `v1` | pass: IIA 1.000 against family-wise line 0.000 |
| `v3` | resid_2, 1 direction(s) | numerical | `(map (eqc "(") @v1)` | `v1` | pass: IIA 1.000 against family-wise line 0.000 |
| `v4` | resid_3, 1 direction(s) | numerical | `(aggregate (select @v0 @v0 LEQ) @v2)` | `v0`, `v2` | pass: IIA 1.000 against family-wise line 0.000 |
| `v5` | resid_3, 1 direction(s) | numerical | `(aggregate (select @v0 @v0 LEQ) @v3)` | `v0`, `v3` | pass: IIA 1.000 against family-wise line 0.000 |
| `v6` | resid_4, 1 direction(s) | numerical | `(sequence_map minus @v5 @v4)` | `v4`, `v5` | pass: IIA 1.000 against family-wise line 0.000 |

## Unresolved ledger

Nothing is unresolved.

## How to read it

How to read the definitions (token sequences; every value is computed at each position of the input):
- `tokens` is the token at each position; `indices` is the position (0, 1, 2, ...).
- `(select KEYS QUERIES PRED)` makes, for each query position q, the set of key positions k where PRED(KEYS[k], QUERIES[q])
  holds. PRED is EQ, NEQ, LT (key < query), LEQ, GT, GEQ, TRUE or FALSE.
- `(aggregate SEL VALUES)` at q is the mean of VALUES over the selected positions (0 if none is selected).
- `(aggregate_cat SEL VALUES)` at q is the one value all the selected positions hold.
- `(selector_width SEL)` at q is the number of selected positions.
- `(map F X)` applies F at each position: `id`, `inc` (x+1), `dec` (x-1), `neg`, `(eqc c)` (1 if x == c else 0), and so on.
- `(sequence_map F X Y)` applies F to X and Y at each position: `plus`, `minus` (X - Y), `times`, `equal`, ...
- `@vN` is the value of part vN; `$name` is an invented operation, applied at each position (its code is listed).

