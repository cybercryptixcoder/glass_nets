# The starting vocabulary (version v1)

**Frozen at the git tag `vocab-frozen` (see `docs/build/STATUS.md` for the commit).** After the freeze, §9 applies.

*Written by the builder, session 1, Nov 27, 2026 (`docs/build/DIARY.md`). Required by `docs/spec/SPEC.md` §6 and
`docs/spec/BUILD.md` §4 step 4: the primitives the box begins with, published and frozen (git tag `vocab-frozen`)
before the bench-writer writes S1 and S2. The code is `bench/accept/vocab/` (importable; `test_vocab.py` holds its
known-case tests; the domain tables below are printed from the code by `describe.py`, so the two cannot drift).*

## 1. How a definition is written (the encoding)

A definition is one S-expression: an operation applied to arguments, nested. `INTERFACE.md` adopts this encoding.

| symbol | meaning |
|---|---|
| `name` | a base feature or an operation of the domain (tables in §8) |
| `3`, `-0.5` | a numeric constant |
| `"x"` | a token constant (token sequences only) |
| `@id` | a reference to another part of the same blueprint, with that part's declared type |
| `$name` | an invented primitive from the box's library (§4) |

Examples:
- chess: `(count (and own_P (not defended)))`, the number of the mover's undefended pawns (5 symbols);
- chess, through a part: `(and @move_to (not defended))`, where `move_to` is the blueprint's own part giving the
  square the network's chosen move goes to;
- numeric sequences: `(argmax (eq @r (bcast (add n -2))))` with `@r = (rank x)`, the position of the second largest
  element (AlgZoo's task), written as two parts;
- token sequences: `(aggregate_cat (select indices @opp EQ) tokens)` with `@opp = (map dec (sequence_map minus
  @length indices))` and `@length = (selector_width (select tokens tokens TRUE))`: Tracr's reverse, as three parts.

Values are computed over a batch of inputs. Booleans are 0 and 1. `not` gives 1 where its argument is 0 or less;
`and` and `or` are minimum and maximum; `count` counts values above 0.

## 2. Length and the constants budget

- **Length** is the number of symbols in the expression. A part reference and an invented primitive each count as
  one symbol. The maximum length **L** is fixed per domain (§8): chess 6, numeric sequences 7, token sequences 12,
  fluid fields 6, network weights 6.
- **Constants:** at most **5** per definition (`SPEC.md` §6; only Shreyas changes this), counting every numeric or
  token literal and every table entry, after every invented primitive the definition uses has been expanded.
- `vocab.admissible(text, domain, part_types, library)` returns whether a definition is in the grammar, well typed,
  of an allowed root type, and within L and the constants budget. The scorer of record applies its own copy of
  these rules; this is the builder's.

## 3. The network's own choice, and other parts

A definition may use another part of the blueprint by `@id`. That is the only way it can refer to the network's own
choice (its chosen move, its planned line): through a part of the blueprint that computes it. A definition never
calls the network. So Rosetta's look-ahead primitives, which ran the network to get its line, are not base features
here; they exist only as parts the blueprint must compute itself (`BUILD.md` §4 step 4).

## 4. Invented primitives (R8)

The box may add two kinds, each recorded with its provenance (which run made it, from which evidence). Both are
checked by `vocab.check_library(domain, library)`, which `vocab.admissible` runs on the library a definition uses.
- **A named expression** (`{"kind": "expression", "definition": text}`): a closed definition in this grammar, used
  later as the one symbol `$name`. It is itself a definition: at most L symbols, at most 5 constants after expanding
  the named expressions it uses, no circular use, and no part references (a named expression is a reusable concept,
  not a piece of one blueprint; a quantity with inputs is a new operation). Its constants count, after expansion,
  against the budget of every definition that uses it.
- **A new operation** (`{"kind": "operation", "source": python, "args": [types], "ret": type}`): one Python function,
  using only arithmetic, control flow and `np.<function>` calls, with no imports, no input or output, no string
  literals and no access to the network, its weights or its activations, of at most **200 AST nodes** (the
  `BENCH_BRIEF.md` §6 default; `INTERFACE.md` may fix another number before S2 is written). It is applied as
  `($name arg ...)`, one symbol plus its arguments, and its numeric literals are its constants: at most 5, and they
  count against the budget of every definition that uses it. `vocab.check_operation(src)` checks one;
  `vocab.compile_operation(src)` compiles it.

What binds is the constants budget. Named expressions and operations let a definition be long, but never let it carry
more than 5 fitted numbers.

## 5. Outside the vocabulary

A concept is outside the vocabulary when no expression of length at most L, over the base features and operations
alone (no part references, no invented primitives), with at most 5 constants whose values may be fitted, reaches the
recovery line (`SPEC.md` §6). The counts in §8 are the sizes of those searches: expression skeletons by length, with
each constant one fitted slot. Argument orders of symmetric operations are counted separately, so they are upper
bounds; pruning expressions that give identical values on a probe set (observational equivalence) cuts them further.
The chess count at L = 6 (1.9 billion skeletons) is the largest; the others are 2.5 to 18 million.

Two notes for the exhaustive check:
- `after` needs the positions one move ahead. The child depends only on the squares where FROM and TO are largest,
  so each input has at most its legal moves as children: featurize every legal child of every probe position once
  (about 35 per position), and evaluate `after` by lookup.
- Broadcast values (`bcast` in `seq` and `fluid`) keep a length-1 axis; numpy broadcasting gives them the full shape
  when they meet a full value.

## 6. Inputs each domain expects

- `chess`: `vocab.featurize(fens)`, which computes the per-square board features in the mover's frame (row 0 is the
  mover's first rank, square index rank * 8 + file, as in the Leela input planes) and five position features. The
  feature code is a frozen copy (`chess_board.py`) of Kepler's descriptors and Rosetta's static extras, plus the
  static exchange value of each square for both sides (`see_us`, `see_them`: "play out the exchange", the standard
  swap algorithm; negative for a losing capture).
  - **Looking ahead:** `(after FROM TO EXPR)` plays the legal move from the square where FROM is largest to the square
    where TO is largest and evaluates EXPR in the new position, from the view of the side then to move (0 where the
    move is not legal). FROM and TO may be base-feature maps (for example `(argmax_sq see_us)`, the best exchange)
    or the blueprint's own parts, which is how a definition reaches the network's chosen move. EXPR uses base
    features only (it may contain `after` again, for a second ply); a part reference inside it is rejected.
- `seq`: `{"x": [N, n] array}`, integers or reals.
- `tok`: `{"tokens": list of token lists}`. RASP semantics as Tracr implements them: `aggregate` is the numerical
  mean of the selected values (0 if none); `aggregate_cat` is the selected value when every selected key holds the
  same one, else undefined (None).
- `fluid`: `{field: [N, H, W] array}` for the fields the adapter supplies among `u`, `v`, `p`, `q`, `u_prev`,
  `v_prev`, `q_prev`. Unit spacing, periodic boundaries.
- `weights`: `{"nets": list of {"W": [matrices, first layer first], "b": [vectors]}}`. A missing layer is a zero 1 x 1
  matrix; a shape mismatch gives NaN.

## 7. Tests (`python3 bench/accept/vocab/test_vocab.py`)

79 checks, all passing at publication (and `fuzz_vocab.py`: 812 random admissible expressions across the five
domains, none raising):
- the grammar: admissible definitions pass; over-length definitions, a sixth constant (including one hidden in an
  invented expression), unknown symbols, type errors, syntax errors and undeclared part references fail;
- invented operations: an arithmetic numpy function passes and runs; imports, file reads, `np.load`, network calls,
  dunder access, printing, two functions, string literals and a function over 200 nodes fail;
- the library: a named expression that refers to a part, one longer than L, and circular named expressions fail; a
  checked operation with arguments passes, and its literals count as constants;
- chess: piece counts, a hanging queen, a checking square, king-zone and shift operations, the 20 legal moves of the
  start position; static exchange values on four known positions (+9, +4, 0, +1) and from the other side's view (+9,
  and -4 for a losing capture); one ply ahead with `after`, a move that is not legal giving 0, and part references
  allowed in the move but rejected inside the new position's expression;
- numeric sequences: AlgZoo's second-largest position, written as two parts; counts and prefix maxima;
- token sequences: length, histogram, fraction of previous `x` tokens and reverse, written as chains of parts, give
  exactly the outputs of Tracr's own compiled transformers (`tracr.compiler.lib`) on every test sequence;
- fluid fields and network weights: divergence-free shear flow, vorticity, advection; the singular value, rank and
  layer count of known weight matrices.

## 8. The domains (printed from the code)

### Domain `chess`

Types: `Sq` (a value on each of the 64 squares, mover's frame); `Pos` (one value per position); `C` (a constant).
Root types (what a definition may be): `Sq`, `Pos`. **L = 6.**

Base features:
- `Sq`: `empty`, `own_P`, `own_N`, `own_B`, `own_R`, `own_Q`, `own_K`, `opp_P`, `opp_N`, `opp_B`, `opp_R`, `opp_Q`, `opp_K`, `rel_rank`, `file`, `edge_dist`, `center_dist`, `att_us`, `att_them`, `min_att_us`, `min_att_them`, `pawn_att_us`, `pawn_att_them`, `dist_own_king`, `dist_opp_king`, `own_king_zone`, `opp_king_zone`, `piece_reach`, `pinned`, `defended`, `attacked`, `passed_pawn`, `isolated_pawn`, `doubled_pawn`, `safe_from_opp_pawns`, `safe_from_own_pawns`, `open_file`, `semiopen_us`, `semiopen_them`, `light_square`, `behind_own_pawn`, `in_front_of_opp_pawn`, `hanging_opp`, `hanging_own`, `opp_att_by_lower`, `own_att_by_lower`, `mob_us`, `mob_them`, `check_from_us`, `check_from_them`, `control`, `outnumbered_own`, `outnumbered_opp`, `see_us`, `see_them`
- `Pos`: `in_check`, `n_legal`, `castle_us`, `castle_them`, `halfmove`

Operations (name: argument types -> result type):
- `not`: (Sq) -> Sq; (Pos) -> Pos. 1 where the value is 0 or less, else 0
- `and`: (Sq Sq) -> Sq; (Pos Pos) -> Pos. minimum
- `or`: (Sq Sq) -> Sq; (Pos Pos) -> Pos. maximum
- `add`: (Sq Sq) -> Sq; (Sq C) -> Sq; (Pos Pos) -> Pos; (Pos C) -> Pos
- `sub`: (Sq Sq) -> Sq; (Pos Pos) -> Pos
- `mul`: (Sq Sq) -> Sq; (Sq C) -> Sq; (Pos Pos) -> Pos; (Pos C) -> Pos
- `gt`: (Sq Sq) -> Sq; (Pos Pos) -> Pos
- `ge`: (Sq C) -> Sq; (Pos C) -> Pos
- `le`: (Sq C) -> Sq; (Pos C) -> Pos
- `eq`: (Sq C) -> Sq; (Pos C) -> Pos
- `north`: (Sq) -> Sq. value at the square (+1 rank, +0 file)
- `south`: (Sq) -> Sq. value at the square (-1 rank, +0 file)
- `east`: (Sq) -> Sq. value at the square (+0 rank, +1 file)
- `west`: (Sq) -> Sq. value at the square (+0 rank, -1 file)
- `ne`: (Sq) -> Sq. value at the square (+1 rank, +1 file)
- `nw`: (Sq) -> Sq. value at the square (+1 rank, -1 file)
- `se`: (Sq) -> Sq. value at the square (-1 rank, +1 file)
- `sw`: (Sq) -> Sq. value at the square (-1 rank, -1 file)
- `king_any`: (Sq) -> Sq. maximum over the 8 king-step neighbours
- `knight_any`: (Sq) -> Sq. maximum over the knight-jump neighbours
- `file_any`: (Sq) -> Sq. maximum over the same file
- `rank_any`: (Sq) -> Sq. maximum over the same rank
- `argmax_sq`: (Sq) -> Sq. 1 on the square with the largest value (first by index)
- `argmin_sq`: (Sq) -> Sq. 1 on the square with the smallest value
- `after`: (Sq Sq Pos) -> Pos. (after FROM TO EXPR): play the legal move from the square where FROM is largest to the square where TO is largest (promotion to a queen), then evaluate EXPR, an expression of base features only, in the new position, from the view of the side then to move; 0 where there is no such legal move. A special form: EXPR is evaluated in the child position, not the current one
- `count`: (Sq) -> Pos. number of squares where the value is above 0
- `sum`: (Sq) -> Pos
- `max`: (Sq) -> Pos
- `min`: (Sq) -> Pos
- `bcast`: (Pos) -> Sq. the position value on every square

Expression skeletons by length (constants as fitted slots; part references and invented primitives not counted):

| length | 1 | 2 | 3 | 4 | 5 | 6 | total |
|---|---|---|---|---|---|---|---|
| count | 60 | 1,055 | 34,820 | 1,180,040 | 46,234,545 | 1,839,809,130 | 1,887,259,650 |

### Domain `seq`

Types: `Seq` (a value at each position of the sequence); `Num` (one value per sequence); `C` (a constant).
Root types (what a definition may be): `Seq`, `Num`. **L = 7.**

Base features:
- `Seq`: `x`, `idx`
- `Num`: `n`

Operations (name: argument types -> result type):
- `add`: (Seq Seq) -> Seq; (Seq C) -> Seq; (Num Num) -> Num; (Num C) -> Num
- `sub`: (Seq Seq) -> Seq; (Num Num) -> Num
- `mul`: (Seq Seq) -> Seq; (Seq C) -> Seq; (Num Num) -> Num; (Num C) -> Num
- `max2`: (Seq Seq) -> Seq; (Seq C) -> Seq; (Num Num) -> Num; (Num C) -> Num
- `min2`: (Seq Seq) -> Seq; (Seq C) -> Seq; (Num Num) -> Num; (Num C) -> Num
- `gt`: (Seq Seq) -> Seq; (Num Num) -> Num
- `eq`: (Seq Seq) -> Seq; (Num Num) -> Num
- `neg`: (Seq) -> Seq; (Num) -> Num
- `abs`: (Seq) -> Seq; (Num) -> Num
- `ge`: (Seq C) -> Seq; (Num C) -> Num
- `le`: (Seq C) -> Seq; (Num C) -> Num
- `relu`: (Seq) -> Seq; (Num) -> Num. max(a, 0)
- `mod`: (Seq C) -> Seq; (Num C) -> Num. remainder (integers)
- `floordiv`: (Seq C) -> Seq; (Num C) -> Num
- `where`: (Seq Seq Seq) -> Seq
- `rank`: (Seq) -> Seq. 0-based ascending rank, ties by position
- `sort`: (Seq) -> Seq. ascending
- `rev`: (Seq) -> Seq
- `cumsum`: (Seq) -> Seq
- `cummax`: (Seq) -> Seq
- `cummin`: (Seq) -> Seq
- `lag`: (Seq C) -> Seq. the value c positions earlier (0 before the start)
- `bcast`: (Num) -> Seq. the value at every position (broadcast)
- `sum`: (Seq) -> Num
- `max`: (Seq) -> Num
- `min`: (Seq) -> Num
- `mean`: (Seq) -> Num
- `argmax`: (Seq) -> Num. first position of the maximum
- `argmin`: (Seq) -> Num
- `count`: (Seq) -> Num. positions where the value is above 0
- `at`: (Seq Num) -> Num; (Seq C) -> Num. the value at a position (clipped to the sequence)

Expression skeletons by length (constants as fitted slots; part references and invented primitives not counted):

| length | 1 | 2 | 3 | 4 | 5 | 6 | 7 | total |
|---|---|---|---|---|---|---|---|---|
| count | 3 | 36 | 437 | 5,713 | 78,191 | 1,113,975 | 16,363,487 | 17,561,842 |

### Domain `tok`

Types: `SOp` (a value per token); `Sel` (a selector: query x key booleans); `Pred` (a comparison); `Fn1` (a one-argument function); `Fn2` (a two-argument function); `C` (a constant).
Root types (what a definition may be): `SOp`. **L = 12.**

Base features:
- `SOp`: `tokens`, `indices`
- `Pred`: `EQ`, `NEQ`, `LT`, `LEQ`, `GT`, `GEQ`, `TRUE`, `FALSE`
- `Fn1`: `id`, `inc`, `dec`, `neg`, `not1`, `abs1`
- `Fn2`: `plus`, `minus`, `times`, `equal`, `less`, `greater`, `larger`, `smaller`, `both`, `either`

Operations (name: argument types -> result type):
- `eqc`: (C) -> Fn1. x == c
- `ltc`: (C) -> Fn1. x < c
- `gtc`: (C) -> Fn1. x > c
- `addc`: (C) -> Fn1. x + c
- `mulc`: (C) -> Fn1. x * c
- `modc`: (C) -> Fn1. x mod c
- `subfrom`: (C) -> Fn1. c - x
- `select`: (SOp SOp Pred) -> Sel. select(keys, queries, predicate)
- `sel_and`: (Sel Sel) -> Sel
- `sel_or`: (Sel Sel) -> Sel
- `sel_not`: (Sel) -> Sel
- `aggregate`: (Sel SOp) -> SOp. numerical: the mean of the selected values (0 if none)
- `aggregate_cat`: (Sel SOp) -> SOp. categorical: the selected value (None if none or mixed)
- `selector_width`: (Sel) -> SOp. number selected per query
- `map`: (Fn1 SOp) -> SOp
- `sequence_map`: (Fn2 SOp SOp) -> SOp

Expression skeletons by length (constants as fitted slots; part references and invented primitives not counted):

| length | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| count | 2 | 0 | 12 | 54 | 104 | 1,048 | 3,706 | 17,192 | 103,876 | 430,622 | 2,394,096 | 12,365,968 | 15,316,680 |

### Domain `fluid`

Types: `Field` (a value at each grid cell); `Num` (one value per frame); `C` (a constant).
Root types (what a definition may be): `Field`, `Num`. **L = 6.**

Base features:
- `Field`: `u`, `v`, `p`, `q`, `u_prev`, `v_prev`, `q_prev`

Operations (name: argument types -> result type):
- `add`: (Field Field) -> Field; (Field C) -> Field; (Num Num) -> Num; (Num C) -> Num
- `sub`: (Field Field) -> Field; (Num Num) -> Num
- `mul`: (Field Field) -> Field; (Field C) -> Field; (Num Num) -> Num; (Num C) -> Num
- `neg`: (Field) -> Field; (Num) -> Num
- `abs`: (Field) -> Field; (Num) -> Num
- `sq`: (Field) -> Field; (Num) -> Num
- `dx`: (Field) -> Field. central difference along x
- `dy`: (Field) -> Field. central difference along y
- `lap`: (Field) -> Field. five-point Laplacian
- `div`: (Field Field) -> Field. divergence of (a, b)
- `curl`: (Field Field) -> Field. vorticity of (a, b)
- `adv`: (Field Field Field) -> Field. (a, b) . grad q
- `smooth`: (Field) -> Field. 3 x 3 mean
- `shx`: (Field) -> Field. value one cell along +x
- `shy`: (Field) -> Field. value one cell along +y
- `relu`: (Field) -> Field
- `gt`: (Field Field) -> Field
- `ge`: (Field C) -> Field
- `bcast`: (Num) -> Field
- `mean`: (Field) -> Num
- `max`: (Field) -> Num
- `min`: (Field) -> Num
- `rms`: (Field) -> Num

Expression skeletons by length (constants as fitted slots; part references and invented primitives not counted):

| length | 1 | 2 | 3 | 4 | 5 | 6 | total |
|---|---|---|---|---|---|---|---|
| count | 7 | 98 | 1,407 | 22,547 | 395,969 | 7,366,968 | 7,786,996 |

### Domain `weights`

Types: `Mat` (a matrix per network); `Vec` (a vector per network); `Num` (one value per network); `C` (a constant).
Root types (what a definition may be): `Num`, `Vec`, `Mat`. **L = 6.**

Base features:
- `Mat`: `W0`, `W1`, `W2`, `W3`, `Wlast`, `Wprod`
- `Vec`: `b0`, `b1`, `b2`, `b3`, `blast`
- `Num`: `nlayers`, `nparams`

Operations (name: argument types -> result type):
- `matmul`: (Mat Mat) -> Mat
- `T`: (Mat) -> Mat
- `add`: (Mat Mat) -> Mat; (Vec Vec) -> Vec; (Num Num) -> Num; (Num C) -> Num
- `sub`: (Mat Mat) -> Mat; (Vec Vec) -> Vec; (Num Num) -> Num
- `mul`: (Mat Mat) -> Mat; (Vec Vec) -> Vec; (Num Num) -> Num; (Num C) -> Num
- `abs`: (Mat) -> Mat; (Vec) -> Vec; (Num) -> Num
- `relu`: (Mat) -> Mat; (Vec) -> Vec
- `scale`: (Mat C) -> Mat; (Vec C) -> Vec
- `sum`: (Mat) -> Num; (Vec) -> Num
- `mean`: (Mat) -> Num; (Vec) -> Num
- `std`: (Mat) -> Num; (Vec) -> Num
- `max`: (Mat) -> Num; (Vec) -> Num
- `min`: (Mat) -> Num; (Vec) -> Num
- `count_gt`: (Mat C) -> Num; (Vec C) -> Num
- `frac_pos`: (Mat) -> Num; (Vec) -> Num
- `mv`: (Mat Vec) -> Vec
- `fro`: (Mat) -> Num
- `spec`: (Mat) -> Num
- `trace`: (Mat) -> Num
- `cond`: (Mat) -> Num
- `eig_max`: (Mat) -> Num
- `rank`: (Mat C) -> Num. singular values above c
- `sv`: (Mat C) -> Num. the c-th singular value (0 = largest)
- `rowsum`: (Mat) -> Vec
- `colsum`: (Mat) -> Vec
- `rownorm`: (Mat) -> Vec
- `diag`: (Mat) -> Vec
- `svals`: (Mat) -> Vec
- `norm`: (Vec) -> Num
- `argmax`: (Vec) -> Num
- `div`: (Num Num) -> Num
- `neg`: (Num) -> Num
- `log`: (Num) -> Num
- `ge`: (Num C) -> Num
- `le`: (Num C) -> Num

Expression skeletons by length (constants as fitted slots; part references and invented primitives not counted):

| length | 1 | 2 | 3 | 4 | 5 | 6 | total |
|---|---|---|---|---|---|---|---|
| count | 13 | 170 | 1,385 | 14,165 | 185,729 | 2,336,611 | 2,538,073 |

## 9. After the freeze

From `BUILD.md` §4 step 4: every later primitive is produced by the box itself (R8), with its provenance recorded; a
bug fix that keeps a primitive's meaning gets a ledger row; a change of meaning voids S2, which is then regenerated;
adding a primitive by hand voids S2.
