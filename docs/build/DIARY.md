# Build diary (the finished system, `docs/spec/BUILD.md`)

Every step, decision and failure, with times read from the clock (`TZ=America/New_York date`).

## Session 1 (Nov 27, 2026; two hours, from 00:38)

**00:38.** Session started on Shreyas's instruction: two hours for this build, updates every 10 minutes. Pulled the
repo: `docs/spec/` (SPEC, SPEC_SIMPLE, GAP, BUILD, BENCH_BRIEF, SPEC_CRITIQUE) was committed on `rosetta` on the
evening of Nov 26 (last commit f577f5d7). Read BUILD.md, SPEC_SIMPLE.md, SPEC.md in full, BENCH_BRIEF.md §1 and §6,
GAP.md §3 (C1, C2).

**00:41.** Branch `rosetta-v2` cut from `rosetta` at f577f5d7 and pushed (no bench session had created it). No bench
session exists yet: no `bench/accept/INTERFACE.md`, no draw. Following BUILD.md §5, Shreyas is asked in the first
message which he prefers (he starts the bench session, or the builder launches a fresh agent with BENCH_BRIEF.md as
its whole prompt), together with the resources BUILD.md §4 step 5 lists. Meanwhile: the work that needs no sealed
material (the starting vocabulary; the checker against S0 and the saved S5a cases).

Data check (BUILD.md §4 step 3), present in `~/kdata/chess/`: `LD2.pb.gz`, `744706.pb.gz`, `t1-256x10.pb.gz`,
`lichess_2013-01.pgn.zst`, `puzzles_part.csv.zst`, `net_pb2.py`, `policy_index.py`, `lc0_az_policy_map.py`
(checksums in STATUS.md). These are the files the earlier sprints measured on (Kepler, Rosetta), not re-downloads.
GPT-2 weights: not in the Hugging Face cache; to fetch when Gate 4 needs them. Tracr, JAX and Haiku were not
installed; installing.

**00:42 to 00:51. The starting vocabulary v1 (`bench/accept/VOCABULARY.md`, `bench/accept/vocab/`; ledger row V0).**
Design: one S-expression language for all five domains, so that INTERFACE.md can adopt one encoding; definitions may
refer to other parts (`@id`), which is also the only route to the network's own choice; invented primitives are
`$name` (named expressions, or numpy-only functions of at most 200 AST nodes). Chess base features are a frozen copy
of the 53 per-square facts the earlier sprints used (Kepler's 42 descriptors and Rosetta's 11 static extras) plus five
position facts; Rosetta's look-ahead primitives are left out as base features because they ran the network. L per
domain was chosen from the expression counts so that the bench-writer's exhaustive check stays feasible: chess 6
(1.30 billion skeletons), numeric sequences 7 (11.1 million), token sequences 12 (15.3 million), fluid 6 (7.8 million),
weights 6 (2.5 million). Tests (`test_vocab.py`): 66 of 66 pass after one fix to a test case (the weights rank check
expected the rank of a random matrix above a threshold; replaced by a matrix with known singular values). The Tracr
cross-check passes: length, histogram, fraction of previous x and reverse, written as chains of parts, equal Tracr's
compiled transformers on every test sequence.

**00:48. Error: the 00:48 update to Shreyas was headed "00:50", a time typed by hand before reading the clock (E8
again).** Corrected in the next message. From here every message time is copied from a clock read made just before.

Process note: the ledger row V0 was written after the vocabulary code, before its first commit; CLAUDE.md asks for the
row first. Gate 1's row (G1) is written before its code.

**00:52 to 00:58. The checker v0 on public Tracr programs (`box/`; ledger row G1, written before its code).**
`box/adapters/tracr_net.py` re-implements a compiled Tracr transformer's forward pass in numpy with intervention
hooks at every residual site; it matches Tracr's own `model.apply` to 2e-6 on every residual of four programs
(frac_prevs, hist, reverse, sort_unique). `box/blueprint.py` is the data model; `box/verify.py` the checker: form
(blueprint, wiring, allowed linear locations, vocabulary admissibility, declared upstream), run alone against a line
the caller supplies, interchange interventions on decisive pairs against a family-wise random-subspace control line,
substitution (reported), and the typing rule for lookups. Known cases (`box/test_checker.py`, written before the first
run): 4 true blueprints must be certified; 8 must-fail variants of each must be rejected. First run: 34 of 36. The two
misses: reverse with a wrong or random location for its `length` part was certified. Cause (a translation error): the
pairs were drawn among inputs of equal length, so swapping `length` never changed a prediction, and a part with no
decisive pairs passed silently. Fix: pairs of any lengths, with the blueprint's override mirroring the network's
aligned swap; a rule part that no intervention can test is not certified. Also corrected one test case (reverse's wrong
definition referred to a part it had not declared, so form rejected it before run alone was exercised). Second run:
36 of 36 as expected.

**01:00 to 01:05. The saved S5a cases, first contact (`box/adapters/algzoo_rnn.py`, `box/verify_vec.py`,
`box/test_algzoo.py`).** As given, every saved account is a Python program, so the checker rejects all of them at the
form check, the three real accounts included: none can pass until it is written as a blueprint. M2,2's real account
(run a7c8) was translated by hand into a blueprint of the starting vocabulary (eight parts, locations fitted by least
squares on half the inputs, interventions on pairs from the other half), and the integer copy of the network was
written as a blueprint of its own units (13 parts, each within L = 7 and two constants). Findings:
- Using the vocabulary showed a gap before the freeze: numeric sequences had no `relu` and no clamp with a constant,
  so a ReLU took five symbols. Added `relu` and `max2`/`min2` with a constant to `seq` (vocabulary still v1, not yet
  frozen); `seq` at L = 7 is now 17.6 million skeletons; the 66 vocabulary checks still pass.
- Translation error found and fixed in both checkers: a hit in an interchange intervention counted any output equal to
  the prediction, so a network that ignored the swap could score. A hit is now a change of the output, to the predicted
  value. The Tracr known cases still give 36 of 36.
- On M2,2 the intervention test cannot discriminate: the hidden state has two dimensions and the decision two classes,
  so the best of K random directions, and the same directions in weight-shuffled copies of the network, already score
  0.95 to 1.00 (family-wise lines 0.998 and 0.987 for the real account). Neither the real account nor the copy clears
  them; the copy also fails run alone (0.986 against the real account's 0.9995). This agrees with X0, which reports
  M2,2 as unable to decide anything. The deciding saved cases are M4,3 (hidden 4, three classes) and lc_8_6 (residual 8).

**01:05 to 01:11. The deciding saved case, M4,3 (`box/test_algzoo_m43.py`; output `results/build/s5a_m43_session1.txt`).**
Round 1's real account c928 and a near-copy of the network (its own units with its exact weights) were both written in
the starting vocabulary. Three translation errors in the test harness were found and fixed on the way, none in the
checker's rules:
- Locations fitted by least squares were wrong where units move together on the data (M4,3's first-step units are
  exact multiples of each other): the fit spread a part over several units, and even the network's own unit then
  failed the interventions (IIA 0.05). Units the account names are now declared, and parts linear in located parts
  get the location their definition implies.
- Splitting a unit's weighted sum into partial sums (to fit L) makes parts that are not variables of the network: a
  swap along one moves every unit that reads the same inputs, so they fail (IIA 0.15 to 0.38). This is the checker
  working. Each unit is now one part through an invented operation (five constants: within the budget).
- Class scores located as directions at the last hidden state are not independent either; they are now located at
  the output site, one logit each.
Result (lines from controls: random directions 0.90 to 0.92, weight-shuffled networks 0.31 to 0.39; run-alone line from
the real account's own held-out agreement, 0.9885 minus its spread 0.0024):
- **the near-copy is certified:** run alone 0.9998, all 15 located parts above the line;
- **the real account is rejected:** run alone 0.9902 passes, 22 of its 27 located parts pass, but its channel
  magnitudes m1 and m2 (IIA 0.71 to 0.88) and one class score (0.84) fall below the line.
So on the saved cases the truth tests do what SPEC R9 says they will: a copy is true, so it passes them, while a real
but approximate account can fail them. Counting constants does not rescue this either: the copy uses the network's 32
parameters, the account (as translated) about 50. Gate 1's requirement that the checker certify the real accounts and
reject every copy cannot be met by truth tests on networks this small. It needs R9's own tests (answering without
running, which X0 decides; the bounded reader, which needs the frontier key) or a change to the spec. Reported to
Shreyas as a decision.

**01:12 to 01:16. Vocabulary additions before the freeze (still v1, untagged).** Testing what later tiers need showed
three gaps, each closed with tests (79 checks, all pass):
- invented primitives were not validated: `vocab.check_library` now requires a named expression to be a closed
  definition within L and 5 constants after expansion, with no circular use and no part references, and an invented
  operation (with declared argument and result types) to pass `check_operation`, its literals counting as constants;
- chess had no "play out the exchange": `see_us` and `see_them`, static exchange values per square (standard swap
  algorithm), checked on four known positions (+9, +4, 0, +1) and a losing capture (-4);
- chess could not look ahead at all, so S3's chess rehearsal (an evaluation two plies ahead along forcing moves) and
  S7's look-ahead answers would have been inexpressible: `(after FROM TO EXPR)` plays the legal move from the
  largest square of FROM to the largest of TO and evaluates EXPR (base features only, nestable once more) in the new
  position, from the view of the side then to move; FROM and TO may be parts, which is the route to the network's own
  chosen move. Also `argmax_sq` and `argmin_sq`. Chess at L = 6 is now 1.9 billion skeletons; the exhaustive check
  stays feasible by featurizing each probe position's legal children once (noted in VOCABULARY.md §5).
Earlier step: one test expectation was wrong (see_them from Black's side is +9, not 0; the code was right).

**01:17 to 01:20. Before the freeze: a fuzz test and a fifth Tracr program.** `bench/accept/vocab/fuzz_vocab.py`
evaluates random well-typed expressions of every length up to L in every domain: 812 admissible expressions, none
raised. The checker's known cases now include Tracr's pair_balance (numerical maps compiled into MLPs): its true
blueprint (five parts) is certified and its eight must-fail variants rejected; 45 of 45 known cases as expected
(`results/build/s0_tracr_known_cases_session1.txt`). In the token domain the constants budget cannot bind within
L = 12 (each token constant needs a map and a function symbol), so the "six constants" variants are rejected for
length; the constants rule itself is exercised in the vocabulary tests.

**01:18 to 01:24. The verifier hook (`box/hook.py`, `box/test_hook.py`; output `results/build/s0_hook_session1.txt`).**
The Gate 1 entry point: given a task folder, a folder of true blueprints and a folder of accounts, it prints only
"certified X of N true blueprints, rejected Y of M accounts". Its task-folder and blueprint formats are provisional,
to be rewritten to `INTERFACE.md` when the bench exists. Run on the five public Tracr programs as task folders:
- First run, 1 of 5 as expected. Cause (a translation error): Tracr names s-ops with a process-wide counter
  ("hist_4"), so labels written by one process do not exist in another. Fixed: `TracrNet` uses canonical labels
  (the counter suffix dropped). The direct known cases stay 45 of 45.
- The control line had only 10 to 20 draws behind a 99th percentile, too few to estimate it (translation error).
  The Tracr checker now uses 300 draws. The AlgZoo checker's M2,2 and M4,3 results above used 10 to 20 draws, so
  their lines are, if anything, underestimates: the near-copy's pass and the real account's failures stand.
- Second run, 4 of 5 as expected: on hist, one random-location account passes (IIA 0.878 against a line of 0.282).
  Its random 6-dimensional subspace, in an 18-dimensional residual stream, overlaps the true subspace strongly
  (singular values 0.97, 0.82, 0.66, 0.60, 0.20, 0.08). This is the chance pass a 99th-percentile line allows: the
  account is itself a draw from the control distribution. The direct run, with another random draw, rejected it.
  Must-fail cases built as random locations are not deterministic in small residual streams. Reported, not tuned.

**01:26 to 01:28. A loophole closed in the vector-site checker (a translation error found by reading the code, not
by a case).** A part at site "input" was exempt from location, and any function of the input could be declared there,
so a whole task rule computed from the input would have passed on run alone alone, with nothing located: against
R3, which locates every variable. Now a part at site "input" must be a direct read of the input (an atom, or
`(at x c)`) with no upstream parts, and a "readout" part (the task's decision rule) may use only located parts or
other readouts. The M2,2 and M4,3 verdicts are unchanged (both test blueprints already complied once their decision
parts were marked as readouts): M4,3 near-copy certified; real account c928 rejected (22 of 27 parts); M2,2 both
rejected.

**01:28 to 01:31. Exploratory, for Gate 1 planning: can lc_8_6's census account be completed as a blueprint?
(`box/explore_lc86.py`; `results/build/explore_lc86_session1.txt`).** The account locates only the token identities.
Its census quantities, read by least squares from the last position's residual (fit on half the 46,656 inputs,
scored on the other half), against the same on a weight-shuffled network: 2-cycles R2 0.68 after layer 1 (shuffled
0.01); fixed points 0.39 (0.12); the largest preimage count at a fixed point 0.35 (0.06); all near the shuffled
values at the embedding. So the 2-cycle count is largely present by layer 1, and the other two only partly (perhaps
spread over positions, or not linear). The account cannot yet be written as a blueprint whose census parts pass R3;
locating them is map-maker work (Gate 2's component). No verdict is drawn from this.

**01:28 to 01:31. Gate 2 groundwork: the definer's first search (`box/search.py`, `box/test_search.py`; output
`results/build/search_planted_session1.txt`).** Bottom-up enumeration of the vocabulary by length, dropping any
expression whose values on the probe inputs repeat an earlier one's, constants from a small grid, the last length
streamed rather than stored (the first version stored everything and was killed for memory at 600 probe positions;
now 60 chess positions or 200 sequences as probes). Planted targets, each the value of a known expression, on real
positions (LD2's Lichess data) and random sequences: 6 of 6 recovered at held-out correlation 1.0 on 300 other
inputs, three as equivalent or affine-equivalent expressions (for example `(or defended (not opp_N))`, which is one
minus the planted `(and opp_N (not defended))`). Chess at length 5: 77,248 distinct square-valued expressions below
length 5, 14 seconds. This is the definer's search, not yet the definer: it fits no scale and offset, skips `after`,
and has not met a network.

**01:33 to 01:35.** The definer's search now fits scale and offset (held-out affine R2 1.0 on all six planted
targets). The verifier hook, run on the M4,3 saved cases as a task folder (c928 offered as the true blueprint, the
near-copy as an account), prints "certified 0 of 1 true blueprints, rejected 0 of 1 accounts": the same verdicts as
the direct test, through the Gate 1 entry point.

**01:35 to 01:36.** The search can now use the blueprint's upstream parts as one-symbol atoms (`@id`): definitions
through ingredients. Known case: a target of length 7 in board terms (above chess's L = 6), the number of best
exchanges that take a queen or rook, is found as `(sum @ing)` from an ingredient part, held-out correlation 1.0.
7 of 7 search cases pass.

**01:38. The starting vocabulary is frozen at commit 87642926** (no hold from Shreyas since the 00:41 notice; the
vocabulary files are unchanged since dcd9a1bb; 79 checks and the fuzz test pass on it). The annotated tag
`vocab-frozen` was created on that commit here, but pushing it was refused: HTTP 403 from this session's git proxy
(branch pushes work; tag pushes are denied). Per the proxy's rules this is reported, not worked around. Shreyas is
asked to create the tag `vocab-frozen` on origin at 876429269fc490f8bd81062c2df59a7d6e8b852b (or allow tag pushes).
From now on the vocabulary is treated as frozen: no change of meaning; any bug fix gets a ledger row.

**01:39 to 02:03. Gate 2 rehearsal in miniature: a verdict computed in stages (`box/rehearse_staged.py`; output
`results/build/rehearse_staged_session1.txt` and `.json`).** A builder-made network (not a bench case) on 3,000 real
positions: layer 1 holds three planted ingredients at known units (the mover's hanging pieces, the opponent's hanging
pieces, passed pawns; set by construction) and five random distractor units; later layers are trained to the verdict
relu(g2 - g1) * g3 (held-out R2 0.981). The box's side, blind to which units are planted:
- the search defines units 0, 1 and 2 exactly (`(count hanging_own)`, `(count hanging_opp)`, `(count passed_pawn)`,
  held-out correlation 1.0) and gives the five distractors nothing (0.01 to 0.39);
- from the board alone, within L = 6, it cannot define the verdict (held-out R2 -0.97);
- through the three units as parts it finds `(relu (mul @u2 (sub @u1 @u0)))`, equal to the planted verdict since g3
  is never negative, at held-out R2 0.98, the network's own fidelity to it.
Failures on the way, each fixed: the first training (a learned first layer on standardized features, 2,000 inputs for
3,525 inputs dimensions) generalized badly (held-out R2 below zero), so the ingredients were set by construction;
ties on the probe inputs were broken alphabetically, so a longer equivalent beat `(count hanging_opp)` and that unit
was missed: ties now go to the shorter definition; 60 probe positions were too few for a verdict that is zero on most
positions: the verdict search now runs over position-level expressions only, on 1,000 probes.
**And it exposed a gap in the frozen vocabulary:** chess had no `relu`, so relu(a - b) * c could not be written within L
even through its ingredients. Vocabulary v2 (commit f0b3b680) adds `relu` to every numeric type that lacked it (and
clamps against a constant in chess, and the token function `relu1`). v1 had been frozen at 01:38 but its tag never
reached origin, and no bench session had written anything against it, so nothing is voided. 80 vocabulary checks and
the fuzz test (826 expressions) pass; the search's seven cases still pass. The tag `vocab-frozen` now sits locally on
f0b3b680; pushing it is still refused (HTTP 403).

**02:05 to 02:15. The loop closed on the rehearsal (`box/rehearse_staged_check.py`; output
`results/build/rehearse_staged_check_session1.txt`).** The blueprint the search produced (the three ingredients at
layer-1 units, the verdict through them at the output) was put to the checker's interventions on the rehearsal
network, with a must-fail variant written before the run (u0 and u1's locations swapped). First run: the true
blueprint was rejected, because the random-direction line was 1.0: the verdict part sits at the one-dimensional output
site, where a "random" direction is the part's own (a translation error: such a draw is not a control). Fixed: random-
direction controls are drawn only at sites of more than one dimension. Second run: the true blueprint is certified
(run alone 0.998; ingredient IIA 0.80, 0.94, 0.95 against lines of 0.54 from random directions and 0.08 from
weight-shuffled networks); the swapped variant is rejected (IIA 0.13 and 0.03). M4,3's verdicts are unchanged after
the fix (near-copy certified; c928 rejected, 22 of 27 parts). So, on a toy: found by the search, certified by the
checker, end to end.

## Session 2 (Nov 27, 2026; four more hours, from 05:21)

**05:21.** Shreyas (05:21): no frontier model (DeepSeek V4.1 Flash only, through Luminal); Colab units as needed; the
tag cannot be created ("work with what we've got"); all other decisions approved or left to the builder's
recommendation; four more hours, and by the end "something that works like the vision". Decisions taken on that
approval: the bench-writer is launched by the builder as a separate agent (BENCH_BRIEF.md plus a section of facts:
no private repositories, sealed material in `/home/claude/bench_private/` and `bench/accept/sealed/`, its own branch
`bench-v2`, deadline 09:00), and the builder never opens its sealed files; R9 stays as written (copies caught by R9's
own tests); the reader test runs with DeepSeek Flash, noted as weaker than the spec's reader; commit f0b3b680 is the
frozen vocabulary.

**05:23 to 05:35. The box on compiled transformers, blind (`box/tracr_box.py`, `box/tok_fast.py`; output
`results/build/tracr_blind_session2.txt`).** Given only the compiled network (weights, token interface, outputs when
run), the box maps the residual directions each sublayer writes into variables (one-hot groups are categorical, other
directions numerical), reads from the weights which directions each sublayer reads and whether it is attention or an
MLP, defines each variable in order from the vocabulary and earlier parts, invents operations when nothing matches,
assembles the blueprint and certifies it with the checker. Translation errors found and fixed on the way, in order:
input parts (the tokens and positions at the embedding) cannot be tested by swaps, since positions are the same in
every input: they are now reported as the network's inputs; invented operations were not compiled in the Tracr checker;
the output part must reproduce the decoded output itself, not just match it one to one; candidates that average token
values were proposed (token-valued atoms are now only compared, selected on and carried); a variable written by an MLP
was defined with an attention operation (operations are now restricted by the sublayer that wrote the variable);
internal categorical labelings are fixed only by how later parts compare them, so each categorical variable now offers
its other exact labelings (ranked: non-negative, starting at 0, short) as atoms, and a later part that uses one commits
it, recursively; candidates undefined at some positions scored on the rest (full coverage is now required).
Result: **5 of 5 programs recovered and certified**, every variable defined exactly. Reverse comes out as Tracr's own
program (length, length minus position, minus one, read the token there). For selector width, Tracr stores
1/(count + 1) in an attention output; the vocabulary has no such operation, and the operation search invented
`1/(x+1)` and its inverse `1/x - 1` (provenance recorded in each blueprint).

**05:38 to 06:00. Readable blueprints, the live reader test, and the box on Leela.**
- `box/report.py` writes each blueprint as a reader sees it (top level, parts with location, definition, wiring and
  intervention result, invented primitives with provenance, unresolved ledger, how to read the notation):
  `results/build/blueprints/*.md`.
- **Reader test, live (DeepSeek V4.1 Flash, low effort; `box/reader.py`, `box/run_tracr_reader.py`,
  `box/run_reader_swap.py`; outputs `results/build/reader_tracr_session2.json`, `reader_swap_session2.json` and the
  answers in `results/build/blueprints/*_reader_*.txt`).** Eight fresh inputs per program. With the box's blueprint:
  100% of outputs correct on all five programs. Controls fixed before the run: no blueprint 0% to 67%; definitions
  rotated between parts 0% to 48% on three programs but 100% on reverse and sort (the task can be guessed from the
  inputs, so that control is weak there). A stronger control, added after reading those answers: another program's
  certified blueprint, 0% (reverse), 19% (sort), 0% (hist) against the true network. The reader follows the blueprint.
- **Translation error fixed in the Tracr checker:** a swap took the source input's padded positions beyond its length;
  it now swaps only the source's real positions. Known cases still 45 of 45.
- **The box on Leela LD2 (`box/leela_box.py`; `results/build/leela_box_session2.json`,
  `results/build/blueprints/leela_ld2.md`).** Top level: LD2's value is a linear readout (an invented operation, four
  weights and an offset) of four block-10 dictionary variables (434, 193, 304, 493), held-out R2 0.806; control line,
  the same search over 16 variables drawn from outside the 64 most important, best of three draws, 0.445. Certified by
  interventions: swapping each variable's board map between positions moves LD2's value as the top level predicts,
  correlation 0.752 to 0.813 against a line of 0.191 (controls swap an outside variable instead). The four parts cannot
  be defined from the board within L (held-out correlation 0.05 to 0.43 against a control line of 0.46): all four are
  unresolved. Next level down started: block 8 (`box/leela_levels.py`).
- `box/run.py`: one entry point (`python3 -m box.run tracr <program> | staged | leela`).

**06:06 to 06:50. The box on a trained network with a known answer (Gate 2 rehearsal; the development analog of
S3).**
- First attempt (`box/learned_staged.py`): an MLP on all 3,525 input features of 6,000 positions, trained only on the
  staged verdict relu(g2 - g1) * g3 (hanging pieces, passed pawns). It did not learn the verdict (held-out R2 -0.49;
  on ten planes, -0.01). Cause: the verdict is zero on 95% of positions (309 of 6,000 are not). A test case the network
  cannot learn says nothing about the box, so it was replaced, not tuned.
- The known case (`box/learned_net.py`): a 640-32-32-1 ReLU MLP on ten per-square planes of 20,000 positions (seven
  are distractors), trained only on v = relu(A - B) * C, with A = (sum att_us), B = (sum att_them) (all attacks by the
  mover and by the opponent) and C = (count own_P); v is non-zero on 40%. Held-out R2 0.990, error 1.85 points against
  a spread of 18.6; the rounded output equals v on 68.6% of held-out positions.
- The measurement first, on the known answer (checker v0, `results/build/learned_truth_session2.json`): the true
  account (A, B and C at layer 1 by their probe directions, the verdict through them) is rejected: interventions on A
  score 0.011 and on B 0.000 against a line of 0.077; C passes (0.097). The must-fail accounts (locations permuted; A
  and B exchanged) are rejected. The probe directions of A and B have cosine 0.96: the network holds their difference
  (probe R2 0.9975) much better than either (0.94, 0.95).
- **A translation error in the vector-site checker:** an intervention on one part moved the readout of another part
  at the same site. Ledger row C2 was written first (06:46) and `box/test_joint.py` before the change. Checker v1
  writes the smallest change that sets the part's readout to the source's and holds every other part at the site. On a
  toy with a non-orthogonal code (z = (A, A + B), output A * B), v0 moves B's readout on 89% of pairs when it swaps A
  and scores the exact mechanism at 0.10 and 0.11; v1 moves it on none and scores 1.000 and 1.000; locations exchanged,
  a wrong definition and a random direction are rejected under both. The test as first written also expected v0 to
  reject the toy's true account; v0 certifies it at 0.10 against a line of 0.08, so the defect shows in the scores, not
  in v0's verdict, on that toy (recorded in C2). Session 1's vector-site cases are being rerun under v1.
- A translation error in the harness: the checker was given training positions too, so the run-alone agreement it
  reported (0.855) mixed training and held-out positions. It now sees held-out positions only (0.686).
- Leela block 8: the first run died after training the block-8 dictionary; the untrained-network control on all
  20,000 positions, run on the CPU, did not finish inside the time limit. Restarted with caching and a 4,000-position
  control (2,800 training, 1,200 held-out).

**06:50 to 07:05. The box, blind, on the trained network.** `box/learned_box.py`. A first version ranked candidate
variables by the output's sensitivity to each and chose mixtures such as (add (sum control) (count (west own_P))); the
verdict through them failed (validation R2 -197). The map now takes the directions the output depends on (the top
eigenvectors of the output gradient's covariance at the site) and chooses the shortest expressions that read them. Result
at layer 1: (sum control), which equals A - B on all 20,000 positions, and (count own_P), which is C; the verdict
(mul @v1 (relu @v0)), validation R2 0.989 against the network's own output. Checker v1 certifies it on held-out
positions the box never used: interventions 0.424 and 0.102 against a line of 0.078, run alone 0.686, the true
account's own agreement. The network holds A - B as one variable, not A and B, and the box reports that variable.

**07:05 to 07:40. The box on the bench's S0 networks (`box/bench_s0.py`).** The bench-writer pushed development draws
to branch `bench-v2`: S0 Tracr networks with the residual stream in an arbitrary orthonormal basis and no labels. The
box now recovers a frame from the weights and the given inputs, runs the same box in it, and writes the blueprint in
the bench's format with every location turned back into the network's basis. First attempt wrong: I assumed the shared
constant direction lies in the span of the token embeddings; it does not (frame error 0.25 on the reference). Fixed:
the smallest vector with unit inner product with every embedding, plus the one direction nothing else uses, its sign
chosen by how sparse the input weights become. On the bench's public reference network the blind blueprint passes every
requirement of the bench's own scorer at the line and recovers 4 of 4 sealed variables. On the 30 development
networks: frame recovered on all 30 (error below 1e-14); output reproduced exactly on 13. Two format problems caught by
the bench's checks: provenance must name the run and the evidence; the bench's vocabulary passes invented operations
whole sequences (and does not allow `is`), so they are written position by position.

**07:40 to 15:26.** Interrupted: this session paused, and the bench-writer agent stopped at a usage limit (its draws and
scorer are on `bench-v2`; its report never arrived).

**15:26 onward (Shreyas: thirty more minutes, then stop).**
- Tried to score the development draws against their sealed answers by pointing the bench's scorer at the sealed
  folder, the flow the bench's README describes (the scorer reads them, the builder never opens them). The permission
  system refused it as unauthorized access. Not worked around. Without the sealed answers the scorer checks format
  only, so no sealed-answer score exists for the development networks.
- Instead, the box's own step 5: its checker on 150 fresh inputs: all 13 exact blueprints certified (run alone 1.000;
  interventions 0.913 to 1.000 against family-wise lines 0.000 to 0.651).
- Three one-layer networks were open. Tabulated by token and position, each outputs (token + index) mod m. Ledger
  row BX6 written first; the definer now also tries a modulus after an arithmetic sequence_map. All three are then
  defined exactly, (map (modc m) (sequence_map plus @v0 @v1)), and certified.
- Full rerun with the change and the box's own certification: 16 of 30 development networks certified (the 13 from
  before, all still certified, and the three (token + index) mod m networks); the reference certified; the five public
  Tracr programs rerun, all five still certified. On three networks whose output the box does not reproduce, the
  certification step stopped with an error (an empty sampling range); they were not certified before either.
- Not done in this session: scoring against the sealed answers (refused, above); the 14 remaining S0 networks (six
  write several variables through one sublayer, which the frame recovery does not split yet); S1 to S7.
