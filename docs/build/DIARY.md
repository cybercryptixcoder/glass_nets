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
