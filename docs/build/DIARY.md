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
