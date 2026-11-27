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
