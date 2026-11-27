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
