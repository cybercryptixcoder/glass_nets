# Build status (the finished system, `docs/spec/BUILD.md`)

*Kept current after every step. Times from the clock. Branch: `rosetta-v2`.*

## Where things stand

- **Bench session:** none yet. Shreyas asked (session 1, first message) whether he starts it or the builder launches
  it with `docs/spec/BENCH_BRIEF.md` as its whole prompt.
- **Asked of Shreyas, at the start (BUILD.md §4 step 5):** a frontier-model API key (observer, reader, director,
  critic roles); a GPU budget; a spending budget per gate; the original chess files or their sources and checksums.
- **Blocked until the key arrives:** R10 (the reader test) and the director's judgment roles: "blocked: no key".
- **Starting vocabulary:** in progress (`bench/accept/VOCABULARY.md`, `bench/accept/vocab/`). Not frozen.
- **Gate 1 (checker):** not started. Ledger row G1 to be written before its code.

## Data (BUILD.md §4 step 3)

Files in `~/kdata/chess/`, as measured by the Kepler and Rosetta sprints (SHA-256):

| file | sha256 |
|---|---|
| `LD2.pb.gz` | `04ba214ce1277ece9b781e8a1ec1ef2dd4d8f294f496cc8e6fef325f298db60a` |
| `744706.pb.gz` | `430f9ac229eafc1b4c305ba20adf7e4b4c5d7a0a3f032307398ac8e349b722c8` |
| `t1-256x10.pb.gz` | `bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406` |
| `lichess_2013-01.pgn.zst` | `aa40b3671fa3cf1072eb182892cd90b0e1e003a4a5943492f64b77e7f3fd1635` |
| `puzzles_part.csv.zst` | `437c3da45fee902edc2124645f962add7920eeea28dae135cad6ca6acaac4475` |

GPT-2 small and medium: not downloaded yet (needed from Gate 4, S7 month rollover).

## Next

1. Publish the starting vocabulary with its code and expression counts; tag `vocab-frozen` (BUILD.md §4 step 4).
2. Ledger row G1; the checker (`GAP.md` C2) against S0's public circuits and the saved S5a cases (three real accounts
   must pass; the copies, near-copies, the shuffled-net account and the disguises D1 to D6 must fail).
