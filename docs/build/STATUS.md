# Build status (the finished system, `docs/spec/BUILD.md`)

*Kept current after every step. Times from the clock. Branch: `rosetta-v2`. Last updated: session 1, Nov 27, 01:28.*

## Where things stand

- **Bench session:** none yet. Shreyas was asked at 00:41 whether he starts it or the builder launches it with
  `docs/spec/BENCH_BRIEF.md` as its whole prompt, and whether the private repositories `glass_nets_bench_sealed` and
  `glass_nets_bench_dev` exist. No answer yet.
- **Asked of Shreyas (BUILD.md §4 step 5), no answer yet:** a frontier-model API key; a GPU budget; a spending budget
  per gate; the go for formal Gate 1 attempts.
- **Blocked: no key.** R10 (the reader test) and the director's judgment roles.
- **Decision asked of Shreyas (01:12):** copy rejection on small networks. On the saved M4,3 cases the checker's truth
  tests certify a near-copy and reject the approximate real account (`docs/build/DIARY.md` 01:05 to 01:11;
  `results/build/s5a_m43_session1.txt`). Options: gate copy rejection on R9's own tests (X0's estimator, run by the
  bench-writer; the bounded reader, which needs the key), or a new named version of R9 / Gate 1. The spec is unchanged.

## Done

- **Starting vocabulary v1** (`bench/accept/VOCABULARY.md`, code `bench/accept/vocab/`, ledger V0; fuzz test:
  812 random admissible expressions, none raise): five domains in
  one S-expression language; L = 6 (chess), 7 (numeric sequences), 12 (token sequences), 6 (fluid), 6 (weights);
  5 constants after expansion; part references (the only route to the network's own choice); invented primitives
  checked; chess static exchange values and one-ply look-ahead (`after`). 79 of 79 known-case checks pass, including
  four Tracr circuits that equal Tracr's compiled models. **Not yet tagged `vocab-frozen`** (planned about 01:40
  unless Shreyas says hold).
- **Checker v0** (`box/`, ledger G1, in progress):
  - Tracr adapter (`box/adapters/tracr_net.py`): numpy forward pass with interventions, equal to Tracr's own within
    2e-6 on four programs.
  - AlgZoo RNN adapter (`box/adapters/algzoo_rnn.py`).
  - Checks (`box/verify.py`, `box/verify_vec.py`): form (blueprint, wiring, allowed linear locations, vocabulary
    admissibility, declared upstream), run alone, interchange interventions on decisive pairs (a hit is a change to
    the predicted output) against family-wise lines from random-direction and weight-shuffled controls, substitution
    (reported), the lookup typing rule.
  - Known cases: Tracr S0-kind, 45 of 45 as expected (5 true blueprints certified, 40 must-fail variants rejected;
    `results/build/s0_tracr_known_cases_session1.txt`). Verifier hook (`box/hook.py`, counts only): 4 of 5 programs
    as expected; one chance pass of a random-location account on hist (`results/build/s0_hook_session1.txt`). Saved S5a cases: every saved account, as given, is code and
    rejected (the three real ones too, until written as blueprints); M2,2 cannot discriminate (two dimensions, two
    classes: control lines 0.95 to 1.00); M4,3: near-copy certified, real account c928 rejected (see the decision
    above).

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

1. Tag the vocabulary `vocab-frozen` and push the tag (unless Shreyas says hold).
2. Gate 1: once a bench exists, wire the checker to the bench's verifier hook (`INTERFACE.md`); meanwhile, write
   lc_8_6's real account as a blueprint (the third saved must-pass case) with an attention-only transformer adapter.
3. Follow Shreyas's decision on copy rejection; if R9's own tests gate, prepare the estimator interface (answering
   without running) for X0's outcome.
