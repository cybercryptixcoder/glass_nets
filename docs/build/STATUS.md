# Build status (the finished system, `docs/spec/BUILD.md`)

*Kept current after every step. Times from the clock. Branch: `rosetta-v2`. Last updated: session 2, Nov 27, 15:32.*

## Session 2 result (Nov 27, 05:21 to 07:40 and 15:26 to 15:56)

What the box does now, end to end: it takes a network and returns a blueprint (variables, where each sits, what each
computes in the frozen vocabulary or through earlier parts or an operation it invents, the wiring, the certification
by interventions against control lines, and what stays unresolved). Plain description: `docs/build/BOX.md`.

| Target | What the box returned | Certification |
|---|---|---|
| The bench's S0 networks (Tracr, arbitrary basis, no labels), dev-001 to dev-003, 30 networks | frame recovered on 30; output reproduced exactly on 16 | 16 of 30 certified by the box's own checker on fresh inputs (`results/build/bench_s0_session2.json`); not scored against the sealed answers (the permission system refused pointing the scorer at them) |
| The bench's public reference network | the true blueprint, blind | the bench's own scorer: every requirement at the line; 4 of 4 sealed variables recovered |
| Five public Tracr programs (blind) | all five recovered | 5 of 5 certified; live reader test (DeepSeek V4.1 Flash): 100% with the blueprint, controls 0% to 67% |
| A trained network with a known answer, relu(A - B) * C | (sum control) = A - B, (count own_P) = C, top level relu(part 1) * part 2 | certified by checker v1 (0.424 and 0.102 against 0.078; run alone at the true account's agreement) |
| Leela LD2 | top level: four block-10 variables, R2 0.806; block 8: each top-level variable a readout of four block-8 variables (R2 0.71 to 0.97) | top level certified (0.75 to 0.81 against 0.19); 11 of 16 block-8 links certified; the parts' board definitions unresolved |

Checker changes this session: the vector-site checker v1 (CHANGES.md C2: an intervention holds the other parts at its
site; session 1's verdicts unchanged). Box changes: BX1 to BX6.

Missing against the vision (SPEC.md): scores against the sealed answers (the scorer needs them and this session may not
point it there); S0's remaining 14 networks (six write several variables through one sublayer, which the frame
recovery does not split yet; the rest need definitions the search does not reach); the S1 to S7 tiers (no adapter
runs on them yet); R9's copy tests; the frontier reader; the director and editor; the fluid and weights domains; the
bench-writer's report (it stopped at a usage limit; its draws and scorer are on `bench-v2`).

## Where things stood after session 1

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
  four Tracr circuits that equal Tracr's compiled models. **Frozen as v2 at commit f0b3b680 (02:03)** (v1, frozen at
  87642926 at 01:38, lacked relu in chess; its tag never reached origin and nothing had been written against it). The
  tag `vocab-frozen` exists locally on f0b3b680; pushing it gets HTTP 403 from this session's proxy, so **Shreyas is
  asked to create `vocab-frozen` on origin at f0b3b680e23700bdcadfc314bc99969dc18b9c49.**
- **Definer's search v0** (`box/search.py`, ledger G2, groundwork): bottom-up enumeration with
  observational-equivalence pruning, fitted scale and offset, upstream parts as atoms; 7 of 7 planted definitions
  recovered at held-out correlation 1.0. **Staged rehearsal** (`box/rehearse_staged.py`, builder-made network): the
  three planted ingredients defined exactly, the five distractors not; the verdict defined through the ingredients at
  held-out R2 0.98, where the board alone gives -0.97; the blueprint the search produced is then certified by the
  checker's interventions, and a swapped-location variant rejected (`box/rehearse_staged_check.py`).
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

1. **Freeze:** done (v2, f0b3b680); the tag `vocab-frozen` must be created on origin by Shreyas (tag push refused
   here, 403).
2. **Waiting on Shreyas:** who writes the bench (and the two private repositories); the frontier-model key; GPU and
   per-gate budgets; the go for formal Gate 1 attempts; the copy-rejection decision (R9's own tests, or a new named
   version of R9 / Gate 1); whether the T2 real accounts stay must-pass under R3.
3. **Gate 1, when the bench exists:** rewrite `box/hook.py`'s loaders to `INTERFACE.md`; run the hook on the
   development draw's S0, S1, S4 and S5a.
4. **Gate 2 groundwork (no resources needed):** the definer's search with fitted scale and offset and with `after`;
   the map-maker (variables at every layer, located by interventions rather than by fitting, as M4,3 showed); a
   builder-made staged planted network (not sealed) to rehearse S3's shape; locate lc_8_6's census quantities.
