# Build brief: start here

*For the agent who builds the finished system. Written Nov 26, 2026, from 19:30 ET (clock). Shreyas's instruction to
you is "Build". This file says what that means:*
- *what is fixed, and what is yours to decide;*
- *what you must not do;*
- *what to ask for, and when;*
- *how you will know you are done.*

*It does not tell you how to build anything.*

---

## 1. What you are building

A box that takes a trained network, a behaviour and inputs, and returns a verified **blueprint** of how the network
produces that behaviour. The blueprint is complete, honestly typed, runnable on its own, faithful at every level,
navigable, and usable by a mind.
- The plain-words picture is `SPEC_SIMPLE.md`.
- The authority is `SPEC.md`: 15 requirements (R1 to R15), each with a test, and a sealed acceptance suite (S0 to
  S7).
- You are done when §9 below holds, and not before. §9 is `SPEC.md` §7 plus the final report.

---

## 2. What is fixed, and what is yours

**Fixed. You may not change these during the build:**
- `SPEC.md`: the requirements, their tests, the lines, the pass counts, and the defaults in §6.
- The sealed-bench protocol (§5 below).
- The starting vocabulary, once you have published it (§4, step 4).
- The rules in `CLAUDE.md`.

**Yours: everything else.**
- The architecture and the code layout.
- The methods, the order of work inside a gate, and which existing code to reuse, rewrite or discard.
- Which models play which roles.
- How the components of `GAP.md` §3 are actually divided. They are a default decomposition, not a design you must
  follow. Any structure that passes every requirement's test is acceptable.

**For this build, `SPEC.md` is the intent the code serves.** `docs/glass/INTENT.md` stays as the record of the
earlier target until Shreyas approves replacing it, as Rosetta's own sprint intent did (`docs/rosetta/INTENT.md`).

**The frozen starting vocabulary is a real constraint on your design.** It is what makes S2 a test of invention, not a
test of what you coded by hand.

**If you believe the spec is wrong**, do not change it, and do not quietly work around it:
1. Write the evidence (run, file, numbers) in `docs/build/DIARY.md`.
2. Propose a new, named version of the spec to Shreyas.
3. With his approval, record the intent error in `docs/glass/INTENT.md`, as `CLAUDE.md` requires, and issue the new
   version.
4. Keep working on everything the problem does not touch.

This is the project's intent-error routine (`CLAUDE.md`, "A failure is one of two things").

---

## 3. Read in this order

1. `SPEC_SIMPLE.md` (short).
2. `SPEC.md`, in full. It is the authority whenever documents disagree.
3. `GAP.md`: where the project stands, requirement by requirement; the default components; the six gates.
4. `SPEC_CRITIQUE.md` and `GAP.md` §8: why several tests look the way they do. Read these before you think a test is
   needlessly strict.
5. `docs/rosetta/FINDINGS.md` and `RESULTS.md`: the last build, where it broke, and its code (`rosetta/`), which is
   the natural starting point for chess.
6. Background, only when you need the evidence behind a requirement:
   - `docs/review/HANDOFF.md`;
   - `docs/review/NEXT_BUILD.md` and `REVIEW_LOG.md`;
   - `main:docs/literature/SYNTHESIS.md`.

**Superseded:**
- the designs in `docs/review/NEXT_BUILD.md` §3;
- `docs/review/PREREG_FIRST_EXPERIMENTS.md` (X0 to X8), which is optional for you. The bench-writer runs X0's main
  arm, and its outcome decides whether R9's estimator test gates (`SPEC.md` R9). Where the experiments conflict with
  `SPEC.md`, the spec wins.

---

## 4. Before you write any component

1. **Branch.** Cut a new branch from `rosetta`: `rosetta-v2`, unless Shreyas names another. If the bench-writer has
   already created it on origin, check it out instead. Never edit `constructor-attempt`.
2. **Ledger.** Before each gate's code, write one row in `docs/glass/CHANGES.md`, labelled G1 to G6 (not R, which
   names requirements). This follows Rosetta's practice of one row per build step. Fill the six fields `CLAUDE.md`
   requires, in the form that applies to new components:
   - **observed failure:** the requirements not yet met;
   - **cause:** the component that does not exist yet;
   - **change:** what the gate builds;
   - **test:** the gate's pass tests, from `SPEC.md`;
   - **cases not used to design it:** the certification draw;
   - **verdicts that must not change:** the earlier gates' passes.

   A later change to a measurement inside a gate gets its own row, as `CLAUDE.md` requires.
3. **Data.** Check that the data the targets need is present:
   - the three Leela networks (LD2, 744706, t1);
   - the Lichess January 2013 games and puzzles;
   - GPT-2 small and medium weights.

   **Where the files go.** The chess files go in `~/kdata/chess/`. `kepler/` reads `$KDATA` (default `~/kdata`), but
   Rosetta's code hardcodes `~/kdata`, so keep the default. By file name, the code expects:
   - `LD2.pb.gz`, `744706.pb.gz` and `t1-256x10.pb.gz`: the three Leela networks;
   - `lichess_2013-01.pgn.zst`: Lichess standard rated games of January 2013;
   - `puzzles_part.csv.zst`: the start of the Lichess puzzle database. A partial download is expected:
     `kepler/targets/chess/data.py` stops reading at the cut;
   - `net_pb2.py` (compiled from lczero-common's `net.proto`), `policy_index.py` and `lc0_az_policy_map.py` (from
     lczero-training): support files, imported from that folder and never copied into the repo.

   **Where they come from.** The repo does not record exact sources. `docs/kepler/DIARY.md` says only that the
   networks came from lczero storage, and the setup script that `kepler/targets/chess/leela.py` names
   (`scripts/kepler/chess_setup.sh`) was never committed. So:
   1. First ask Shreyas for the original files, or their URLs and checksums. S7's known answers were measured on
      them.
   2. If he cannot supply them, fetch them by name: the networks from lczero storage (`storage.lczero.org`; t1 is
      the distilled t1-256x10, as `kepler/targets/chess/leela_t.py` says), and the games and puzzles from
      `database.lichess.org`.
      - Check each network against its recorded behaviour before use: the opening and mate-in-one checks in
        `docs/kepler/DIARY.md` (line 32), and t1 solving 49 of 50 mate-in-one puzzles without search (line 230).
      - Record every file's checksum in `docs/build/STATUS.md`.
      - Never substitute a different network silently.

   **What is derived from them** is rebuilt in two steps:
   1. `NET=LD2 python3 scripts/kepler/chess_build.py`, and the same for 744706. This writes positions, outputs,
      activations and concepts under `~/kdata/chess/<net>/`.
   2. The steps in `docs/rosetta/STATUS.md` ("Data outside the repo").

   **Other setup:**
   - There is no dependency file. Install what you need:
     - `torch`, `python-chess`, `transformers`, `scikit-learn`, `zstandard` and `protobuf`;
     - for compiled transformers, Tracr, which needs `jax` and `dm-haiku`.
   - Keys, if any, are in the gitignored `lab/.env`. A fresh workspace may have none.
   - If a download is blocked, ask Shreyas to supply the files. Meanwhile, work on whatever does not need them:
     AlgZoo, the modular-arithmetic networks, the saved copies (`lab/models`, `lab/constructions/`), and the
     interface.
   - `bench/` already holds unrelated tools (`find/`, `arc/`, `weights/`). The acceptance suite lives in
     `bench/accept/`.
4. **Publish the starting vocabulary**, then freeze it.
   - Write `bench/accept/VOCABULARY.md`, with its code in `bench/accept/vocab/`. You choose its contents. Rosetta's
     chess primitives and Kepler's 38 physics terms are the obvious seeds.
   - **It must cover every domain the suite uses:**
     - chess boards;
     - numeric sequences, both integer and real-valued (AlgZoo M4,3 reads real numbers);
     - token sequences for language models, including Tracr's operations (select, aggregate, map and
       sequence-map), so that S0's true circuits can be written in it;
     - fluid fields;
     - network weights, the input of S6's weight-reading rehearsal.
   - **For each domain it gives:**
     - the base features and the base operations, as importable code in `bench/accept/vocab/`;
     - the grammar for combining them, and how a definition is encoded. `INTERFACE.md` adopts this encoding;
     - the maximum definition length L. Choose L so that the bench-writer's exhaustive check (below) is feasible,
       and state how many expressions of length at most L each domain has.
   - **The constants budget is not yours to set:** 5 numeric constants per definition, counting every literal and
     table entry (`SPEC.md` §6; only Shreyas changes it).
   - **Invented primitives** (R8) are bounded as `BENCH_BRIEF.md` §6 says. Each is either:
     - a named expression in the grammar, used in later definitions as one symbol; or
     - a new operation: Python code that does not call the network or read its weights or activations, within the
       size limit `INTERFACE.md` fixes.

     Length counts each invented primitive as one symbol. Constants are counted after expansion, so invention adds
     structure, never fitted numbers.
   - **The network's own choice.** A definition may refer to the network's chosen move, or its planned line, only
     through a part of the blueprint that computes it. It never calls the network. So Rosetta's line primitives,
     which run the network to get its line, are seeds only in that form.
   - **Outside the vocabulary.** A concept counts as outside it when no expression of length at most L, within the
     constants budget, reaches the recovery line. The bench-writer checks this for every S2 concept.
   - **The freeze.** Push the vocabulary to the build branch and tag the commit `vocab-frozen`.
     - It comes before the bench-writer writes S1 and S2.
     - Do not wait for `INTERFACE.md` to write it.
   - **After the freeze:**
     - every later primitive must be produced by the box itself (R8), with its provenance recorded;
     - a bug fix that keeps a primitive's meaning gets a ledger row;
     - a change of meaning voids S2, which is then regenerated;
     - adding a primitive by hand, as the line-swing quantity was added in Rosetta, voids S2.
5. **Ask Shreyas, at the start, for:**
   - **a frontier-model API key** for the observer, reader, director and critic roles. This workspace has only
     DeepSeek V4.1 Flash, which could barely apply board rules (`docs/rosetta/RESULTS.md` §8);
   - **a GPU budget** for your own work: decomposition at chess and language-model scale, the fluid rebuild, and
     Tier B. The bench's training (S6 and the planted networks) is the bench-writer's to ask for;
   - **a spending budget per gate.**

   Do not wait idle for the answers. Gate 1, and much of Gate 2 (search, library learning, operation search), need
   none of them.

---

## 5. The sealed bench: what you may and may not see

The bench is the finish line, so you must not be able to fit to it.

**Who writes it.** A separate session, given `BENCH_BRIEF.md` as its whole brief.
- **Preferred:** Shreyas starts that session himself.
- **Acceptable,** if he asks you to: launch a fresh agent with `BENCH_BRIEF.md` as its entire prompt, and accept only
  its final report, which describes the interface and nothing sealed.
- **If no bench session exists when you begin,** ask Shreyas in your first message which he prefers. Until he
  answers, do the work that needs no sealed material:
  - publish the starting vocabulary;
  - build your checker against S0, whose circuits are public, and against the saved copies and disguises of S5a,
    which are not sealed. They are the round 1 and 2 constructions in
    `lab/constructions/<target>/<run>/construction.py`, and the disguises D1 to D6 generated from them. Both are
    listed, with run ids, in `scripts/glass/info_measure.py` (`CASES`), and their results are in
    `docs/glass/info_measure_T2.txt`. The same list holds three real accounts that must pass.

**What you get from it:**
- development draws, which you use while building (see "formal attempts" below);
- a scorer that prints scores and pass/fail only;
- `bench/accept/README.md`, which says how to call the scorer;
- `bench/accept/INTERFACE.md`: the executable contract your box's output must meet. It includes a reference true
  blueprint for a public S0 model, which the scorer passes.

**How the two sessions connect.** `BENCH_BRIEF.md` §1 has the full protocol.
- **The branch.** Both work on the build branch.
- **The order.**
  - You push the vocabulary and tag it `vocab-frozen`.
  - The bench-writer pushes a draft `INTERFACE.md` at any time.
  - It completes `INTERFACE.md`, with its reference true blueprint, after the tag.
- **The sealed development answers** live in a private repository, `glass_nets_bench_dev` by default.
  - When `docs/build/STATUS.md` shows "Draw <id> ready", clone or pull that repository into the gitignored
    `bench/accept/sealed/`, without opening any file in it.
  - The generator and the certification material are in a second private repository. You never get access to it.
- **Asking for a draw.** Add the line "Draw request: <gate>, <tiers>" to `docs/build/STATUS.md`, push, and tell
  Shreyas in the same step. A new bench session may be needed to serve it.
- **Large weights** come as release assets of the public repo, with checksums in `bench/accept/README.md`.

**What you never do:**
- open `bench/accept/sealed/`, or anything the bench-writer marks sealed;
- read the generator;
- decode answers from the scorer;
- edit anything under `bench/accept/`, except `VOCABULARY.md` and `bench/accept/vocab/` before the freeze.
  - The scorer there is a development copy of the scorer of record, which the bench-writer keeps privately.
  - Report a scorer bug in `docs/build/STATUS.md`. The bench-writer fixes it as a new, named version.

If you see sealed content by accident, tell Shreyas: that tier is regenerated with new concept families.

**Certification.**
- At the end, a separate session (the bench-writer, or a fresh one) generates a fresh draw from a seed Shreyas gives.
- That draw includes concept families never used in any development draw.
- It is scored once against your finished box, and the result goes to Shreyas and to you together.
- You never see the certification draw before it is scored.

**Formal attempts and retries.**
- Exploratory scoring on development draws while you build is allowed.
- A formal gate attempt is a scoring run used for a gate decision. Each formal attempt uses a draw that no earlier
  formal attempt used: ask for a new one.
- A draw for a formal attempt holds only the tiers its gate scores.
  - S6 is costly to train, so it is drawn afresh only for Gate 3, Gate 6 and certification.
  - Within one gate's retries, S6's networks may be reused, with only pass or fail printed for them.
- The lines' spreads and control lines are computed from the first three development draws, and are then frozen
  (`SPEC.md` §6).

---

## 6. The order of work

The six gates in `GAP.md` §4. Each ends with a gate report in `docs/build/GATE_<n>.md`, giving what passed, the
numbers, and what failed, plus a message to Shreyas.

**Every gate needs Shreyas's go before its first formal attempt** (`CLAUDE.md`: ask before starting a round, and
before spending money).
- Ask early, in one message with whatever else the gate needs (the last column).
- Keep building while you wait. Only formal attempts and spending wait for the go.

| Gate | What must pass (development draws) | Ask for, with the go |
|---|---|---|
| 1. The finish line | The bench is written by the separate session. Through the bench's verifier hook, your checker certifies the sealed true blueprints of S0, S1 and S4, and rejects every S5a account. You see only the counts | Nothing else: no spending is needed. The bench-writer asks for the bench's own GPU time |
| 2. The make-or-break | S2 meets its pass counts (`SPEC.md` §6). On every S3 network, every sealed intermediate variable is recovered, and every sealed edge between them appears in the matched parts' upstream lists, directly or through a chain of your parts. S0's circuits are located and S1's rules defined. A sealed variable is matched by one of your parts, or by one composite part your blueprint defines; the scorer does not search for combinations. Extra parts and edges, and the overview, are judged at Gate 3 | The budget for this gate, if it needs paid model calls or GPUs |
| 3. Complete blueprints | S0 to S4 run on their own and are complete within the line. On S1 to S4, R2, R5 and R12 pass, and S5b's networks get no story. S6's rehearsals pass their R3 to R7 parts, and the fluid rehearsal passes R15 | The GPU budget, if your work on the rehearsals needs it |
| 4. Real known answers | S7: each known answer reproduced by one part, or one composite part, at its line (`BENCH_BRIEF.md` §3) | The data (§4, step 3) |
| 5. Minds and hands | The reader and editor tests (R10, R11) | The frontier-model key, which the reader test needs |
| 6. No hands | The whole suite, including the full S6, on a development draw, end to end from one start configuration, with no human input | Shreyas's go for certification, and the seed |

**Then certification** (§5). Only after it, and with Shreyas's go, the real flagships:
- the chess networks t1 and LD2;
- the rebuilt fluid emulator;
- the lying mechanism, and weight reading, on 7–8B models (Tier B, if GPU time is supplied; otherwise "blocked",
  as `SPEC.md` R13 says).

**Missing resources.**
- Without the frontier-model key, R10 (the reader test) and the director's judgment roles are marked "blocked: no
  key" in `docs/build/STATUS.md`. The rest continues. The build is not done until the block is resolved.
- Without the GPU, the same rule applies to the parts that need it.

**After a Gate 2 failure:**
- keep trying other ways within Gate 2;
- work that does not depend on Gate 2 may continue: adapters, the reader rendering, the editor's plumbing;
- do not score Gates 3 to 6 until Gate 2 passes, or until Shreyas decides otherwise.

**Gate 2 decides whether the approach works.** Spend the effort there before polishing anything else. When a part
falls short, try another way and test it, quickly and many times. Do not stop at a report of why it does not work
(Shreyas's working rule, `CLAUDE.md`). A gate that cannot pass after real attempts is reported with the named
component and the evidence. It is not lowered.

---

## 7. When something fails

Follow the routine in `CLAUDE.md`:
- **A translation error** (the code does not do what the spec says): fix the code.
- **An intent error** (what the spec asks is wrong): record it with its evidence and propose a new version of the
  spec (§2 above).

**Never:**
- lower a line;
- change a pass count;
- reuse a development draw for a retry;
- look at sealed material;
- choose a threshold by looking at the results it judges.

---

## 8. Keeping the record

- **Diary:** `docs/build/DIARY.md`, with every step, decision and failure. At each gate, add a one-line pointer in
  `docs/glass/DIARY.md`, the project diary `CLAUDE.md` names.
- **Status:** `docs/build/STATUS.md`, kept current after every step: what is done, what is next.
- **Times:** from the clock only. Run `TZ=America/New_York date` in its own call and copy its output. Never type a
  time by hand.
- **Commits:** under Shreyas's identity (cybercryptixcoder, official.shreyas28@gmail.com), with no AI co-author or
  session lines in the message. Push to origin after every step, so nothing is lost if a session ends.
- **Certification** runs from a commit you tag `cert-candidate`. It must hold everything the box needs: the code, the
  start configuration, and the concept library grown during development.
- **Keys:** only in the gitignored `lab/.env`. The repo is public.
- **Messages to Shreyas:** when a result lands. Plain English and plain markdown; no artifacts or widgets.

---

## 9. Done

You are done when all of these hold:
- **Certification** passes on the fresh draw: every requirement on S0 to S6, at the pass counts of `SPEC.md` §6, end
  to end with no human input (`SPEC.md` §7).
- **S7** reproduces its known answers on the real Tier A networks, with completeness reported.
- **"Fully interpretable"** is claimed for a real network only once its matched-architecture rehearsal (S6) passes,
  as `SPEC.md` R4 requires. Until then, its completeness is reported with an interval.
- **The final report** is in `docs/build/REPORT.md`: what passed, with the numbers, and what remains unresolved and
  where.

**After that,** with Shreyas's go, the flagships run (§6).
- Tier B runs if GPU time is supplied. Otherwise it is marked "blocked: no GPU" in `docs/build/STATUS.md`, not
  failed.
- The flagships produce findings, not pass or fail. They are added to the report.
