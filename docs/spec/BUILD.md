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
- You are done when `SPEC.md` §7 says you are, and not before.

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

**If you believe the spec is wrong**, do not change it, and do not quietly work around it:
1. Write the evidence (run, file, numbers) in `docs/build/DIARY.md`.
2. Propose a new, named version of the spec to Shreyas.
3. Keep working on everything the problem does not touch.

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
- `docs/review/PREREG_FIRST_EXPERIMENTS.md` (X0 to X8), which is optional. X0 informs the copy test, R9. Where the
  experiments conflict with `SPEC.md`, the spec wins.

---

## 4. Before you write any component

1. **Branch.** Cut a new branch from `rosetta`: `rosetta-v2`, unless Shreyas names another. Never edit
   `constructor-attempt`.
2. **Ledger.** Before each gate's code, write one row per gate in `docs/glass/CHANGES.md`, as Rosetta did with its
   rows R1 to R3. Each row gives:
   - what the gate builds;
   - its pass tests, from `SPEC.md`;
   - the cases not used to design it (the certification draw);
   - the verdicts that must not change.
3. **Data.** Check that the data the targets need is present:
   - the three Leela networks (LD2, 744706, t1);
   - the Lichess January 2013 games and puzzles;
   - GPT-2 small and medium weights.

   These live outside the repo, in `$KDATA` (default `~/kdata`, gitignored; `docs/rosetta/STATUS.md` lists how to
   rebuild what is derived from them). This workspace's shell may not be able to download them. If they are
   missing, ask Shreyas to attach or supply them, and meanwhile work on whatever does not need them: AlgZoo, the
   compiled and planted networks, and the bench interface.
4. **Publish the starting vocabulary**, then freeze it.
   - Write `bench/VOCABULARY.md`: for each domain, the base features and the base operations your definition
     language starts with. You choose them. Rosetta's chess primitives and Kepler's 38 physics terms are the obvious
     seeds.
   - It must be published before the bench-writer writes S2, because S2's concepts are written to lie outside it.
   - After that, the starting vocabulary is frozen. Every later primitive must be produced by the box itself (R8),
     with its provenance recorded.
   - Adding a primitive by hand, as the line-swing quantity was added in Rosetta, voids S2.
5. **Ask Shreyas, at the start, for:**
   - **a frontier-model API key** for the observer, reader, director and critic roles. This workspace has only
     DeepSeek V4.1 Flash, which could barely apply board rules (`docs/rosetta/RESULTS.md` §8);
   - **a GPU budget**: decomposition at chess and language-model scale, the fluid rebuild, the planted language model,
     and Tier B;
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
  - build your checker against S0, whose circuits are public, and against the saved copies and disguises of S5,
    which are not sealed.

**What you get from it:**
- development draws, which you use freely while building;
- a scorer that prints scores and pass/fail only;
- `bench/README.md`, which describes the interface.

**What you never do:**
- open `bench/sealed/`, or anything the bench-writer marks sealed;
- read the generator;
- decode answers from the scorer.

If you see sealed content by accident, tell Shreyas: that tier is regenerated with new concept families.

**Certification.**
- At the end, a separate session (the bench-writer, or a fresh one) generates a fresh draw from a seed Shreyas gives.
- That draw includes concept families never used in any development draw.
- It is scored once against your finished box, and the result goes to Shreyas and to you together.
- You never see the certification draw before it is scored.

**Retries.** A gate retried after a failure uses a new development draw, never the same one twice.

---

## 6. The order of work

The six gates in `GAP.md` §4. Each ends with a gate report in `docs/build/GATE_<n>.md`, giving what passed, the
numbers, and what failed, plus a message to Shreyas.

| Gate | What must pass (development draws) | Ask before starting |
|---|---|---|
| 1. The finish line | The bench is written (by the separate session). Your checker certifies the true answers of S0, S1 and S4, and rejects every case in S5 | Nothing, unless training the S6 rehearsals needs a GPU: then ask for that budget |
| 2. The make-or-break | S2 and S3 recovered at the pass counts in `SPEC.md` §6 | The budget for this gate, if it needs paid model calls or GPUs |
| 3. Complete blueprints | S0 to S4 run on their own and are complete within the line; S6's R3 to R7 parts pass | The GPU budget, if the rehearsals need it |
| 4. Real known answers | S7 | The data (§4, step 3) |
| 5. Minds and hands | The reader and editor tests (R10, R11) | The frontier-model key, which the reader test needs |
| 6. No hands | The whole suite, including the full S6, end to end from one start configuration, with no human input | Shreyas's go for certification, and the seed |

**Then certification** (§5). Only after it, and with Shreyas's go, the real flagships:
- the chess networks t1 and LD2;
- the rebuilt fluid emulator;
- the lying mechanism, and weight reading, on 7–8B models (Tier B).

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

- **Diary:** `docs/build/DIARY.md`, with every step, decision and failure.
- **Status:** `docs/build/STATUS.md`, kept current after every step: what is done, what is next.
- **Times:** from the clock only. Run `TZ=America/New_York date` in its own call and copy its output. Never type a
  time by hand.
- **Commits:** under Shreyas's identity (cybercryptixcoder, official.shreyas28@gmail.com), with no AI co-author or
  session lines in the message.
- **Keys:** only in the gitignored `lab/.env`. The repo is public.
- **Messages to Shreyas:** when a result lands. Plain English and plain markdown; no artifacts or widgets.

---

## 9. Done

You are done when all of these hold:
- **Certification** passes `SPEC.md` §7 on the fresh draw.
- **Tier A** (`SPEC.md` §3) has blueprints that meet §2 on S7's real networks.
- **Tier B** has run end to end with completeness reported, if Shreyas supplied the GPU for it. If he did not, it is
  marked "blocked: no GPU" in `docs/build/STATUS.md`, not counted as failed.
- **The final report** in `docs/build/REPORT.md`: what passed, with the numbers, what the box found on the real
  networks, and what remains unresolved and where.
