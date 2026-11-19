# Read this first: the working rules for this repository

*Binding for every session and every agent working on this repo. Set by Shreyas on Nov 18, 2026 after rounds 1–2
showed what happens without them (`docs/glass/LEARNINGS.md` E12, `docs/PROJECT_REVIEW.md` §7).*

## Read before doing anything
0. **CURRENT WORK (Nov 19 onward): `docs/glass/ATLAS.md`.** The mechanism atlas build (decompose → explain → certify): why the plan changed, intent, components, status and the ordered next steps. Keep its Status and Next sections current after every step.
   - Background for it: `docs/glass/UNITS_FINDINGS.md` (what the units program established; one-page summary first) and `docs/glass/UNITS_PROGRAM.md` (that program's plan and status; concluded Nov 19).
1. `docs/glass/INTENT.md`: what we are building and why. The code serves this document.
2. `docs/glass/CHECKLIST.md`: current state.
3. `docs/glass/CHANGES.md`: the change ledger.
4. `docs/glass/EVIDENCE_LADDER.md`: what the judge measures.
5. `docs/PROJECT_REVIEW.md`: the history and why we went in circles.

## The build routine: intent first, then implementation, never patches

1. **State the intent** as a question with a yes/no answer, and the decision each answer leads to.
2. **Build the measurement first, and prove it on known cases before any run.** Cases that must pass, cases that must
   fail (including attempts to beat it, written *before* the measure is designed), and controls.
3. **Implement exactly the intent.** Check the implementation against the intent on the known cases. A mismatch here
   is a translation error: fix the code, don't touch the intent.
4. **Before spending, show that the experiment can answer its question:** how many runs, and which result means yes
   and which means no.
5. **Run, then read the outputs by hand** (files, transcripts, notes) before trusting any summary number.
6. **A failure is one of two things.**
   - A translation error: the code didn't match the intent.
   - An intent error: what we asked for was wrong.
   An intent error is recorded in `INTENT.md` with its evidence, the intent is revised, and the code is rebuilt from
   it. Nothing else counts as a "fix".

## How to work (Shreyas, Nov 19)
- **Building is the goal.** Experiments exist only to answer a question a specific build step needs. Name that step in the pre-registration.
- **When a component falls short, try another way and test it,** quickly and many times if needed. Don't stop at a report of why it doesn't work.
- **Work continuously.** Go from one step straight into the next; never idle between steps. A scheduled check-in is only a safety net that restarts work if the session has stopped. It is not a cadence.
- **Update Shreyas when a result lands,** not on a timer.

## Hard rules
- **No code change without a `CHANGES.md` entry first:**
  - the observed failure (run id + file);
  - the cause, shown by a minimal reproduction;
  - the change;
  - a test that fails before and passes after;
  - cases not used to design it;
  - the verdicts that must not change.
  A change that can't fill every field is an experiment: behind a flag, off by default, never used for claims.
- **The measurement is frozen during a round.** A change is a new, named version.
- **No pass line or threshold is chosen by looking at the results it will judge.** Lines come from controls.
- **One kind of change per round:** judge or reader, never both.
- **Free experiments first:** re-score and re-read saved runs before paying for new ones. Runs are small and
  question-driven (minutes, not hours).
- **Before claiming anything new,** compare it against prior work.
- **Ask Shreyas** before spending money, starting a round, or changing `INTENT.md`.

## Standing rules
- **Times in docs and messages come only from the system clock** (`TZ=America/New_York date`). Never type a time by hand: this error has happened repeatedly (E8; Nov 18 19:35; several times on Nov 19). Run `date` in its OWN call, read its output, then write the entry copying that value exactly. Never put `date` and a hand-written time in the same command (the text is written before the clock is read).
- Plain English and plain markdown in docs and messages. No artifacts or widgets.
- Document every step, decision and failure (`docs/glass/DIARY.md`), and keep the checklist current.
- Keys live only in the gitignored `lab/.env`. The repo is public: never commit, push or print them.
- Never edit the branch `constructor-attempt`.
