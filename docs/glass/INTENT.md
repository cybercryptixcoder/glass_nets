# Glass Networks: the intent (what we are building, stated so it can be checked)

*Version 2, Nov 18, 2026. Version 1 is what rounds 1 and 2 ran under. Every change below names the mistake in
version 1 and the evidence for it. Code is rebuilt from this document. If the code and this document disagree, the
code is wrong. If this document turns out to be wrong, it is changed here first, with evidence, and the code
follows.*

## The goal (unchanged)

A public library of **constructions**: readable programs that stand in for a whole trained network. Each one ships
with a **certificate** that states, with numbers, how true it is and how much of an explanation it is, so that
anyone can run it, test it and argue with it. A **reader** (an AI) writes them. A fixed **judge** certifies them.

## The changes from version 1

| # | Version 1 (what we asked for) | The mistake, and the evidence | Version 2 |
|---|---|---|---|
| 1 | A construction must reproduce ≥ 95% of decisions AND verify every stage AND be shorter than the network | These can't hold together for networks with a messy remainder. Readers swung between exact copies and short accounts the judge rejected. Evidence: `DIAGNOSIS_ABOVE_32.md` §2, M8,8 v2a's measurement, and ARC's own account of M16,10 | **An account with a stated tolerance.** Exactness is a number on the certificate, not a gate |
| 2 (REOPENED 11:35: tested and failed as a gate; see DIARY 11:35 and `info_measure_step1.txt`) | "Not a copy" is judged by counting the numbers written in the file | Beaten by four ways of writing the same numbers. It passed a shuffled-network control. Evidence: `REPORT_ROUND2.md`, corrections | **Information versus fidelity against the copy frontier** (`EVIDENCE_LADDER.md`, rung X). Adopted only after it separates every saved copy from every known account (see "Rebuild order", step 1) |
| 3 | One verdict: complete / not verified | A binary verdict hid real progress: the census rule at 92.4%, and E5 in round 1 | **A certificate profile**: behaviour with a confidence interval, worst slice, mistakes, inside correspondence per claimed variable, information versus the copy frontier, controls. The catalog shows the profile. Any named levels get thresholds from controls, never from our results |
| 4 | Every stage must have a verified variable, including the embedding | Real accounts whose variables aren't linear at the embedding failed there for 10 straight submissions (lc_8_6, run b) | **Every variable the account places inside the network is tested.** Which stages are covered is reported as coverage, not required |
| 5 | Judge feedback: "FIX FIRST: <stage>", then "now compress" | It made the reader optimise the measure (100 steps shaving constants on lc_6_5 run b) while its understanding stayed in notes the judge never reads | **Feedback says where the account is wrong**: which inputs it misses, which claimed variables fail. It never hands the reader a count to drive down for its own sake |
| 6 | Reader v2 (stage-by-stage prompt) as the default | Added from one anecdote per change; never shown to help (`CHANGES.md`) | **Reader v1 is the default; v2 is off.** Changed back by the change rule, not a fix |

## Deliberately NOT changed (no evidence yet)

- **The forced submission every 6 steps.** It is a suspected cause of shallow investigation on big networks
  (`DIAGNOSIS_ABOVE_32.md` §4), but not shown. It gets tested as a question first.
- **The reader model** (DeepSeek V4.1 Flash). Whether a stronger model is needed is a question for after the
  measurement works.
- **The truth tests** (rungs 1–4). They were calibrated on known answers and never shown to be wrong.

## How development runs (Shreyas, Nov 18): a micro-experiment loop

Build by many small end-to-end experiments, minutes each: see what works, what helps, what hurts, and continue from
there. A "round" (a full batch of runs) is only for testing a generation that has already been shaped by many micro
experiments, never for testing a single new piece. Each micro experiment still states its question and the decision
it informs, and it is logged in the diary.

## How experiments are run (new)

1. **Question first.** Every experiment states the question, the decision each outcome leads to, and the number of
   runs needed to tell the outcomes apart. If it can't decide anything, it isn't run.
2. **Free experiments first.** 757 saved run files and their notes can be re-scored and re-read in minutes, with no
   AI calls. Most questions about the judge are answered this way.
3. **Small runs.**
   - One or two targets, 2–4 runs each, all four keys in parallel, a step cap set by the question.
   - Expect 15–30 minutes, not 4 hours. Round 2's 4 hours came from 40 jobs on 4 keys, most of them asking
     questions the judge couldn't answer.
4. **Warm starts.** Readers already understood several networks; their notes are saved. To test whether a reader can
   *write up* an account (as opposed to *finding* one), start it from those notes instead of paying for discovery
   again.

## Rebuild order (each step needs Shreyas's go)

1. **Build and prove the measurement. No AI calls.**
   - Compute the copy frontier for each AlgZoo and longest-cycle network.
   - Assemble the must-fail set: every saved copy, plus new disguises written *before* the measure is coded, plus
     shuffled controls.
   - Assemble the must-pass set: M2,2, M4,3, the lc_8_6 census rule, the month-line construction.
   - The criterion, decided now: every must-fail file lies on or above the frontier, and every must-pass file lies
     clearly below it.
   - If that can't be met, the idea is wrong. Say so, and go back to the intent.
2. **Re-score everything already saved** (rounds 1 and 2) under the version-2 certificate. The question: did we
   already produce explanations above 32 parameters that version 1 hid? This also decides what the catalog can
   show.
3. **One small warm-start run** on two networks (lc_8_6 and M16,10), with version-2 feedback. The question: given its
   own understanding, can the reader write an account that beats the copy frontier?
4. **Only then the ladder,** with the results deciding the next intent change.
