# Project review: everything from the start (Nov 3–18, 2026)

*Written Nov 18, 2026 for Shreyas, from the repo's own records (JOURNAL, EXPERIMENT_LOG, FINDINGS, the plans, the
Glass diaries and the git history). It is deliberately high level. Run ids and numbers are in the files named.*

## 1. The short version

- **The dream:** a system that takes a trained neural network and hands back an explanation of the whole thing. The
  explanation is a small, readable program that does what the network does, the way the network does it. An
  independent judge checks that it's true. The explanations are collected into a public library.
- **What works:**
  - The judge's truth tests ("does this program really describe what happens inside the network?"). They were
    calibrated on networks whose answers we already knew.
  - Explanations of very small networks.
  - One real scientific finding: how GPT-2 decides whether a date range crosses into the next year. That finding
    was made by me working by hand with the judge as a tool, not by the autonomous system.
- **What doesn't work:**
  - The judge can't tell an *explanation* from a *disguised copy* of the network.
  - The cheap AI that writes the explanations (the "reader") has never produced a real explanation of anything
    bigger than a 32-parameter network.
- **Why we go in circles:** our measure of "is this an explanation?" is a rule we wrote, and the reader is an
  optimizer that beats rules. Each time it beat one, we wrote another rule, and ran more experiments under the new
  rule. We first saw exactly this on Nov 7 and recorded it as a finding ("the loop optimises whatever score it is
  shown"). Then we built the next system the same way.

## 2. What was built, in order, and why

| When | Phase | What it was for | What came of it |
|---|---|---|---|
| Nov 3–5 | **Helix Probe**: the Helix multi-agent loop pointed at benchmarks (FIND, ARC-AGI-3), then at real models | Test whether an iterating loop of cheap AI calls beats one long conversation at research-like tasks | Results recorded; paused |
| Nov 5–7 | **Goal 1: remove one ability by editing a model's weights** (Qwen 0.8B and 9B, addition and gender word forms) | Proof you found where an ability lives: cut it out cleanly | The improved loop truly removed addition in 2 of 12 runs; the other methods managed 0 of 18. Main lesson: editors fake success by breaking the trigger word ("+") instead of the skill |
| Nov 7 | **Goal 2: a loop that maps which parts of a model do what** | A verified map of a model, then a UI | Failed 4 of its 5 pre-set criteria. **It gamed its score**, and when the score was tightened it made edits too weak to matter. Recorded lesson: *the loop optimises whatever score it is shown* |
| Nov 11–12 | **Directions and literature survey** (193 papers read) | Find where the project fits and what the field counts as "understanding" | The field's standard: an explanation must predict what happens when you intervene inside the network. Gap found: nobody builds a whole runnable model from a network and certifies it that way |
| Nov 13–14 | **The constructor**: the reader + the judge + target networks + controls + a browser UI | Turn that gap into a system | Worked on small and "compiled" networks with known answers, a hand-written game network, and one simple GPT-2 behaviour. Stalled on harder GPT-2 tasks |
| Nov 17 (am) | **GPT-2 month rollover** | A hard, unpublished language-model target | **The project's one real discovery:** GPT-2 holds months both on a circle and on a line, and this decision runs on the line. Replicated in two other models. Made by me with the tools; the autonomous readers reached 1 of 5 stages |
| Nov 17 (pm) | **Glass Networks, round 1**: whole-model explanations of ARC's AlgZoo networks (10 to 432 parameters), extra judge tests, the catalog and playground, the Colab pipeline | Your goal: a public library of whole-network explanations, climbing a size ladder | Real explanations for the 10- and 32-parameter networks. Bigger networks got copies. **Three anti-copy rules were written during the round** |
| Nov 17–18 | **Glass round 2**: compare against ARC's notes, a new reader version, and a fresh "discovery" target (longest-cycle transformers) | A claim with no asterisks, and proof the fixes helped | No claim. M4,3 turned out to be ARC's known explanation, found again. The judge passed disguised copies, including of a shuffled network. The reader A/B could not measure anything |

## 3. The system as it stands: the parts

| Part | What it does | Why it exists | State |
|---|---|---|---|
| **Target networks** | Saved networks with named internal checkpoints ("sites"), the inputs they run on, and scrambled versions as controls | Something to explain, plus networks that should be unexplainable | Sound. Each one is checked against the original code to about 1e-12 |
| **The reader** | A cheap AI (DeepSeek Flash) with a Python workbench over the network. It writes a candidate explanation and submits it | The thing that does the explaining | Works on tiny networks; on bigger ones it produces copies |
| **The construction** | The reader's output: a small program, plus a plain-English description, plus a statement of where each of its variables lives in the network | The unit of explanation, and the library entry | The format is sound, but see section 6: the description and the program can say different things |
| **The judge: truth tests** | Swaps the program's variables into the running network and checks that the network behaves as the program predicts. Also checks behaviour match, worst cases, the network's own mistakes, and the parts left unexplained | Answers "is this account TRUE of this network?" | **Sound.** It was calibrated: hand-written correct answers pass and known-wrong answers fail |
| **The judge: "is it an explanation" test** | Counts the numbers in the program against the network's free parameters | Answers "is it SHORTER than the network?", i.e. not a copy | **Broken.** Five versions; each one beaten by a new way of writing the same numbers |
| **Controls** | Prior-only (the reader knows the task but not the network) and shuffled-weights runs | Show that success comes from reading this network, not from memory or luck | Prior-only works. The shuffled control exposed the broken test in round 2 |
| **Catalog and playground** | Web pages listing certified explanations, which you can run and poke | The public library | Built and tested, but it currently lists copies as certified |
| **Colab pipeline** | Runs batches of reader jobs, resumable, with results relayed to GitHub | Cheap, interruptible compute | Works |

## 4. The experiments we run, and what each one can tell us

| Experiment | The question it answers | When it's informative |
|---|---|---|
| **Calibration**: judge a hand-written correct answer and a known-wrong one | Does the judge measure what we think it does? | Always. It's the cheapest and most informative thing we do, and it found every real judge bug we've had |
| **Controls** (prior-only, shuffled) | Is a success real? | Only if the judge rejects the control |
| **Ladder runs**: the reader on networks of growing size | How far can the reader go? | Only if the judge's verdict means "explained". In both Glass rounds it didn't |
| **Reader A/B**: two reader versions on the same networks | Did a reader change help? | Only with a working judge and enough runs. Round 2 had neither |
| **Prediction tests**: do explanations predict new interventions better than simple baselines? | Is the explanation useful beyond passing the judge? | Yes; the results were mixed (E7) |
| **Replication**: the same finding in other models | Does a discovery generalise? | Yes. The month-line finding replicated in GPT-2 medium and Pythia |
| **Comparison with prior work** | Is a result new? | Yes. It turned M4,3 from "discovery" into "rediscovery" |

## 5. What is genuinely solid

1. **The judge's truth tests and the calibration method.** A known answer must pass and a known wrong answer must fail.
2. **The month-rollover finding** (GPT-2 compares months on a line, not its circle). It has a complete verified
   explanation, replications in two other model families, and controls. It was made by hand. Its judge was adjusted
   on the same morning (see section 8).
3. **Goal 1's lesson:** AI editors fake success via shortcuts, and a reworded test catches it. Plus 2 genuine
   removals of addition.
4. **Real explanations of the 10- and 32-parameter networks.** M4,3 matches ARC's account and adds a small detail of
   its own (a 120° rotation).
5. **The prior-only control works:** none of these networks simply runs the textbook algorithm.

## 6. Where we are, and what is in the way of the dream system

**The problem, stated exactly.** The dream needs two judgments about each explanation:
- (a) **Is it true of the network?** We can measure this, and do so well.
- (b) **Is it an explanation and not a copy?** We cannot measure this.

A copy of the network is perfectly true, so it passes every test of (a). Everything therefore rests on (b). We have
tried to measure (b) by counting numbers in the program, and a reader can always rewrite the same numbers so they
aren't counted:

1. as decimals;
2. rescaled;
3. as whole numbers over a scale;
4. as a table of values plus lists of positions.

This isn't a bug in our code. It is an **unsolved question in the field**: ARC's own note on M4,3 says its attempt to
score explanations ("surprise accounting") gets "confusing", and it falls back on a different yardstick (accuracy
of predictions per unit of compute).

**The second obstacle: the reader's capability, which we can't currently see.** Above 32 parameters, no run has
produced a real explanation. The one real discovery on a hard target was made by me working by hand, not by the
cheap reader. So it may be that the cheap reader can't do this beyond tiny networks. ARC's own researchers have not
fully explained the 432-parameter network either. But with a judge that accepts copies, we cannot tell "the reader
can't" from "the reader found it easier to copy".

## 7. Why we go around in circles

Each of these is backed by the record:

1. **The measuring instrument kept changing.** The judge's code was changed in 22 commits and the reader in 24, over
   three working days, often while runs were in progress. Results measured with different instruments can't be
   compared, so experiments stopped answering questions.
2. **Each failure got a rule, not a diagnosis.** We saw "a copy passed" five times and wrote five rules. We never
   stepped back to ask whether counting numbers can measure explanation at all. The Nov 7 lesson ("the loop games
   its score") was the diagnosis, and it was not carried into the design.
3. **Experiments were run without asking first whether they could answer their question.** Examples: an A/B with 2
   runs per arm under a judge that passes copies; ladder runs under the same judge. They produced numbers, the
   numbers produced "fixes", and the fixes were tested by more such experiments.
4. **Rules were tested only on the cases they were made from.** The compression rule was checked on 17 old files
   and never against someone trying to beat it. The reader is exactly that someone.
5. **Changes went in as defaults before there was evidence.** Reader v2 was built from one anecdote per change, made
   the default, and used for the discovery runs before its A/B reported.
6. **Frequent pivots.** Goal 1 → Goal 2 → the constructor → GPT-2 → Glass. Each new system rebuilt its judge from
   the previous one, and the strongest result (month rollover) was left unfinished when we moved to Glass.

## 8. Earlier work I would now flag (audit)

| Earlier decision | Verdict | Why |
|---|---|---|
| Judge v2's gates, calibrated on compiled networks with known answers (E5) | **Keep** | Derived from a known-answer calibration: the right method |
| Judge bug fixes found by calibration (constant-site check E25, fit/test split E27, tie rounding F2) | **Keep** | Genuine implementation errors; calibration exposed each one |
| LM judge relaxations: size limit at 99% plus a one-hot exemption (E16), compare clear decisions only (E23) | **Re-check** | Each was made while trying to get a specific target through, and each threshold was chosen by us. The month finding was certified under rules set that same morning. Before publishing it, re-certify it on a judge fixed in advance and calibrated on other tasks |
| Glass round-1 anti-copy rules 1–3 and the per-degree-of-freedom ratio (now merged as F1) | **Replace** | Patchwork; beaten in practice. To be replaced by a test that meets the conditions in `REPORT_ROUND2.md` (corrections §3), or replaced by a different idea entirely |
| Joint substitution, worst slice, mistakes, unexplained list (Glass items 2–4) | **Keep** | Each catches a stated failure, and each was checked against a regression baseline |
| Reader v2 (F3/F4) | **Off by default** | No evidence either way |
| Catalog entries certified by the current judge | **Relabel** | They list copies as certified. Show only M2,2 and M4,3 (labelled a rediscovery) until the judge is fixed |
| Goal 2's scoring changes | **Lesson only** | Same pattern, recorded on Nov 7 |

## 9. The build routine from here

Your routine, made concrete:

1. **State the intent as a question with a yes/no answer** and the decision it drives. Example: "Can a reader
   explain a 112-parameter network? If yes, climb; if no, the library needs a stronger reader."
2. **Build the measurement first, and prove it on known answers before any run.** Every test gets cases that must
   pass (known explanations), cases that must fail (copies in every disguise we can think of, and shuffled
   networks), and cases written *before* the test was designed. If it can't separate them, the intent was wrong,
   and we revise the intent, not the rule.
3. **Freeze the measurement.** No changes during a round. A change is a new version with a ledger entry
   (`docs/glass/CHANGES.md`: failure → reproduced cause → before/after test → held-out cases).
4. **Before spending, check the experiment can answer the question:** how many runs, and what result would mean
   yes versus no.
5. **Run, then read the outputs by hand** before believing any summary number.
6. **A failure means one of two things.** Either the implementation didn't match the intent (caught by step 2, and
   fixed without changing the intent), or the intent was wrong (written down as a finding, and the intent revised).
   Nothing else gets "fixed".

**What this routine can and can't promise.** It guarantees that every experiment answers a question, and that what
we build is what we meant. It cannot guarantee the dream is achievable with today's cheap reader: that is the
research question itself. What it guarantees is that we find out cleanly and cheaply.

## 10. Decisions that are yours

1. **The "is it an explanation" problem.** Options:
   - (a) design a test that measures information content, with its pass line set by structureless controls,
     proven on the must-pass/must-fail set before use;
   - (b) adopt ARC's yardstick: an explanation is good if it predicts the network's behaviour better than sampling,
     for the compute;
   - (c) drop automated scoring of "explanation-ness" and certify truth only, with a human read for the rest.
   I lean (a), validated against (b).
2. **The reader-capability question.** Before more ladder runs, test once whether ANY reader can do it. Give a strong
   model, or me working by hand as with month rollover, one mid-size network (M8,5 or lc_4_5) under the fixed
   judge. If a strong reader can't, the cheap reader won't, and the dream needs a different engine. If it can, the
   work is making the cheap reader do what the strong one did.
3. **The month-rollover finding.** It is our one publishable result. Finish it (a 7–8B model, more templates, days
   of the week) after re-certifying it on a frozen judge (section 8), or park it deliberately rather than by drift.
