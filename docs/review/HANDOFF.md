# Handoff: building the next Glass Networks

*For the agent who will build the next version. Written Nov 26, 2026, 03:59 ET (clock), at the end of a two-run
research review of the whole project. This is not a specification. It gives you the picture, the reasons behind it,
and where to read. How to build it is yours to decide.*

---

## What this document is, and what it is not

- **It is** the vision, the evidence for it, and a reading map.
- **It is not** a design you must implement. Nothing here fixes an architecture, a file layout, an order of work or
  a technology. The review that precedes it (`NEXT_BUILD.md`) proposes designs. Treat them as one researcher's
  hypotheses, not as requirements.
- **Why it is written this way.** Shreyas wants you to build from understanding, not from a checklist. A specification
  would lock in one agent's guesses about how to do something that has never been done. The review's own record shows
  what those guesses are worth: two independent checks found real errors in it on the same night it was written
  (`CRITIQUE.md`, `VERIFY_CHANGES.md`). The goal and the reasons are well supported. The particular designs are not.

---

## The picture: what should exist when this works

Someone hands the box a trained network. It could be a chess engine, a learned physics simulator, or a language
model. They ask what the network does, how, and why.

What comes back is not a score or a heat map. It is a **model of the model**:
- It is a program that runs and reproduces the network's behaviour, including its intermediate states, not only its
  answers.
- It is layered: the network's own variables, the operations that compute them, the algorithm those operations
  form, and the knowledge that algorithm embodies, stated so a person can learn it.
- Every claim is tied to the network by tests that intervene on it, and says what kind of guarantee backs it.
- What is not yet explained is listed honestly, with how large it is and what kind it is.

A person reads it like a debugger for the network's mind. Pick a chess position and step through the network's
computation, from the threats it registers to the line it looks ahead along and the reason it prefers a move. At
every step you can see where the explanation and the network agree, and where they part.

**The questions Shreyas wants it to answer**, which should pull the build:
- What strategies does a chess network use that people have not named?
- How does a learned fluid model get its result? (As built, the current one is not cheap at all. A target that has
  to be cheap must be rebuilt first: see `NEXT_BUILD.md` §6, flagship 2.)
- What gives a language model the ability to lie, and what is its fingerprint?
- How do language models read weights, and can that be done on purpose?

**The test of success is discovery, not description.** The box should find, without being told, things the people
running it did not already know. Recovering a textbook algorithm from a network trained on that textbook proves
little.

---

## The reasons, in short (each argued in full, with sources, in `NEXT_BUILD.md`)

These are the findings the vision rests on. Each is a *why*. Decide for yourself what each one implies for the build.

1. **The engine that produced every hard result was a strong agent doing research by hand, not the code.**
   - The automated reader/judge loop worked only where the mechanism was of a kind the reader already knew.
   - Each sprint then built a new bespoke pipeline.
   - So the thing to build is what makes a capable researcher (agent or person) compound their work, not another
     one-shot pipeline.
   - Sources: `NEXT_BUILD.md` §1; `docs/PROJECT_REVIEW.md`.
2. **Understanding has levels, and nothing connected them.**
   - Each instrument produced one level: a top-level program, a parts list, or a law read from behaviour.
   - The one hard-target result that held up, the GPT-2 month line, has the shape of a trace through levels. Even it
     is partial: its comparison step is untraced.
   - Sources: `NEXT_BUILD.md` §1b, Gap 1.
3. **Every instrument explains in a vocabulary someone supplied in advance.**
   - Task features, Fourier angles, 61 chess concepts, 38 physics terms, group theory.
   - Anything new comes back only as "residual". About a fifth of a chess network's value sat there, located but
     never named.
   - Discovery needs a way to grow vocabulary from what the network itself computes.
   - Sources: Gap 5; `docs/kepler/FINDINGS.md`.
4. **The object interpreted was the decision, not the computation.**
   - Almost every test asks "is my quantity at this site, and is it used?"
   - The network's own steps, intermediate results and control flow were looked at only through an analyst's
     variables.
   - Separately, everything that held up rested on inference-time evidence. Reading raw weights helped language
     models very little.
   - Source: §1b.
5. **Difficulty is structure, not size.**
   - A clean 226k-parameter network was easy. Dense 112-parameter ones were not, until an exact decomposition handled
     them.
   - A ladder of ever-bigger networks climbs the wrong axis.
   - Sources: Gap 6; `docs/glass/UNITS_FINDINGS.md`.
6. **Telling an explanation from a disguised copy is the central unsolved problem, and the weakest part of any
   design.**
   - A copy of a network is perfectly true, so tests of truth cannot reject it.
   - Rules about the text of a program were each beaten.
   - "Fewer bits than a copy" failed the repo's own pre-registered test: it rejected every real explanation.
   - The most promising lead in the record is ARC's framing: a copy can answer a question only by being run.
     `EVIDENCE_LADDER.md` rung Y; `NEXT_BUILD.md` Gap 2, and experiment X0, which re-scores saved runs at no cost.
7. **An optimiser pointed at a score games it.**
   - Goal 2 gamed its score, and the cheap reader copied.
   - High-volume search on a cheap model has worked here before. Helix Evolve passed the best human construction on
     an open problem. But that was against a score that could not be faked.
   - Any search you build is only as good as how hard its score is to game.
   - Sources: §3.11 and §3.12; `helix_evolve/docs/RECORDS.md`.
8. **Targets trained on functions people already wrote down can only confirm.**
   - Real discovery needs networks that know something we don't: strong games, emulators trained on finer truth, or
     language models.
   - It also needs a yardstick of networks whose algorithms are known but published nowhere, so that recovery can be
     told apart from recall.
   - Sources: §6; the calibration ladder.

---

## What you inherit

**Instruments that work**, with what each is good for:

| What | Where | Why it matters |
|---|---|---|
| A causal judge, calibrated | `lab/judge.py`, `lab/lmjudge.py` | Interchange, substitution and necessity tests that caught real errors |
| Controls | Throughout: shuffled, untrained, permuted, prior-only, a memoriser | They are why the results can be believed |
| Exact weight decomposition with "rule or lookup" labels | `lab/spd.py`, `lab/atlas.py` | Faithful parts lists, including for networks no short program fits |
| A blind decompiler for group-structured networks | `lab/decomp/` | The closest thing to blind, exhaustive recovery, inside a fixed family of structures |
| Law extraction, probes and teachability tests | `kepler/core/`, `kepler/targets/` | The physics law read with its coefficients and regimes; ported Leela networks |
| A predict-first experimental loop | `probe/` (Helix Probe) | Lifted a weak model on black-box tasks, mostly by making it experiment at all |
| A volume search engine | Separate repo `cybercryptixcoder/helix_evolve` | Thousands of programs scored against a score that cannot be faked |
| A large literature log | Branch `main`, `docs/literature/` | 793 paper entries plus a 193-entry core survey |

**What failed, and why.** This is as important as what worked: `NEXT_BUILD.md` §1, §2 and §4, and
`docs/glass/LEARNINGS.md`.

You are free to keep, rebuild or discard any of it. The review's own opinion is in §4 (keep / scrap). Justify your
choices from the record.

---

## Where to read, in order

1. `docs/review/BRIEF.md` (two pages): the whole argument in short.
2. `docs/review/NEXT_BUILD.md`, in this order:
   - §0 (the answer in short);
   - §1b (what the system interprets, and why that framing limits it);
   - §2 (fifteen gaps, each with evidence);
   - §3 (one researcher's design);
   - §6 (the flagships and what "decoded" would mean for each).
   Read §3 as a proposal to argue with, not as a plan.
3. `docs/review/CRITIQUE.md` and `docs/review/VERIFY_CHANGES.md`. They show where the review was wrong, and how
   confidently it was wrong.
4. `docs/review/PREREG_FIRST_EXPERIMENTS.md`: nine small experiments (X0–X8) that test the premises the design leans
   on. Run them, change them or replace them. They are offered because each answers a question a build would
   otherwise guess at.
5. `docs/review/REVIEW_LOG.md`: every finding (F1–F60), every decision and its reason, with sources.
6. The project's own record, the primary source:
   - `docs/PROJECT_REVIEW.md` (the history, and why the project went in circles);
   - `docs/glass/INTENT.md`, `EVIDENCE_LADDER.md` and `LEARNINGS.md`;
   - `docs/kepler/FINDINGS.md`, `docs/decomp/RESULTS.md`, `docs/glass/ATLAS_RESULTS.md`,
     `lab/constructions/INDEX.md`.

   Where the review and the record disagree, the record wins.

Branches: the review lives on `review-next-build`, cut from `liar`, which is the furthest code branch. `main` holds
the literature log and lags behind the code.

---

## The few things that are not yours to change

These come from the project's rules (`CLAUDE.md`) and from what the record shows happens without them:
- **No claim without a measurement built first and proven on known cases,** including cases written in advance that
  must fail, and controls.
- **Pass lines come from controls, never from the results they judge.** A measurement is frozen while it judges.
- **Exploration is free, but only confirmed work becomes a claim.** Label which is which.
- **Ask Shreyas** before spending money, starting a paid round, or changing `docs/glass/INTENT.md`. The new
  direction is not yet in `INTENT.md`: proposing that change is likely your first conversation with him.
- **Keys live only in the gitignored `lab/.env`.** The repo is public: never commit or print them.
- **Never edit the branch `constructor-attempt`.**
- **Times in docs come from the system clock,** never typed by hand.
- **Document every step, decision and failure** (`docs/glass/DIARY.md`), in plain English and plain markdown.
- **Commit under Shreyas's name** (cybercryptixcoder / official.shreyas28@gmail.com) with no Claude attribution
  lines.

---

## What is yours to decide

Everything else:
- the architecture;
- what gets built first;
- which instruments to keep;
- which language models play which roles, and whether a cheap model at volume, a strong agent, or both does each
  part;
- how the explanation object is represented;
- which flagship to aim at first;
- whether the review's measures, loop and roles are right at all.

If you find a better way to reach the picture above, take it, and write down why.

---

## Where things stand

- **Resources.**
  - The only model key in this workspace is DeepSeek V4.1 Flash, through Luminal (`lab/luminal.py`). A frontier
    model needs a key from Shreyas.
  - GPU work has run on Colab sessions, which have no GitHub credential, so results reach the repo only when relayed
    by hand.
  - No budget is approved yet.
- **Nothing in X0–X8 has been run.**
  - X0's main arm needs no money: it re-scores saved runs, and it tests the weakest part of any design.
  - Whether to start there is your call.
- **Open decisions for Shreyas:**
  - which experiments to approve;
  - whether the new direction goes into `INTENT.md`;
  - whether `CLAUDE.md`'s pointer to current work should move from `docs/glass/ATLAS.md` to this handoff.

## How Shreyas works (from `CLAUDE.md`)

- **Building is the goal.** An experiment exists only to answer a question a build step needs.
- **When something falls short, try another way and test it,** quickly and many times if needed. Don't stop at a
  report of why it doesn't work.
- **Work continuously,** from one step straight into the next.
- **Tell him when a result lands,** not on a timer.

## How you will know it is working

Two signals, both argued in `NEXT_BUILD.md` §0. Use them, sharpen them, or replace them with better ones.

- **The discovery score.** Blind recovery of algorithms that are known to the experimenters but published nowhere.
- **The residual curve.** On a real network, does each round name, verify and teach something new, so that the part
  explained by knowledge (not by the network's own errors) grows beyond what the classical reference library reaches?
