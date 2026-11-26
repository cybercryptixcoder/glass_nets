# Glass Networks, the next build: the brief

*Two pages. The full argument, with evidence for every claim, is `NEXT_BUILD.md`; how the review was done is
`REVIEW_LOG.md`; the first experiments are pre-registered in `PREREG_FIRST_EXPERIMENTS.md`; an independent critique
by a fresh agent is `CRITIQUE.md`, and what survived checking is folded in. Nov 26, 2026.*

## Where the project stands

- **The instruments are strong:**
  - a calibrated causal judge;
  - controls that caught real errors;
  - an honest memorisation label;
  - a blind decompiler;
  - a law extractor;
  - a teachability test.
- **The engine was not the code.** Every result on a hard target came from a Claude session doing research by hand,
  in a new pipeline per target. The automated reader only succeeded where the mechanism was of a kind it already
  knew.

## What stops it from being the magic box: five assumptions, not size

1. **One artifact, one leap, one judge.** Understanding is a hierarchy (variables → operations → algorithm →
   knowledge) built in many verified steps. Today nothing connects the levels.
2. **Concepts known in advance.** Every instrument explains in a supplied vocabulary. Anything new comes back as
   "residual": the unexplained 22% of a chess network's value (LD2) was located but never named.
3. **The decision, not the computation.** Almost every test asks "is my quantity at this site, and is it used for
   the decision?", not "what steps does the network take?". Separately, every result that held up rested on
   evidence from inference, and an LLM reading raw weights predicts edit effects barely better than one reading none
   (0.17 against 0.11).
4. **A size ladder.** Difficulty is structure, not size: a clean 226k-parameter net was easy, and dense
   112-parameter nets were not.
5. **Targets trained on functions we wrote down.** Only the implementation could be new. The one physics target
   was built with its answer sealed inside it.

## What the box must be

- **The output: a Glass Model.** A verified, executable model of the network at five levels, with:
  - a ledger of everything still unexplained (memorised, noise, or structured but unnamed);
  - a typed guarantee on every claim.
  
  A person reads it as a step-by-step trace of the network's own algorithm on any input.
- **The measures: six numbers instead of "verified / complete":**
  - faithfulness at every level;
  - completeness;
  - **answering without running**, the test against copies: used to estimate what the network does, does the
    explanation beat simply running the network, or its cheapest copy, for the same compute? A copy can only answer
    by being run. (Length in a shared vocabulary is reported too, but not as a gate: the repo's own pre-registered
    test showed that "fewer bits than a copy" rejects real explanations of small networks.);
  - whether it makes checking the network cheaper, where the domain allows a proof;
  - whether a fresh reader can predict the network from it;
  - whether it lets you edit the network with every effect predicted first.

  The last two are scored against baselines that use no explanation, because attribution patching once
  out-predicted the project's own construction.
- **The method: a loop driven by the residual that invents its own vocabulary.**
  1. Map what the network does.
  2. Find the variables it actually uses.
  3. Write programs that define them.
  4. Compose them into its algorithm, and compare that with the textbook one.
  5. Turn what recurs into new named primitives.
  6. Test what a person can learn from it.
  7. Confirm under frozen tests.
- **The engine: both, in three layers.**
  1. **A toolkit first:** every action and every score as code.
  2. **A Helix-style volume engine** on a Flash-grade model, for the sub-problems with precise, cheap scores:
     definitions of found variables, laws and closures, experiment design, abstraction. In the cases we have, that
     is where a cheap model at volume did best: Helix Evolve passed the best human construction on an open
     inequality, with no outside input, and went much further with a summary of published methods. These scores
     are truth tests, and a copy passes truth tests, so every search runs in a restricted program language (a
     budget of constants, no weights, no fitted sub-models) against a set of fitted copies that must fail.
  3. **A strong agent as director:** it picks the residual, writes the scorer, decides what is confirmed, and writes
     the account.

  The box improves verifiably at three timescales:
  - inner: the frozen score;
  - middle: the residual curve;
  - outer: the discovery score.
- **Whether this suits Helix is a test, not a given.** Helix's open question is how to get past the model's prior,
  and its own research case says a score alone only re-weights what the model already proposes. The network can
  also show the proposer evidence from outside that prior (counter-examples, minimal pairs). X6 runs both, with
  Helix's prediction as the null.
- **The flywheel, last.** Search, verify, then distil the verified explanations into a trained reader, and repeat.
  It is fed by interpretability self-play: a puzzle-maker builds networks with hidden algorithms and sealed keys, and
  an interpreter decodes them. This is the part whose ceiling does not depend on how many good researchers exist.

## Keep, scrap, rebuild

- **Keep, as parts:**
  - the judge's causal tests (scored on intermediate states, not only answers);
  - the controls and the memoriser test;
  - the decompiler (the first structure recogniser);
  - SPD with rule/lookup labels;
  - Kepler's law extraction and teachability;
  - the change ledger (at the boundary between exploring and confirming);
  - the catalog, as a view of the Glass Model;
  - Helix Probe's loop as the skeleton, and Helix Evolve's engine.
- **Scrap:**
  - the one-leap reader/judge loop as the centre;
  - the size ladder;
  - the copy rules (replaced by answering without running, and by restricted languages with must-fail sets);
  - fixed per-study vocabularies;
  - a bespoke pipeline per sprint;
  - templated LM tasks whose input is the analyst's feature row.

## The flagships, and the open seat in each

| Flagship | The open seat |
|---|---|
| A chess network's strategies | How look-ahead is implemented and combined with heuristics across all positions (named as open by Jenner et al.). The record's unexplained 22% is in a convolutional network (LD2); the transformer (t1) needs its baseline measured first, and the residual must be split into knowledge and the network's own errors |
| How a fluid model is cheap | The implicit closure and long-step integrator inside a coarse emulator. The target must be rebuilt first (coarse-grained, long steps); published work shows such emulators learn real corrections (Kochkov et al. 2021), and reading them is the open part. As built, the current model costs about 47× its solver's arithmetic |
| What lets an LLM lie | How the instruction's tokens gate a shared lying circuit; whether lies with no instruction use the same parts; a fingerprint defined by that mechanism that transfers across model families |
| How LLMs read weights | A mechanistic account inside the reader model; nobody has opened one. First find an open model that reads at all: on the record, most do not |

## The first experiments (none run; X0–X6 pre-registered, X7–X8 described in NEXT_BUILD §10)

| # | Question | Decides |
|---|---|---|
| X0 | On saved runs, does answering without running accept the real explanations and reject every copy? (free) | Whether the test against copies works |
| X1 | Does a frontier reader recover accounts known to exist (month rollover, countdown, planted networks) where Flash stalled? | Who holds the reasoning roles |
| X2 | Can unpublished-algorithm networks be made, and what does today's toolkit recover blind? | The discovery yardstick |
| X3 | Does our rebuilt coarse fluid emulator reproduce the published gain over the coarse solver? | Whether flagship 2 has a target |
| X4 | Is LD2's quiet-position residual knowledge or error, and can a cycle define channel 22? | The definition engine (with X6) |
| X5 | Does Helix Probe's loop recover planted mechanisms inside a network? | The loop's skeleton |
| X6 | Does Helix-scale volume find what 59 one-shot programs missed, with the score alone or with the network's evidence shown? | The definition engine (with X4), and whether the network gets Helix past its prior |
| X7 | Which heads read the chat-template tokens and pass the lying mode on? | Flagship 3's first object |
| X8 | Which open model reads weights above its controls? | Whether flagship 4 has a subject |

Order: X0 first (free), then X2 and X5 (cheap, and they build the yardstick), then X1, then X4 and X6 together,
with X7 and X8 alongside, then X3. Before any of it: X1 needs a frontier-model key (this workspace has only
DeepSeek V4.1 Flash), X3, X7 and parts of X4 need GPU time, and spending needs Shreyas's approval.
