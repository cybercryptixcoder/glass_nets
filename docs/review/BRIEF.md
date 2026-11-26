# Glass Networks, the next build: the brief

*Two pages. The full argument, with evidence for every claim, is `NEXT_BUILD.md`; how the review was done is
`REVIEW_LOG.md`; the first experiments are pre-registered in `PREREG_FIRST_EXPERIMENTS.md`. Nov 26, 2026.*

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
   "residual": the chess network's 22% was located but never named.
3. **The decision and the weights, not the computation at inference.** Every result that held up rested on
   evidence from inference. Weights alone predict edit effects at 0.17; the hidden state predicts them at 0.97.
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
  - length in a shared, growing vocabulary (this beats disguised copies by construction);
  - whether it makes checking the network cheaper;
  - whether a fresh reader can predict the network from it;
  - whether it lets you edit the network with every effect predicted first.
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
  2. **A Helix-style volume engine** on a Flash-grade model, for the sub-problems with exact scores: definitions of
     found variables, laws and closures, experiment design, abstraction. This is where a cheap model at volume is
     strongest: Helix Evolve beat the best human construction on an open inequality this way.
  3. **A strong agent as director:** it picks the residual, writes the scorer, decides what is confirmed, and writes
     the account.

  The box improves verifiably at three timescales:
  - inner: the exact score;
  - middle: the residual curve;
  - outer: the discovery score.
- **Why this suits Helix.** Helix's open question is how to get past the model's prior. Here the network's own
  variables are an exact signal from outside that prior.
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
  - the copy rules;
  - fixed per-study vocabularies;
  - a bespoke pipeline per sprint;
  - templated LM tasks whose input is the analyst's feature row.

## The flagships, and the open seat in each

| Flagship | The open seat |
|---|---|
| A chess network's strategies | How look-ahead is implemented and combined with heuristics across all positions (named as open by Jenner et al.); that is where the residual sits |
| How a fluid model is cheap | The implicit closure and long-step integrator inside an end-to-end coarse emulator. The target must be rebuilt first (coarse-grained, long steps) |
| What lets an LLM lie | How the instruction's tokens gate a shared lying circuit; whether lies with no instruction use the same parts; a fingerprint defined by that mechanism that transfers across model families |
| How LLMs read weights | A mechanistic account inside the reader model; nobody has opened one |

## The first six experiments (pre-registered, none run)

| # | Question | Decides |
|---|---|---|
| X1 | Does a frontier reader pass where Flash stalled? | Who holds the reasoning roles |
| X2 | Can unpublished-algorithm networks be made, and what does today's toolkit recover blind? | The discovery yardstick |
| X3 | Does a coarse-grained fluid emulator beat the coarse solver? | Whether flagship 2 has a target |
| X4 | Can a strong agent's cycle define chess channel 22? | The definition engine (with X6) |
| X5 | Does Helix Probe's loop recover planted mechanisms inside a network? | The loop's skeleton |
| X6 | Does Helix-scale volume find what 59 one-shot programs missed? | The definition engine (with X4) |

Order: X2 and X5 first (cheap, and they build the yardstick), then X1, then X4 and X6 together, then X3.
