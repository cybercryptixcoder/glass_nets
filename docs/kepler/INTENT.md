# Kepler: intent

*Set by Shreyas on Nov 24, 2026 (09:20). This file states what Kepler is for; the code serves it.*

## The ask, in Shreyas's words (condensed)

> Build, as ambitiously as possible, the most advanced interpretability tool in the world: a single tool that can
> make real discoveries, not just at a basic level but at a fundamental one. Use it on more complex things: how a
> (small) video model trained on fluids does fluid dynamics well, and a model that plays chess, to see whether we can
> extract real chess strategies that have never been known before. Extract and understand things at a level of
> abstraction from which real concepts and real learning can be taken, and put them together so that we can really
> learn from them. Build on the shoulders of giants, but build something that has never been done before.

## The question

Can one engine turn a trained network into **verified, human-learnable knowledge** (laws, concepts, strategies),
across networks as different as a fluid video model and a chess network, without being told the answer?

- **Yes** means: on each target, Kepler states things in human terms (an equation, a rule written as a program, a
  concept with example positions) that pass its checks below, and at least one statement is something a person can
  learn from the model that was not put there by the analyst.
- **No** means: the statements fail the checks, or only restate what the analyst already supplied.

## What counts as a finding

A finding is a claim about the model, stated in human terms, that passes all of these:

1. **Fidelity.** The statement predicts the model (its internal activity or its outputs) on inputs not used to make it.
2. **Cause.** Changing the model along the statement (patching, steering, removing) changes its behaviour the way the
   statement says.
3. **Level.** It is stated at the level a person can use: a law with coefficients, a rule as a program, a concept with
   examples and counter-examples. Not a list of neurons.
4. **Novelty, measured.** Each finding says how much of it a reference library of known concepts (physics terms;
   classical chess knowledge) already explains, and compares against prior work. "New" is a measured residual, not
   a feeling.
5. **Faithfulness.** If the model memorizes rather than computes, Kepler must say so. It must not invent a law for a
   lookup table (the A41 lesson).

## Ladder of abstraction

Kepler works up four levels, and every level is checked before the next one builds on it:

1. **Where is a concept?** (concept atlas: which layers carry which known quantities)
2. **What are the model's own units?** (decomposition into sparse features)
3. **How does it compute?** (mechanisms, located and causally tested)
4. **What does it know?** (laws, strategies: compact statements that reproduce the model, ranked by how much of the
   model they explain, with the part no known concept explains reported as the frontier)
