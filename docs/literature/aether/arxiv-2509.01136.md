# Heads or Tails: A Simple Example of Causal Abstractive Simulation

- **Key:** arxiv-2509.01136
- **Authors / organisation:** Gabriel Simmons (single author; the printed contact address is a UC Davis e-mail, no institutional byline or acknowledgements section appears in the PDF).
- **Date:** 2025-09-01 (v1, single version; no later revision listed)
- **Link:** https://arxiv.org/abs/2509.01136
- **Kind:** paper (a 14-page technical note, not an empirical study)
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

The author wants a formal answer to the question "does this language model simulate that system?" that does not rest on the benchmark du jour. The starting observation is that practitioners in the language-model-simulation line of work (social-science replication, generative agents, role-play claims) disagree about whether models simulate at all, and that the disagreement is usually about the untested assumption rather than the evidence. The proposed resolution is to reuse the existing formal apparatus of causal abstraction and use it to define what it means for a simulator to stand in for a referent *for a particular observer*, so that simulation claims become provable or refutable relations between causal models rather than aggregate score improvements. The note explicitly stages itself as illustrative: it does not argue for the hypothesis, only shows how to operationalise it, and the arguments are promised in future work. A secondary aim is philosophical — to give a precise reading of the claim that models are role-playing, by making the observer's interpretive work part of the definition rather than an implicit assumption.

## What they studied

Nothing empirical in the modelling sense: no networks, layers, features or datasets are examined. The objects of study are *formalisations* — a small hand-built causal model of a coin toss as the referent, a language model recast as a probabilistic causal model (input tokens and per-step random draws as exogenous variables, emitted tokens as endogenous ones), and an observer consisting of a causal model of the referent plus explicit maps: a distribution connecting referent high-level states to sim inputs, and a state-mapping function from sim outputs back to referent states. The unit of analysis is therefore the **triple (simulator, referent, observer)**, and the worked example is a single prompt family — three phrasings of a coin flip — answered with one token. One empirical footnote does report a small run of a real model against these definitions, but no model name, size or prompting protocol is given.

## How they did it

The note proceeds by definition-then-example. First it fixes a causal-model tuple (exogenous and endogenous variables, ranges, structural equations, admissible interventions) and its probabilistic version with a distribution over contexts. Then it states the central hypothesis — a simulator simulates a referent *for an observer* when the observer's causal model of the referent is a causal abstraction of the simulator — and defines the three pieces it needs: a referent as its true causal model, a simulator as a probabilistic causal model whose exogenous variables are split into those the observer controls (the prompt) and those it cannot (the sampling randomness), and an observer as a tuple carrying its own causal model of the referent, a prior over referent contexts, an intervention distribution, the stochastic map from referent states to prompts, and the deterministic map from output tokens back to referent states. The coin is then written out concretely: an exogenous heads-causing or tails-causing state, an endogenous landing side, a deterministic structural equation, a uniform prior standing in for human ignorance of toss mechanics, and an intervention that sets the coin down on a chosen side. The language model is written out equally concretely: greedy or top-2 sampling is folded into a deterministic function of the next-token distribution and a uniform draw, so the whole generator becomes a causal model with a fixed input length and output length. Independence is assumed between the prompt and the sampling draws, and the rest of the note works with a non-intervening observer so that prompts are marginalised over referent states through the observer's map. Finally the simulation criterion is written as a commutativity equation — the observer's distribution over referent states must equal the push-forward of the simulator's output distribution through the state map — plus an approximate version with a distance and a tolerance, and four worked cases are computed by hand against it.

## What counts as an explanation here

Not a mechanistic account of the network; an **existence claim of a specific algebraic kind**. The "explanation" is that a stated causal model of the phenomenon, together with a stated interpretation of the model's outputs, commutes with the model's own input-output distribution — i.e. a proof obligation, dischargeable exactly or up to a stated tolerance. It is deliberately partial: it can predict the joint distribution over the abstract variables under the observer's prompting policy and nothing about internals, and it quantifies over observers, so the same model can satisfy the relation for a lenient state map and fail it for a strict one. This is the causal-abstraction form in its purest version in this folder — a replacement model over the abstract variables with an explicit, inspectable interpretation layer instead of a fitted one.

## How they judged success

Total variation distance between the referent distribution entailed by the observer's model and the distribution obtained by pushing the simulator's outputs through the observer's state map, compared against a tolerance. Failure is diagnosed into three sources the formalism separates cleanly: the sampling function (a model whose per-step distribution is close to correct is shown to fail categorically under greedy decoding while a stochastic decoder recovers it), the model's distribution itself (a strongly skewed distribution fails even with a good decoder while a balanced one succeeds), and the observer's state map (a model that emits single-letter answers fails a map that only recognises full words, and succeeds under a widened map that accepts both spellings). The success case is proved by computing both sides and showing them equal. There is no benchmark, no baseline comparison, no held-out split, and no control beyond the arithmetic itself.

## What they found

The definitions are consistent and the four cases separate as designed: one failure is attributable to decoding, one to the model's probabilities, one to the interpretation the observer applies, and the fourth computes to equality and hence counts as a simulation for that observer. The single empirical footnote backs the decoding point: a model with a mildly heads-biased distribution scores a total variation distance of 0.500 with greedy decoding — the maximum for a two-outcome problem — and about 1.06 × 10⁻² (with a standard deviation of 3.0 × 10⁻³) with top-2 sampling, on the same underlying probabilities. That gap is the note's most useful concrete number: identical model, identical distribution, one decoding choice apart, and the verdict flips from categorical failure to near-exact simulation.

## Limits and open problems they name

The hypothesis itself is assumed, not defended; the author says explicitly that arguments for it come later. The formalism is worked only for single-turn, fixed-length input and output and never uses an intervention in the worked example (the observer's intervention distribution is a point mass on doing nothing), so the causal part of causal abstraction is set up but not exercised. Observers are treated as given, which means the criterion inherits the observer's model of the referent — including whatever it gets wrong — and there is no procedure for choosing the state map or the tolerance other than stipulation. The note names three extensions it does not develop: padding to handle variable-length input and output (with the open question of whether padding tokens may causally influence generation), a per-turn decomposition of multi-turn dialogue into a trajectory of simulation quality, and sequence-level search decoders like beam search, whose treatment it defers. Beyond what the author names: no LLM is examined, the empirical footnote carries no model identity or sampling protocol, and the abstraction relation is checked at the level of the output distribution rather than of internal states, so nothing here is white-box.

## What we take from it

Framing and a measure, plus one sharp warning, and it is a *shallow* entry — a note, not a study.

- **Framing**: a simulation/abstraction claim is a triple, and the observer's interpretation layer is part of the claim. For our predictive white-box model this is a clean way to state what we are trying to predict: not "the behaviour" but "the behaviour under a declared interpretation, for a declared prompt distribution". Two of our recurring measurement failures — a probe that separates states nobody can name, and a metric that improves because the tolerance moved — are exactly what an undeclared observer hides.
- **Measure**: total variation distance between the abstract model's distribution and the model's distribution pushed through the state map, with a stated tolerance. This is the honest general form of every "KL recovered" figure in this folder, and it makes the interpretation explicit rather than smuggling it into the metric's implementation.
- **Warning (the transferable one)**: the decoding-step result is a reminder that in these batteries the *decision procedure* is as much a part of the measurement as the model is. The same probabilities failed outright under argmax and passed under stochastic sampling. Any battery we publish that scores a network by its most-likely answer is reporting the interaction of the model with the decoder, and a reported failure should be re-run under a stochastic read-out before it is blamed on the network.
- **Limitation to record**: interventions are defined and then not used, which is the same gap the folder has documented from the other side (a causal abstraction is only informative where it is exercised under interventions that the abstract model predicts). This note supplies the vocabulary for testing whether a declared abstract model predicts held-out interventions; it does not perform that test.

## Tags

unit: whole-model
form: causal-abstraction, replacement-model
method: other (formal definition and hand computation)
eval: prediction/simulation
goal: understanding, evaluation-of-methods
