# Finding Interpretable Prompt-Specific Circuits in Language Models

- **Key:** arxiv-2602.13483
- **Authors / organisation:** Gabriel Franco; Lucas M. Tassis; Azalea Rohr; Mark Crovella (Northeastern University)
- **Date:** 2026-02-13 (v2, updated 2026-05-13)
- **Link:** https://arxiv.org/abs/2602.13483
- **Kind:** paper
- **Read from:** full text (v2 HTML; body plus Appendices A-L, references skipped)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Circuit tracing usually produces one graph for a task, aggregated over a prompt distribution. The authors ask what is lost by that aggregation: if the tracing is done one prompt at a time, do the circuits actually differ, and if so, along which axis of prompt variation? Their larger goal is a mechanistic account of why models behave inconsistently across prompt wordings - which they open by naming as a practical problem, not an aesthetic one. They also want per-prompt tracing to be cheap enough to be a routine unit of analysis, rather than something that needs a counterfactual dataset and many forward passes. The paper is therefore as much about making a per-input explanation affordable as about what it finds.

## What they studied

GPT-2 Small, Pythia-160M and Gemma-2 2B on Indirect Object Identification, Greater-Than and Gender Pronoun (the last two mostly for reproduction against a prior method); multilingual IOI in English, Spanish, French and Portuguese on mGPT and Gemma-2 2B; and illustrative single prompts in Gemma-2 2B for factual recall and in-context translation. The unit of analysis is unusual for this literature: an **attention firing** (a specific head attending from a destination token to a source token), and the object it produces is a graph whose nodes are components and whose edges are **signals** - a component's output projected onto one singular direction of that head's query-key matrix. Circuits are sets of edge-singular-vector pairs, clustered by Jaccard distance. There is no feature dictionary and no learned dictionary at any point.

## How they did it

For a fixed head and token pair, the pre-softmax score is a bilinear form whose matrix is the product of the query and key weights; its SVD gives paired left and right directions. The residual stream is decomposed into per-component outputs, and every (component, singular direction) pair becomes a candidate signal. Candidates are scored by integrated gradients against the post-softmax attention weight, and the solver greedily removes the top-scoring candidate, recomputing the weight, until it falls below a threshold. Signals removed this way form the explanation for that firing. The circuit for a prompt is built by seeding at components with a direct effect on the chosen logit direction and recursing upstream. Two things distinguish this from its predecessor: candidates are formed as a component-by-direction cross product rather than identifying components and subspaces separately, and the objective is the true softmax weight rather than a linear surrogate. A separate appendix shows that attention bias and rotary position embeddings can both be absorbed into the token vectors, leaving a single matrix per head and therefore a single SVD. Signals are then described in natural language by an automated pipeline that retrieves top-activating token-pair contexts from a Pile subset, asks an explainer model for one sentence, and has a judge model compare top contexts against random ones.

## What counts as an explanation here

A per-prompt graph plus per-edge one-sentence descriptions. The causal content is local and attention-specific: a signal is in the explanation because removing it drops *that* attention weight below a threshold. Nothing in the object is required to explain the answer - the paper says explicitly that circuits are built from attention-causal components, and the edges exhaust the pre-softmax score vector but that is an accounting identity for attention, not for the residual stream. The explanation is partial by construction and is meant to predict attention decisions on this input, not outputs.

## How they judged success

Four checks, of unequal strength. Circuit size against the predecessor method (nodes and edges, mean with standard error) while holding the attention-counterfactual criterion. Structural agreement with reference circuits from patching-based methods (Path Patching, edge pruning, ACDC, attribution patching) via precision, recall and F1. Causal effect on downstream task performance of ablating or boosting the selected signals, against random-signal ablation and boosting controls, with the residual-stream perturbation of the intervention reported as cosine similarity and norm ratio. And an interpretability criterion with actual statistics: a one-sided Fisher exact test of whether the judge accepts an interpretation more often on top contexts than on random ones, with false-discovery-rate control across signals at 5 per cent, alongside median judge accuracy with interquartile ranges against a sparse-autoencoder baseline from prior work. Clustering quality is judged by whether the dendrogram blocks line up with known prompt-template labels - a qualitative read of coloured dendrograms, not a scored index.

## What they found

Circuits are 2x to 10x smaller than the predecessor's at the same attention criterion, and most selected signals use a single singular direction. The fraction of signals passing the interpretability criterion varies sharply with model: about 63 per cent in GPT-2 Small, 50 per cent in Pythia-160M, 31 per cent in Gemma-2 2B, with median judge accuracy 0.70, 0.65 and 0.60 against 0.76 for the SAE baseline. Prompt-level circuits cluster, but along *model-dependent* axes: GPT-2 splits cleanly by name-role order, Pythia by surface wording with many small clusters, and Gemma by neither cleanly. The mechanism story is the paper's best result - in GPT-2, the same name-mover head (9,9) carries near-antipodal incoming signals in the two template families, because one template lets the model tag the indirect object as the second item of a parallel pair while the other forces generic structural matching. In multilingual IOI, components are largely reused across languages but signals are language-specific, and cross-language circuit distance correlates with typological distance from URIEL+ for the syntactic family (r = 0.83) and the genetic family (r = 0.88) in mGPT, with 0.80 and 0.84 in Gemma; the other four feature families are not significant. Clustering separability improves as the unit of comparison moves from nodes to edges to edge-singular-vector pairs. A translation trace shows two cross-lingual heads reading the English demonstration noun from a French determiner, which appear only when tracing the correct answer's logit direction and vanish from the runner-up's trace.

## Limits and open problems they name

The threshold is set by a heuristic knee in the distribution of attention weight times context length, and no principled dataset-agnostic rule exists. The recursion stops at attention firings and does not trace what caused an MLP to emit a signal. Interpretations rest on a language model as judge and are called suggestive rather than conclusive, with an explicit warning against treating them as ground truth. The typological correlation rests on six language pairs and is offered as suggestive. From my side of the reading, three additional limits stand out: every validation of causal relevance is on *attention weights or task accuracy*, never on predicting an unseen intervention; the ablation arm removes many rank-1 signals at once, so which signal did the work is not identified; and the clustering evidence is dendrograms read by eye, with no null for "would random circuits also cluster by template".

## What we take from it

The transferable instrument is the *routing-versus-content split applied per input*: holding the component set roughly fixed, the paper measures whether the incoming signal changed, and that is what separates a reused head doing the same job from a reused head doing a different job. Two mechanical rules follow for a predictive white-box model. First, every edge we publish carries the subspace (or equivalently the rank-1 projector) it travelled in, not just its endpoints - because two circuits with identical nodes and identical edge sets can differ entirely in the directions on those edges, and only the direction-level representation recovers the template split. Second, an intervention on a rank-1 signal perturbs the residual almost not at all (cosine near 0.99, norm ratio near 1) while still moving attention and task accuracy, which is the cheapest available demonstration that small, targeted, in-distribution edits are enough to test a hypothesis. The warning to carry is the opposite of the paper's own framing: a single-forward-pass explanation with no counterfactual input is *unfalsified* on new prompts until it is converted into a prediction-and-check, and the paper never runs that check - so we should treat per-prompt circuits as hypotheses to be scored against held-out interventions, not as results.

## Tags
unit: head
form: graph
method: attribution
eval: ablation
goal: understanding
