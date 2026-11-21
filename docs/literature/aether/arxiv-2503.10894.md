# HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks

- **Key:** arxiv-2503.10894
- **Authors / organisation:** Jiuding Sun, Jing Huang, Christopher Potts (Stanford University); Sidharth Baskaran (Confirm Labs); Karel D'Oosterlinck (Ghent University); Michael Sklar, Atticus Geiger (Pr(Ai)R Group) — affiliations from the byline block.
- **Date:** 2025-03-13 (v1); this entry reads v3, 2025-04-25
- **Link:** https://arxiv.org/abs/2503.10894
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Distributed alignment search is the strongest supervised route to a localised concept feature, but it has an unstated dependency: someone has to decide *where* to fit. In practice that means sweeping layers and token positions, and the field has settled on heuristics (intervene above the last entity token) that the authors note are known to fail for some entities. Their goal is to move the site from a hyperparameter to a prediction: give the method a natural-language description of the concept and let a learned model choose the layer, the token pair and the subspace. The larger aim is the automation of interpretability for models too large to hand-search, and the paper is explicit that the same machinery raises the risk of an interpretability method that steers instead of explaining — so a substantial part of the paper is spent on the fidelity problem rather than on the score.

## What they studied

Llama3-8B as the fixed target model, with the intervention site restricted to the residual stream of one chosen layer (the paper searches layers but each HyperDAS is fitted for a layer). Unit of analysis is a *subspace of the residual stream at a token position*, i.e. a feature located in both space (which token) and direction (which low-rank orthogonal matrix). The task data is the RAVEL benchmark: five entity domains (cities, Nobel laureates, occupations, physical objects, verbs) with 21–23 attributes in total and train/test prompt splits between a hundred and a few hundred examples per attribute; the paper reprints the full per-attribute counts in its appendix.

## How they did it

A transformer hypernetwork consumes three inputs: a short natural-language instruction naming the concept, and the target model's hidden states on a base prompt and on a counterfactual prompt over every layer. Two extra cross-attention blocks per decoder block let the instruction representation read the base and counterfactual activations. The final concept vector then does two jobs. First, it scores every base-token × counterfactual-token pair through a bilinear attention, with an extra "do not intervene" column, and a column softmax turns these into weights; the weighted mix of counterfactual hidden states is the replacement value for each base token. Second, it produces a vector that defines a Householder reflection, which rotates a learned orthogonal rank-128 basis; the rotated basis is the subspace that gets patched. The discrete token selection is softened by the softmax during training and snapped to a one-to-one assignment by a double argmax at evaluation. Training minimises a cross-entropy loss on the RAVEL counterfactual label plus a sparsity penalty on columns of the alignment matrix that exceed total weight one, with the penalty ramped up over the second half of training. Specific fitting choices they describe: masking the base prompt's attribute-mention tokens so the hypernetwork cannot see which attribute the question asks about, and an optional symmetric variant that randomly swaps base and counterfactual order so the same site is used for reading and writing the concept.

## What counts as an explanation here

A located intervention site: a layer, a token pair, and a rotated orthogonal subspace, packaged as a *replacement-model-like* object — the hypernetwork is a programmable map from concept description to intervention, not a description of the network in words. It is partial in the same way DAS is partial (one attribute, one layer, linear subspace), but it is more predictive than a hand-fitted DAS in one respect that matters to us: a trained HyperDAS predicts the intervention site for a concept it was not fitted to individually, and its Householder vectors are inspectable as a geometry of attributes.

## How they judged success

RAVEL's Cause, Iso and Disentangle scores (Cause = the targeted attribute moved; Iso = an untargeted attribute did not move; Disentangle = their average), all computed under one-to-one snapped interchange interventions. Baselines: multi-task DAS at the same layer with the manually chosen token position, an SAE featurizer in the benchmark's own setting with a Cause–Iso width sweep, and four ablations of HyperDAS itself — removing cross-attention, removing the DAS rotation matrix (whole-vector patch), and replacing the transformer concept encoder with a vector lookup. Two fitting regimes are compared: one HyperDAS per entity domain, and one model trained on all domains at once. A layer sweep from the embedding to deep layers reports the Disentangle curve, with the best layer chosen between 10 and 15. Compute, GPU memory, token-selection counts, a sparsity-weight sweep, and a cosine-similarity analysis of Householder vectors by attribute are all reported.

## What they found

- Per-domain HyperDAS beats multi-task DAS on the pooled Disentangle score (84.7 against 76.0), and does so by raising Cause far more than Iso: on cities the Cause score goes from 55.8 to 70.8 while Iso rises from 77.9 to 93.9, and on verbs Cause goes from 74.3 to 93.0.
- The gain is not uniform: on Nobel laureates and occupations the per-domain model is essentially level with multi-task DAS on Cause (55.4 vs 56.0 and 50.4 vs 50.7), while the pooled average is carried by the domains where the entity-token heuristic was evidently wrong.
- A single model trained across all five domains still beats multi-task DAS (80.7) but sits about four points under the per-domain models, and it loses badly on cities (58.8 Cause) — specialisation buys the top number.
- The symmetric variant is not a free improvement: per-domain it drops the pooled score to 76.9, mostly from failing on the verb domain (42.3 Cause), and the all-domain symmetric model collapses to 54.8 with Cause near zero in three domains — a strong hint that enforcing the read-site equals the write-site assumption is an empirical commitment, not a neutral convenience.
- The ablation is sharp: removing the rotation matrix leaves Cause at 0.8 — the method only "works" because it edits a subspace — and replacing the transformer concept encoder with a lookup table drops Disentangle to 31.0, so the cross-attention over target activations is where most of the localisation comes from.
- Different layers pick different sites: counterfactual-side interventions stay on entity tokens at every depth, but base-side interventions are erratic at early layers, become the entity token at middle layers, and at deep layers move onto syntax tokens in JSON-formatted prompts.
- Cost: roughly 2.4× the compute of multi-task DAS for one epoch on the city/country task, but memory that does not grow with the number of attributes, so the pooled all-attribute model needs less memory than the per-attribute baseline.
- Householder vectors cluster by attribute more tightly than by entity type, and latitude and longitude (the one pair RAVEL identifies as entangled) learn near-identical subspaces.

## Limits and open problems they name

The method only succeeds if the concept really is mediated by a linear subspace, and the authors cite work suggesting non-linear mediation exists. They also state plainly that supervised interpretability can produce false positives and that "steering versus explaining" is not fully settled; their mitigations are design choices (masking, a sparsity constraint that forbids blended sources, snapping to one-to-one at test) rather than a guarantee, and they report that with too little or too much sparsity the model finds intervention patterns that score well under weighted interventions and fail under the snapped evaluation. Loading the hypernetwork from pretrained Llama parameters gave no measurable benefit in their appendix, with scaling left open. Whether the attributed site is the site the model actually uses is tested only through the benchmark's intervention score.

## What we take from it

- **Warning, and the most important thing in the paper for us: a learned localiser can score high by intervening on a blended mixture of hidden states, and the only thing that separates that from an explanation is an evaluation constraint that forbids it.** Their snapping of the alignment to one-to-one at test, and their report of the pathological weighted-intervention models, is the discipline we should copy whenever we fit anything to choose its own intervention site: constrain the intervention's shape at test time, and publish the weighted score next to the constrained one.
- **Warning: mask the input feature that tells the method the answer.** Their appendix shows the hypernetwork learning to skip intervention whenever the base attribute differed from the instruction's attribute — a shortcut that makes Iso look excellent while no disentangled subspace is ever learned. Any of our fit-to-localise batteries must hide the label-bearing part of the prompt from the fitting model, and this is the cheapest known control for that failure.
- **Method to reuse: predict the site, then check the prediction against a grid search.** Giving the localiser a concept description and letting it pick layer, token and subspace is the difference between "we can fit a feature if you tell us where" and a method that generalises across concepts; the placebo is the manual site, which is exactly what their baseline is.
- **Measure: probe the localiser's own geometry.** Cosine similarity among the Householder vectors recovered the known latitude/longitude entanglement without being trained for it, which is a cheap consistency check that a fitted localiser's internal representation of attributes matches the structure the benchmark already reports.
- **Framing: a symmetric method is a stronger claim than an asymmetric one, and the data says the claim is often false.** Their symmetric collapses suggest that reading a concept and writing it need not use the same token, which is a direct caution against our own habit of assuming a single site serves both directions.

## Tags

unit: feature | direction | layer
form: direction | replacement-model | steering
method: DAS | patching | decomposition | training-interpretable | other
eval: patching | benchmark | faithfulness | downstream-task
goal: control/editing | understanding | evaluation-of-methods
