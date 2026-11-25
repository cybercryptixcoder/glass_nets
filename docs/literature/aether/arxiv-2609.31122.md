# LocUS: Head Selection and Subspace Projection for Targeted Activation Steering

- **Key:** arxiv-2609.31122
- **Authors / organisation:** Irene Tallini, Lorenzo Basile, Valentino Maiorca, Francesco Locatello, Alberto Cazzaniga — Université Côte d'Azur / Inria (Maasai team) and Area Science Park, Trieste; IST Austria
- **Date:** 2026-09-25
- **Link:** https://arxiv.org/abs/2609.31122
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1, 59,973 clean chars / 1,640 lines; abstract, §1–§7, Limitations, Appendices A.1–A.4 and C, Tables 2, 3, 4, 5, 6 read cell by cell; Appendix B and Figures 1–6 read from prose and captions only, not inspected as images)
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Difference-of-means steering takes the offset between mean activations of two contrastive sets and adds it at every layer of the residual stream. The authors' complaint is that this vector inherits every axis along which the two sets differ, not just the target property, so the edit carries whatever was spuriously correlated with the labels into the model. Their first instinct — localise the edit to fewer heads — is, as they demonstrate in their own ablation, not enough: a difference-of-means update confined to selected heads is still unconstrained in direction. So the paper's aim is a two-axis localisation, a sparse set of attention heads and, inside each, a sparse subspace, where both are chosen by the model's own output vocabulary rather than by the contrastive data. The larger goal is a steering operator whose side effects are structurally limited rather than empirically monitored.

## What they studied
Three instruction-tuned and base model families of 7–8B — Mistral-7B-Instruct-v0.3 / v0.1, Llama-3.1-8B-Instruct / base, deepseek-llm-7b-chat / base — plus a scaling arm on deepseek-67B. Unit of analysis is the (layer, head) pair and, beneath it, the raw per-head output before the output projection; the intervention is on the raw head output at the last token position. Three behaviours are steered: toxicity rate on 2,546 TET prompts, positive-sentiment rate on IMDb continuations, and a 0–100 sycophancy trait score on the Chen et al. persona data, judged by Qwen2.5-32B-Instruct.

## How they did it
The instrument is a token set, not a direction. A property dictionary is built automatically from the fit split by scoring each word by its pointwise contribution to the KL divergence between the property-bearing and property-free unigram distributions, keeping the top 70 words in each direction, giving 139–140 words that index columns of the unembedding matrix — directions that decode onto those tokens at the final logits.

For head selection, each head's output is pushed through the final RMSnorm and reconstructed from a sparse subset of those columns with Simultaneous Orthogonal Matching Pursuit, which forces one shared atom set to explain all calibration examples rather than fitting each example separately. The explained-variance ratio of that reconstruction ranks heads, and heads above μ + 2σ of the model-wide distribution are kept; the count that yields (36 to 49 in the 7–8B cells, 166 to 186 at 67B) becomes the steering site count.

For the subspace arm, because the dictionary lives in residual-stream space but the intervention is wanted on the raw head output, they push it backwards through the output matrix and a linearised RMSnorm — the per-token reciprocal norm replaced by a calibration-set average — which gives a per-head dictionary in raw-head space, a logit lens read at the head rather than at the residual stream.

The steering vector is then the projection of the difference-of-means update onto the per-head atoms, applied only at heads above threshold, and the whole update is rescaled so its total magnitude matches the unrestricted baseline's. Ablations separate the two axes: head selection only, subspace projection only at the residual stream of every layer, projection at every head without selection, and both. The coefficient is the single swept quantity and is selected per cell on a validation split under a capability gate.

## What counts as an explanation here
A localisation with a geometric constraint: a small named set of (layer, head) sites and a small named token set per site, both derived from the model's own weights plus a keyword dictionary, with a causal claim that edits confined to them move the target behaviour. It is partial by construction and the paper says so — it explains where to intervene and in what subspace, not how the property is computed. The prediction the form licenses is narrow and real: which heads will transmit an edit on this property, and which directions inside them will not bleed.

## How they judged success
Three numbers per cell: the target metric on a held-out test split, MMLU 5-shot on a held-out half of the 57-subject set, and wiki perplexity over 20k article openings. The operating point is chosen by a two-condition gate on validation — perplexity at most twice unsteered, MMLU at least 0.99 of unsteered — and then the lowest admissible target. Baselines are unsteered, layer-wise difference-of-means, and the two published head-level methods ITI (per-head logistic probes selecting the same k heads) and SMH (one layer, heads aligned with the steering vector). Fairness of the head-level subspace arm is protected by equalising parameter count — intervening on the raw head rather than the projected output so H heads do not cost H×d.

## What they found
Head selection alone carries most of the gain and beats layer-wise difference-of-means in every cell of the ablation, while projection on top of it protects MMLU further. Projection without selection fails in the two predictable ways: at every head it loses MMLU as difference-of-means does, at every layer's residual stream it keeps MMLU but understeers.

Against difference-of-means and against ITI, the full method has the lower target in eight of nine cells, at 92% to 99% relative reduction on Mistral and deepseek for toxicity and sentiment (Mistral toxicity 0.237 to 0.020, sentiment 0.630 to 0.009) and much smaller effects on sycophancy (Mistral 66.5 to 52.0; deepseek 57.0 to 14.9). The ninth cell is sycophancy on Llama-8B, where the score moves the wrong way (80.9 to 83.8) and where, the authors note, no variant at any strength moves it, so they file it as a difference-of-means failure rather than a method failure.

The price is fluency: the chosen operating point is the one that keeps MMLU at the unsteered value, and in most cells it is paid in perplexity instead (Mistral sentiment 6.8 to 11.6, deepseek sycophancy 10.7 to 20.4, Llama sentiment 8.5 to 13.6, Llama toxicity 9.4 to 14.3, all still within the twice-unsteered gate).

On the external-corpus test — dictionary, head scores, subspaces and vector all fitted on Jigsaw rather than TET — the method still beats difference-of-means on all three models (85/75/99% against 60/72/89%) with MMLU within 0.005. At 67B about 3% of 6,080 heads are selected and the ordering holds on all three tasks (toxicity to 0.000 against 0.020, sentiment 0.009 against 0.387). Total degrees of freedom used are 0.4–0.8% of an unprojected layer-wise update.

The paper's own explanation for why EVR selection beats probe selection is a dissociation worth recording: Fig. 5 shows the property is linearly decodable from a broad set of heads, while Fig. 2 shows only a few write strongly onto it — probes rank by detection, explained variance ranks by expression, and the causal arm wants the second.

## Limits and open problems they name
The method only applies where the property is carried by some lexical difference, so a property with no characteristic vocabulary has no dictionary and no atoms. Fluency degrades more than factual recall, and capability is measured only by MMLU and perplexity, so open-ended reasoning is not covered; the authors propose a fine-grained controllability benchmark as future work. The composition with a better vector estimator is left open, as is varying the curation level of the dictionary — the one lever they suggest for fine-grained control, since removing demographic atoms when steering toxicity is a design choice the current experiments do not study.

Two gaps the text does not name are visible in its own tables. The per-head projection uses a linearised RMSnorm with a corpus-averaged reciprocal norm, and no experiment varies that approximation. And the head-count rule is a threshold on a model-wide distribution, so the cardinality is an artefact of how peaked the explained-variance distribution happens to be in that model — 36 heads here, 44 there, 166 at 67B — with no size sweep on the selection side to say whether the count matters.

## What we take from it
- **The unembedding is a free per-unit dictionary, and reading it at the head rather than at the residual stream is the transferable trick.** Pushing W_U back through the output matrix and a linearised norm gives, for every head, the small set of final-logit directions that head can actually reach. Our white-box model needs exactly this object to say what a unit is *for* in the reader's terms rather than the writer's, and the cost is one matrix multiply per head with no training and no forward pass.
- **Detection and expression are different head rankings, and the causal arm wants expression.** A logistic probe ranks heads by how well the property can be read off them; explained variance ranks them by how much of their write lands on the property. Only the second beats the baseline in the steering cells. This is the folder's cleanest statement that a decodable unit is not a writable one — and it is the same dissociation earlier entries met between probe accuracy and patch effect, here imported into a method rather than reported as a caveat.
- **Localising the site does not constrain the direction, and the ablation proves it.** Projection without selection loses MMLU, selection without projection understeers, and only both do the work. Any intervention our model proposes has to declare both coordinates — where in the graph and which subspace inside it — because either one alone is a different failure mode, not a partial version of the same success.
- **A warning about two summary statistics chosen inside the paper.** The μ+2σ head rule and the n = 20 atom budget are both read off fit-time curves with no sweep, and the budget rule itself (the smallest n at which the average head reconstructs a quarter of its variance) would give anywhere from 14 to 25 across the twelve cells; 20 is their median. That is a defensible default and a weak commitment. When our model ships a hyperparameter justified this way, the cell-by-cell range and the median should be in the paper, because a value that generalises only to the median of a distribution is a value that will be wrong in a third of conditions.

## Tags
unit: head | layer | weights
form: steering | direction
method: decomposition | steering | weight-analysis | ablation
eval: ablation | downstream-task | benchmark
goal: control/editing | safety/auditing | evaluation-of-methods
