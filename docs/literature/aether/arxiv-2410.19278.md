# Applying sparse autoencoders to unlearn knowledge in language models

- **Key:** arxiv-2410.19278
- **Authors / organisation:** Eoin Farrell, Yeu-Tong Lau (equal contribution), Arthur Conmy; MATS Program 6.0
- **Date:** 2024-10-25 (v1); v2 2024-11-02
- **Link:** https://arxiv.org/abs/2410.19278
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Unlearning a hazard category from a language model is normally done by fine-tuning the weights, and the results are hard to inspect: you cannot tell whether the capability was removed or merely hidden behind a new output preference. The authors ask whether sparse-autoencoder features give an intervention that is both effective and legible — you point at a named feature, push it down, and watch what the model says. Their larger goal is verifiable removal at a scale larger than the single-fact editing methods allow, and they are explicit that the interpretable route only wins if it can be shown to actually delete knowledge rather than suppress its expression. They test it on the biology slice of the WMDP benchmark. This entry is the static-clamping baseline that later SAE-unlearning work measures itself against.

## What they studied

Two instruction-tuned Gemma models, gemma-2b-it and gemma-2-2b-it, at residual-stream sites. For gemma-2b-it they trained a 16k-feature SAE at layer 9 on 120M OpenWebText tokens with L0 20, choosing the layer by activation patching on a sample of the multiple-choice questions. For gemma-2-2b-it they used the open Gemma Scope SAEs trained on the base model and applied them to the instruct model. The unit of analysis is the individual SAE feature — its decoder direction and its activation at a token position — and, at the multi-feature end, a selected set of 10, 20 or 50 of them.

## How they did it

Data is the WMDP-bio subset, 1273 four-option questions on biosecurity and hazardous biology. Because random guessing gets about a quarter of four-option questions right, they restrict every evaluation to the questions the base model answers correctly under all 24 orderings of the options — 172 of 1273 for gemma-2b-it and 522 of 1273 for gemma-2-2b-it — so a correct answer is evidence the knowledge is present. The intervention is a clamp: wherever a selected feature is positive, its activation is overwritten with a fixed negative constant −c, and positions where it is zero are untouched; they also tried multiplying by a negative constant and clamping to a multiple of the dataset maximum, and kept the fixed negative clamp because it produced comparable unlearning with fewer side effects, especially over several features. Feature selection has two variants. The first is a sparsity filter: measure each feature's sparsity on a forget corpus (WMDP-bio text) and a retain corpus (WikiText), drop features above a retain-sparsity cutoff (typically 0.01), sort the survivors by forget-corpus sparsity and take the top N. The second is attribution: for each question, take the correct logit minus the mean of the wrong ones, backpropagate to the intervention layer, dot with each feature's decoder column, weight by activation, sum over positions excluding special tokens, take the top 20, clamp each at −20 to check that it moves the answer, then discard features that break more than about 0 to 3 of 300 unrelated MMLU questions. Side effects are measured two ways: accuracy on a hand-picked MMLU subset (high-school US history, geography, college computer science, human aging), again restricted to the all-24-permutation survivors (97 questions for the small model, 300 for the 2B), and cross-entropy loss added over 50k OpenWebText tokens. The comparison arm is RMU, swept over steering coefficients 100/200/400, retain-loss weights 100/300/500 and layers 3/7/11.

## What counts as an explanation here

A named feature plus a measured dose-response: the claim is that pushing feature #9163 down lowers the probability of the correct option on this kind of question, with the loss added staying near zero, and that the response is specific because randomly chosen features that fire on the same prompt do not produce the same trade-off. The explanation is local and interventionist rather than a complete account — nothing here says how the feature's suppression propagates, and the authors say the knowledge is still in the weights since only activations are modified. It is a predictor at the level of "what changes at what clamp strength" for one question and one feature, not a model of the mechanism.

## How they judged success

Three named metrics: WMDP-bio accuracy as the fraction of all-permutation-correct questions still answered correctly, which is the unlearning score; loss added on 50k OpenWebText tokens, read as the general-damage axis; and selected-MMLU accuracy, read as the unrelated-knowledge axis. Plots are Pareto fronts of loss added against WMDP accuracy and MMLU accuracy against WMDP accuracy, with the RMU sweep as a cloud of comparison points. For the single-feature case they additionally count how many of the 24 option orderings were unlearned, on the argument that a truly removed knowledge should look like guessing rather than a consistent wrong answer. The encoder-versus-decoder experiment is the control that matters: they keep the original encoder for detection but clamp a *random* feature whenever a selected feature is positive, so the intervention has the same locations and the same magnitude but the wrong direction.

## What they found

gemma-2b-it scores 560/1273 (44.0%) on WMDP-bio at baseline. On the single-feature case study, clamping #9163 to zero does nothing, the probabilities barely move through −5, the correct option's probability starts falling between about −10 and −15 and the answer flips to the model's preferred wrong option by roughly −18 to −20 with negligible loss added, and beyond that the probabilities stop changing while loss added rises by orders of magnitude; the unrelated MMLU set is undamaged out to −30. That question's unlearning holds for 19 of 24 orderings, and across features and prompts the permutation count ranges from 2 to 24. In the multi-feature comparison on gemma-2-2b-it at layer 3 (16k features, L0 about 59), 10 features is the most effective setting — cheaper in loss added than 20 or 50 — while on the MMLU axis feature count shows no clear trend. Against RMU at matched WMDP-bio reduction, the SAE method costs more MMLU accuracy but adds less cross-entropy loss on OpenWebText. Replacing the decoder directions with random ones substantially degrades unlearning on gemma-2-2b-it, whereas the same swap on gemma-2b-it leaves unlearning intact — a disagreement between the two models that the authors report without resolving.

## Limits and open problems they name

The headline limit is the one they lead with: on side effects the method is similar to or worse than RMU, and they conclude that either SAE quality or the intervention technique would need substantial change before SAE unlearning is competitive with fine-tuning-based methods. Only activations are modified, so the knowledge survives in the weights — the authors note this may still be useful for API-only settings. Three future directions are named: how SAE width and quality affect unlearning, a head-to-head against activation steering from paired bio and non-bio prompts, and how the intervention propagates into features at later layers. Two further limits are visible in their own text: the multi-feature results are figure-only Pareto plots with no tabulated numbers, and the two models disagree on whether the decoder direction carries the effect, so the mechanism claim is unstable across scale.

## What we take from it

Three things, in order of usefulness. First, **a direction, not a magnitude**: ablating a feature to zero changed nothing and a small negative clamp changed nothing, while the answer flipped as the clamp crossed a threshold band and then the probabilities stopped moving — so "this feature matters" is a claim about a signed push past a threshold, and a zero-ablation baseline cannot falsify a feature's causal role. Any ablation we run has to include the negative-clamp arm before we conclude a unit is inert. Second, **the permutation-surviving subset as a knowledge gate**: because four-option chance is 25%, they only count questions stable across all 24 orderings, which converts accuracy into evidence that knowledge was present, at the price of discarding 87% of the benchmark for the small model. That is the right shape for our evaluation and we should report the retained fraction beside every number, as they do. Third, **the random-decoder control as the template for testing whether an intervention works by direction or by disruption** — same positions, same magnitude, wrong vector — and their honest report that it disagrees across the two models is the warning that comes with it.

## Tags

unit: feature
form: steering
method: SAE | ablation | steering
eval: ablation | downstream-task
goal: safety/auditing | control/editing
