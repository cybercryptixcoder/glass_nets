# Thought Anchors: Which LLM Reasoning Steps Matter?

- **Key:** arxiv-2506.19143
- **Authors / organisation:** Paul C. Bogdan, Uzay Macar, Neel Nanda, Arthur Conmy; MATS (ML Alignment & Theory Scholars) program. Author order determined by coinflip; Nanda and Conmy are joint senior authors.
- **Date:** 2025-10-27 (v4; v1 2025-06-23)
- **Link:** https://arxiv.org/abs/2506.19143
- **Kind:** paper
- **Read from:** full text — HTML v4, 97,190 clean chars; abstract, §§1–9, references, Appendices A–N.
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Interpretability is built for a single forward pass, but reasoning models compute across hundreds of self-generated tokens, and the tools for saying *which part of a trace mattered* are thin. The authors argue that the sentence is the right unit — coarser than a token, finer than a paragraph, and usually close to a human-named reasoning step — and that importance should be defined causally rather than by reading the text. Their larger goal is a principled decomposition of a reasoning trace into steps and dependencies between them, so that reasoning can be debugged and monitored rather than merely displayed. They are explicit that this sidesteps arguments about CoT faithfulness: a sentence's effect is measured by resampling, not by trusting what it says.

## What they studied

DeepSeek-R1-Distill-Qwen-14B (48 layers) is the main subject, with R1-Distill-Llama-8B as a replication; a later section uses Qwen3-30B-A3B on MMLU for scale. The MATH dataset supplies 20 problems solved correctly between 25% and 75% of the time, with one correct and one incorrect trace each (40 traces, mean 144.2 sentences / 4,208 tokens). The unit of analysis is the sentence, typed by an eight-category taxonomy (problem setup, plan generation, fact retrieval, active computation, uncertainty management, result consolidation, self checking, final answer emission) assigned by GPT-4o. A secondary unit is the attention head, via a "receiver head" score.

## How they did it

Three methods on the same traces, then a fourth for scale. (1) **Forced-answer importance** (the baseline): at each sentence, interrupt and append a boxed-answer prompt 100 times and score the answer distribution. (2) **Resampling importance**: at each sentence, sample 100 rollouts continuing from the original prefix without that sentence (base condition) and 100 with it (intervention), and take the KL divergence between the two final-answer distributions. (3) **Counterfactual importance**: the same, but keeping only rollouts whose replacement sentence differs semantically from the original (cosine below the dataset median of 0.8 under all-MiniLM-L6-v2), which removes sentences that could not have been said otherwise. For white-box evidence, they average each head's token-to-token attention matrix into a sentence-to-sentence matrix, score each head by the kurtosis of its column means below the diagonal (ignoring the four nearest sentences), and call high-kurtosis heads "receiver heads". For structure, they suppress all attention to one sentence and measure the average token log-KL on every later sentence, normalised per target column; then they extend this to 2,492 MMLU problems whose non-reasoning accuracy is under 50%, computing a causal graph per correct trace and reading the mean effect on each subdiagonal to get a distance profile.

## What counts as an explanation here

A causal graph over sentences: node size is counterfactual importance, edges are measured influence of one sentence on another, and the graph is annotated by sentence function. It is deliberately partial and black-box-first — the explanation predicts the answer distribution under intervention on a sentence, and no claim is made that the graph is a mechanism. The white-box half (receiver heads) is proposed as convergent evidence for the same structure rather than as the explanation itself. The authors call the result a "tentative CoT circuit" and a proof-of-concept toolkit.

## How they judged success

No benchmark score; the evidence is agreement across independent instruments plus explicit causal controls. Convergence: receiver-head scores, resampling importance and attention-suppression local maxima pick the same sentences in the case study. Reliability: split-half correlation of per-head kurtosis (r = .84), correlation among the top-16 heads' sentence scores (mean r = .56 against r = .35 for arbitrary head pairs), and a paired t-test of receiver scores across sentence types. Validation of the masking approximation against the resampling measure (mean r = .20 across traces, .34 for pairs under five sentences apart, positive in 90% of traces). Causal control: ablating receiver heads versus a **layer-matched random** set of heads at 128 / 256 / 512 heads. Cross-model replication on Llama-8B, smoothing checks on the KL (Laplace α = 1.0, Jeffreys α = 0.5), and a linear probe for the sentence taxonomy (layer 47, group-5-fold CV, macro-F1 0.71) to show the labels are not free text.

## What they found

Plan generation and uncertainty management sentences carry the highest counterfactual importance and receive the most receiver-head attention, while active computation — which the forced-answer baseline ranks highest — gets relatively little (receiver-score gap t > 4.0, p < .001). Sentence frequencies: active computation 32.7%, fact retrieval 20.1%, plan generation 15.5%, uncertainty management 14.0%, problem setup 2.4%, final answer emission 0.7%. The case study's pivotal sentence is a plan to convert to decimal (sentence 13), which the forced-answer measure misses entirely. The causal graph of a solved problem shows a self-correction scaffold (proposal → discrepancy → re-check → explanation of the error). Across MMLU domains, easy questions and maths-heavy subjects produce stronger short-range links and weaker long-range links (r = .44 and r = −.54 with subject accuracy, p < .001); the mean trace length there is 90.1 sentences. Ablating the top 512 receiver heads drops MATH accuracy to 27.7% against 37.3% for an equal, layer-matched random ablation (t[31] = 2.55, p = .02; baseline 64.1%), and 256 heads barely moves either arm (48.8% / 52.7%). In the reasoning model, the highest-scoring sentences get 1.8× more attention through the top-16 heads than in the base Qwen-14B, with no such gap for Llama-8B.

## Limits and open problems they name

Resampling is expensive — 100 rollouts per sentence, 95% CI at worst ±10% — so the black-box measure does not scale to thousands of traces without adaptive budgets. Counterfactual importance goes high-variance when fewer than ten semantically different replacements exist. Sentences can be overdetermined or independently sufficient, which the design does not separate. Receiver-head scores are confounded by **sentence position** in two opposing ways (attention dilution pushes later sentences down, proximity pushes them up), and the authors only mitigate by ignoring the four nearest sentences. Attention suppression forces the model off-distribution. The taxonomy is GPT-4o-labelled from text, and receiver-head location differs across the two models. They present the work as preliminary.

## What we take from it

Three transferable items. (1) **The counterfactual filter as the price of an importance claim**: an intervention whose replacement is semantically identical to the original measures nothing, and here the fix is one sentence — keep only pairs below the dataset's median embedding similarity and report the retention — which is the same discipline the folder has been deriving for contrastive directions, applied to a natural-language unit rather than a vector bank. (2) **A named baseline that ranks a different unit**: forced-answer scoring systematically credits arithmetic over planning because it credits whatever the final answer depends on late, and the paper's disagreement between the two measures is the evidence for its own construct — we should carry that paired-baseline habit into any step- or sentence-level attribution we publish. (3) **The layer-matched random ablation** is the cheapest causal control in the folder: it removes the obvious confound (receiver heads cluster in late layers) and turns "this head matters" into a comparison at identical depth, and the 512-versus-256 reversal in the paper is a reminder that the control has to be run at the strength where the effect is claimed. Also worth recording: reasoning at sentence level is affordable where token-level is not (≈100× cheaper than resampling), which is a pragmatic argument for the unit choice.

## Tags

unit: circuit | feature | layer
form: graph | causal-abstraction
method: ablation | attribution | probing | other
eval: ablation | faithfulness | prediction/simulation
goal: understanding | safety/auditing | debugging
