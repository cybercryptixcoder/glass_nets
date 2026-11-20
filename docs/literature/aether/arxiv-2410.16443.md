# Improving Neuron-level Interpretability with White-box Language Models

- **Key:** arxiv-2410.16443
- **Authors / organisation:** Hao Bai, Yi Ma (UIUC; HKU); the HTML byline also prints UC Berkeley, with a footnote stating the work was done entirely at UC Berkeley. Code and project page at crate-lm.github.io.
- **Date:** 2025-02-27 (v4; v1 2024-10-21)
- **Link:** https://arxiv.org/abs/2410.16443
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Sparse dictionary learning is the field's standard way to make neurons of a language model interpretable, but it is applied after training, so the dictionary is a lossy re-description of representations the model was never asked to make sparse. The authors ask whether the alternative is to make sparsity part of the architecture from the start, so that the units you read are the units the model actually computes with. Their target is a "white-box foundation model": a network whose representation is engineered — by a stated optimisation objective — into sparse, low-dimensional, axis-aligned codes. The claim they want to test is narrow and empirical: under the same size budget, same data and same interpreter, do architecturally sparse neurons explain better than ordinary MLP neurons? The larger goal is to make prediction-under-intervention possible by construction, because a code that is genuinely sparse and linear needs no approximation to be edited.

## What they studied

GPT-2-shaped autoregressive language models (1, 3, 6 and 12 layers; d = 128 or 768, h = 4d), pretrained on the uncopyrighted Pile, with parameter counts matched against GPT-2 (e.g. 81.2M vs 123.6M at the base configuration). The unit of analysis is the **neuron activation** — for their model the sparse code A_t coming out of the ReLU in the ISTA block, for GPT-2 the post-nonlinearity MLP hidden activations. Layers 1-12 are compared individually and averaged, with the final layer excluded in both models. Companion objects: SAEs trained on top of both architectures (width multipliers 1, 4, 16), used as the post-hoc baseline and as a test of whether built-in sparsity leaves anything for an SAE to find.

## How they did it

They take the CRATE architecture (Yu et al.), which is an unrolled optimisation of a coding-rate-reduction objective, and adapt it to next-token prediction: causal mask inside the multi-head subspace self-attention (MSSA) block, and an over-complete ISTA block replacing the MLP, where the code is found by a fixed small number of shrinkage-thresholding steps (t = 2 in practice) against an orthogonal dictionary, with a relocated skip connection so the output is the dictionary times the code rather than the code plus a residual. They pretrain on the Pile with Adam at 100B tokens for the small models and 160B for base, matching GPT-2's token budget. Then they run the Bills et al. auto-interpretation loop on every neuron: an explanation model writes a description from a handful of top-activating excerpts (plus random and quantile-sampled ones), a separate simulation model re-scores excerpts from the text alone using only that description, and the score is the correlation between simulated and true activations. Three metric variants are used — OpenAI top-and-random, OpenAI random-only, and the Anthropic short-excerpt variant — with cheaper open models substituted for GPT-4 and text-davinci-003, and a check that the substitutions do not change the between-model comparison. Controls: a parameter-matched GPT-2, a token-loss-matched earlier checkpoint of their own model, and SAEs trained on both architectures.

## What counts as an explanation here

A natural-language description per neuron, produced by an LLM and validated by whether it lets another model reproduce the neuron's activation profile. The explanation is partial by construction — one sentence per unit, no composition, no circuit — and its predictive content is exactly the simulation correlation. The architectural claim is a structural rather than an explanatory one: the code is meant to be the model's own sparse representation, so that a description of a unit is a description of computation, not of a post-hoc decomposition. Sparsity here is not a descriptive statistic but an objective term, and the paper's stronger claim is that steering a unit is loss-free because no encoder/decoder round trip is involved.

## How they judged success

Auto-interpretation score (Pearson correlation between simulated and true activations, averaged over neurons, layers and models), its **variance across layers** as a steadiness measure, the three metric variants as sensitivity checks, an SAE gain test (score with SAE minus score without, per architecture), a next-token validation loss curve against GPT-2 as the capability control, and a single-checkpoint ablation separating interpretability from training loss. All of it is comparative between two architectures under the same interpreter, which is the design's main virtue and its main limit.

## What they found

- Interpretability is higher for CRATE than GPT-2 on all three metrics at 2, 3, 6 and 12 layers; the headline is a **103% relative improvement** on OpenAI random-only at 6 layers. The 1-layer model is the exception and ties.
- Layer-wise variance is much lower for CRATE at the larger sizes — e.g. at 6 layers, random-only variance 1.87 vs 18.39 and Anthropic 2.01 vs 32.56 — so the gain is not one lucky layer.
- Capability is worse: validation loss 2.29 vs GPT-2 base's figure on the curves, training loss converges more slowly, and a base model of 81.2M parameters is compared against GPT-2's 123.6M for the same depth.
- The SAE comparison is mixed and honestly reported. On the long-context OpenAI metrics the GPT-2 SAE matches GPT-2 and stays behind CRATE; on the short-excerpt Anthropic metric **the SAE beats CRATE** and the authors read this as the metric rewarding monosemanticity that sparse codes have anyway.
- The delegation test (Table 3): adding an SAE to GPT-2 buys more than adding one to CRATE on both OpenAI top-and-random (+6.5 vs −10.2) and Anthropic (+38.1 vs +34.8), which is their evidence that the built-in code is already closer to what the SAE would be searching for.
- The checkpoint ablation: a later, better-loss checkpoint scores at least as well (6.8 vs 6.6 average TaR), and the 2-layer model beats the 1-layer on both loss and interpretability — so the gain is not a by-product of being worse.

## Limits and open problems they name

Performance is below GPT-2, which they attribute to the sparsity penalty, and they flag the performance-versus-interpretability trade-off as the central open problem. Their own appendix admits the activation-sparsity figure is counter-intuitive — the "sparse" model's evaluated activations come out denser than GPT-2's hidden layer, because the two matrices being compared are not the same object — which is a real comparability caveat rather than a footnote. The steering claim is supported qualitatively (activating a neuron and reading logit changes) with no quantitative fidelity measure. Training is expensive (32 GPUs for four days).

## What we take from it

- **A method** for the prediction-under-intervention problem: if the code is the computation, an edit needs no reconstruction, so an intervention's expected effect is computable in closed form rather than estimated through a lossy surrogate. This is the strongest argument in the folder for designed-in sparsity over post-hoc dictionaries.
- **A measure** worth copying: report the **variance of the interpretability score across layers**, not just its mean. A mean score can be produced by a couple of excellent layers and tells you nothing about whether reading a random layer works — and the paper's steadiest comparisons are all in that column.
- **A warning** about auto-interpretation as a yardstick: the SAE wins one metric and loses two on the same models, and the explanation in the paper is that the short-excerpt metric rewards sparsity per se. Any comparison of two representation schemes through an LLM interpreter has to print the per-metric profile, because the metrics disagree in sign.
- **A shortcut for our own read-outs:** the delegation experiment is cheap and general — fit the post-hoc dictionary you would have fitted anyway, and if it adds nothing, that is direct evidence the representation already expresses what you were looking for.

## Tags

unit: neuron | layer
form: dictionary | description
method: training-interpretable | decomposition | weight-analysis
eval: human-rating | prediction/simulation | downstream-task
goal: understanding | control/editing | evaluation-of-methods
