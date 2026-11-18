# Are Language Models Aware of the Road Not Taken? Token-Level Uncertainty and Hidden State Dynamics

- **Key:** arxiv-2511.04527
- **Authors / organisation:** Amir Zur, Atticus Geiger, Ekdeep Singh Lubana, Eric Bigelow — Pr(Ai)²R group, Stanford Linguistics (Zur); Goodfire (Geiger); Harvard Center for Brain Science and NTT Research Physics of Intelligence group (Lubana); Harvard Psychology and NTT Research (Bigelow)
- **Date:** 2025-11-06
- **Link:** https://arxiv.org/abs/2511.04527
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Estimating how unsure a reasoning model is at a given moment of a chain of thought is hard, because a single sampled token can send the model down a completely different reasoning path — the uncertainty is over continuations that were never generated, not over the finished text. The authors ask whether the model internally represents those untaken paths. Their instrument for the ground truth is the expensive one: Forking Paths Analysis (their own earlier work) estimates a per-token outcome distribution by resampling continuations at that position, costing on the order of millions of generated tokens for one trace. So the paper's larger goal is amortisation — can hidden states stand in for that simulation, both as a predictor of the outcome distribution and as a guide to where an intervention will actually take effect? It sits between the folder's belief-readout line and its steering line: the same trace, read and written at the same positions.

## What they studied

One small instruct model, Llama-3.2 3B Instruct, prompted zero-shot with a "think step by step" prefix, with Gemma-2 2B Instruct used only as a second embedder for a control in the second experiment. Data are only a handful of multiple-choice reasoning items drawn from GSM8K, AQuA and GPQA — the paper is explicit that the method it is pegged to is too expensive to scale past a few examples, and item selection keeps a question only if its majority answer appeared 4–6 times in 10 samples, i.e. where the model is between 40% and 60% confident. The unit of analysis is the **residual-stream activation at a single token position** along a base path, with a difference-in-means direction over a middle layer (the worked example uses layer 12 at token 200) as the object both read and written.

## How they did it

(1) **Ground-truth uncertainty.** Sample one base path with its token probabilities; at each token index, re-prompt the model with the base prefix plus each of the top-N alternate tokens at that position and resample S continuations; extract a final answer from each continuation (an appended "therefore the answer is" scaffold) and aggregate into a weighted outcome distribution over answers per token index, weighting by the token and continuation probabilities. A Bayesian change-point detector then assigns each token index a probability of being a change point. (2) **Does uncertainty predict controllability?** Build a difference-in-means steering direction for a target answer from n = 500 generations ending in that answer against 500 ending elsewhere, picking the (layer, token index) whose direction gives the best held-out linear classification accuracy as a separator, then add that same fixed direction to every token of the base path and sample k = 10 continuations. Steering success is the fraction ending in the target answer minus the base path's own outcome probability for it — a residual-outcome measure, not a raw success rate. Success is then plotted against the base outcome probability across positions on log–log axes and correlated. (3) **Can hidden states replace the resampling?** Train a linear probe at each token index to regress the full outcome distribution from the activation, using KL divergence to the distribution as the loss and reporting held-out KL. The control is the same probe trained on another model's embedding of the same text at the same token — the hypothesis being that if the two embedders carry the same semantic information, any gap between the KLs is model-specific information not recoverable from the text.

## What counts as an explanation here

A **predictive read-out plus a controllability map**, not a mechanism. The object claimed is that the hidden state at a token carries a compressed version of the model's distribution over future answers — a distributional prediction about what the model would do from here, licensed and calibrated against an explicit resampling ground truth. This is a genuinely predictive form of explanation, the kind our goal wants, but at a very small scale: one direction per answer, one layer, one position, and a distribution over the (four) answer options of a multiple-choice item. Nothing is said about which components compute the distribution, and the paper does not claim the state is legible.

## How they judged success

Held-out classification accuracy for choosing the layer and token of the steering direction (about 80% for the position used in the main example); Pearson correlation between steering success and base outcome probability across token positions (R = 0.57 on the main example, R = 0.64 averaged over the four examples, both computed on log–log axes); and held-out KL divergence for the distribution-predicting probes, compared against three references — a probe on a different model's embeddings, a uniform-distribution baseline (1.53) and a one-hot majority-class baseline (0.85) — with the layer curve reported per layer rather than a single best layer. There is no ablation of the direction's magnitude, no seed repetition, and no statistical test; the correlation is a descriptive number, and the paper's own Discussion treats it as indicative.

## What they found

- **Controllability tracks uncertainty, and the whole effect can vanish at one token.** In the worked AQuA example, the base path reaches the wrong answer D while the correct answer is B; at token 114 ("loss") the outcome distribution sharply shifts away from B, and steering success at exactly that point drops to nearly zero, while it stays comparatively high before it. The change points found by the resampling and the drops in steering success co-locate, which is the paper's central figure.
- **The correlation is moderate, not tight:** R = 0.57 on the worked example, 0.64 averaged over the four examples, on log–log axes — so uncertainty at a position is a useful but partial predictor of whether an intervention there will work.
- **The best separating positions are not the informative ones for control.** The highest-accuracy separating (layer, token) pairs sit late in the trace and in middle layers (~80% classification at layer 12, token 200 in the example) — i.e. near the end, where the answer probably already appears in the text — while the paper's control question concerns positions earlier in the reasoning. The authors note this mismatch explicitly rather than resolving it.
- **Hidden states beat another model's embeddings at predicting the outcome distribution:** held-out KL 0.11 against 0.19 at layer 8, with both probes most predictive in the middle layers (around 6–10 of either model). Both beat the uniform (1.53) and majority (0.85) baselines. The gap persists and widens in later layers, where the second model's probe degrades, which the authors read as evidence of model-specific decision information not present in the text embeddings.
- **The honest scale of the result:** ten AQuA items for the prediction experiment, four items for the steering experiment, one base path per item. The paper does not present these as benchmarks.

## Limits and open problems they name

The resampling ground truth costs millions of tokens per trace, which is why only a handful of examples are used and why the paper is framed as a demonstration rather than an evaluation. The steering direction is a single difference-in-means vector per answer selected by separator accuracy, so the choice of layer and token is fit, not given, and the reported correlation is computed on the same small set of items the direction was built from. Correlation between uncertainty and controllability is not causation: the paper shows that positions where the model is uncertain are positions where steering works, not that uncertainty causes steerability. There is no limitations section in this build; the open problem it points at is whether the prediction can be made cheap enough to run at scale, which is exactly the folder's question about amortising expensive causal ground truth with internal read-outs.

## What we take from it

- **A cheap-good/best/bad triple for a predictive read-out**: a baseline that ignores the model (uniform), a baseline that ignores the item (majority class), and the control that isolates the claim (another model's embedding of the same text). The third is the transferable one — it prices "is this the model's state or just the meaning of the words" with one extra probe, and it is the same logic as `arxiv-2611.09945`'s adapter-off oracle line and this folder's repeated warning that a read-out can be reading the prompt.
- **Report steering success as a residual over the base distribution, not a raw rate.** Because the base outcome probability varies wildly along the trace, a raw success rate confounds "the direction works" with "the model was already likely to answer that". Subtracting the base probability makes the curve comparable across positions, and it is the difference that lines up with the change points.
- **The steering-curve-versus-belief-curve coincidence is the folder's missing synthesis, at miniature scale.** The paper plots, on one axis, how uncertain the model is at each token (from resampling) and how much an intervention moves it — and finds the intervention weakest exactly where the model has committed. That is `arxiv-2603.05488`'s committed-belief curve and `arxiv-2603.17199`'s cue curve on the same axis as an *actionability* curve, and it is a shape our own write-ups should adopt: for every position we claim to be able to intervene on, print the base outcome probability beside the effect size.
- **A warning about where the direction was fit.** The best-separating position is chosen by accuracy on the same kind of data, lands late in the trace, and is then applied everywhere — so "we can steer at any position" is really "we can steer with a direction fit near the end, applied earlier". This is the read-out-layer/actuation-layer separation `arxiv-2608.28859` reports and a reminder that choosing a direction by any held-out score still requires a separate actuation test at the positions of interest.
- **Scale discipline worth copying in the other direction**: four items is far too few for any general claim, and the paper is careful not to make one — but a reader skimming the abstract could easily take "activations predict the future outcome distribution" as a general result. The habit is to write the item count into the sentence that carries the claim.

## Tags

unit: direction | layer
form: probe | steering
method: probing | steering | ablation
eval: prediction/simulation | ablation
goal: understanding | evaluation-of-methods
