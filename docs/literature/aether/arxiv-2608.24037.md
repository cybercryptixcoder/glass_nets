# Curved Inference II: Sleeper Agent Geometry — Extending Interpretability Beyond Probes

- **Key:** arxiv-2608.24037
- **Authors / organisation:** Rob Manson (independent; robman.fyi)
- **Date:** 2026-08-25 (v1; document dated 2025-07-31 internally)
- **Link:** https://arxiv.org/abs/2608.24037
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1, ~98k chars of extracted page read end to end: abstract, §1–§2, §3.1–§3.6, §4.1–§4.4, §5–§8, references, and Appendices A–E including the full unnormalised-trajectory formalism, the prompt table, the inter-rater reliability block, the test-selection protocol and the complete per-strategy statistical tables). Figures 1–5 and the metric share tables were not inspected as images; every number below is a prose sentence or a table cell, and several formula-bearing quantities are blank in the HTML text layer, so I quote only what is legible.
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Anthropic's sleeper-agent result — a linear probe reading a trained trigger with very high accuracy — is treated in this paper as a validity problem rather than a success: the clean linear boundary may be an artefact of how the backdoor was injected, and deception that arises naturally during training need not be linearly separable at all. The author therefore asks whether a *supervised* detector is the right instrument in the first place, and proposes to read the same models with an unsupervised geometric quantity instead. The larger goal is a deception monitor that does not need labelled examples, does not need a direction, and that could survive models learning to suppress linear indicators. A secondary and more interesting goal is methodological: to argue that apparent failures of a monitoring signal may be failures of the *labels* the signal is scored against.

## What they studied

Two small open-weight decoder-only models, Gemma3-1b (26 layers, hidden 2,048) and LLaMA3.2-3b (28 layers, hidden 3,072), both with RoPE, both used frozen. The unit of analysis is not a neuron, feature or direction but the **residual-stream trajectory of each generated token** — the sequence of unnormalised residual vectors across depth — reconstructed at "double resolution" by keeping the states after attention and after MLP separately, so ~2× the layer count of trajectory points. Five prompt variants (honest, strategic, persuasive, deceptive, malicious) share one user question about whether to disclose a prior recommendation, differing only in the prefixed simulated "Thought:" history and the final visible line. Each variant is sampled 100 times per model.

## How they did it

Prompts are generated, 100 completions per strategy per model, with activations captured at every token position and every sublayer boundary — but with layer normalisation undone, so the trajectory keeps its true magnitudes rather than the rescaled, re-centred version the forward pass actually feeds downstream. On each trajectory they compute three things: **salience**, the step length under a pullback metric induced by the unembedding (so distances are measured as changes in logit space rather than in raw activation coordinates); **curvature**, a three-point discrete estimate that tolerates unequal step sizes, evaluated under the same metric; and **semantic surface area** A′, a per-step linear combination of salience and curvature with a single scalar weighting γ (set to 1 throughout, giving equal weight to the two components). Surface area is then normalised by response length.

Labels are not the intended strategies but a consensus of three LLMs (Gemini 1.5 Flash, Claude 3.5 Sonnet, GPT-4o Mini) shown only the visible response and asked for two orthogonal readings: transparency (high/medium/low) and response type (explanatory/evasive/manipulative). Two labelling regimes are carried through every test — the full majority-vote set of 500 per model, and a "unanimous" subset where all three judges agree on both dimensions (LLaMA 201/500 = 40.2%, Gemma 293/500 = 58.6%). Distributions fail Shapiro–Wilk, so Kruskal–Wallis is used for the three-level transparency comparison and Mann–Whitney U for response type, with Cohen's d, η² and Cliff's δ as effect sizes and 1,000-resample bootstrap intervals for group means. Absolute surface-area magnitudes are never compared across models; only standardised effects and the *ordering* of groups are treated as evidence.

## What counts as an explanation here

A continuous geometric scalar per response — how much work the representation did to arrive at its answer — that is claimed to vary systematically with what kind of answer it is. It is not a mechanism, not a direction, not a label: nothing is localised to a component and nothing is intervened on. The claim is predictive in the weak sense that a higher-or-lower surface area should sort responses into transparency/type strata, and the explanation is "meaning construction has shape, and the shape is informative". Completeness is not claimed. Notably the paper explicitly declines behavioural ground truth, describing its target as the shape of inference rather than whether a response is in fact deceptive.

## How they judged success

Two-way: a distribution test on A′ (does it separate the label groups, with non-parametric p-values and standardised effect sizes), and a **precision-of-labels test** — re-run every test on the unanimous subset and ask whether effects hold, strengthen, or vanish. Classification quality itself is reported honestly as a measurement: mean pairwise agreement 0.79/0.79 (Gemma, transparency/type), 0.63/0.73 (LLaMA), with Krippendorff's α 0.309 and 0.443 for Gemma and 0.364 and 0.523 for LLaMA. The author's own named validation criterion is that genuine signal should *grow* as labels get cleaner rather than disappear.

## What they found

On LLaMA3.2-3b the signal is large and survives every strategy in the unanimous set: Honest d = 2.15, Strategic η² = 0.85, Persuasive d = 4.15, Deceptive d = 1.02 (η² = 0.448), Malicious d = 2.22, with p between 0.027 and <0.001. On the full sets the same strategies are significant at very small p (for example Strategic at p < 0.000001 on both dimensions). The sign is firm: explanatory/high-transparency responses carry higher surface area than evasive/low ones (e.g. Honest low-transparency mean 1,418 vs high 2,490; Malicious 1,220 vs 2,400).

Gemma3-1b is much weaker and the paper's headline lives here: on the honest strategy, transparency and response type go from p = 0.555 and p = 0.310 in the full set to **p = 0.048 and p = 0.048** (d = 1.24) in the unanimous set — the improvement from non-significance to significance that the abstract advertises. Strategic strengthens to p = 0.003 (d = 1.51) on response type. Two strategies cannot be tested at all after filtering (persuasive, malicious: effect sizes 1.07 and 0.28, p = 0.138 and 0.794), and the deceptive strategy collapses to a single class — all 61 unanimous responses labelled low-transparency and evasive, so there is no contrast left to test. Absolute scale differs by 6.7× between models (~1,500 for LLaMA, ~10,000 for Gemma), which is treated as an architectural signature rather than a confound.

## Limits and open problems they name

Classifier dependence: LLM consensus is a proxy for human judgement, evasive-vs-manipulative is genuinely ambiguous, and the reported α values (0.31–0.52) are fair to poor. Selection bias: unanimous filtering discards 40–60% of data and preferentially keeps unambiguous cases, which is the opposite of the hard cases deployment would contain; the paper says this may *overestimate* reliability on edge cases. Only two small earlier-generation models, one prompt family, five variants. Metric sensitivity: curvature is a finite-difference estimate, small trajectory deviations late in the stack can produce large metric shifts, and A′ is a composite whose two halves are not separately validated. And no behavioural ground truth at all — the geometric quantity is correlated with a text-based label, not with anything the model did differently.

## What we take from it

- **A habit to steal, and a warning that comes with it.** The precision-of-labels re-run (same statistic, cleaner labels, does the effect grow or shrink) is a cheap diagnostic this folder has been circling for months from the other side — it is the same move as conditioning an agreement score on the model's own answer, and as reporting an absolute-quality number beside a relative win. But the arithmetic of the re-run needs care before use: with n = 100 → 34 (Honest, LLaMA) and an effect moving from "small" to d = 2.15, the p-value improvement is consistent with a *stronger* effect in the clean subset and also with a smaller-but-cleaner sample; the paper reports the two regimes as a validation of the signal and does not report the same statistic on the discarded, disagreed-upon responses, which is the arm that would decide it. Read it as "selection can manufacture significance" as easily as "precision reveals it", and note that the one strategy the paper leans on hardest (Gemma honest, 0.555 → 0.048) is also the one where the unanimous subset is 63 of 100 responses and the low-transparency group is 60 of 63 — a group ratio that makes the test nearly a one-group test.
- **The construction worth keeping is double-resolution, unnormalised trajectories.** Sampling after attention and after MLP separately, and refusing the LayerNorm rescale, gives a trajectory whose step magnitudes are meaningful; that is a genuinely different object from the per-layer residual-state read-outs this folder mostly uses (probe on one layer, patch one site, lens one block), and it is cheap — no training, no labels, one forward pass. If we ever want a "how much work did the model do here" covariate to accompany a predictive claim, this is a defensible way to get one.
- **A caution about metric composites.** A′ folds salience and curvature into one number at γ = 1 with no ablation of the two components, and the one place the paper inspects them separately is a claim that their contributions "exceed" some threshold that the text layer drops. That is the same defect the folder has logged for seven-metric batteries and for cost metrics that manufacture winners: a composite whose parts are never shown to carry the result is a description, not a measurement. Any geometric covariate of ours should report salience and curvature separately before the sum.
- **Framing, one line:** the paper's own best sentence is that apparent detection failures may be measurement failures rather than absent patterns — true, and also exactly the claim every under-powered interpretation experiment makes about itself, which is why the discarded stratum has to be scored.

## Tags

unit: layer
form: description
method: other
eval: benchmark
goal: safety/auditing
