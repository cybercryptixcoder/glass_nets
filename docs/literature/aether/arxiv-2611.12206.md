# A Geometric Invariant of Attention Predicts Head Specialization from Weights Alone

- **Key:** arxiv-2611.12206
- **Authors / organisation:** Miguel P. Bento (single author); Feedzai, Lisbon, Portugal — work done independently of his role there.
- **Date:** 2026-11-09 (extended abstract, NeurReps workshop at NeurIPS 2026, PMLR proceedings)
- **Link:** https://arxiv.org/abs/2611.12206
- **Kind:** paper (workshop extended abstract)
- **Read from:** full text (HTML, ~12K characters: abstract, §§1–4, Table 1, Table 2, references — the whole paper; there is no appendix)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Almost every claim about what an attention head does in a modern LLM is established with data: patch activations, train probes, run crafted inputs, correlate with attention patterns. The author asks the complementary question — how much of a head's *role* is already fixed by its numbers, before any forward pass. The motivation is analogy with physics: a theory usually has far fewer invariant observables than free parameters, so a head should be characterisable by quantities that survive every reparameterisation the architecture permits. The concrete target is rotary-position-encoding models, where heads are widely split into "content" and "positional" families; the paper tries to turn that split into a scalar computed from weight matrices alone.

## What they studied

Attention heads in four bias-free RoPE language models spanning three orders of magnitude: SmolLM2-135M, TinyLlama-1.1B, SmolLM2-1.7B and Mistral-7B. The unit of analysis is the individual head, specifically the query–key sector, with induction heads — the best-studied circuit type in the field — as the behavioural label to be predicted. No other circuit families are tested.

## How they did it

The attention logit is written as a static content term plus a positional perturbation that the RoPE rotation at each relative offset adds to it. The author defines a single scalar, called rho, as the size of that position-dependent term relative to the content term, aggregated over a set of relative positions sampled on a logarithmic grid so that the geometrically spaced RoPE frequency bands are all exercised up to a stated context horizon. Because the attention mechanism is invariant under a residual symmetry group that leaves the query–key bilinear form's trace-like quantity fixed, rho is claimed to be basis-independent, so it can be computed from stored weights without an input. A proposition shows rho = 0 iff the positional perturbation vanishes at every sampled offset, in which case the logits are position-independent for every input. Empirically, induction scores are measured the standard way — repeated random sequences — and rank-correlated with rho across heads in each model. As an application, YaRN context extension is applied only to the top half of heads ranked by rho, leaving the rest at their original positions, and short- and long-context perplexity are compared with a baseline and with uniform YaRN.

## What counts as an explanation here

A single weight-derived scalar per head, plus the symmetry argument that licenses calling it an invariant. It is explicitly a *capacity* statement rather than a behavioural one: a high-rho head may still ignore position on a given input, but a head with rho = 0 cannot depend on position at all. Completeness is not claimed — the author notes that a full invariant signature would need the eigenspectrum, singular values and a commutator term, and that the framework covers only the query–key sector of RoPE architectures. Prediction is in the negative direction only: low rho is put forward as necessary but not sufficient for induction.

## How they judged success

Spearman rank correlation between rho and induction score across all heads of each model, with significance against the head count; the rank percentile of the ten best induction heads within the rho distribution; and, for the application, perplexity at 512 and 4,096 tokens for baseline, uniform YaRN and rho-selected YaRN on Mistral-7B. No ablation of the log-spaced sampling grid, no shuffled-weight or random-direction control, and no comparison against a data-based head classifier.

## What they found

The correlation is negative and significant in every model — Spearman between −0.55 and −0.74, p < 1e-40 — and the ten strongest induction heads always sit in the low-rho tail: bottom 14% (SmolLM2-135M, r = −0.74), bottom 7% (TinyLlama-1.1B, r = −0.71), bottom 17% (SmolLM2-1.7B, r = −0.55) and bottom 30% (Mistral-7B, r = −0.59). The author reads the imperfection as expected, since induction is a multi-head circuit and low rho is necessary but not sufficient. In the application, targeted YaRN on Mistral-7B gives short-context perplexity 12.92 against 19.37 for uniform YaRN and 7.63 for the untouched baseline, and long-context perplexity 130.34 against 162.96 — both better than uniform, both far above baseline, with the author noting the evaluation sits near the context limit and inflates the long-context figures.

## Limits and open problems they name

A workshop-length proof of concept. Only four RoPE models, all bias-free, all up to 7B; only the query–key sector; the context-extension experiment is one model within its native window and is called preliminary. Rho is sensitive to the softmax-shift direction, which leaves attention patterns unchanged — the author argues weight decay discourages large components there but does not measure whether it does. The correlation is a population statistic across heads, and the paper does not test whether a low-rho head outside the top ten is also an induction head.

## What we take from it

A method and a framing. The framing is the useful part: a weight-only quantity that is a *ceiling* on what a head can do (zero rho forbids positional dependence) is a different kind of claim from a probe accuracy, and it costs no inference — which is exactly the direction our predictive model wants, since a ceiling computed from weights can be checked against any behavioural battery for free. The measurement discipline is also worth carrying: the headline is a *negative* correlation plus a rank-percentile statement about the top-k heads, not an accuracy, and the author is explicit that the direction of the claim is necessary-not-sufficient. The rho-selected YaRN result is the kind of cheap downstream use a scalar invariant earns: if the quantity ranks heads by what they can couple to, spend the intervention only on the ones that can respond. Caveat for us: n = 4 models, one sector, one downstream task, and no shuffled-weight control — so treat it as a hypothesis worth testing on our own models rather than an established property.

## Tags

unit: head | weights
form: direction
method: weight-analysis | other
eval: prediction/simulation | downstream-task | benchmark
goal: understanding
