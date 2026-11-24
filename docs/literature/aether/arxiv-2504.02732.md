# Why do LLMs attend to the first token?

- **Key:** arxiv-2504.02732
- **Authors / organisation:** Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličković, Razvan Pascanu; University of Oxford, National University of Singapore, Google DeepMind
- **Date:** 2025-04 (v4 dated 2025-08-05)
- **Link:** https://arxiv.org/abs/2504.02732
- **Kind:** paper
- **Read from:** full text (arXiv HTML v4, ~69k chars clean; abstract, §1-§6, Appendices A-D including the two proofs, Tables 1-3 as cells; figures 1-10 captions only, not inspected as images)
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Attention heads in trained decoder-only models put a large share of their mass on a single early position, most often the first token. Most prior work treats this as a defect to be mitigated (quantisation, KV-cache, streaming attention). The authors take the other side: since the pattern is produced by gradient descent and appears everywhere, it should be doing work, and the paper asks what work - specifically why it lands on the first position. The proposed answer is that deep and long-context Transformers tend to over-mix their token representations (rank collapse, representational collapse, over-squashing), and an attention sink is a learned brake on that mixing. The larger goal is a mechanistic account of a training-emergent attention pattern rather than a mitigation recipe.

## What they studied
Decoder-only Transformers: Gemma 7B and the LLaMA 3.1 family (8B with 32 layers / 1,024 heads, 70B with 80 layers / 5,120 heads, 405B with 126 layers / 16,128 heads), plus LLaMA2-style 120M-parameter models the authors pre-train from scratch to control context length and data packing. The unit of analysis is the attention head and its attention distribution over the first `T=64` positions; secondarily the value-vector norm of the first token and the per-token representation at the last layer.

## How they did it
Two theoretical steps and then four measurements. First, they show that the rank-collapse condition implies the representational-collapse condition but not the reverse, so the two are distinct phenomena tied respectively to depth and to context length. Second, they extend an existing over-squashing bound to multi-head attention: the norm of the Jacobian between an input token's representation and an output token's representation is bounded by a sum over attribution paths of a product of head-averaged attention weights, so lowering any single weight on a path multiplies the sensitivity down. A sink is therefore a cheap way to shrink that product for *all* paths - and since the bound carries the depth and context length in it, larger and longer-context models should need stronger sinks. They test this with a perturbation probe (swap one word for a synonym, holding token count fixed) in Gemma 7B with and without the beginning-of-sequence token, then pre-train 120M models at five context lengths with total token budget held constant, then measure a per-head sink metric across the three LLaMA sizes over the same 170 prompts, then sweep pre-training packing and masking strategies (causal vs intra-document; whether the beginning token is fixed in first position) and test inference with and without it, reporting validation loss and seven downstream benchmarks.

## What counts as an explanation here
A mechanistic/functional account: an analytic bound on inter-token sensitivity, plus the claim that a specific learned pattern (attention to position zero) implements the brake, plus a per-head scalar (the sink metric) and a per-token norm that names the physical mechanism (attending to a low-norm value vector makes the attention output approximately zero, so the block degrades into a residual-only no-op). It is predictive in a restricted sense - it predicts which models will have stronger sinks - not a per-prompt predictor of behaviour.

## How they judged success
Sink metric defined in the manner of prior work: fraction of heads whose attention on the first token exceeds a threshold (ε=0.8 in the headline tables), averaged over 170 prompts from a public probe set. Supporting evidence: representational distance from the per-token mean as a function of prompt length (a rank-collapse proxy), spread of per-token norms, Jacobian-norm behaviour estimated by a single-token lexical perturbation, validation loss held equal across pre-training runs, and downstream benchmark scores (ARC-easy, ARC-challenge, PIQA, SIQA, HellaSwag, Winogrande, RULER at 4k context) with and without the first token.

## What they found
- Perturbing one mid-prompt word moves the representations of other tokens much more strongly once the first token is removed; attention maps become visibly smoother, which raises the Jacobian proxy.
- Sinks scale as predicted: sink metric 45.97% for LLaMA 3.1 8B, 73.49% for 70B, 78.29% for 405B (ε=0.8). The authors quote roughly 80% of heads forming strong sinks in 405B, and note that in a typical 405B prompt about 80% of attention can sit on that token.
- In their own 120M pre-training runs, sink prevalence rises monotonically with training context length and is essentially absent at a context of 128; the growth is progressive during training, and all runs reach comparable validation loss.
- Whether the first token is a special beginning-of-sequence symbol is largely irrelevant to *formation*: without one in training, the sink forms on whatever token is first. But whether it was **fixed** in first position matters to *use*: with causal masking and a fixed beginning token, pre-training sink metric 90.84 and validation loss 2.69, and removing the token at inference drops the metric to 0.05 and loss explodes to 7.56. Intra-document masking without a fixed beginning token gives 28.23.
- Removing the first token from Gemma 7B at inference collapses benchmark scores (ARC-e 80.77 to 28.49, HellaSwag 80.61 to 27.35, PIQA 81.72 to 52.77) and takes RULER at 4096 context from 82.57 to 0.00. The damage is worst exactly where the theory says it should matter most - long context.
- A worked head in Gemma 7B layer 1 behaves as an if-else: sharply attend to a preceding apostrophe when present, otherwise attend to the first token, whose value vector has the lowest norm. The sink is thus an "approximate no-op" write into the residual stream.

## Limits and open problems they name
The over-squashing bound deliberately does not push derivatives through the softmax, so it is a qualitative guide, not a tight prediction; queries and keys are assumed independent of values for the proof. The sink metric is a thresholded head count, so it is a coarse summary and its absolute values are not comparable across papers. Pre-training experiments are 120M-parameter models on limited token budgets, and the LLaMA measurements are observational - training pipelines across sizes are assumed comparable but not verified. Perturbation experiments are a small number of hand-built prompt pairs. They do not show that the *same* mechanism explains sinks that form on non-first tokens (delimiters, repeated token), and they flag that at sufficiently long context every such mechanism is expected to disperse. Sink rate is also indistinguishable from a proxy for layer activity in their own plots - they note the middle layers look more active, matching prior work.

## What we take from it
Three things. (1) **A measure.** The Jacobian-norm bound is the cleanest justification in this folder for a *path-product* attribution over attention head pairs: sensitivity between two token positions is bounded by a product of head-averaged attention weights along the path, so a single weight maps monotonically to a single predicted effect. We can use it as a cheap analytic surrogate to check a discovered circuit's predicted influence before any ablation, which is exactly the "predict under intervention" shape our model needs. (2) **A warning about reading a per-unit statistic without its function.** A head that looks maximally "meaningless" (it attends to a token whose value vector is tiny, i.e. it writes nothing) is doing the most useful job in the layer - it is implementing a no-op. Any of our own "inactive head" or "inactive feature" readings must be reported alongside the norm of what that unit *writes*, because low contribution is not low function. (3) **A design lesson for probes.** Removing the sink at inference does not degrade quality gracefully - RULER falls to zero - so ablating a structural, input-statistics-driven component can read as "this feature is essential" when it is really "we broke the arithmetic of the block". Our own ablations of such components need a matched repair (restore an equivalent no-op, not the original token) before we call the component necessary.

## Tags
unit: head, layer
form: description
method: weight-analysis, other
eval: patching, ablation, benchmark
goal: understanding, science
