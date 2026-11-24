# Attention Sinks as Spectral Spikes: A Mechanism Analysis of Gated Attention

- **Key:** arxiv-2611.20403
- **Authors / organisation:** Seojin Kim, Yehjin Shin, Noseong Park; KAIST
- **Date:** 2026-11-16
- **Link:** https://arxiv.org/abs/2611.20403
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1, 120,213 chars clean; abstract, §1-§7, Appendices D.1-D.8, E.1-E.3, F.1-F.4, G.1-G.3 and the limitations appendix H read as text; Tables 1-14 read off column ordering and cross-checked against prose restatements; Figures 1-9 read from captions and the prose describing them, not inspected as images; Appendices A, B, C and I located but not read for the algebra - only the statements of Theorem 3.1, Proposition 3.2, Theorem 4.1 and Corollary 4.2)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

A post-SDPA output gate - an elementwise sigmoid applied to attention output, head by head - is known to nearly abolish the attention sink (the paper restates the drop in first-token attention as 46.7% to 4.8%) and is now shipping in production architectures. The published explanation for why it works is behavioural: the gate makes attention sparse and query-dependent. The authors find that unsatisfying, and ask a structural question: what mechanical property lets an output gate suppress sink-aligned structure specifically, rather than uniformly shrinking everything? They also want the backward half of the story, because a separate line of work argues sinks create gradient sinks and that massive activations are a normalisation response to that gradient pressure. The goal is a forward-and-backward account of one architectural knob, i.e. a mechanism claim rather than a performance claim.

## What they studied

Gated and ungated Transformer language models: Qwen-1B (an existing post-SDPA gated model from the gating paper), Pythia-410M and Pythia-2.8B trained matched by the authors, plus released Softpick-340M checkpoints for one external comparison. Data is WikiText-103 validation (16 sequences of 512 tokens, EOS-separated) and FineWeb-Edu for the 15B-token training runs. The unit of analysis is the **token-space covariance of a single attention head's output** - a n x n matrix over positions rather than over features - together with per-source-token attention column mass and per-source value gradients.

## How they did it

They first express the post-attention output Y = A V, and under an isotropic-value approximation show its token-space covariance reduces to a weighted sum of rank-one outer products of attention columns. A high-mass column is therefore a candidate low-rank "spike" in that covariance. They formalise a sink-induced spiked model: the covariance has one structured direction, and that direction is called sink-induced if it lies mostly inside the span of the normalised high-mass attention columns. Then they treat the elementwise gate as a diagonal contraction per token row and derive that the gated signal's energy equals the ungated amplitude times a factor measuring how much of the feature direction survives each row's gate - so a spike is attenuated to the extent the gate contracts the tokens that carry it. An idealised thinning argument links the contracted strength to the random-matrix detection threshold above which a rank-one spike separates from a pure-noise spectrum, giving a prediction: gating should move the leading eigenvalue back toward the noise edge, or erase the outlier. Empirically they measure four spectrum statistics before and after the gate at every layer, plus the alignment of the leading eigenvector with a data-chosen sink subspace built from the top column masses (at most eight columns), against a random-subspace baseline. Controls include non-standardised covariance, replacing the elementwise gate with its per-token feature average (a scalar gate that keeps token-wise strength but drops feature selectivity), projecting out the sink subspace, permuting the attention-column-to-value pairing while keeping both matrices, interpolating the value Gram matrix between isotropic and learned, and a position-controlled EOS substitution. For the backward pass they bound the value gradient of a source token under a gate by the gated effective column mass, then test it with a fixed-upstream-gradient counterfactual: recompute the first-token value-gradient contribution with and without the gate while holding attention weights and incoming gradients fixed.

## What counts as an explanation here

A spectral mechanism plus a gradient bound: two inequalities that both read off quantities available from one forward pass and one backward pass (attention column mass, gate values, value norms, upstream gradient norms). It is deliberately partial. The authors state that the random-matrix threshold story holds only for an idealised isotropic reference, that a learned elementwise gate also reshapes the noise so the covariance is not a standard spiked model, and that the gradient result is local - conditional on fixed attention weights and fixed upstream gradients. The explanation predicts a direction and a magnitude proxy per mode and per sink column, not the training trajectory or the final loss.

## How they judged success

Four spectrum statistics per layer (out-of-bulk eigenvalue mass, leading eigenvalue over the noise edge, effective rank, sink-subspace alignment) compared pre- versus post-gate within the same model and across matched gated/ungated training runs; the random k/T subspace baseline for alignment; a scalar-gate control for feature selectivity; a sink-subspace projection to check that the removed structure is the sink's; the attention-value permutation control; a downstream benchmark table of eight tasks plus WikiText and LAMBADA perplexity on matched 15B-token training; and the local gradient-attenuation ratio against an RMS gate-mass proxy, expecting the two to track across layers. No pre-registration and no confidence intervals are reported; the paper argues from consistent direction across models, scales (410M vs 2.8B), context lengths (4K vs 8K) and formatting conventions rather than from significance tests.

## What they found

Gating reduces out-of-bulk spectral mass, pulls the leading eigenvalue closer to the noise edge and raises effective rank, and the reduction survives dropping feature standardisation. The scalar (feature-averaged) gate control fails to reproduce it: in the standardised covariance the retention of highly sink-aligned directions is 0.740 against 0.945 for weakly aligned ones, and the gap shrinks to 0.957 vs 0.990 under the scalar gate, while raw energy retention is nearly identical (0.047 vs 0.044) - so the effect is feature-selective rather than uniform shrinkage. Backward: baseline models develop first-token value-gradient amplification of 25.40 in mid-to-late Pythia-410M layers against 5.67 gated, and the fixed-upstream counterfactual puts the local attenuation well below one and tracking the RMS effective-mass proxy. Training-time checkpoints show attention concentration, gradient amplification and first-token representation dominance co-emerging in the baseline and jointly suppressed when gated; massive-activation magnitudes fall by 3.5x-4.75x on Pythia-410M and by more than an order of magnitude on Qwen-1B. At 2.8B the direction is unchanged. The matched-training benchmark average is 44.96 ungated vs 46.06 gated. Notably, the EOS counterfactual shows the sink follows a special token when one is inserted mid-sequence, and position 0 keeps a residual gradient signature even when it is no longer the top attention sink.

## Limits and open problems they name

The spectral threshold argument is an idealised reference, not a theorem about learned gates; noise reshaping by the gate is unmodelled. The gradient statement is local and does not imply all gradients are smaller. Not all observed spectral suppression is explained by sink-induced modes - the gate attenuates leading structure outside the sink subspace as well. Scope is small models trained by the authors plus one released model; the broader generalisation is left open. The 8K/8K backward measurement ran out of memory and is missing.

## What we take from it

A **reference-null for structure**, first: an observed low-rank direction should be scored against the random-matrix noise edge or a random-subspace baseline computed from the same shapes, so a "dominant direction" claim carries a number saying how far above chance it sits. Second, a **selectivity control** we can reuse cheaply on any read-out: replace the selective mechanism with its own average and show the effect disappears, which is the difference between a mechanism and a scale knob. Third, a **warning about metric geometry**: the same gate looks non-selective in raw energy retention and strongly selective in standardised covariance, so which normalisation a statistic lives in decides the conclusion - our probe and dictionary statistics need the same disclosure. Fourth, the fixed-upstream-gradient counterfactual is the intervention recipe this folder keeps asking for: hold the more expensive half of the forward pass fixed, toggle only the component under test, and compare. Fifth, a cross-reference for the offset/drift cluster: the paper's own EOS experiment shows a sink's location is set by token identity and position, not by semantics, the same lesson as the chat-format boilerplate entries.

## Tags

unit: layer | feature
form: description | direction
method: weight-analysis | ablation | patching | other
eval: ablation | prediction/simulation | downstream-task
goal: understanding | control/editing
