# Cross-Layer TopK: Dynamically Allocating Sparsity Across Layers in Transcoders

- **Key:** arxiv-2611.09233
- **Authors / organisation:** Anyuan Zhuo, Shaohua Zhang, Tianle Li, Yuan Lin, Diandian Gu, Jun Wang, Pinyan Lu; ByteDance Seed / Shanghai University of Finance and Economics / The University of Hong Kong
- **Date:** 2026-11-06
- **Link:** https://arxiv.org/abs/2611.09233
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Cross-layer transcoders (CLTs) are the dictionary that circuit tracing depends on: a feature read from
the residual stream at one layer is allowed to write to the MLP outputs of every later layer, so a single
sparse feature can express computation spread across depth. Training them is awkward for a reason that has
nothing to do with features — every layer's reconstruction loss is coupled to every other layer's, because
every earlier layer's features contribute to every later layer's output. The existing recipe inherits
JumpReLU from sparse autoencoders, where sparsity is reached indirectly through a penalty whose end level
cannot be predicted from its coefficient, so hitting a target sparsity means repeated search over that
coefficient at CLT scale. The obvious alternative, plain TopK, fixes the count exactly but fixes it
*per layer*, which is wrong here because layers are not equally hard to reconstruct. They want a sparsity
mechanism that is exact like TopK and adaptive across depth like JumpReLU, so that the sparse dictionary a
circuit-tracer is built on is not silently wasting its budget in the easy layers.

## What they studied

Qwen3-8B-Base (36 layers), activations from the Olmo 3 corpus (pretraining text, reasoning chains and
instruction SFT mixed 5:2:1), with CLTs at latent size m = 32,768 and a per-token budget of 360 active
latents for the three TopK arms. The unit of analysis is the **latent in a layer** — the allocation
decision is a per-token ranking over the pooled set of L·m candidate latents — and the outputs examined are
reconstruction at each layer and, downstream, attribution graphs built on the replacement model.

## How they did it

Each layer's encoder maps that layer's MLP input to a vector of pre-activations, as in an ordinary SAE.
Instead of taking the top-k inside each layer, they pool the pre-activations of all 36 layers for a token,
score each latent by its rectified pre-activation multiplied by the L2 norm of the decoder columns it would
write through (its total output to layers l..L), and keep the global K = 360 largest; each kept latent
retains its rectified value and contributes through its decoder to all layers at or above its source
layer. Reconstruction loss sums squared error over all 36 layers, with MLP inputs and outputs normalised
to an expected norm of root n beforehand so the per-layer errors are commensurable, plus a dead-latent
auxiliary loss that runs the *same pooled top-K* restricted to latents that have not fired in the last 250
steps. Training is 3.2B tokens, AdamW, effective batch 32,768 tokens, with the budget K annealed from 3,600
down to 360 over the first quarter of training. The baselines share the hyperparameters and the 360-token
budget: **per-layer TopK** (10 latents per layer by construction), **Cross-Layer BatchTopK** (top-K over all
latents of the whole micro-batch, with a per-latent threshold calibrated after training on 1,000
micro-batches), and **JumpReLU** with the quadratic L0 penalty against a target of 360. Comparisons are FVU
on held-out validation data, dead-latent fraction, and the replacement and completeness scores computed on
attribution graphs for 100 filtered prompts; the allocation experiment plots per-layer active counts against
per-layer FVU; two scaling sweeps vary m over 2¹³–2¹⁵ and k over 5–20, each with a separately trained model.

## What counts as an explanation here

Not an explanation of model behaviour — an explanation of *how much of a dictionary's budget should sit at
which depth*, expressed as the learned allocation profile itself, plus the account of the mechanism that
produces it (the scoring rule, and the empirical regularity that a layer's allocation tracks how hard that
layer is to reconstruct). The dictionary is the explanatory artifact; completeness and replacement scores
are the paper's way of asking whether that artifact is a good substrate for someone else's mechanism
claim. There is no claim about what any feature means.

## How they judged success

Fraction of variance unexplained per layer and averaged over layers, on held-out validation data; the
fraction of latents dead over the last 250 optimiser steps; circuit **replacement score** (fraction of
end-to-end path influence from token embeddings to the target logit that stays on feature nodes rather than
error nodes, i.e. a complete error-free path is required) and **completeness score** (influence-weighted
average over nodes of the fraction of incoming influence coming from non-error nodes, which grants partial
credit), both defined following Ameisen et al. and averaged over the same 100 prompts for every
architecture; training step time on 128 GPUs with a batch of 8,192 tokens and k = 10, split into compute
and communication, across a baseline and three decoder kernels; and log-log power-law fits for the scaling
sweeps. The prompt filter for the graph evaluation is worth naming because it is a design choice, not an
implementation detail: keep a prompt only if the model predicts the ground-truth token greedily, that token
is rarer than 10⁻⁴ in the pretraining corpus, it has not appeared earlier in the prompt, and the prediction's
cross-entropy exceeds 0.2.

## What they found

At m = 32,768 and equal budget, Cross-Layer TopK gives **FVU 27.07% against 27.64% for per-layer TopK**,
with 3.0–3.2% dead latents for all three TopK arms against **41.2% for JumpReLU**, whose achieved sparsity
stalls at 452 active latents despite the 360 target (the quadratic penalty fades as L0 approaches the
target, so reconstruction takes over near it), leaving it at 32.17% FVU with a *larger* budget.
Cross-Layer BatchTopK reaches 26.98% during training but **29.04% at inference** once batch selection is
replaced by a calibrated threshold, because a latent's threshold is estimated from the batches in which it
was most active and is therefore biased upward for rare latents. The allocation itself is the interesting
object: per-layer TopK's FVU profile runs about 5–12% in the first layers, above 40% in layers 15–19, and
under 4% in the last layer, whereas Cross-Layer TopK gives 12–14 latents to most of layers 10–22, peaks at
14.36 in layer 16, and withdraws to **0.90 in layer 36**; across layers 5–27 the latents a layer receives
correlate with its per-layer FVU at **r = 0.85**. The first layers are the exception and get more than their
low FVU would predict (13.8 in layer 3), which they attribute to early features writing to all later layers
at little extra cost. It is a trade, and they print both sides: **27 of 36 layers improve, largest gain 2.0
points in layer 1, and the nine exceptions are exactly the last nine layers, largest loss 2.2 points in
layer 35**, with JumpReLU worse in every layer despite its larger budget. On circuits, Cross-Layer TopK
scores **replacement 0.617 against 0.577 (per-layer TopK), 0.578 (BatchTopK) and 0.589 (JumpReLU)**, and
completeness 0.902 against 0.894 / 0.889 / 0.897. Sparse decoding with a fixed per-token budget gives
0.355 s per step against 1.025 s for the latent-sharded baseline (2.9×, with compute cut 65% and
communication 67%). Reconstruction follows **FVU ∝ m^-0.16** over 2¹³–2¹⁵ and **FVU ∝ k^-0.13** over k = 5–20.

## Limits and open problems they name

A single model, a single budget and one training length: 3.2B tokens, chosen because the reconstruction
loss had plateaued, with no seed variation reported anywhere. The FVU advantage over per-layer TopK is
0.57 points averaged over depth, and the circuit advantage is 0.04 of replacement and 0.008 of completeness
on 100 prompts — no confidence intervals, no per-prompt spread, and no second corpus. They state they did
not scale further because a CLT's decoder grows with the square of the number of layers, so each large run
is expensive. On the allocation itself they report the early-layer exception and attribute it to cheap
writes, but do not test whether the extra early budget is optimal or an artefact of the scoring rule,
which multiplies by the decoder norm of the whole write path. And they cite, without resolving, the result of Lange et al. that CLTs are *incentivised*
to learn unfaithful circuits — the reconstruction-quality argument for a better CLT does not by itself
establish that the resulting graph is closer to the model's computation.

## What we take from it

Three things, in order of usefulness. **A measurement-design framing:** this is the folder's cleanest case
of a *capacity constraint deliberately moved to the coordinate where it belongs*. The constraint "at most
K active latents per token" is real and mechanistically meaningful (it is the sparsity of the
representation), while the constraint "exactly k per layer" was inherited from the SAE template and is not
— depth is heterogeneous in a way anybody can measure by running the fixed-budget arm first. The general
recipe: when a hyperparameter has to be set per site, run the uniform-budget arm once, look at the residual
across sites, and if it varies systematically, pool the budget and let the training decide. That is cheap
and it applies well past dictionaries. **A warning about the reconstruction ladder:** the ordering of the
four arms is not recoverable from the reconstruction number alone, because the arms differ in *achieved*
budget — JumpReLU at 452 latents loses to a 360-latent arm, BatchTopK wins at training and loses at
inference, and the paper's own explanation of each reversal is legible in the dead-latent column and the
train/inference column. Our own dictionary comparisons should print achieved L0 and dead fraction beside
every FVU, or the comparison is between budgets rather than between architectures. **A caution to carry into
every circuit claim:** completeness of 0.90 means a tenth of the influence at the nodes that matter arrives
through an opaque error node, and that is the *best* arm; a mechanism claim built on an attribution graph
inherits that residual, which is the folder's reconstruct-versus-explain separation in arithmetic form. And
the transferable engineering point — a fixed per-token budget turns the selection rule into a known sparse
pattern, which is what makes sparse decoding possible at all — is the same reason a fixed interpretation
unit is worth choosing even when a data-dependent one reconstructs better.

## Tags

unit: feature | layer | weights
form: dictionary | replacement-model | graph
method: decomposition | transcoder | other
eval: faithfulness | completeness | downstream-task
goal: understanding | evaluation-of-methods
