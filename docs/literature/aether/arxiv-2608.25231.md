# Output Dilution: Redundant but Fragile Representations in MoE Models

- **Key:** arxiv-2608.25231
- **Authors / organisation:** Orion Reblitz-Richardson; Distiller Labs (sole author). Code and outputs at github.com/deepsteer/deepsteer.
- **Date:** 2026-08-25 (v1, the only version)
- **Link:** https://arxiv.org/abs/2608.25231
- **Kind:** paper
- **Read from:** full text (abstract through SS6, references, Appendices A–D, Tables 1–3 and Appendix B.1/C.3 tables read one value per cell; Figures 1–4 recorded through captions and restating prose only, never as images)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Sparse mixture-of-experts blocks look like an interpretability gift: the representation is split into named,
discrete modules, so if some concept concentrates in a few experts there would be somewhere to prune, monitor
or edit. The author asks the question directly for moral content - do MoE experts specialise, and does the router
send moral inputs anywhere in particular - and treats a negative answer as informative rather than as a failed
paper. Wrapped around that null is a second, larger question inherited from the author's own companion paper:
whether probing accuracy is an adequate summary of "how strongly a model represents this feature". The stated
goal is to reconcile two models that look identical on the field's standard readout and behave differently under
perturbation, and to name the mechanism.

## What they studied

OLMoE-1B-7B (16 layers, 64 experts per layer, top-8 routing, 6.9B total / 1.3B active parameters, hidden
dimension 2048, expert intermediate dimension 1024 gated SiLU MLPs, learned 2048-to-64 router, load-balancing
auxiliary loss at 0.01) against dense OLMo-2 1B (16 layers, 1.5B parameters, hidden 2048) as the architectural
control - same lab, same layer count and width, comparable active parameter count, both base checkpoints. Both
are read on the same 240-pair moral probing set: 40 minimal pairs per Moral Foundations Theory foundation,
deterministic subsample (seed 42) of a 1,200-pair parent, split 192 training pairs / 48 test pairs (384 / 96
texts) with foundation balance preserved. Units of analysis are, in turn, whole layers, individual experts
(1,024 expert-layer combinations), the router, the aggregated FFN output, and 11 training checkpoints spanning
OLMoE's published 244-checkpoint record (steps 5K to 1.2M, 20B to 5,033B tokens).

## How they did it

Five arms, all on one MacBook Pro M4 Pro in float16. (1) **Layer probing plus fragility, identical on both
models**: hooks on post-layer hidden states, mean-pooled, one `nn.Linear(d,1)` trained with BCE, Adam at 1e-2,
50 epochs, logit sign as the threshold, accuracy on the 96-text test set; then Gaussian noise at five magnitudes
on a log grid, accuracy averaged over 10 noise seeds per magnitude, and the critical noise defined as the smallest
magnitude at which the seed-mean accuracy falls below 0.6, with layers that never fall censored at the grid
maximum so the aggregate is a mean over all 16 layers. (2) **Per-expert probing that bypasses the router**:
capture the pre-MoE hidden state, mean-pool it, and push the same vector through all 64 expert FFNs in one
batched einsum per layer, then fit one probe per expert. (3) **Router content analysis**: softmax routing
probabilities and top-8 selection frequencies conditioned on moral versus neutral text, reported per expert and
per layer. (4) **Component perturbation**: the same noise protocol applied at three places inside the block -
router logits before softmax and top-k (so selection and weights move), individual expert outputs with routing
held clean, and the final aggregated output as the control - over a seven-point grid extending below the
standard one, probes trained on clean outputs and evaluated on perturbed ones. (5) **Output-scale measurement**:
forward hooks on the FFN module's output before residual addition, standard deviation of the mean-pooled output
across 100 texts, per layer, for both models on the same inputs. Plus a checkpoint trajectory arm re-running the
per-expert and router analyses at 11 checkpoints, scoring concentration with the Gini coefficient of per-expert
accuracy and stability with the Jaccard overlap of the top-5 experts between adjacent checkpoints.

## What counts as an explanation here

A mechanism with a number attached: the claim is that the fragility difference is caused by output dilution, and
the evidence is a measured scale gap (the feedforward block's contribution to the residual stream, standard
deviation across inputs, is 74x smaller on average in the MoE than in the dense MLP) that is shown to be of the
same order as the fragility gap (4.2x), and that is corroborated by the within-model component ordering. It is
partial by design: the unit is the layer and the expert, not a feature, and the paper is explicit that raw
critical noise is behaving as a scale meter rather than as a property of the representation as such - the
companion work's RMS-normalized control is cited as the reason a fixed-architecture comparison can read it as
intended. It predicts: greater sparsity should give greater fragility, fine-tuning should move fragility more in
MoE than in dense models of matched active size, and aggregation schemes other than weighted averaging should
give different fragility profiles. None of the three predictions is tested here.

## How they judged success

Two-model probe-accuracy comparison at matched dataset and protocol; critical-noise means with a stated censoring
convention (cap at the grid maximum, and the alternative drop-them value reported alongside, 9.56 versus 9.13);
the Gini coefficient of per-expert accuracy against stated hypothetical bands for no / mild / strong
specialisation; top-5 Jaccard against the random baseline of 5/64; router preference as an absolute difference in
mean routing probability with a 0.5% detection threshold; a three-way perturbation-target ranking with 10 noise
seeds per cell; and the layer-by-layer output-scale ratio between the two models on identical inputs. There is no
ablation of the mechanism itself (no re-run with the aggregation replaced), and no statistical test of the 74x
and 4.2x pairing - the argument is that the two are the same order and that the component ordering is consistent
with the scale account.

## What they found

Probing accuracy is identical and uninformative: both models peak at 99.0% (OLMoE at layer 13, OLMo-2 at layer
12), both have onset at layer 0 and breadth 1.0, and they differ only in early layers, where OLMoE reaches
79-86% at layers 0-3 against 94-97% for the dense model. Fragility diverges: mean critical noise 0.92 for OLMoE
against 3.81 for OLMo-2, the 4.2x in the abstract, with the dense model at or above 3.0 on 12 of 16 layers and
at or above 10.0 on the last three, while the MoE reaches 3.0 only on layers 13-15 and sits at or below 0.3 on 9
of 16 layers. (The text is internally inconsistent about this figure: the Figure 1 caption reports 5.1x with
means 0.84 versus 4.25, and the panel (b) summary of the component arms reports 9.1 and 0.56 against the body's
9.56 and 0.6. All three versions are recorded here; the abstract and conclusion both use 4.2x.)

Specialisation is absent rather than weak. 1,020 of the 1,024 per-expert probes exceed 75% accuracy, the four
exceptions sit at 72-75% on early layers, no expert at any layer falls below 60%, and at the peak layer every
expert exceeds 84% (mean 93.0%, minimum 84.4%). The Gini coefficient stays between 0.016 and 0.023, which the
appendix translates as roughly a 1.3:1 accuracy ratio between the best and worst expert, an order of magnitude
below the paper's own "mild specialisation" band of 0.05-0.15. The router barely distinguishes moral from neutral
inputs: maximum routing preference 1.8% (layer 12, expert 37, absolute difference in mean routing probability) and the
95th percentile across all 1,024 combinations below 2%.

The component ranking inverts the naive expectation. Perturbing the router is nearly harmless (σ* 9.56 with 8 of
16 layers never fragile), perturbing individual
expert outputs is intermediate (σ* 1.8, all layers fragile), and perturbing the aggregated output is the most
fragile of the three (σ* 0.6, all layers fragile). The explanation is the scale: the aggregated MoE output has a
standard deviation of 0.003-0.008 at layers 0-8, against router logit scale around 0.5, so a fixed absolute noise
level is enormous relative to the former and modest relative to the latter. Across all 16 layers the dense MLP
output is 74x larger on average, ranging from 5x at layer 2 (an unusually large MoE output there) to 167x at layer
0, with the ratio above 60x at 9 of 16 layers.

Training changes nothing. At step 5,000 - about 0.4% of training - the peak-layer mean already sits at 92.1% with
1,006 of 1,024 probes above 75%, and it stays in a 92-94% band through step 1.2M (93.6% at 10K, 93.2% at 200K,
93.7% at 1.2M). Peak-layer Gini stays between 0.012 and 0.018 at every checkpoint, and the top-5 expert overlap
between adjacent checkpoints hovers at the random value of about 0.08 across a 0.0-to-0.25 range. The paper's
compressed causal account: probe accuracy survives because a linear head can amplify a small but faithful
signal; specialisation does not appear because the router is content-agnostic, so every expert sees the same
pre-MoE state; fragility rises because the absolute threshold for disrupting a 0.003-scale signal is far lower
than for a 0.3-scale one; and the router is robust because its logits live at a scale the tested noise rarely
reaches.

## Limits and open problems they name

One MoE family, so generalisation to Mixtral, DeepSeek-MoE or Qwen-MoE is open and the higher-sparsity prediction
is untested. Everything runs on mean-pooled representations while routing is per-token, which may mask per-token
routing effects - and per-expert probing here applies each expert to the whole pooled vector, which is not what the
router would actually do. Linear probes only, with the acknowledgement that a nonlinear probe might extract the
small-scale signal more effectively and shrink the apparent gap, plus the counter-argument that the scale
measurement does not depend on the probe. The dense comparison is same-lab but not identical: different data mix,
hyperparameters and training duration confound architecture. The measured scale is the feedforward block's
contribution to the residual stream, not the residual itself, and the author is careful to say so: because the
residual carries the pre-MoE state forward, a probe on the full hidden state reads both, and the fragility gap
arises because noise calibrated to the residual is large relative to the MoE contribution. English, MFT foundations
only. A more concrete omission the author makes explicit: the companion work's C15 fragility-locus result under
insecure-code LoRA was not replicated on OLMoE, so the fine-tuning prediction is stated as a prediction.

## What we take from it

A measure and a warning, both directly usable. **The warning:** this paper and its companion are the same
instrument, the same dataset and the same single author, and one now claims the fragility metric reads scale as
intended while the other withdrew its layer-depth gradient because it was an activation-scale artifact under RMS
normalization. Whether or not one thinks the fixed-architecture reading here is legitimate, the report does not
run the RMS control on the MoE-versus-dense comparison, and a 74x scale gap mapped onto a 4.2x fragility gap is
exactly the pattern the companion's own v2 called a scale artifact. The rule we should copy: **a critical-noise
or margin-like threshold quantity is never compared across two models without printing both the raw threshold
and the threshold normalized to that site's own activation RMS, and where the raw ratio is much larger than the
normalized one the finding is described as a scale reading rather than as a claim about how securely the
information is encoded** - the normalized version is the one that says anything about robustness "beyond scale".
**The measure worth stealing:** the *component* variant of the same protocol - hold the trained probe fixed and
inject the same absolute noise at each internal site in turn (router logits, expert outputs, aggregated output),
then rank the sites by the noise that breaks the readout, with each site's natural standard deviation printed on
the same row. Here the ranking is the whole mechanism (router 9.56, experts 1.8, aggregate 0.6), it is the
fifteen-minute experiment in their own compute table, and it is the cleanest cheap demonstration in the folder that an intervention's
"fragility" is a ratio to the site's own working scale rather than a property of the site.
**And one construct that is unusual enough to name:** the Gini coefficient of per-module probe accuracy as a
concentration statistic, with hypothetical bands pre-stated for absent, mild and strong specialisation and the
observed 0.016-0.023 sitting an order of magnitude below even the mild band - a negative result with a scale
attached to it, rather than an eyeballed boxplot.

## Tags

unit: layer | feature | whole-model
form: probe | description
method: probing | ablation | decomposition | other
eval: ablation | prediction/simulation | faithfulness | benchmark
goal: understanding | safety/auditing | evaluation-of-methods
