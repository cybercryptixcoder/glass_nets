# Learning a Generative Meta-Model of LLM Activations

- **Key:** arxiv-2602.06964
- **Authors / organisation:** Grace Luo, Jiahai Feng, Trevor Darrell, Alec Radford, Jacob Steinhardt; UC Berkeley (Radford independent; Steinhardt also affiliated with Transluce)
- **Date:** 2026-02-06
- **Link:** https://arxiv.org/abs/2602.06964
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1, 88.4k chars after stripping; abstract, §1–§7 including Discussion, Limitations and Future Directions, Tables 1–5 as cells, Appendices A–D statements and Table 10–12; figures 1–16 read from captions only and not inspected as images)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Dictionary methods — PCA, sparse autoencoders — carry structural assumptions (linearity, sparsity, additive atoms) that are guesses about what activation space looks like. The authors want an activation model that assumes nothing about structure and instead learns the distribution of the activations themselves, so that "structure" is whatever the learned density says it is. Two payoffs are claimed: such a model is a prior that can project an edited activation back onto the set of activations the network actually produces, and it is an encoder whose own internal units might be more monosemantic than dictionary features. The larger goal is an interpretability tool that improves by training more compute rather than by finding a better assumption.

## What they studied

Residual-stream activations at the middle layer of Llama-3.2-1B (width 2,048; layer 7) and Llama-3.1-8B (width 4,096; layer 15), drawn from one billion FineWeb tokens, one activation per token position with the BOS token excluded and sequences capped at 2,048 tokens. The unit of analysis is the whole single-token activation vector, and secondarily the units ("meta-neurons") inside the generative model. Five dense-activation diffusion models were trained on 1B activations (0.5B, 0.9B, 1.7B and 3.3B parameters at 3, 6, 12 and 24 layers), plus one 3.4B model on 8B activations and a multi-layer variant conditioned on layer index.

## How they did it

They train an unconditional diffusion model with a flow-matching objective on single-token activations: the forward process linearly interpolates a real activation with Gaussian noise, and a denoiser MLP regresses the velocity pointing from data to noise. The architecture is a stack of Llama-style SwiGLU blocks with residual connections, no attention (one token at a time), and timestep conditioning by multiplicative modulation of each block's gate pre-activation; model width is twice the activation dimension and the gated MLP expands by a further factor of two. Activations are standardised to zero mean and unit variance before training, and a producer–consumer pipeline (vLLM plus nnsight) streams cached activations so a billion-scale corpus fits in memory. Training is one epoch at batch size 4096, learning rate 5e-5 with cosine decay, on a single A100; the longest run took 5.6 days. Three applications follow. (1) **On-manifold steering**: take an activation, add a scaled steering vector, then re-noise it to an intermediate timestep and integrate the learned velocity field back down — an activation-space copy of SDEdit — which is supposed to pull the edited point back toward the data manifold while keeping its semantic content. (2) **Quality checks**: Fréchet distance between 50k generated and 50k real activations, PCA overlays of generated versus real against deliberately degraded models simulated by truncating sampling steps, and ΔLM loss when reconstructed activations are injected back into the model. (3) **Meta-neuron probing**: run 113 binary concept tasks through 1-D probing on the denoiser's own gated MLP neurons, using an activation noised to a chosen timestep, with baselines of SAE features, raw layer outputs and raw MLP neurons, and selecting the best neuron by validation AUC.

## What counts as an explanation here

Not a description, a circuit or a dictionary of human-readable features in the usual sense. The explanation is a **generative surrogate for the activation distribution** — a model you can sample from, score against, and steer with — plus, as a second product, a set of units in that surrogate that each answer one concept. It is deliberately structure-free: the paper's claim is that whatever is interpretable about activations will show up as the surrogate gets more accurate, not that the authors know what it is. Prediction is real but distributional: the surrogate predicts what activations look like, not what the network will do.

## How they judged success

Fréchet distance (with the train-versus-validation FD as an irreducible-error floor and SAE reconstruction as a more favourable baseline); ΔLM loss on 2,048 held-out OpenWebText sequences, for both Llama-8B-Base and Llama-8B-Instruct; concept-versus-fluency Pareto frontiers with 95% bootstrap intervals, scored by LLM judges on 0–2 and 0–100 scales and, in a larger check, by a five-point sentiment classifier with negative log-likelihood for fluency over 1k prefixes; 1-D and dense probe AUC on 113 concepts with 95% bootstrap intervals; a pre-filtering validation (top-512 candidate neurons against all features); and a power law in FLOPs of the form loss = E + A·C^(−α), with downstream-task scaling laws fitted separately on checkpoints of the compute-efficient frontier.

## What they found

- **Generation fidelity beats a sparse autoencoder despite the harder setting.** Unconditional sampling from noise reaches FD 0.53 against an irreducible floor of 0.22 on Llama-1B (3.3B model), where an SAE reconstructing from real activations gets 1.99; on Llama-8B the 3.4B model gets 5.93 against an SAE's 6.91 and a floor of 2.60. ΔLM loss is 0.0513 for the generative model against 0.1976 for the SAE on Llama-8B-Base (0.0860 vs 0.2224 on Instruct), and the authors read this as the SAE's sparsity bias pushing its reconstructions off-manifold while theirs stay close to the real distribution.
- **The diffusion loss is a usable receipt for downstream quality.** Loss follows a smooth power law with irreducible error 0.52 and exponent 0.169, and halving the remaining gap takes roughly a 60× increase in compute. Better-loss checkpoints steer better on the sentiment task and probe better on 1-D concepts, and the probing curves show no plateau — the paper's headline for scaling.
- **On-manifold post-processing shifts the concept–fluency frontier outward in all three steering settings**: SAE feature steering (500 random directions on Neuronpedia descriptions), persona-vector elicitation on Llama-8B-Instruct, and difference-of-means sentiment steering, with the gain concentrated at relative coefficients of 1 or more, i.e. where the steering vector's norm exceeds the average activation norm. Since the model doing the post-filtering was trained on Base activations and still helps on Instruct, the prior transfers across a fine-tuning boundary.
- **Meta-neurons are the strongest 1-D encoder tested**: AUC 0.84 on Llama-1B against 0.70 for the SAE and 0.79 for raw MLP neurons, and 0.87 on Llama-8B against 0.76 and 0.82. A generative model trained on 1B activations (0.84) beats every read-out of the 8B model's own activations (0.77–0.82).
- **But dense probing is where the claim collapses to a tie.** Using all features rather than one, the generative model is indistinguishable from raw layer outputs and raw MLP neurons (both 0.92 on Llama-1B, 0.94 on Llama-8B) and beats only the SAE, which the authors concede means the tested concepts are already distributed in the raw activations and there is little headroom for any activation encoder.

## Limits and open problems they name

Single-token modelling only, so no cross-position structure; the model is unconditional, and conditioning on the clean activation rather than a noised one could reduce information loss for steering; one residual-stream layer, with a multi-layer model only explored in an appendix and performing slightly worse on the middle layer (FD 0.66 against 0.55 for the layer-specialist). Two things they do not name but the numbers show: the 1-D advantage is measured against baselines that were all evaluated under the same top-512 candidate filter, and the dense result implicitly bounds the 1-D result — a single unit that outperforms a dense linear read-out of raw activations is a statement about linearity of the raw read-out, not necessarily about the unit. The on-manifold correction is a fixed two-hyperparameter projection (timestep 0.5, 20 steps) that was never swept against alternatives, and its cost is 20 denoiser forward passes per token, which is not reported in the throughput terms that the steering literature this folder has read uses.

## What we take from it

A **framing and a warning**, in that order. The framing: this is the first entry in this folder whose surrogate is a density rather than a decomposition, and it makes "off-manifold" a measurable property (ΔLM loss when a reconstructed activation is injected) rather than an adjective — which is exactly the read-out a predictive white-box model needs for an edit, because the failure mode of an intervention is not that the target concept does not move but that everything else degrades. The rule to keep: **any intervention our model proposes ships an on-manifold check — a reconstruction score or a density score computed on the edited activation itself and compared with the same score on the unedited one — rather than only the task metric after the edit.** The warning is the dense-probing table: the authors' strongest unit-level claim survives only in a metric (1-D probe AUC) whose headroom is created by restricting the read-out to one scalar, and the honest baseline that uses all of them shows no advantage at all. That is the folder's recurring lesson in its sharpest form — **a per-unit interpretability score is only as meaningful as the dense baseline it is compared against, so any single-unit claim our model makes is reported beside the all-units score for the same task.**

## Tags

unit: feature | layer
form: replacement-model | description
method: decomposition | probing | steering | training-interpretable
eval: probing | steering | downstream-task | prediction/simulation
goal: understanding | control/editing | evaluation-of-methods
