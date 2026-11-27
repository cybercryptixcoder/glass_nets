# When Can We Trust the Sparse Lens? A Certification Framework for SAE Faithfulness

- **Key:** arxiv-2606.18383
- **Authors / organisation:** Dibyanayan Bandyopadhyay, Asif Ekbal; Indian Institute of Technology Patna
- **Date:** 2026-09-26 (v2; v1 2026-06-16)
- **Link:** https://arxiv.org/abs/2606.18383
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

The field uses sparse autoencoders as windows into frozen language models, but nobody had stated the condition under which that window can be trusted as a statement about the model rather than about the autoencoder. The authors want a mathematical certificate: a post-hoc upper bound on the population risk of the untouched language model, obtained by passing through the SAE. The larger goal is to convert a qualitative reading practice ("the features look interpretable, so the decomposition is informative") into a decidable question with a number attached, and to say when the answer is no. They are explicit that this is *predictive* faithfulness only — no claim about whether individual features are meaningful, causal, or human-legible.

## What they studied

Three frozen model–SAE pairs: GPT-2 Small with a residual-stream dictionary at layer 6 (24,576 features), Gemma-2B at layer 12 (16,384), and Llama-3-8B at layer 30 (131,072, EleutherAI Top-k SAE), with a layerwise sweep over layers 4, 8, 12, 16, 20, 24, 28, 30 on Llama. Inputs are 32-token English C4 sequences, scored with a smoothed bits-per-dimension loss at smoothing 0.5 so the loss is bounded. The unit of analysis is the whole model's risk, mediated by an SAE reconstruction at one layer — not a feature, head or circuit.

## How they did it

Run the base model to the target layer, encode the hidden state with the pretrained SAE, keep a Top-k sparse code (k = 64 for every model at evaluation, applied to ReLU/L1 activations for GPT-2 and Gemma and to raw encoder pre-activations for Llama's native Top-k dictionary), decode back to activation space, and resume the ordinary forward pass. That patched network is the SAE proxy. On a disjoint calibration stream of 2.24M tokens they record every feature that ever appears in a Top-k support, forming a pool; the proxy is then restricted to that pool at evaluation. Three held-out quantities are estimated and plugged into a Hoeffding-based bound at 95% confidence: the proxy's own risk, the per-example gap between the base model's loss and the proxy's loss, and the rate at which a held-out input's Top-k support falls outside the calibration pool. A more conservative variant makes the proxy-risk concentration term hold simultaneously over all pools of the same size, paying a binomial combinatorial penalty. They then re-run certification at each patch layer, and separately run a horizon-conditioned rollout — feed base or reconstructed activation through h downstream blocks and measure the divergence at each horizon — to separate local reconstruction quality from downstream amplification. Two ablations: shuffle which coefficient belongs to which feature identity (preserving values, sparsity and pool size exactly), and threshold the pool by feature frequency.

## What counts as an explanation here

A number, not a description. The "explanation" of whether the SAE lens is faithful is a certified upper bound on the frozen model's population risk that routes through the SAE, compared against the loss of an uninformed uniform predictor. The bound is deliberately decomposed into three named terms so that a loose certificate localises *why*: bad proxy, bad reconstruction, or bad coverage. It is an incomplete object by design — they state that the direct bound on the model is tighter and that theirs is not offered as the tightest bound, only as the one that says whether the SAE-mediated reading carries predictive information. No features are named, no circuits are drawn, no predictions are made about what any individual latent does.

## How they judged success

Non-vacuity: the certified bound must fall below the uniform-predictor baseline (log₂ of vocabulary size, 16.9687 bits for Llama and GPT-2 at this tokenisation). They report the first evaluation sample size at which the bound crosses that line, and they label this crossing explicitly as a descriptive statistic with no simultaneous confidence guarantee. Supporting measures: a layerwise decomposition of the three bound terms plus the combinatorial term, an "asymptotic floor" that fixes the measured terms and lets only the finite-sample corrections vanish, output-fidelity metrics against the base model (KL divergence, top-1 token agreement, absolute gold-token log-probability gap), and zero-shot accuracy on WinoGrande, PIQA and HellaSwag as a behavioural cross-check that they explicitly say does not validate the theorem. The shuffling ablation is the control: if sparsity and coefficient values alone determined fidelity, permuting identities should change nothing.

## What they found

All three pairs certify at practical sample sizes: first crossings at 2,000 (GPT-2 Small), 3,008 (Gemma-2B) and 5,008 (Llama-3-8B) evaluation samples, with bounds of 13.962, 16.906 and 16.464 bits against baselines of 15.617, 17.966 and 16.969. The direct bounds are much tighter (7.617, 6.338 and 5.593 respectively), which is the point: the SAE-mediated bound is only interesting as a statement about the decomposition. On Llama the depth effect is severe and decomposable — from layer 4 to layer 30 proxy risk falls 15.713 to 6.206 bits and the reconstruction gap 10.42 to 0.92, output fidelity improves on every measure (KL 12.051 to 1.112 bits, top-1 agreement 0.008 to 0.670), and the *asymptotic floors* at layers 4, 8, 12 and 16 are 26.29, 25.24, 19.32 and 17.05 bits, all above the baseline, meaning those layers can never certify no matter how much data you collect; layers 20, 24, 28 and 30 floor at 12.52, 10.72, 10.55 and 7.17 and cross at 450k, 150k, 90k and 10k samples. The horizon rollout shows later layers win on both axes — lower local distortion and flatter growth with horizon — so the effect is not just a shorter path to the output. GPT-2 Small shows almost no depth sensitivity, which the authors take as evidence the Llama pattern is a property of the pair, not of depth in general. Calibration data buys coverage, not fidelity: for Llama η falls 0.2466 to 0.0029 as calibration grows 0.28M to 2.24M tokens while proxy risk (~6.207) and reconstruction gap (~0.915) barely move. Feature-identity shuffling substantially increases the reconstruction gap on all three pairs with values, sparsity and pool size held fixed. Certificates stay non-vacuous under 15% token corruption and go vacuous on random-token input.

## Limits and open problems they name

"Faithfulness" here means only risk-level behavioural fidelity; nothing in the framework speaks to semantic or causal interpretability, and the authors say so twice. The headline non-vacuity result depends on the calibration pool covering nearly the whole dictionary — pool-to-dictionary ratios of 0.981 to 0.997 — which they concede is near-saturation and test with a frequency threshold that degrades the certificate as the pool shrinks (retaining only features seen at least 50 times makes the bound vacuous at every evaluated size). The cross-model comparison is explicitly non-causal: model family, SAE release, layer and hookpoint all vary together. The first-crossing statistic carries no joint guarantee. One primary SAE checkpoint per family, one dataset (C4), one tokenisation of "risk" (smoothed BPD), no downstream safety property certified. Their own shuffling ablation is acknowledged not to establish interpretability either, since a non-interpretable dictionary could be equally permutation-sensitive.

## What we take from it

- **A framing we should adopt wholesale:** a white-box object is useful only if you can state what it preserves *about* the black box, in a quantity with a direction and a baseline; the discipline of subtracting a proxy's own performance from the model's and charging the difference to named failure terms is exactly the reporting shape our predictive model needs.
- **The asymptotic-floor calculation is the cheapest instrument in the paper:** fixing the measured terms and letting only the sample-size corrections vanish tells you, before you spend any compute on more data, whether a patch point *can* ever be informative. We should run this on every candidate site in our own model: it converts "we need more samples" into "this location cannot work", which is a decision, not a measurement.
- **The horizon-conditioned rollout is a reusable separator:** local fidelity versus downstream amplification, measured by feeding base and reconstructed state through the same subnetwork at increasing depth. Any time our model proposes that a component's reconstruction is faithful, this tells us whether the error is small or merely short.
- **A warning that sits directly on our survey's theme:** the guarantee holds precisely where the feature pool covers almost the whole dictionary. Near-total coverage is what makes the combinatorial term small and what makes the bound honest — but it is also the regime in which the sparse decomposition is doing the least compression work. A certificate that only holds when the dictionary has collapsed back to (nearly) the full activation space is a real constraint on what sparsity buys, and the frequency-thresholded table is the paper's own demonstration that the certificate dies as the pool gets selective.
- **A clean control to copy:** shuffling coefficient-to-identity assignment while preserving values, sparsity and pool size. It is one line of code and it rules out the null that fidelity is a function of activation statistics alone.

## Tags

unit: whole-model
form: replacement-model
method: ablation | weight-analysis
eval: prediction/simulation | downstream-task | benchmark | ablation
goal: understanding | evaluation-of-methods
