# Shaped SAEs: Expanding the Toolkit for Mechanistic Interpretability

- **Key:** arxiv-2610.27375
- **Authors / organisation:** Oveys Delafrooz Noroozi, Bhagyashree Puranik, Upamanyu Madhow; University of California, Santa Barbara
- **Date:** 2026-10-19
- **Link:** https://arxiv.org/abs/2610.27375
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

A sparse autoencoder assumes that the coordinate system it is handed is the right one to be sparse in, and buys its dictionary out of whatever variance the activations happen to have. The authors' question is whether the *input geometry* is a free design choice: if you linearly reshape the activations before the dictionary sees them, do the learned atoms become more semantically selective at the same feature budget? Their particular shaping is borrowed from the J-space literature — average the Jacobian from the layer to the output over a corpus, take its SVD, and project out the low-gain directions on the argument that what a layer writes downstream matters more than what happens to be large. Separately, they propose a new dictionary parameterisation in which the encoder's matching rule and the decoder's synthesis use the *same* unit-norm atoms, so that atom selection scores alignment rather than a free-inner-product-plus-bias and coefficients are least-squares fits rather than pre-activations. The larger goal is to expand the space of available sparse representations rather than to settle the field's argument about whether SAEs are worth having.

## What they studied

Gemma-2-2B (d = 2304, 26 layers) at residual-stream layers 12 and 20, plus GPT-2 Small (d = 768) at layers 7 and 8 in the appendix. Dictionaries of 16K and 32K atoms, k ∈ {40, 80, 160, 320, 640} active atoms per token, trained on the uncopyrighted Pile subset for 7,000 steps (~28.7M tokens), AdamW at 2e-4, with a small auxiliary coefficient-magnitude penalty at λ = 0.01. The unit is the dictionary atom (a feature) and, in the shaping-only arm, the coordinate of the projected residual stream. The live subspace uses r = 1536 of 2304 directions, with r ∈ {768, 1152, 1536, 1792} swept.

## How they did it

Estimate the averaged Jacobian once per layer by reverse-mode differentiation over 1,000 prompts truncated to 64 tokens with the final block as target, take its SVD, and define the live space as the span of the right singular vectors with the largest r singular values. Shaping is then a rank-r projection of each residual activation onto that span; the quiet component is discarded, and for the live arm the reconstruction error is measured only on live coordinates, normalised by r. Two SAE arms are trained on top: the raw MF-SAE on unshaped activations and the live MF-SAE on projected ones. The MF-SAE maps the input through a learned affine encoder to a latent, scores every unit-norm atom by its rectified inner product with that latent, keeps the top k, reconstructs the latent as the least-squares combination of the same atoms, and maps back through a learned affine decoder. Selection scores are detached and the selected coefficients are recomputed differentiably.

They also run shaping *without* any SAE — probing the shaped coordinates directly, which separates the effect of dropping quiet directions from the effect of sparse coding. For the live SAE they patch the model three ways: replace the activation with the reconstruction, replace it with the reconstruction plus the exact quiet residual restored, or replace it with the projection alone (which bounds the cost of dropping quiet directions without any SAE). Baselines are dense activations, held-out records of SAEBench TopK SAEs at matching k and at a token budget matched to their own training, Gemma Scope JumpReLU at 16K and 32K paired to the nearest average L0, and a 32K Matryoshka BatchTopK at L0 = 40. Evaluation is SAEBench sparse probing (36 binary tasks, probes on the top-k_probe class-discriminative features), the 113-dataset SAE Probes benchmark, fraction of variance explained, fraction of alive atoms, cross-entropy recovered under each patch mode, and an automated interpretability detection score from an LLM-written one-line explanation tested on held-out activating windows.

## What counts as an explanation here

A feature dictionary — a set of atoms with a scoring rule and a reconstruction map — with the novel claim being that the *metric* in which the dictionary lives is itself a learned, shaped object. The MF-SAE's interpretability story is geometric rather than verbal: atoms are unit directions, coefficients are non-negative projections, and the outer linear maps are described as whitening and centring in the signal-processing sense. It is explicitly partial. The paper does not claim the atoms are the model's computational units, cites the benchmarks that say SAE features do not reliably beat dense or supervised baselines for causal localisation and steering, and states that it is not trying to resolve that question.

## How they judged success

SAEBench sparse-probe accuracy averaged over 36 tasks as a function of the number of probe features; mean test accuracy over the 113 SAE Probes datasets; PVE; fraction of alive atoms; CE recovered = (CE_zero − CE_rec)/(CE_zero − CE_clean) with CE_zero from zero-ablating the activation; and a balanced-accuracy detection score from an LLM judge over 12 shown and 8 held-out windows against 8 random-feature windows. Controls and ablations: a no-SAE shaping arm at four values of r, a matching-token-budget TopK baseline separate from the 300M-token checkpoint, sparsity sweeps at both dictionary sizes, a 32K replication, a GPT-2 replication, and an ablation removing the linear encoder and decoder from the MF-SAE. No random-direction or shuffled-label arm is run for the shaping claim, and no confidence intervals are reported anywhere in the main text.

## What they found

The two contributions do not stack, and the paper's own summary is the honest statement of that. Shaping without an SAE is a clear win on SAEBench: the projected coordinates beat the raw ones at both layers and at every r tested, and are comparable at layer 12 and slightly better at layer 20 on SAE Probes — so discarding the quiet 768 directions does not remove information the probes need and makes it easier to reach. But once a dictionary is fitted, the gain largely evaporates. Both MF-SAEs beat every available baseline at both layers and both headline sparsities on SAEBench; raw and live MF-SAE are similar in most panels, with the live arm consistently better in exactly one (layer 12, k = 80); and on the 113-dataset SAE Probes benchmark the improvement over baselines is only marginal, which the authors note is consistent with that benchmark's original negative result.

The reconstruction table is where the arithmetic is visible and where the honest reading differs from the abstract. The raw MF-SAE gets PVE 0.86–0.92 and recovers 97–99% of CE. The live MF-SAE gets 0.79–0.86 PVE *on its own projected input* and recovers 89–94% when patched alone — but the projection-only arm already recovers 92–95% without any SAE, so most of that shortfall is the price of dropping the quiet subspace, not of sparse coding; restoring the quiet residual recovers 99% in every setting. Two properties the MF-SAE does earn cleanly: dead atoms are basically absent, 81–99% of the live dictionary alive against 62–89% for the raw arm, with no auxiliary dead-latent loss; and the linear projections are load-bearing, their removal significantly reducing probe accuracy in both the raw and live arms. The architectural argument behind that ablation is a clean piece of algebra — in a TopK SAE an outer linear map can be absorbed into the free encoder rows and decoder columns and is therefore redundant, whereas in the MF-SAE tying encoder and decoder atoms up to one shared linear map makes those maps the only thing that lets the two differ.

## Limits and open problems they name

Combining semantic shaping with the matched-filter dictionary yields gains only at tight feature budgets; how best to combine them is stated as open. The live/quiet split is justified empirically by a rapidly decaying Jacobian spectrum but is not analysed in depth, and the quiet space in particular is named as a direction for future work — the Jacobian is a linearisation averaged over contexts, so a nonlinear or context-dependent shaping is untouched. The live PVE is measured on the projected input and is therefore not comparable to the raw PVE, which they flag rather than paper over. Coverage is two small models, residual-stream hooks only, one corpus, SAEBench and SAE Probes only.

## What we take from it

A **warning about the reach of a representation choice, and one transferable design rule**. The warning, which is the most useful thing in the paper and which the authors state but do not emphasise: a shaping step that provably helps the *coordinates* can stop helping once a *learner* is inserted downstream, because a dictionary fitted to the raw space was already allowed to find a comparable projection. Their own numbers make the point — shaping without an SAE wins on both benchmarks, shaping plus an SAE wins in one panel out of four — so any claim of the form "our preprocessing makes features more semantic" has to be measured both with and without the learner, or it is measuring what the learner would have done anyway. This is the folder's familiar confound (a pipeline whose gain belongs to an arm it is compared against) in its cheapest form, and the no-SAE arm costs one logistic regression.

The design rule is worth copying for the opposite reason: it costs nothing and buys coverage. Encoding and decoding with the *same* unit-norm atoms, with a shared learned affine map as the only asymmetry, gives 81–99% alive atoms with no dead-latent machinery, and the tying argument — a TopK SAE's outer linear maps are absorbable and therefore redundant, this one's are not — is a small piece of analysis that tells you which architectural knobs are real. The measurement habit the paper supplies for our own dictionary work is the three-way patch (reconstruction alone, reconstruction plus the discarded component restored, discarded component alone), which separates "my dictionary loses information" from "my input transform loses information" with two extra forward passes and no retraining.

## Tags

unit: feature | direction
form: dictionary
method: SAE | decomposition
eval: downstream-task | ablation | benchmark
goal: understanding | evaluation-of-methods
