# Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data

- **Key:** arxiv-2404.01413
- **Authors / organisation:** Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Davilov, Dhruv Pai (Stanford), Henry Sleight, John Hughes, Tomasz Korbak, Rajashree Agrawal (Constellation), Andrey Gromov (University of Maryland, College Park), Daniel A. Roberts (MIT & Sequoia Capital), Diyi Yang, David Donoho, Sanmi Koyejo (Stanford); arXiv:2404.01413v2
- **Date:** 2024-04-29 (v1 2024-04-01)
- **Link:** https://arxiv.org/abs/2404.01413
- **Kind:** paper
- **Read from:** full text (arXiv HTML v2, ~72.8k chars; abstract, §1–§4, references, Appendices A–F; figures not inspected)
- **Relevance to us:** 3 (central for the data-provenance thread; peripheral for probe design)

## What they wanted, and why
The model-collapse literature argued that a model trained on its own outputs degrades with each generation until it is useless. Almost every such study assumed a *replacement* regime: each round's synthetic set overwrites the last, keeping dataset size fixed across iterations. The authors point out that this is the wrong physical picture for real training corpora, which have grown monotonically (Llama 1's 1.4T tokens, Llama 2's 2T, Llama 3's 15T). If data accumulate instead of replace, does the curse still hold? The paper's ambition is to show that the catastrophic conclusion is an artefact of one modelling choice, and to supply the arithmetic that explains why.

## What they studied
Three deep-generative setups plus one analytic one, all across model-fitting *iterations* — a sequence of freshly initialised models rather than a single model trained continuously. (1) Causal transformer language models — GPT-2 at 9M parameters and Llama 2 at 12M, 42M and 125M — pretrained for one epoch on TinyStories, a 470M-token corpus of short kindergarten-level stories that was itself machine-generated. (2) GeoDiff, a geometric diffusion model, on 40,000 molecular conformations down-sampled from GEOM-Drugs, 50 diffusion steps per prediction. (3) Convolutional VAEs (five layers, 32/64/128/256/512 channels, 128-dimensional isotropic-Gaussian latent) on CelebA at 64×64, 100 epochs per iteration, 163K sampled images per round. (4) A closed-form sequence of ridgeless linear regressors. The unit of analysis is the *dataset* and its error trajectory over generations, not any internal representation.

## How they did it
For each round n ≥ 2 the previous round's model samples a fresh dataset the same size as the original, and the trainer either (R) discards the old data and trains only on the new synthetic set, or (A) concatenates: the accumulated pile of all previous real and synthetic data. Sampling temperature was 1.0 in the main runs and 0.3 in one arm. Evaluation is cross-entropy on a held-out real split for the language models, the standard weighted variational lower bound for GeoDiff, and reconstruction-plus-KL for the VAEs. Four ablations guard the comparison: a "replace-multiple" arm that keeps replacement but grows the synthetic set to the *same size* the accumulate arm would have (the obvious confound, since accumulate sees more tokens per round); a temperature arm; a one-fifth-dataset arm at one and three epochs; and an arm varying how much training the *first* iteration gets, to rule out a bad first synthetic round. The theory uses the Mobahi/Dohmatob framework in which each iteration fits a linear model to targets drawn from the previous iteration's fitted model, extended to the accumulating covariate-and-target stack.

## What counts as an explanation here
A generative account of a failure mode: a closed-form expression for test error as a function of iteration count, derived from an explicit data-generation process, matched against measured error curves in four different settings. It is meant to be a mechanism claim rather than a description — the theorem names the quantity (the accumulated noise coefficient) whose summability does the work. It predicts the qualitative shape of every curve (linear divergence versus plateau) but nothing about any individual model's internals.

## How they judged success
Test error versus model-fitting iteration — cross-entropy for the language models, test loss for GeoDiff and the VAE, mean squared parameter error for the linear arm. Baselines are the replacement regime itself and, critically, the replace-multiple matched-size ablation. The theoretical claims are exact expectations under stated assumptions (full-rank design, T ≥ d + 2, isotropic features), and the numerics are checked against the closed form.

## What they found
Replacement collapses in every setting: GPT-2 9M validation cross-entropy moves 1.82 → 2.39 at iteration 4 and 2.91 at iteration 10, while accumulation sits at 1.74 at iteration 4 — *below* the first iteration. The same pattern holds for all four Llama 2 sizes, for GeoDiff's molecular loss over 8 iterations, and for the VAE, which under replacement degenerates until every generation is the same mode. The matched-size control is what makes the headline credible: a replacement arm grown to the accumulating set's size still degrades (GPT-2 9M at 2.18 at iteration 4 against 1.74 accumulate), so the benefit is not token count. Lower temperature collapses faster under replacement (5.82 at iteration 4, 9.85 at 10 for temperature 0.3). The theorem explains it exactly: with accumulation the ith round enters the design with weight 1/i, so its noise contributes at 1/i², and the sum is bounded by π²/6 times the base noise-to-signal prefactor; with replacement the error is that same prefactor times n, linear and unbounded; the halfway "replace-multiple" case lands at log(n). Two honest exceptions: the VAE's accumulated error does creep up slowly even though it never collapses, and the paper concedes that one prior study reported fast deterioration under accumulation on a smaller architecture and leaves the discrepancy unresolved rather than explaining it away.

## Limits and open problems they name
Model scale is tiny — TinyStories models top out at 125M and the datasets are the toy-but-realistic kind, so no claim here is tested at web scale. The synthetic data are all sampled stochastically; deterministic generation (temperature 0) is named as untested. Real-data growth across iterations, varying synthetic-to-real schedules, and filtering of generated data (as RLHF does) are all left to future work. Most importantly the authors separate four distinct phenomena that have all been called model collapse — unbounded error blowup, collapse to a few modes, collapse to uniformity, and amplification of inherited artefacts — and say plainly that they have only measured the first. Their own VAE result shows that avoiding the first does not avoid the second: the accumulated model loses variance along minor axes and still reconstructs them. And they argue with a rival paper rather than proving it: they re-read the accumulating arm of a prior StyleGAN2 experiment as plateauing, where its authors concluded accumulation only delays collapse.

## What we take from it
- **Method:** the replace-versus-accumulate-versus-replace-multiple three-arm design is the template for any of our own iterative-training or self-distillation study, because the middle arm is what converts "more data" into a separable explanation.
- **Measure:** report error-versus-iteration curves with the noise-to-signal prefactor named, since the theorem says the plateau *height* is set by noise per dimension over sample count, not by the recursion.
- **Warning:** "the model degrades" and "the model narrows" are different failures with different arithmetic, and a fix for one is not evidence about the other. Our diversity findings (hivemind's within-model saturation, verbalized sampling's Tulu-3 stage ladder) describe the second kind, and this paper explicitly does not address them.
- **Framing for the provenance thread:** inherited training data is not a fingerprint that persists because of some property of the student — it is a weighted average whose weights decay as 1/i. A signature that survives is one carried by the *early* data; that is a testable prediction about which habits should and should not persist across a distillation lineage.

## Tags
unit: whole-model
form: causal-abstraction
method: other | decomposition
eval: prediction/simulation | downstream-task
goal: understanding | evaluation-of-methods
