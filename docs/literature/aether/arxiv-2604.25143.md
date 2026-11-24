# Gradient-Direction Sensitivity Reveals Linear-Centroid Coupling Hidden by Optimizer Trajectories

- **Key:** arxiv-2604.25143
- **Authors / organisation:** Yongzhong Xu (independent; no institution stated in the v1 HTML, contact is a personal address and a GitHub repo under the same handle as the carried prior papers)
- **Date:** 2026-04-28
- **Link:** https://arxiv.org/abs/2604.25143
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

The paper asks which parameter-space directions actually correspond to feature formation, and argues that the standard answer — top singular vectors of a rolling window of optimiser steps, which the author calls "update SED" — is measuring the wrong object. The motivation is that optimiser steps combine momentum, per-coordinate adaptive scaling and weight decay, and in a shared-encoder multitask model they additionally sum competing objectives, so the direction the optimiser moves is not the direction the loss is steepest. The author swaps the estimator for a rolling SVD of raw loss gradients at the current parameters and re-runs the same coupling diagnostic. The larger goal is a diagnostic of feature formation in *parameter* space that can serve as a monitor — the paper's own framing is that this is a methodological correction to a family of trajectory-based low-rank analyses, not a new theory.

## What they studied

A 2-layer pre-norm transformer encoder, d_model 128, 4 heads, d_ff 256, GELU, trained on modular arithmetic mod 97 (add, sub, mul, a²+b²). One single-task model per operation and one multitask model with a shared encoder and four linear heads over the summed loss. AdamW, lr 1e-3, weight decay 1.0, beta2 0.98, batch 512; checkpoints every 25 steps single-task and every 200 steps multitask. The unit of analysis is the attention-weight vector (131,072 parameters) and, on the representation side, a centroid vector per probe input in R^256 defined as the gradient of the correct logit with respect to the flattened input embedding, following the Linear Centroids Hypothesis of Walker et al. (2026, arXiv:2604.11962).

## How they did it

Two estimators of a spectral-edge basis are compared on the same checkpoints, probes and perturbation magnitude. Update SED takes the top-K right singular vectors of a rolling window of W consecutive AdamW steps on attention; gradient SED takes the top-K right singular vectors of a rolling window of W gradients of the loss on attention, evaluated at the current weights on a fixed batch of 512 examples resampled once at the start of analysis. Multitask adds a third: per-operation gradients, one backward pass per op, SVD'd within each op's own window. Coupling is measured by a finite-difference sensitivity: perturb the attention parameters by ±eps*v (relative eps 0.005 of the attention norm, everything else frozen), recompute every probe's centroid, take the mean squared norm of the central-difference response, and divide by the median response to 20 random Gaussian unit directions. That ratio R_k is the paper's single number. Ablations sweep window size, rank, perturbation magnitude and probe seed. A causal experiment then projects attention gradients (or, in a second version, the AdamW update) onto a rank-3 subspace and measures the step at which test accuracy reaches 0.98 across five conditions: natural update, remove the SED subspace, keep only it, remove a freshly drawn random 3-subspace, keep only that.

## What counts as an explanation here

A surrogate quantity: a low-rank subspace of attention-parameter space, claimed to be the subspace along which centroid features form, validated by a perturbation-response ratio against random directions. It is explicitly partial and explicitly non-causal — the author's own conclusion is that the recovered directions are diagnostic of feature formation but not uniquely responsible for it. Nothing here predicts a held-out quantity; the coupling ratio is a same-checkpoint, same-data measurement, and the causal test resolves to a statement about rank rather than about direction.

## How they judged success

The metric is R_k, the centroid-response ratio over the median random-direction response, reported as a peak and as a mean over the top 3; the null is 20 random Gaussian directions, not a shuffled or matched control. Robustness is checked by a four-axis ablation table (W, K, eps, probe seed), a three-seed reproduction of the cross-op table, and a rank-90 centroid-matrix trajectory that is reported as independent of the estimator. The causal experiment is graded by grokking step (test accuracy 0.98 with patience 20) over three seeds per operation, with the five intervention conditions as the comparison set and the random subspace as the load-bearing control.

## What they found

Update SED gives a centroid-coupling ratio of about 9x for addition and 3-4x for the other three operations, decaying after grokking. Gradient SED on the same checkpoints gives peaks of roughly 110-330x across the four operations on seed 42 and mean peaks of about 120-220x across three seeds — a 30-100x change from a one-line change of estimator. In the multitask model update SED falls to R_k within 0.1-0.9, below random, even though all four operations still reach at least 0.99 test accuracy; per-op gradient SED gives 20-45x for all four operations at the end of training. The window sweep is the honest part: R_1 falls monotonically from 648 at W=1 to 212 at W=20 (the paper's canonical setting) to 140 at W=40, and the author states that the W=1 value is near-tautological because the gradient and the centroid are built from the same Jacobians, so in single-task the rolling window is a denoiser and the temporal structure is not itself load-bearing. A per-example gradient SVD at a single timestep gives a rank-K basis with R_k of roughly 270-568 that is largely orthogonal to the mean gradient (cosine 0.05-0.35), which the author reads as a centroid-coupled subspace of at least four dimensions. The causal experiment is negative for direction and positive for rank: with gradient projection, keeping only the SED rank-3 accelerates grokking 2.20x and keeping only a fresh random rank-3 accelerates 2.36x, while removing either has no effect (1.02x and 1.00x) — so any rank-3 attention update accelerates and the choice of subspace is within seed variation. The 1.21x effect of removing the SED subspace under update projection disappears (0.98x) once the projection is applied before AdamW, which the author diagnoses as optimiser internal state leaking a full-gradient component into a projected update.

## Limits and open problems they name

Single architecture, single modulus (97), single optimiser and single hyperparameter setting; the window and rank were inherited from prior work and, the author acknowledges, not formally ablated (the W sweep in this paper does ablate the window, though the canonical rank-3 is only spot-checked). The centroid is the simplified single-logit form rather than the full J-transpose-1 centroid, the two agree on the rank-90 trajectory (62/76, 96/88, 23/18 at init, mid, post-grokking) but diverge mid-training (top-1 principal cosine 0.04). The "op-spread shrinks from 2.6x to 1.8x" claim is described as suggestive but not statistically tight at three seeds, with per-op peaks varying by a factor of about three across seeds. Sensitivity of R_k to the fixed analysis batch and its reseeding is not measured. The rank-3 acceleration is untested at other weight decay, learning rate, batch size, or projection targets, and the weight-decay-versus-variance-reduction explanation for it is explicitly left open, with a lambda=0 control that groks in no condition and therefore does not separate the two accounts.

## What we take from it

The transferable object is the estimator/pipeline split: this paper is direct evidence that a spectral quantity computed downstream of the optimiser is a different random variable from the same quantity computed at the gradient, and that the difference can cross the random-direction baseline. For our white-box model the useful artefact is the negative causal result — a subspace can be maximally diagnostic (100x+ coupling, reproducible across seeds) and simultaneously non-causal and interchangeable with a random subspace of the same rank at the same intervention site, so a coupling number alone can never license a mechanistic claim. The stimulus, the measurement protocol (finite-difference centroid response with a random-direction median as the null) and the multitask decomposition (never aggregate gradients across objectives before asking which directions matter to one objective) are all reusable. Two warnings: the paper's own headline numbers are computed at W=20 while its window sweep shows the quantity falling monotonically with W, so the reported values are a chosen averaging timescale, and the "op-dependence disappears" claim is supported by three seeds whose per-op spread overlaps.

## Tags

unit: direction | weights
form: direction
method: decomposition | weight-analysis | other
eval: ablation | downstream-task | benchmark
goal: understanding | evaluation-of-methods
