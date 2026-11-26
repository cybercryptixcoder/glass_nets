# Polymorphism Is Rotation: Operational Mechanistic Interpretability from a Two-Layer Transformer to Pythia-70m

- **Key:** arxiv-2605.24577
- **Authors / organisation:** Jordan F. McCann (independent researcher)
- **Date:** 2026-05-23
- **Link:** https://arxiv.org/abs/2605.24577
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Independently trained transformers of the same architecture look like they have learned the same features, and the field's headline evidence is the cosine between matched sparse-autoencoder decoder columns, which routinely lands in the high 0.9s. The author asks the sharper question - does that number mean the dictionary actually *works* on the other model - and finds it does not: an autoencoder trained on one seed reconstructs another seed's activations worse than a constant-mean predictor. The paper then explains the failure and repairs it with a single matrix multiplication. The larger goal is an operational standard for mechanistic claims: a small set of numerical bars fixed before analysis, so a mechanism claim either clears a threshold or has failed in a specific way, rather than being re-scoped into a story.

## What they studied

Two systems: a fully-interpreted 104,448-parameter two-layer transformer (d_model 64, 4 heads x 16, ReLU MLPs of 256, no biases, RMSNorm folded, analytic positional features, 40-token vocabulary, 64-token context) trained on bounded-depth Dyck-3 with three per-position heads (bracket identity, clamped depth, sticky validity flag); and nine publicly released independently-trained Pythia-70m seeds (d_model 512, 6 layers) on The Pile. The unit of analysis moves between whole models (as rotation equivalence classes), residual-stream bases, SAE encoder and decoder matrices, individual heads and MLPs (weight-level decomposition), and diff-of-means steering vectors.

## How they did it

Four pre-registered bars are defined first: behavioural (forward KL between a spec and the model under 1e-4), parametric (max per-entry weight MSE after searching the architecture's symmetry group, under 1e-3), causal (Pearson r between predicted and measured patch effects, above 0.99) and predictive (the same for ablation losses). Five lenses then describe every component - direct weight decomposition, SAEs and transcoders per site, causal interventions, a polyhedral enumeration of ReLU regions, and a constructive spec built by folding the trained primary seed itself. Two toy cohorts separate the issues: five seeds sharing frozen input/output projections with the primary (isolating interior weights from boundary relabelling) and five with fully independent initialisation. Cross-seed alignment is a multi-start search over head permutations, per-head rotations, neuron permutations and scalings, plus a Cayley-parameterised Riemannian refinement of the residual rotation. The functional test is to train an SAE on one seed's residual stream and apply it, unchanged, to another seed's; then to fit an orthogonal Procrustes rotation between the two seeds' centred activation matrices from one batch and re-apply. At Pythia scale the same audit runs over 7 residual sites and 36 cross-seed pairs, with the rotation's Frobenius distance and eigenvalue angles compared against closed-form Haar predictions. Steering transfer is measured by building three diff-of-means vectors on the anchor model and applying them unchanged to the others across an alpha sweep.

## What counts as an explanation here

A rotation: an explicit orthogonal matrix mapping one model's residual basis onto another's, claimed to hold to first order and at every internal site. It is partial by design - the paper says the rotation explains most of the polymorphism and leaves a second-order non-orthogonal residual near the output boundary - but it is predictive in a concrete sense, since it says in advance which of another model's feature reads and steering vectors will transfer cleanly, approximately, or with inverted sign. Around it sits a second, formal kind of explanation: the constructive spec, a folded copy of the trained model with per-tensor role labels, whose status the paper is unusually honest about (its causal and predictive bars pass by construction).

## How they judged success

Behavioural KL over 10^7 (sample, position, head) triples; parametric max MSE after symmetry search; Pearson r between predictions and measured mean-ablation and patch effects; SAE explained variance within seed versus across seeds, including negative values; Procrustes rotation Frobenius distance against the random-orthogonal prediction sqrt(2 d_model); a two-sample Kolmogorov-Smirnov test of the rotation's eigenvalue spectrum against Haar measure on the special orthogonal group, pooled and per pair; per-feature firing-magnitude correlations across seeds before and after rotation; and dose-response ratios of cross-seed to within-seed effects for steering. Two controls carry unusual weight: a Gaussian-noise covariance-matched baseline for SAE reconstruction (explained variance 0.94-0.98, better than an unrotated cross-seed SAE) and an adversarial decoy that row-averages the depth unembedding, which separates from the real model by a factor of 977 on the behavioural bar.

## What they found

Naive cross-seed SAE transfer is catastrophic: negative explained variance at every internal toy site, down to -6.56 at 32x expansion, and -2.11 to +0.75 across Pythia sites. One-batch Procrustes restores reconstruction to within 0.025 EV of the within-seed ceiling on the toy and to 0.85-0.99 on Pythia. The rotation is not a small perturbation: its Frobenius distance from identity is 31.99 (10th-90th percentile 31.94-32.03) at d=512, matching sqrt(2 x 512) = 32.00 to 0.1%, and its eigenvalue angles are statistically indistinguishable from Haar on SO(512) (KS statistic 0.0027, p about 1.000, pooled over 28,672 eigenvalues; per-pair statistics 0.0068-0.0104). Meanwhile decoder-cosine universality reproduces at 98-99% of features above cosine 0.5 with mean max-cosine 0.91-0.93, so the standard metric is measuring the half of the round trip that agrees. Firing-pattern overlap across seeds rises from essentially zero to 0.25-0.44 at Pythia and 0.03-0.51 to 0.27-0.74 at toy scale - most, not all, of the gap. Steering falls into three regimes at toy scale (clean when the vector is pinned by a shared output projection, partial with roughly a 4x dose penalty, inverted when it lives in the rotated subspace) and collapses to inverted-universally at Pythia, where nothing is shared. Cross-checkpoint within one Pythia run gives a rotation of 16.81, about half the cross-seed random value, with naive explained variance -0.86 lifted to +0.73. Joint optimisation of weight and activation alignment never clears the parametric bar at any mixing weight, and the weight-optimal and activation-optimal rotations sit in disconnected basins. Within-seed, only the behavioural bar is substantive (1.74x its threshold, localised to the depth head); causal and predictive pass tautologically because integrated gradients makes the predictor an identity with the measurement - and the paper says so, having first documented that attribution patching anti-correlates with measured patch effects on converged models (r from -0.632 to +0.585 across five seeds, and 0.05 versus 0.98 for integrated gradients at Pythia).

## Limits and open problems they name

No frontier-scale test; the toy uses coordinated shared-I/O training for its primary cohort; the behavioural bar is a quantified near-miss rather than a pass; 10^7 samples rather than 10^8; the cross-seed predictive bar collapses into the causal one because both use component-level mean ablation (the columns are bit-identical, which the paper states); single-component enumeration will not scale to billion-parameter models without a sampled estimator that has not been validated; and the cross-checkpoint result is one checkpoint pair at one layer of one model. The residual firing-pattern gap after rotation is attributed to the unshared output boundary but not decomposed.

## What we take from it

Three things. First, a warning that reframes the folder's oldest stability question: if the residual basis differs between two independent fits by a near-uniform random orthogonal draw, then reporting a "recovered direction" from one fit and checking its stability in another fit's coordinates is measuring the rotation, not the mechanism - comparison has to happen after an explicit Procrustes fit, and the fit's distance from identity should be reported as the size of the change being discarded. Second, a concrete method: one batch, one Procrustes, and the resulting matrix is a testable object whose distribution (Frobenius distance and eigenvalue angles against closed form) can be checked rather than assumed, and whose invariant subspace predicts, before any intervention is run, which of our read-outs will port and which will come back with the wrong sign. Third - the methodological move most reusable of all - the decoy control: build a sibling model that differs in exactly one documented way and show the bars separate it by orders of magnitude, because a bar that cannot fail distinguishes nothing.

## Tags

unit: whole-model, weights
form: replacement-model, direction
method: weight-analysis, SAE, decomposition, patching, ablation, steering
eval: ablation, patching, prediction/simulation, faithfulness
goal: understanding, evaluation-of-methods
