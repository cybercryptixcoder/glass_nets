# Neighbor Crowding Predicts Feature Stability in Sparse Autoencoders

- **Key:** arxiv-2611.33182
- **Authors / organisation:** Denis Federico Lim (KAIST / University of Pennsylvania / University of Cambridge / UC Berkeley), Naveen Dasari, George Flint & Kevin Zhu; work conducted in the Algoverse AI Research programme. arXiv 2611.33182, PMLR workshop paper (Symmetry and Geometry in Neural Representations, 2026)
- **Date:** 2026-11
- **Link:** https://arxiv.org/abs/2611.33182
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

An SAE dictionary is only worth reading if its entries survive a different random initialisation; otherwise a "feature" is a fact about the optimiser's starting point rather than about the network. Prior work established that TopK SAEs share only a minority of features between seeds, and that the honest way to find the survivors is to train several SAEs and match them. That is expensive, and it scales with the number of models one wants to inspect. The authors ask a cheaper question: can the probability that a given latent will reappear be predicted from statistics of the run that produced it, before any second seed exists? Their larger goal is a single-run quality signal for dictionary entries, so that a reader of an SAE can tell which of its features are likely to be properties of the model and which are noise.

## What they studied

TopK SAEs trained on four language models: Pythia-70m (layer 3 of 6, width 512, 8,192 latents, a 16x expansion), Pythia-160m (layer 6 of 12, 12,288 latents), Pythia-410m (layer 12 of 24, 16,384 latents) and GPT-2 small (layer 6 of 12, 12,288 latents). Main sweeps vary the TopK sparsity k over 64, 128 and 256, hold three seeds fixed, and record six training-token checkpoints from 50M to 1B. The unit of analysis is the individual latent (SAE feature), represented by its unit-normalised decoder column, and the label is its stability across seeds.

## How they did it

All seeds of a given setting see the same data stream in the same order, so the only thing that differs between runs is initialisation; the authors are explicit that this makes seeds easier to match than they would be in an independent-shuffle regime. The SAEs use the standard TopK recipe with tied initialisation of the encoder to the decoder transpose and an auxiliary reconstruction loss with a small coefficient to revive dead latents, which keeps every seed comparable. Stability is labelled by the convention inherited from earlier seed-dependence work: normalise each decoder column, compute the cross-dictionary cosine matrix, use many-to-one argmax matching, and call two features the same when the best match exceeds 0.7; a feature is stable only if its best match in *every* partner seed clears the threshold, unstable if none does, and partially-matching features are dropped from the classifier entirely. Six statistics are computed on a held-out 1.31M-token stream from a single run: activation frequency, mean activation conditional on firing, the mean cosine of each decoder column to its ten nearest neighbours inside the same dictionary (neighbor crowding), the per-token MSE increase from zeroing the feature, encoder column norm, and encoder bias. Statistics are z-scored and a class-balanced, L2-regularised logistic regression is scored by mean AUROC over stratified five-fold cross-validation. Transfer is tested three ways, source seed to target seed, early checkpoint to later checkpoint, and Pythia-70m to the larger Pythia models and to GPT-2 small, with the standardisation either carried over from the source or refitted on the target's (unlabelled) statistics.

## What counts as an explanation here

A per-latent stability score: a scalar probability that a decoder column will be recovered in another seed, decomposed into six named geometric and dynamical statistics. It is deliberately not a semantic description of the feature and says nothing about what the feature means or does. It is meant to be partial and predictive, a screen that tells the reader which entries of the dictionary have earned the right to be interpreted at all, plus an interpretation of *why* the successful ones survive (they sit where the dictionary is crowded).

## How they judged success

AUROC of the stability label, always reported beside the activation-frequency baseline, with the headline number being the difference from that baseline. Controls: a permutation test with 1,000 relabellings to show the classifier sits at chance when labels are shuffled; a rotated-partner null, where the other seed's dictionary is multiplied by a random orthogonal matrix so its geometry is kept and its correspondence destroyed, against which neighbor crowding is required to sit at chance; an activation-overlap check correlating each stable feature's firing pattern with its cross-seed match; a five-seed relabelling in place of three; and a re-calibration of the matching threshold per k using a null distribution of random-feature similarities. Leakage guards: a floor of 100 firings so the classifier cannot win by identifying dead latents, and exclusion of partially-matching features.

## What they found

Stability falls as k rises. At 1B tokens roughly 64% of latents are stable at k=64, about 56% at k=128, and near zero at k=256, which is not explained by dead latents (69.3% of latents are dead at k=256, but even among live ones only about 16% are stable) nor by a miscalibrated threshold (re-calibrating per k still leaves k=256 far below the others). The six statistics reach AUROC 0.868 at k=64 against 0.606 for activation frequency alone, a gap of about 0.26; the classifier fitted on one seed lands within 0.0013 of another seed's own ceiling. Neighbor crowding alone reaches 0.816, ahead of reconstruction contribution (0.802) and mean activation (0.780), and remains the strongest single predictor inside GPT-2 as well (0.698). Removing crowding costs 0.032 AUROC, comparable to removing mean activation (0.037), while every other statistic costs under 0.01, so the full classifier is carried by two statistics rather than six. Cross-seed transfer is near-lossless, cross-scale transfer from Pythia-70m costs 0.003 to 0.012 AUROC at Pythia-160m and Pythia-410m, and cross-family transfer to GPT-2 small costs 0.051 but only after re-standardising on the target; with the source's scaling the score collapses to 0.414, below chance, because one statistic's spread differs by two orders of magnitude between families. Transfer across training budget is the weak case: fitted at 50M tokens and applied at 1B it scores 0.697, about a third of the ceiling's gain, even though the statistics are strongly predictive within the 50M checkpoint itself (0.937). The rotated null leaves crowding at about 0.501, and matched pairs fire on the same tokens (median correlation 0.90) while the second-best match, the nearest within-dictionary neighbour and a frequency-matched random feature do not (at most 0.03).

## Limits and open problems they name

Two model families and four model sizes, with the two largest targets trained for only 500M tokens. Most results use three seeds. One model's 1B run was resumed from the start of the data stream, so its token count is not 1B unique tokens. The fixed data order shared by all seeds is the authors' own named confound and they do not test a reshuffled stream; it likely inflates the mean best-match cosine. Untested: other layers, other SAE architectures, and whether stability survives data reordering. And their strongest stated caveat: matched features respond to the same inputs, which is not the same as playing the same causal role, and establishing that would need interventions.

## What we take from it

- **A warning.** A feature-level claim of the form "SAE feature X in model M" is contingent on the seed, and the share of latents that reproduce can be a bare majority at best and near zero at a larger k. Any white-box model built on dictionary entries should carry a per-entry survival estimate, not a dictionary-level claim.
- **A method.** The six-statistic screen is cheap, width-agnostic, and transferable across model families after refitting only the standardisation on unlabelled target activations, so stability labels need to be paid for once. That is exactly the budget shape we want for a prediction model that must generalise to a model we cannot afford to re-fit.
- **A measure with a null.** The rotated-partner null is the cleanest control in the paper: it keeps the geometry and destroys the correspondence, so a predictor that still scores above chance is measuring density rather than recurrence. Any geometric predictor we report should be run against a geometry-preserving, correspondence-destroying null.
- **A caution about the label.** Stability here is cosine matching at a fixed threshold, and the causal role question is left open; the paper says so. A stable feature is a stable *direction*, which is a weaker object than a stable *function*.
- **A scale-dependent failure we should expect.** The cross-family collapse to 0.414 with the source scaling is the strongest evidence in the paper that a z-scored statistic's weight comes from its spread, not its meaning; a predictor of ours that stacks heterogeneous statistics must state where its scaling came from.

## Tags

unit: feature
form: probe
method: SAE, probing, weight-analysis
eval: prediction/simulation, benchmark, ablation
goal: understanding, evaluation-of-methods
