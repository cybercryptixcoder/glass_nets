# On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders

- **Key:** arxiv-2605.31518
- **Authors / organisation:** Elana Simon (Stanford), Etowah Adams (Columbia), James Zou (Stanford)
- **Date:** 2026-05-29
- **Link:** https://arxiv.org/abs/2605.31518
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1: abstract, sections 1-8, all six main figures' captions, Tables 1 and S7 read as cells, appendices A-E skimmed by section heading and by the passages referenced from the main text; figures 1-6 and the supplementary figures were not inspected as images)
- **Relevance to us:** 3 (central)

## What they wanted, and why

The same sparse-autoencoder recipe produces an almost death-free dictionary on GPT-2 and a mostly dead one on AlphaFold3, and nobody could say why. Dead features are wasted capacity, and a dictionary that silently loses most of its latents is worse than a small dictionary because the reported size no longer means anything. The authors want a diagnosis that can be computed *before* committing compute to a training run, and a fix that follows from the diagnosis rather than from trial and error. The larger goal is a dictionary whose effective size is predictable from cheap statistics of the activations it is trained on.

## What they studied

TopK sparse autoencoders (with ReLU and JumpReLU variants in the appendix) trained on residual-stream or hidden activations from 454 model-layer combinations across language, vision, protein and genomic models, plus synthetic activations with a controlled amount of one dominant dimension. The unit of analysis is the *dimension of the input activation* and the *feature* of the autoencoder: each dimension's mean and spread, and each feature's alignment with the activation mean. There is no claim about concepts in the underlying model; the object is the fitting problem itself.

## How they did it

They define an outlier-severity ratio: the norm of the activation mean divided by the norm of the per-token standard deviation, computed after per-token LayerNorm so that between-token scale differences do not mask it. Decomposing a token's activation into the mean plus its deviation from the mean splits each feature's pre-activation into a constant shift (the feature weight dotted with the mean) and a signal (the weight dotted with the deviation). They derive the probability that a feature is born permanently negative, and the probability that it never wins the top-k competition, as closed-form functions of the severity ratio and the sample count, and check both against synthetic activations with the ratio set by hand. Then they sweep real layers, correlate the ratio with measured initial death rates, and separate the two death pathways by training with dummied-out components: freeze the bias, add an auxiliary reconstruction loss for dead features, or both. Finally they test the prediction head-on by centring the activations - folding the geometric median into the encoder bias at initialisation - and reading the resulting death counts, concept coverage and monosemanticity.

## What counts as an explanation here

A mechanism with a closed-form consequence: the account is that activation geometry sets the pre-activation shifts, shifts decide which features can ever fire, and therefore the death rate is a property of the layer's input statistics. The explanation predicts a number (a death rate for a given severity and top-k) and a remedy (remove the mean, or whiten if the residual is low-rank). It is complete only for this death pathway: the authors state explicitly that heavy tails and low-rank covariance produce death the ratio does not capture, and they measure an effective-rank statistic to cover the second.

## How they judged success

Spearman rank correlation between the predicted and the observed death rate across the 454 layer sweep (reported separately for death by negative pre-activation and death by top-k competition); monotonicity on synthetic data where the ratio is set directly; agreement of the analytic curve with observed rates; an ablation grid over training steps, bias freezing and auxiliary loss; concept coverage on a protein model against a fixed set of known annotations at a fixed F1 threshold across three dictionary sizes; mean maximum cosine similarity between learned and ground-truth directions on synthetic data; and a monosemanticity score computed with an independent embedding model. Controls include an auxiliary-loss baseline, a LayerNorm baseline, a four-times-larger dictionary, and centring by the arithmetic mean versus the geometric median.

## What they found

The severity ratio predicts initial death with rank correlation 0.82 for the negative-pre-activation pathway and 0.89 for the top-k pathway across the 454 layers, with no fitting; synthetic sweeps are monotone with perfect rank correlation and match the analytic curve. High-severity layers do not recover on any practical schedule: death plateaus at 75-90 percent after two million steps once the ratio is 30 or more, because recovery of the negatively-shifted features requires the bias to learn the mean and the bias converges slowly (99 percent of the mean inside 200k steps at a ratio below 5, but only 50-70 percent after two million steps at 30 or more). The top-k pathway recovers separately and quickly, around 200k steps, because living features shrink and free slots; the auxiliary loss helps mostly by preventing *new* deaths created during that shrinkage rather than by reviving old ones, which explains why it stops working in the worst cases. Centring drops initial death from 83 to near zero on one protein model and from 98 to under 5 on AlphaFold3, and the surviving features are better: on synthetic data the mean maximum cosine to the true directions goes from 0.38 to 0.97, and on the protein model a centred 2048-feature dictionary recovers more annotated concepts (100) than an uncentred 8192-feature one (73), out of 187 at a 0.7 F1 threshold. Centring fails on one genomic model whose covariance is so low-rank that four components carry almost all the variance; whitening there reaches zero death on every affected layer. Centring with the arithmetic mean is itself unsafe on four models where a few extreme tokens pull the mean far from the typical token, and the geometric median fixes that.

## Limits and open problems they name

The focus is TopK autoencoders on residual-stream activations; ReLU and JumpReLU variants have additional training-time death from sparsity pressure that is not analysed. The analytic formula assumes Gaussian signal projections, so on heavy-tailed activations it over-predicts death, and the authors only offer kurtosis as a cheap flag for where it breaks down. Why outlier severity differs across architectures and domains is left open, as is how LayerNorm - a standard preprocessing step motivated by hyperparameter transfer rather than by death - interacts with these dynamics across more models.

## What we take from it

A pre-registration measure for any dictionary we train on frozen activations: **report the mean-norm-over-spread-norm ratio of the layer's inputs, after per-token normalisation, before we report the dictionary's dead or live fraction, because otherwise a difference between two dictionaries may only be the geometry of the inputs.** A warning that also applies to probes and to any read-out we fit on top of frozen features: a large offset makes a randomly initialised read-out fail at initialisation, and no amount of further training recovers it inside a practical budget - so centring (by a robust location estimate) is a change we should make before interpreting a failure as absence of a concept. And a reporting pattern worth copying: the authors separate two death pathways that share a cause but differ in recovery timescale, and then re-explain the effect of an existing mitigation (the auxiliary loss) as acting on one pathway only - which is what a mechanism, rather than a hyperparameter sweep, looks like. Cross-reference the folder's own later paper on the same ratio in a different setting (run 701), the neuron-community coverage entry (run 696) for the same "a partial dictionary's reach is a property of the pipeline" caution, and the absorbed-component entry (run 700) where out-of-scope variance is declared rather than silently occupied.

## Tags

unit: feature | direction | layer
form: description | dictionary
method: SAE | weight-analysis | other
eval: prediction/simulation | downstream-task | benchmark
goal: understanding | evaluation-of-methods
