# Activation Probes Are Reliable With a Handful of Positive Examples

- **Key:** arxiv-2610.39766
- **Authors / organisation:** Riya Tyagi (MIT), Stefan Heimersheim (FAR.AI); NeurIPS 2025 Mechanistic Interpretability workshop; code at github.com/Centrattic/scarcity-probes; Pivotal Research Fellowship, compute acknowledged from Max Tegmark
- **Date:** 2026-10-29
- **Link:** https://arxiv.org/abs/2610.39766
- **Kind:** paper
- **Read from:** full text (all ~34.8k chars of the arXiv HTML v1, main text plus Appendices A.1–A.8; no figure pixels inspected, so every number below is one stated in prose by the authors)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Activation probes are proposed as a monitoring instrument for a model that might be deceptively aligned, but the realistic monitoring problem is not class-balanced: aligned examples are abundant and genuine misalignment instances are rare. The paper asks whether a usable probe can be trained from one to three verified positives when thousands of negatives are available. The authors note that earlier low-data probing studies manufactured scarcity by shrinking both classes together, or by using moderate imbalance near a five-per-cent positive rate, so the "few positives against many negatives" regime had not actually been measured. The larger goal is a monitoring story in which a scarce, high-stakes class can still be detected, with the abundance of the easy class treated as the resource rather than as noise.

## What they studied

Two tasks, deliberately one off-policy and one on-policy. Spam classification on Enron-Spam, probing Gemma-2-9b after layer 20 (the layer a prior sparse-probing baseline had selected), with out-of-distribution transfer to an SMS spam set, and the same task repeated across the Qwen3 family at 0.6B, 1.7B, 4B, 8B, 14B and 32B at layers 15/15/18/18/24/32. Honesty detection on the MASK benchmark, probing Llama-3.3-70B-Instruct after layer 50. The unit of analysis is a linear direction over an aggregated activation vector, with four aggregation functions (mean, max, last token, softmax-pooled) and four probe families: plain linear, sparse-autoencoder probe over the top 1024 class-discriminative latents, a non-parametric cosine-centroid probe, and an attention probe over token positions.

## How they did it

Each dataset is first filtered so the probe is asked about the concept rather than about the model's factual knowledge: spam emails are kept only where a few-shot belief query yields a positive Yes-minus-No logit difference (max over several tokenisations), and MASK examples are kept only where the model answered the belief prompt correctly before the pressure prompt induced the dishonest answer. Probes are then trained either on an imbalanced pool (1750 negatives against 1–150 positives for Enron, 340 negatives for MASK) or on a balanced pool with the same number of positives, and both arms are scored on an identical balanced held-out test set of 50/50 classes so that AUC and recall at a one-per-cent false-positive rate are not flattered by the training skew. Hyperparameters are fixed across the main results rather than tuned per arm, on the explicit argument that a real analyst with three positives cannot spare any for validation; a cross-validation arm appears in the appendix instead. Separately they upsample the positive class with gpt-4o-mini, generating in batches of at most five, rejecting duplicate or malformed batches, and sweeping factors of two to twenty. Finally the whole spam experiment is rerun down the Qwen3 size ladder. Ten random seeds for the Llama and Gemma probes (five for Qwen), each shuffling the splits.

## What counts as an explanation here

A probe: a single direction over hidden states, with a scalar score per input, used as a detector. The paper does not claim the direction is the mechanism. Its claim is narrower and predictive in one specific sense — how detection quality should be expected to move as the size of the scarce class, the size of the abundant class, the upsampling factor and the model scale change. It is explicitly a claim about the instrument, not about what the model represents.

## How they judged success

AUC on the balanced held-out set, and recall at a false-positive rate of one per cent, which the authors argue is the operationally relevant metric when false alarms are expensive in a monitoring setting. Both are reported for the imbalanced and balanced arms across the full range of positive counts, aggregated over the five best-AUC probe configurations, with per-probe curves in the appendix. Out-of-distribution generalisation to the SMS set is a separate arm. Controls include the fixed-hyperparameter choice, the cross-validated arm, the balanced test set, ten seeds, and a per-probe breakdown that surfaces where a family behaves differently from the aggregate.

## What they found

More negatives than positives beats balanced training for small positive counts: for the top five probes in the 1–10 positive range, imbalanced training gains on average 0.045 AUC and 0.093 recall points on Enron, and 0.055 AUC and 0.034 recall on MASK. The advantage largely evaporates in the 50–150 range (0.000 AUC, 0.015 recall on Enron; 0.007 and 0.041 on MASK). Generalisation to the out-of-distribution SMS set does not follow: +0.012 AUC but −0.031 recall at 1–10 positives, and −0.011 AUC with −0.059 recall at 50–150, which the authors call inconclusive rather than supportive. Upsampling by a factor of two is worth approximately doubling the real positives (0.026 AUC, 0.104 recall), with degradation at ten- and twenty-fold, which they attribute to probes fitting the synthetic style; the aggregate figure caption instead names threefold as the best factor across thirteen architectures and ten seeds, so the factor and the per-factor ranking are not stated consistently in the paper. On scale, larger Qwen models are generally more positive-sample-efficient, most clearly in the scarce regime, but the trend is not monotonic — the appendix notes 1.7B outperforming 4B on the out-of-distribution set. Probe family matters in a way the authors treat as a warning: attention probes have lower AUC under imbalance yet often reach high recall when trained balanced, and SAE probes keep improving at ten- and twenty-fold upsampling where the cosine-centroid probe degrades sharply.

## Limits and open problems they name

The off-policy task is easy for modern language models, so success there is no evidence about detecting a novel or complex misalignment. The utility of upsampling in the high-stakes case is doubtful for a reason that bites: if the only system able to generate examples of its own misalignment is the suspect model, the quality and diversity of those examples cannot be verified, and the authors' own human review found generated spam samples highly similar — data collapse in miniature. Only a single layer is probed per model, and the right layer for a concept is not known in advance. The belief filter uses a permissive threshold of zero logit difference, so confidently-held and highly-uncertain items are pooled; the authors propose a margin and a contrastive-prompt filter as future work. An SAE feature-selection caveat is also stated: the unnormalised mean-difference criterion can prefer a high-magnitude, low-discrimination latent, and it was kept only because the prior baseline found normalisation unnecessary at large top-k.

## What we take from it

A measure and a reporting convention. If our white-box model is ever asked to predict behaviour on a class that is rare in whatever corpus we have, the design question is not "how many positives" but "how many negatives, and where does the marginal positive stop helping" — and the shape of that answer here is a crossover between the scarce and adequate regimes, with the benefit concentrated below about ten positives. The warning is the out-of-distribution column: the arm that wins in-distribution at the smallest positive counts loses recall out-of-distribution, so a sample-efficiency claim made on held-out data from the same distribution does not transfer, and the two columns must be printed together. The second transfer is the fixed-hyperparameter choice as a *design* argument rather than a shortcut — the paper states plainly that with three positives there is no validation set, then runs the cross-validated arm anyway so a reader can see what tuning would have bought. And the family-dependence result is a caution for any probe benchmark we publish: an aggregate over probe families describes the mixture of the families sampled, since attention and cosine-centroid probes move in opposite directions under the same treatment.

## Tags

unit: direction
form: probe
method: probing
eval: benchmark | downstream-task
goal: safety/auditing | evaluation-of-methods
