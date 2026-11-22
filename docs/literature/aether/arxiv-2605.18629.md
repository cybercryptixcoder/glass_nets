# Aligned Training: A Parameter-Free Method to Improve Feature Quality and Stability of Sparse Autoencoders (SAE)

- **Key:** arxiv-2605.18629
- **Authors / organisation:** Michał Brzozowski, Neo Christopher Chung — byline prints "Affiliation: University of Warsaw, Poland" once as a single line for both authors, with the corresponding-author address on a Samsung AI Center, Warsaw domain; licence banner **CC BY 4.0**
- **Date:** 2026-05-18 (v1); v2 revised 2026-06-02
- **Link:** https://arxiv.org/abs/2605.18629
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Sparse autoencoders are the field's default dictionary-learning tool for turning a residual stream into feature directions, and the authors take two defects as given: a large share of dictionary entries never activate at all, and dictionaries trained on the same data with different random seeds come out different. Their diagnosis is that both are symptoms of one underconstrained optimisation — a degree of freedom nobody constrains — rather than two unrelated pathologies that need separate add-ons. They name that degree of freedom with a one-number diagnostic: the inner product between a feature's encoder row and its own decoder column. Well-aligned entries are the interpretable ones; near-orthogonal ones are training artefacts. The larger goal is that a dictionary can be reported as *the* dictionary for a model rather than as one seed's arbitrary draw, which is what any downstream claim about a feature implicitly assumes.

## What they studied

The unit of analysis is the **dictionary entry** of an SAE (one encoder row paired with one decoder column), and, secondarily, the whole decoder dictionary as a set of directions whose cross-seed agreement is measured. Models: layer 8 of Pythia 160M (residual dimension 768, float32) and layer 12 of Gemma 2 2B (residual dimension 2304, bfloat16), with pretrained SAEs on Pythia 70M layer 3 and LLaMA 3 8B IT layer 25 used only for the diagnostic histograms. Architectures: ReLU SAEs (four sparsity penalties swept), TopK, and BatchTopK. Dictionary sizes 4K, 16K and 65K; training on 50M tokens of The Pile, evaluation on OpenWebText, following the SAEBench protocol, with an appendix run at 500M tokens.

## How they did it

The geometric move is a reparameterisation, not a loss term. Take a free unconstrained matrix as the trainable encoder; the actual encoder row is that matrix projected onto the affine hyperplane on which the inner product with the feature's own decoder column equals one — a shift along the decoder direction by exactly the amount needed. The constraint therefore holds by construction at every step, the decoder and both biases stay unrestricted, and no hyperparameter is added. Because each row now lives on an (n−1)-dimensional hyperplane, one element of the free matrix can be fixed to zero and left untrained, so the aligned network has slightly fewer parameters than the baseline while reaching the same set of weights. They argue the constraint is the weak form of weight tying (which forces encoder equal to transposed decoder) and is analogous to putting a convolutional inductive bias on a linear layer. Comparisons are head to head at **matched achieved L0 sparsity**, across models, dictionary sizes, penalties and architectures, with three seeds for the p-annealing figures. Appendices add the code for the projection in a dozen lines, an alternative pseudoinverse derivation, a weight-tying comparison, dictionary-size sweeps, the 500M-token run, and the diagnostic's correlation studies.

## What counts as an explanation here

A **constraint on the training objective plus a diagnostic for the resulting dictionary** — a claim that a named geometric quantity of the weights (the encoder-decoder inner product per entry) causes a named pathology (dead entries, seed divergence), expressed as a parameterisation that removes the freedom to violate it. It is a partial explanation: it says why entries die and why seeds disagree, not what any live feature means, and it is predictive only in the weak sense that the recipe should transfer to an architecture whose loss it does not touch (which they test on TopK and BatchTopK). It also carries a claimed *measurement* of dictionary identity across runs, in the form of a pairwise set-to-set similarity.

## How they judged success

Four families, all on named SAEBench metrics. Reconstruction: explained variance and recovered cross-entropy (original model cross-entropy held apart from the zeroed-activation reference), compared as Pareto curves across sparsity. Dead features: the fraction of dictionary entries that never activate — the paper's baseline figure is roughly 20% for standard ReLU SAEs at dictionary size 4K, and it plots the alive fraction directly. Stability: **mean min cosine similarity (MMCS)** between the decoder dictionaries of two independently trained SAEs — for each entry in one dictionary the best-matching entry in the other, averaged; this is a set-overlap number, not a per-entry one. Downstream: the SAEBench Spurious Correlation Removal benchmark, where latents encoding a spurious attribute are zero-ablated and classifier accuracy on a balanced held-out set is read. Controls are the standard objective, weight tying, p-annealing alone, and SAEBench state-of-the-art checkpoints at the 500M-token scale.

## What they found

The diagnostic is bimodal across all four model/architecture combinations checked with pretrained dictionaries, and it tracks two existing quality proxies without needing a second training run: correlation with maximal cosine similarity is reported at Pearson 0.65, and with autointerpretability at 0.32. Aligned training improves reconstruction across both models, all three dictionary sizes and all four sparsity levels; dead features fall to near zero where the standard objective sits around a fifth of the dictionary, and this holds for TopK and BatchTopK and at 500M tokens; cross-seed MMCS is higher throughout training and the gap opens early, which the authors use as evidence it is structural rather than an initialisation effect. Combined with p-annealing, the dead-feature fraction goes to near zero where p-annealing alone plateaus at roughly 73% alive. Weight tying reduces dead features but costs reconstruction, while the hyperplane constraint improves both. Two cells cut against the story and are printed: on Gemma, SCR is marginally better for standard training at the very lowest sparsity (L0 at or below 110) before alignment overtakes it, and the p-annealing comparison shows p-annealing alone with a marginal cross-entropy advantage at the highest sparsity. A concurrent companion paper by the same authors argues that the reported stability of archetypal SAEs is an artefact of shared initialisation — relevant here only as the standard they are holding themselves to.

## Limits and open problems they name

Their own stated limit is scope: two model families, a fixed set of architectures, and the question of whether the benefit survives at larger scale or different training regimes is left open. Two more a reader should add. First, the stability result is a **decoder-dictionary** number: MMCS compares directions across two seeds and never asks whether the two dictionaries agree on *which token* a feature fires on or whether the same entry is the one a downstream reader would name — so a high MMCS is compatible with two dictionaries whose live features are paired but whose firing patterns differ. Second, the headline stability figures are curves rather than intervals: no confidence band over seed pairs and no per-entry frequency distribution is printed, only the mean best-match similarity, so the reader cannot see whether the improvement is a shift of the whole population or the removal of a tail. Third, the dead-feature claim is measured on the training distribution's token budget, which is exactly the confound the folder has already recorded from the starvation side: an entry that never fires in 50M tokens is not the same object as an absent entry.

## What we take from it

A **method and a warning**. The method is the cheapest structural explanation of a unit-level pathology the folder has met: one scalar per dictionary entry that is computable from the weights alone, requires no labelled data and no second run, correlates with expensive proxies, and is *causally actionable* because constraining it removes the pathology — which is the shape our white-box account wants for its own unit-statuses (a number read off the weights, then verified by an intervention that only that number predicts). The warning is about what a set-level stability metric licenses: MMCS is the mean best-match similarity between dictionaries, and the folder has spent five runs asking for the *distribution* of that agreement, not its mean; without the per-entry frequency histogram and a check that paired entries agree on their activations, "stable dictionary" means "stable directions", which is weaker than "stable features". Also worth copying: the paper closes an alternative explanation (shared initialisation) by naming it and pointing at the companion paper that tests it directly, and it prints the two benchmark cells that favour the baseline.

## Tags

unit: feature | dictionary-as-set | weights
form: direction | constraint-on-objective | replacement-model (reparameterisation)
method: SAE | training-interpretable | decomposition | weight-analysis
eval: downstream-task | benchmark | ablation | prediction/simulation
goal: understanding | control/editing | debugging
