# IDEEA: training-free Input-Dependent stEEring via Activation cluster matching

- **Key:** arxiv-2609.02089
- **Authors / organisation:** Zheng Wang, Muchen Li, Renjie Liao & Yan Leng; University of British Columbia / Vector Institute for AI / Canada CIFAR AI Chair, with Leng at UT Austin McCombs
- **Date:** 2026-09-02
- **Link:** https://arxiv.org/abs/2609.02089
- **Kind:** paper
- **Read from:** full text (v1 HTML: abstract, §1-§6, Tables 1-5, Appendices A-K)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Training-free activation steering is cheap next to fine-tuning, but almost every training-free method fits one direction once and applies it to every input: the difference between positive and negative activation means, per head or per layer, is a constant. The authors argue this is the wrong object for a different reason than "the direction is inaccurate" — they argue the activations that carry a concept are spread over several distinct sub-regions of the representation space, so a single averaged direction can line up with at most one sub-region and leaves the others untouched. Their motivating analogy is explicit and is the most useful sentence in the paper: the optimal push toward a fixed target concept varies from input to input exactly as the gradient of a fixed loss varies from input to input. The goal is to get input-dependence without any training, by keeping a small pool of directions that all encode the same concept and choosing among them per input.

## What they studied

Six open-weight instruction-tuned models — Llama2 7B, Llama3 8B, Mistral 7B, Qwen2.5 7B, Gemma2 2B and 9B. The unit of analysis is the per-head multi-head-attention output activation, indexed by (layer, head), collected at the position of the last input token. Intervention is restricted to the top-K heads ranked by held-out linear-probe accuracy for the concept label, a head-selection rule taken straight from prior per-head steering work. Four concept targets: truthfulness (TruthfulQA, 6-shot free generation judged by two fine-tuned Llama2 7B judges), social-behaviour traits of the zero-sum dictator game, political polarity (TwinViews), and toxicity mitigation (directions fit on PKU-SafeRLHF, evaluated out-of-distribution on TET).

## How they did it

Four stages. (1) Collect per-head activations on a contrastive conversation set labelled +1/-1 for the concept, and rank heads by probe accuracy, intervening on the top-K only. (2) Run K-means separately on the positive and the negative activation support of each head, with equal counts `n_c` on both sides so that a discrete direction vector `v = C+_i - C-_j` exists for any pairing of centroids. (3) Choose the pairing: there are `n_c!` bijections and they keep the one that maximises the average pairwise cosine among the `n_c` chosen directions — a coherence criterion, on the argument that all clusters describe the same concept so the resulting directions should be mutually aligned. This is a quadratic assignment problem, solved by exhaustive enumeration over `n_c` in 2..6; for larger `n_c` they give both an approximation algorithm and a linear-assignment relaxation against the mean direction, which is exactly the static mass-mean direction, solved with the Hungarian algorithm. (4) At inference, pick one direction from the pool by one of two rules: *min-perp*, the pool member most aligned with the current activation's own direction (minimising perpendicular distance from the activation), or *nearest-cluster*, the direction associated with whichever centroid of either sign is closest to the activation. Clustering is repeated ten times per head with the same seed sequence and the lowest-inertia fit is kept. Strength is a per-method grid; for their methods `K` is swept over small multiples of the head count, `alpha` over {2,4,6}, and `n_c` over 2..6, all inside a 5-fold outer cross-validation so no evaluation item informs its own configuration.

## What counts as an explanation here

A pool of directions with a stated selection rule, plus a geometric picture: the positive activation support of a head is multi-modal, clustering recovers the modes, and steering with a cluster-matched direction maps the negative support onto the positive support mode by mode rather than rigidly translating it. It is a mechanism sketch rather than a model of the network — nothing in the method predicts a behaviour from weights, and the multi-modality claim is illustrated with PCA/KDE plots and one quantitative distance statistic, not tested as a hypothesis about the model's computation.

## How they judged success

TruthfulQA truth rate times informativeness rate is the headline, with both factors reported, on the explicit ground that truth alone is gameable by refusing. Four training-free baselines — per-head static mass-mean steering, layer-level contrastive activation addition, an SAE feature add-back where a public feature suite exists (only three of six models), and a covariant spectral subspace edit. For the social-behaviour task the fit is on a purely synthetic contrastive corpus and the evaluation is on 240 payoff settings sampled five times, with ordinary-least-squares standard errors and a *system-prompt* reference arm that names and then fully describes the target persona. Ablations: the number of clusters, removal of the bijection (let each input pick its own positive and negative centroid independently), and removal of the equal-count constraint (per-side `n_c` chosen by silhouette score). A separate table reports the fraction of non-informative answers and the fraction of refusals per method, and the paper is explicit that its judges label refusals truthful.

## What they found

On TruthfulQA the min-perp variant averages a 34.2% gain in truth-times-info over unsteered across the six models and nearest-cluster 28.5%, against 17.4% for the spectral-subspace baseline and 16.2% for per-head static steering; the abstract's quoted average gain is 9.9% with a 23.5% maximum, which is the improvement over the best baseline rather than over base. Their variants win four of six models outright, lose to the spectral baseline on two, and are the only family that beats base on every model. The refusal breakdown is the sharpest result in the paper: on Llama2 7B the static per-head direction raises truth rate from 0.937 to 0.957 while pushing refusals to 0.306 and non-informative answers to 0.451, collapsing truth-times-info from 0.567 to 0.509, whereas min-perp cuts refusals to 0.081. On the dictator game (Llama3 8B) nearest-cluster gains 292.1% averaged over the four traits, ahead of every training-free baseline and ahead of the weak system prompt; the strong persona prompt, described by the authors as an upper bound rather than a method, reaches 459.5%. On TwinViews min-perp scores 0.506 against 0.124 for the spectral baseline, and the SAE feature collapses to 0.002. On TET toxicity nearest-cluster reaches 0.847 and loses only to the SAE feature at 0.917. Both structural-constraint ablations lose: dropping the bijection costs 0.033 on average and up to 0.066, and letting the two cluster counts differ costs a further 0.041, which they read as the constraints acting as regularisation against clustering noise. Re-running with hyperparameters chosen on the whole evaluation set reproduces the cross-validated table in 26 of 39 cells with a mean absolute difference of 0.007.

## Limits and open problems they name

The one they name is drift: because the top-K heads sit in different layers, perturbing an early layer moves the activations that deeper heads see away from the distribution the clusters were fit on, so the per-head cluster fit is no longer the right partition — and they say quantifying that inter-layer drift is the next step, with single-layer steering as the natural combination. Unnamed but visible in their own tables: the number of clusters matters (gains peak around `n_c` = 5-6 and are modest at 2), the head probe accuracy is what selects where to intervene and it survives only by held-out agreement, the social-behaviour directions are fit on synthetic data generated by one of the models under test, and the whole geometric argument rests on two-dimensional projections with a distance-gap statistic rather than on a test.

## What we take from it

Three things. **A measure**: the sliced 1-Wasserstein gap between the positive support and the shifted negative support, in the full head dimension, used to rank which heads show the effect — the only quantitative geometric statistic here, and the right kind for us, because it asks how much of a distribution a direction reaches rather than how well it separates two means. **A failure mode to instrument**: refusal collapse is an intervention that wins the truth metric by landing every input in the model's most fluent sub-mode, and it is only visible because the non-informative fraction is reported separately; any concept with a rejection-shaped attractor will have this. **A warning about selection rules**: their own ablation shows that the loosest, most adaptive choice of direction is the worst of the three — the pair-chosen-at-inference variant loses to the globally matched pool on every model — so adaptivity bought by relaxing structural constraints loses to adaptivity inside constraints. And the closing partner: the selection rules here are purely geometric proximity on a prompt-position activation, so they inherit run 803's drift problem — the choice is made once and applied as generation proceeds.

## Tags

unit: head | feature
form: direction | steering
method: probing | steering | decomposition
eval: ablation | downstream-task | benchmark
goal: control/editing | understanding
