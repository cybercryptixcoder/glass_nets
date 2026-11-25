# Beyond Head Importance: Collective Computation in Video Transformers

- **Key:** arxiv-2611.11875
- **Authors / organisation:** Yi Yang, Siyuan Liu, Bingbing Nie; Tsinghua University (School of Vehicle and Mobility; State Key Laboratory of Intelligent Green Vehicle and Mobility)
- **Date:** 2026-11-09
- **Link:** https://arxiv.org/abs/2611.11875
- **Kind:** paper
- **Read from:** full text (v1 HTML; supplementary figures read from captions and reporting prose, not inspected as images)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Head-importance rankings are the standard currency of circuit work, and this paper asks what that currency leaves out. The authors argue that a head's contribution depends on two things a ranking cannot see: whether the block's MLP actually uses the head's output, and whether many small-range heads compose across layers into a long-range effect that no single head exhibits. The larger goal is a notion of "collective computation" that survives ablation, so that a set of individually minor components can be shown to carry the behaviour. As a side effect the paper tests whether importance itself can be predicted ahead of time from cheap attention statistics, which is the closest thing here to a weight/activating-statistic reading.

## What they studied

Four video Transformers, all ViT-L encoders with 24 blocks and 16 heads each (384 heads per model): V-JEPA and V-JEPA 2 (feature-prediction pretraining) and VideoMAE and VideoMAE V2 (masked reconstruction). The downstream readout is a linear probe trained on frozen, mean-pooled encoder features over four class-balanced tasks: a ten-class SSv2 action subset, nine instrument-related UCF101 activities, and two binary SSv2-derived tasks (motion direction, action success). Unit of analysis is the attention head throughout, with MLP blocks and residual measurement points as the secondary units; the composition analyses are run mostly on V-JEPA.

## How they did it

Importance is estimated by repeated random joint ablation rather than one head at a time: 3,000 trials per task per model, each ablating a random 25% of heads by *mean replacement* (the head's output is swapped for its mean over training videos and token positions), refitting the linear probe each time, and scoring a head by the average accuracy drop over all trials containing it. Rankings from Action are then used everywhere else, and cross-task transfer is tested by comparing top-25% and bottom-25% orderings, plus a Spearman correlation and a top-set overlap against Action-derived and Object-derived rankings. Representational structure is measured by retaining top or bottom 50% of heads and comparing CKA to the intact model layer by layer, shared-UMAP displacement fitted on the intact model, mean pairwise cosine similarity across videos, and two rank-preservation statistics (inter-video distance Spearman, 10-NN Jaccard). Ten cheap head properties are then used to *predict* importance: mRMR selection plus an ExtraTrees regressor inside repeated six-fold cross-validation, scored by Spearman on held-out heads, with permutation importance. Two follow-ups act on the two strongest cues. For downstream use, the same head ablation is measured twice — after the attention residual and after the MLP residual of the same block — and correlated with importance rank; then the block MLP is zeroed and within-block ranks are re-estimated, and separately only the MLP parameters are fine-tuned (one epoch on unlabelled SSv2, masked feature prediction, everything else frozen) with only the bottom half of heads retained. For cross-layer composition, heads are split into a Local half and a Global half by mean spatial attention distance, a 2%-of-token-norm perturbation is applied at the centre token of 174 SSv2 videos, and central finite differences along four directions give the response magnitude at each position; the headline number is the Local/Global response ratio averaged over the 52 boundary tokens of the source's time slice, tracked layer by layer. Finally, local heads are tested causally by cumulative ablation ordered by attention distance versus by importance, and by replacing the ablated heads' attention with *fixed* spatiotemporal kernels (mean attention averaged over equal relative offsets, row-normalised) to see how much loss stable routing alone can recover.

## What counts as an explanation here

Functionally, a dependency claim of the shape "this head set is load-bearing for these tasks and this representational geometry", plus a mechanism sketch of a *kind*: importance is realised through post-MLP processing, and local heads achieve global reach by composition across depths rather than by single-step attention span. The account is explicitly partial — the conclusion states that ablation and perturbation reveal dependencies without identifying computation pathways — and it is not predictive of the model's output; the one predictive artefact is the property-based ranking model, which is a statistical surrogate for an importance score, not for behaviour.

## How they judged success

Cross-task ablation deltas in accuracy points against the intact model; Spearman rank correlation between independently estimated importance rankings and between ranks and properties; top-set overlap against the chance expectation of 24 heads; CKA against the intact model; UMAP displacement and cosine-similarity drift; distance-rank Spearman and 10-NN Jaccard relative to intact; held-out Spearman from repeated six-fold cross-validation (with a stated interval across splits) and permutation-importance drops; the Local/Global boundary-response ratio per layer; cumulative-ablation curves under three different orderings; and the fraction of ablation-induced accuracy loss recovered by fixed-kernel replacement.

## What they found

Importance is shared across tasks rather than task-specific: ablating the top 25% costs −34.3 / −30.6 / −31.1 / −19.3 points across the four V-JEPA tasks, while ablating the bottom 25% is small and often *positive* (+3.9, +0.8, +1.8, +2.5); the pattern repeats in all four models (top-25% deltas between −6.1 and −51.6 points; bottom-25% between −5.2 and +12.9). Action- and Object-derived rankings agree positively in every model (rho 0.61, 0.43, 0.43, 0.17) with 57, 46, 51 and 34 shared heads in the top quarters against a chance overlap of 24. Retaining high-importance heads preserves cross-video representational structure far better than retaining low-importance heads. Importance is only partially predictable: held-out Spearman climbs from 0.112 with one property to 0.282 with seven and plateaus (V-JEPA; 0.356/0.355/0.418 for the other three), with mean spatial attention distance the dominant cue in every model and the post-MLP representational response second in three. The measurement-point comparison is the sharpest single result: the same head ablation's representational change correlates with importance rank at only −0.05 when read after the attention residual versus −0.21 after the MLP. Zeroing the block MLP reshuffles within-block importance ranks, and MLP-only fine-tuning partially restores the retained-bottom-half model. For composition, Global half wins on long-range response in early layers as expected, but with depth the Local half overtakes it at the boundary tokens; cumulatively ablating spatially local heads first costs *more* than the importance ordering (43.6 vs 34.3 points on Action at 25%), quartile cross-tabs show the two orderings pick largely different heads, and fixed kernels recover 56% of the local-set loss versus 27% of the top-importance-set loss and 75% for the temporally local set.

## Limits and open problems they name

The authors name two: the task suite is a narrow set of small linear-probe classifications, and ablation plus perturbation show dependency without identifying the pathways that produce it. Further limits are visible in the numbers rather than stated — the importance-prediction correlations (0.11–0.42) are far from a usable reader; the zeroed-MLP rank reshuffle has no reported downstream accuracy attached to it; fixed-kernel recovery is reported as a fraction of an ablation loss quoted only for V-JEPA, so its transfer is untested; the low-importance ablations that *improve* accuracy (up to +12.9 points) are treated as a curiosity rather than followed up; and one V-JEPA 2 and one VideoMAE V2 bottom-25% delta is exactly 0.0, so those cells carry no sign.

## What we take from it

A warning and a method. The warning is the one this folder keeps re-learning with new instruments: a function of per-head clean statistics — here an importance score built from ablation drops, and a predictor built from ten such statistics — cannot anticipate the damage from joint ablation of a set chosen by a *different* property (43.6 points lost versus 34.3). The method worth taking is the measurement-point trick: score an intervention's representational effect at two points inside the same block, before and after the MLP, and correlate both with the property you want to explain (−0.05 vs −0.21 here). It converts "the residual made it look flat" into a localisation of where a head's output is actually consumed, using no extra interventions. Also reusable: fixed-kernel replacement as a *routing-only* control, which prices how much of a head set's function is a static stencil rather than input-dependent computation — the closest thing here to a replacement model, and a cheap way to ask whether our own recovered circuit is really a convolution.

## Tags

unit: head
form: description
method: ablation
eval: ablation
goal: understanding
