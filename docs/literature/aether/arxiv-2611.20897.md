# Certified, Not Explained: Exact Token Erasure in Gated Vision State Space Models

- **Key:** arxiv-2611.20897
- **Authors / organisation:** Sadia Parveen, Tehseen Zia; Department of Computer Science, COMSATS University Islamabad, Pakistan
- **Date:** 2026-11-16
- **Link:** https://arxiv.org/abs/2611.20897
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Concept bottleneck models interpose human-readable variables between an image and a label, which buys intervenability but
not provenance: the model can say which concepts it believes and still not say where in the image those beliefs came
from. The authors wanted the *where* to be guaranteed rather than illustrated. Selective state-space models look like a
natural carrier: each token enters the recurrent state through an input-dependent step size, so a token whose step size
vanishes supposedly cannot influence the output, and the step size (or a learned gate on it) would then be a faithful map
of what the model used. The project set out to formalise that as a checkable erasure guarantee, then to audit how much an
explanation such a guarantee actually buys. The larger goal is the one this folder cares about: a statement about a
network that a reader can verify per input instead of trusting a saliency picture.

## What they studied

One model on one task: lung-nodule malignancy on LIDC-IDRI, as a concept bottleneck ("CG-Mamba") whose encoder is a single
gated, bidirectional selective state-space block. Input is the central axial slice of a cube around each nodule centroid,
cut into 64 patch tokens of embedding dimension 192; the block has inner width 384, state size 16 and a width-4 causal
convolution, with the forward and backward mixers initialised from layer 0 of a pretrained Vim-Tiny. Six
radiologist-rated ordinal concepts are read out from six contiguous 64-channel groups, and malignancy is a logistic
regression on the six concept probabilities only, so a clinician can overwrite any concept at test time. The unit of
analysis is the **token**: which patches the prediction is allowed to depend on. Baselines are a small ResNet and a
two-layer ViT, each as a CBM and as an unconstrained classifier, plus the same Mamba block without gates in both forms.

## How they did it

A hard-concrete gate, computed linearly from each embedded token, closes patches; the forward and the flipped backward
scan share one gate pattern so a token cannot arrive through the open direction. Closed tokens are cut off at every point
where they could re-enter: the convolution input, the step size, the B-skip, the multiplicative branch, the residual,
and the pooling weight of the readout (and, for depth, each block's input projection and residual). The authors then prove
three statements: masking only the convolution input empties the scan of closed-token content; with zero pooling weight on
closed tokens the concept and task logits are exactly invariant to what those tokens contain; and across stacked blocks
the projection and residual gates, rather than either step-size gate, are what stop a later block from re-opening a
leaked position. They verify each statement against trained models by a **path-by-path knockout**: hold the gate pattern
fixed, corrupt only the closed tokens (Gaussian noise at SD 50, or air), switch internal paths off one at a time, and
measure the maximum absolute change at the concept and task logits over held-out folds. Training is two-stage: stage A
trains the whole network with the backbone gate under a concept loss, a hinge on gate logits, a recall term on nodule-mask
images and a budget term; stage B freezes the embedding and backbone gate, re-initialises the concept predictors and adds
per-concept gates plus a localisation loss, with nodule masks available for a random half of training patients. The audit
then runs three independent attribution tests on the gate maps — deletion curves with three post-hoc rankers, ROAR-style
retraining after removing top tokens (and a repeat on crops deliberately shifted by three tokens so the centre no longer
covers the nodule), and a label-randomisation test with a noise floor built from seed-to-seed similarity — plus a
decomposition of the standard concept-leakage score.

## What counts as an explanation here

A **certified support set**: for a fixed mask, the statement that nothing outside the open tokens can move the concepts or
the diagnosis, checkable per image and by construction rather than by observation. It is deliberately partial and the
paper says so repeatedly — it is a faithfulness statement about what the model *may* depend on, not plausibility, and it
says nothing about whether the open tokens are the clinically relevant ones. The gate maps themselves are explicitly
withdrawn from the role of explanation: the authors conclude they are a mask-supervised nodule localisation shared across
concepts, not evidence for malignancy. The paper's own framing is the useful part — the guarantee is a property of the
masked model, and the gate is not the explanation.

## How they judged success

Pre-declared pass criteria written down before each experiment and reported with their failures (a table of all of them,
including failures, is in an appendix). Certificate: maximum absolute logit change under closed-token corruption, with
"exact" meaning zero to numerical tolerance in that run. Faithfulness: deletion-max (normalised area under the
mean-absolute-logit-change curve under air removal in ranking order) against random removal; ROAR-lite pooled
out-of-fold AUROC of an unconstrained Mamba retrained after the top 16 of 64 tokens are zeroed; the same on 24 px-shifted
crops with a separately retrained reference. Localisation: energy-based pointing game (fraction of map mass inside the
consensus nodule mask) against a centre prior and an intensity prior, before and after shifting crops. Sanity: a
label-randomisation test comparing permuted-label gate maps with the seed-to-seed similarity floor. Leakage: the Oracle
Impurity Score decomposed into mean diagonal (own-concept predictability), mean off-diagonal, excess over what
correlations among the true concepts imply, and a diagonal-normalised version, with a PCA-to-64 dimension control.
Accuracy: pooled out-of-fold AUROC over 15 runs (5 patient-level folds × 3 seeds) with DeLong tests for paired
comparisons; concept quality by Spearman correlation with reader mean ratings.

## What they found

The knockout follows the algebra exactly: every configuration that masks the convolution input and keeps closed tokens out
of the readout is exact in **15 of 15** runs, and the three configurations that violate a condition — convolution mask
removed, convolution-only masking with step-size-only pooling (largest changes 5.98 at concept and 4.14 at task logits),
and no gating at all (6.69 and 5.40) — break in **0 of 15**. At depth, dropping either step-size gate leaves the guarantee
intact; dropping the projection or the residual gate leaks, with the residual leaking far more, as the proof predicts. The
gated model reaches pooled out-of-fold AUROC **0.8425 ± 0.0085**, level with the ungated Mamba CBM (0.8428) and the CNN
CBM (0.8440), trailing the ViT CBM (0.8880, significant in 2 of 3 seeds) — but the gated model is the only one that saw
nodule masks, and without them it falls to about 0.79–0.81. Concept quality is the model family's weak point: Spearman
correlation is below both other backbones, and a two-block variant improves it while missing the pre-declared accuracy
criterion. The audit is mostly negative. The gate ranks evidence worse than three post-hoc methods on deletion-max, and in
ROAR-lite removing the gate's tokens lowers AUROC to **0.7206** against **0.8265** for random removal but **0.6466** for
removing the central tokens — so the gate loses to the crop geometry, since the crops are centred on the nodule and the
centre block is largely the target. Only on shifted crops does the gate beat the centre prior (0.6655 against 0.8859,
which is *above* the no-removal reference because removing parenchyma regularises), and even there hidden attention
removes slightly more (0.6426, with 81% of tokens shared). The label-randomisation test is the sharpest result: a model
trained on permuted concept and task labels produces backbone gate maps as similar to the real model's as two real-label
seeds are to each other, and localises at least as well — so the maps reflect mask supervision, not the labels. The
leakage decomposition finds that 54–60% of the standard score comes from its diagonal, that partitioning the readout
reduces excess cross-concept predictability, and that the final mask-supervised model sits *at* rather than below the
level implied by concept correlations — so the pre-declared check that it stays below that level fails.

## Limits and open problems they name

The guarantee is conditional on a fixed gate pattern, and in deployment the gate is itself a function of the image: in an
earlier model, replacing closed tokens with air re-opened every one of them, so changing what is hidden can change what is
hidden. It says nothing about the clinical meaningfulness of the open set, and any architecture that hard-masks tokens
gets it for free, so it is not specific to state-space models. Empirically: one public dataset, one axial slice per
nodule although several LIDC concepts are defined on the volume, patients 0001–0600 only, 764 nodules, no external
validation, and a held-out test split abandoned because it had been consulted during development (so all numbers are
cross-validated and mildly optimistic for every model alike, since the pool contains the former validation split).
Backbones differ in parameter count and only the gated model saw masks. Several analyses use one seed. The 3D attempt
underfitted badly and is reported as a statement about the compute budget, not about 3D. One intervention anomaly — in one
fold, predicted concepts beating the readers' values — is reported with two candidate explanations and left open. The
authors also note their third faithfulness criterion was poorly specified, because on centred crops the centre prior is
close to removing the target.

## What we take from it

A **method and a warning**, and the warning is the reusable half. Method: the knockout — hold the mask fixed, corrupt only
what the mask hides, switch candidate paths off one at a time, and require an exactly-zero maximum logit change per run —
is a cheap audit any architecture claiming erasure can be put through, and its positive result is unusually strong
because it is a *prediction of zero*, not of a ranking. For our reading of weights this is the form of claim we should
want wherever we can get it: an exact invariance with a stated precondition, printed with the precondition's own check.
Warning: a structural guarantee does not survive as an explanation of the thing it certifies. The gate maps fail the
label-randomisation test and lose to a trivial spatial prior whenever the prior happens to encode the cropping procedure —
so a support set can be certified on every image and still tell you nothing about why the model answered as it did, and
the only test that separated the two here was retraining with permuted labels and comparing map similarity against the
seed-to-seed floor. Second method: the leakage-score decomposition (own-concept versus cross-concept, each against an
oracle built from ground-truth concepts, with a dimension-matched control) is directly portable to any per-group probe
readout we publish, and its lesson is that an aggregate impurity score can move in the *wrong direction* while the parts
move correctly. Also worth copying: pre-declaring pass criteria and publishing the failures next to the passes, and
deliberately re-running a criterion whose crop geometry made it trivially passable — the shifted-crop ROAR is a template
for how to break your own prior-based baseline.

## Tags

unit: feature | layer | head
form: causal-abstraction | probe | circuit
method: patching | ablation | probing | training-interpretable | attribution | other
eval: ablation | patching | faithfulness | human-rating | benchmark
goal: understanding | evaluation-of-methods | safety/auditing
