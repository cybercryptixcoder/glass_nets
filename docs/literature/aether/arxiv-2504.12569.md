# Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment

- **Key:** arxiv-2504.12569
- **Authors / organisation:** You Rim Choi, Subeom Park, Seojun Heo, Eunchung Noh, Hyung-Sin Kim; Graduate School of Data Science, Seoul National University, and Samsung Electronics
- **Date:** 2025-04-17 (v4 2026-01-16); AAAI 2026
- **Link:** https://arxiv.org/abs/2504.12569
- **Kind:** paper
- **Read from:** full text (v4 HTML and the abs page)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Open-set semi-supervised learning trains on a small labelled set of known classes plus a large unlabelled pool that
secretly contains both known-class and unknown-class items. The paper starts from a split in how prior methods treat
the unknown items: some bolt on a (K+1)-th "catch-all" class and push everything unrecognised into it, others build a
spanning subspace for the known classes and shove anything outside it away. The authors argue both are overreach: the
hole truth is vast and heterogeneous, training shows you only a sliver of it, and any method that forces every
unlabelled point onto a prototype or into a single synthetic class is fitting the sliver. Their goal is a training
objective that keeps uncertain points uncommitted, so that the resulting feature geometry generalises to unknown
classes that were never in the pool at all. This is a step toward OOD detection that is not tuned to the OOD examples
it happens to have seen.

## What they studied

WideResNet-28-2 on CIFAR-10 (six animal classes known, four as seen-OOD) and CIFAR-100 (55/45 and 80/20 known/OOD
splits, grouped by semantic super-class), ResNet-18 on ImageNet-30 (20 known, 10 seen-OOD) and WideResNet-28-4 on
TinyImageNet (100/100), with 25 or 50 labels per class on CIFAR, 5% labels on ImageNet-30 and 10% on TinyImageNet. The
unseen-OOD battery is what they call a diverse set: the other CIFAR's test split, CIFAR-10-C/100-C, ImageNet, LSUN,
SVHN, Gaussian noise, and for ImageNet-30 also Stanford Dogs, CUB-200, Flowers102, Caltech-256, DTD and SUN397. The
unit of analysis is a projection embedding from a dedicated head plus class prototypes (mean embeddings) inside it,
alongside a shared backbone and two readouts — a K-way closed-set classifier and a K-head one-vs-all detector.

## How they did it

Three heads sit on one backbone. The closed-set classifier is trained FixMatch-style (cross-entropy on labels,
consistency on pseudo-labels). The OOD detector is an OVA scheme in the SSB/OpenMatch style: per-class ID-vs-rest
binary heads, entropy minimisation and a soft open-set consistency term on unlabelled data, plus a pseudo-negative
term on confidently identified OOD items. The new part is the third head: a small MLP that projects the backbone
feature to 128-d, where prototypes live and where the selective non-alignment loss acts. Every unlabelled item gets a
gate that requires *both* the classifier's softmax and the detector's ID probability for the argmax class to clear a
threshold; gated items are pulled toward their predicted prototype, and everything else — uncertain items and
confident OOD alike — has the pull term switched off and instead receives only an angular update. That update is the
gradient of a cosine-similarity softmax log-likelihood, which after the standard normalisation is the projection of
the softmax-weighted average of all prototype directions onto the tangent plane, so it is purely rotational: it nudges
the point away from whichever prototypes it currently leans toward, without ever attaching it to one. Prototypes are
refined as a count-weighted blend of the labelled mean and the mean of gated unlabelled items, with a hyperparameter
capping the unlabelled contribution. Labelled items additionally get supervised contrastive alignment plus a
prototype-pull term. They also prove two Rademacher bounds in the appendix: a (K+1)-way catch-all classifier needs
high capacity to represent the collapsed synthetic class and therefore has a loose bound, whereas a detector resting
on a large margin between the ID and OOD regions has a bound that shrinks as that margin grows.

## What counts as an explanation here

A training-time mechanism plus a geometric picture. The mechanism is one line of algebra (the gated pull term
multiplied by a mask, with a tangent-plane projection that makes the unmasked update non-attractive), and the picture
is a two-part claim about the embedding: known-class items end up in compact, high-norm clusters aligned with their
prototypes, unknown items stay low-norm, low-cosine, and dispersed in the space between clusters. The appendix reasons
about the norm of the backbone feature — cross-entropy, consistency and OVA losses all push norm up along whichever
weight vector the point is closest to, in proportion to the cosine, so an item kept misaligned by the angular
repulsion accumulates norm more slowly and in the wrong direction. This is a partial, sign-of-the-effect explanation,
not a predictive model: nothing here says which specific unseen input will be rejected.

## How they judged success

Closed-set accuracy and OOD detection AUROC on the closed-set detector's score, with "overall AUC" defined as the
unweighted mean over all OOD sets, seen and unseen. Four baselines (IOMatch, SSB, ProSub, SCOMatch) under identical
splits and hardware. Three seeds per configuration, reported as mean ± standard deviation. Ablations cover the loss
formulation (a gated-OVA variant and a pull-only cosine variant), the three-way loss combination (with seen, unseen
and overall AUC separated — the only place in the paper where seen and unseen are reported apart), the unlabelled
contribution ratio to prototypes, the OOD-detector gate threshold with the classifier's fixed at 0.99, projection-head
nonlinearity, and the SNA loss weight. No held-out protocol, no significance test, and no correction for the fact that
six to nine OOD sets are averaged.

## What they found

Overall AUC is best in every configuration, up to 7.1 points over the strongest baseline and 3.1 on average, with
closed-set accuracy best in nearly all cells and small regressions on CIFAR-10 and TinyImageNet. Against the two
nearest baselines on CIFAR-10 6/4-50 the gap is 94.9 vs 87.8 (ProSub) and 83.7 (IOMatch); on CIFAR-100 80/20-25, 82.8
vs 80.8 (SSB) and 76.5 (ProSub); on TinyImageNet, 77.1 vs 75.7 (SSB), where the closed-set accuracy actually drops
(60.2 vs 64.3 for SSB). The pull-only cosine variant is the informative ablation: on CIFAR-100 55/45-25 its AUC is
83.4 to the full loss's 84.1 and its accuracy 71.4 to 71.8, but on 80/20-25 accuracy falls to 65.9 against 67.2 — so
the repulsion buys little on the 55/45 split and more when OOD is the minority. The three-way loss table is the one
that shows where unseen generalisation comes from: with no SNA losses, unseen AUC is 77.7; adding the unlabelled
non-alignment term on top of supervised alignment moves unseen to 79.1, while supervised alignment alone gives 77.7 →
the same as the baseline, i.e. supervising the labelled data alone buys nothing on unseen classes and the whole
unseen margin over that baseline comes from the skip operator. Seen AUC,
meanwhile, is *higher* without the unlabelled term (91.5 vs 91.2), so the repulsion pays for itself only on classes it
never saw. Dropping the OOD-detector gate entirely (threshold 0) costs CIFAR-10 14.4 AUC points (80.0 vs 94.4 at 0.5)
while accuracy barely moves — a large effect on the metric of interest and almost none on the other. Adding nonlinearity
to the projection head is worth about 1.7 overall and 2.0 unseen AUC points on CIFAR-100 55/45-25 (84.1 vs 82.4; 83.0
vs 81.2). Results are flat across the SNA loss weight from 0.004 to 0.012 (95.9–96.7 overall on CIFAR-10 6/4-50), with
the default 0.01 not the best.

## Limits and open problems they name

The conclusion names exactly one: performance suffers when unseen classes closely resemble known ones — LSUN under the
CIFAR-100 setting, where scene-centric known classes overlap it at feature level. Future work is feature
disentanglement for that case. They also note, in the main results, that the one baseline beating them on a single cell
(ProSub on LSUN at 80 known classes) does so while being 12 points worse at closed-set classification, i.e. by
misclassifying known items rather than separating unknown ones.

Read off the paper, not stated by the authors: (a) every embedding picture that carries the mechanism is a t-SNE
drawing of the 128-d projection or detector space, and the three quantitative companions to those drawings are a norm
histogram and a cosine histogram shown only as figures — there is no table of either; (b) "overall AUC" averages six
to nine OOD sets unweighted, so the number that appears in every row is dominated by whichever sets are easiest, and
the per-set breakdown lives only in a bar chart; (c) the seed count is three and the ablation tables (2, 3, 4, 5, 6)
carry no variance at all, so the differences those tables are read for — 0.7 AUC between the cosine variant and the
full loss, 1.7 for nonlinearity — are compared without the ± figures Table 1 gives for the same configuration
(Table 1's CIFAR-100 80/20-25 AUC carries ±3.2); (d) the baseline numbers do get ± and they are large on the
ImageNet-30 and TinyImageNet cells (ProSub ±3.4, SCOMatch ±4.2), which is exactly what a table of point estimates for
the ablations cannot see; (e) the hyperparameter most likely to matter for a different OOD mix, the dual-gate
threshold pair (0.99 fixed on the classifier, 0.5 chosen on the detector), is tuned on two configurations and reported
on the same two.

## What we take from it

A mechanism, a measure and a warning. The mechanism: **norm is not free information — it is manufactured by whatever
loss is pulling hardest, and a training objective can be written to spend it only on items it is confident about.** The
appendix is worth keeping for that alone, because it gives the accounting explicitly: cross-entropy, consistency and
OVA gradients all have a component along the feature direction proportional to the cosine to the relevant weight, so
norm growth is an effect of confidence-times-alignment, and a purely tangential update on the uncertain items caps the
norm they can accumulate. For us that reframes a habit: when a white-box model reads a per-sample score off a feature
norm (or a distance that is a proxy for it), the score is reading a *training artefact* with a known sign, and the two
things that set it are the gate threshold and the loss weighting rather than anything about the input.

The measure: the paper's own best evidence for its claim is not its headline AUC but the seen/unseen split in the
three-way ablation, where the unlabelled non-alignment term buys +1.4 unseen and −0.3 seen. That is the right shape for
a generalisation claim about unknown classes: report the seen and unseen columns separately and let the reader see the
trade. Averaging them, as the headline does, produces a number in which a term that only helps on unseen classes looks
merely additive.

The warning, and it is the sharpest one here: **a scaling artefact survives a label swap.** Under this loss, known
items pull toward prototypes and unknown items are held off them, and one of the consequences is a clean gap in
feature norm between known and unknown — which is exactly the statistic a probe would find and report as evidence that
the model "knows what it knows". The gap was created by the objective, not discovered by it, and it exists in a
projection head that was trained for the purpose. That is not a criticism of this paper — it is the honest reading of
its own figure — but it is a constraint on how any reliability probe built on norm or distance from prototypes should
be interpreted, and it lines up with what we have recorded before: a geometric score measures the training objective
that produced the geometry at least as much as it measures the input.

Verified-by-us and reusable: (1) in the three-way ablation, the term that is supposed to generalise to unseen classes
helps unseen and hurts seen — always separate those columns; (2) the gate is doing enormous work on the detection
metric (14 points) and almost none on accuracy, so a detector's quality is set by its filter, not its classifier head,
and any probe comparison across checkpoints has to hold the filter fixed; (3) ablation tables in this literature are
routinely point estimates while the main table in the same paper reports ±3.2 on the same cell — before we treat a
sub-one-point ablation difference as a finding, we check whether the paper prints a variance for that cell anywhere.

## Tags

unit: feature | direction | layer
form: steering | direction | probe
method: training-interpretable | other | probing
eval: ablation | downstream-task | benchmark
goal: understanding | control/editing | evaluation-of-methods
