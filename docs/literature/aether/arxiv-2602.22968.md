# Certified Circuits: Stability Guarantees for Mechanistic Circuits

- **Key:** arxiv-2602.22968
- **Authors / organisation:** Alaa Anani, Tobias Lorenz, Bernt Schiele, Mario Fritz, Jonas Fischer; Max Planck Institute for Informatics / CISPA Helmholtz Center for Information Security
- **Date:** 2026-02-26
- **Link:** https://arxiv.org/abs/2602.22968
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why
Circuit discovery is supposed to recover the computation behind a behaviour, but it is known to be fragile: swap a
few examples in the concept set, or add or remove a handful of them, and the recovered subgraph can change
substantially, and the circuit found on one distribution often fails on a shifted one. That fragility casts doubt on
whether the recovered object describes the concept or the particular examples used to define it, and it is the working
hypothesis of the surrounding literature (variance analyses of attributed circuit discovery, non-robust faithfulness
metrics, interpretability illusions that hold on the discovery distribution and fail elsewhere). The authors want to
convert this from an empirical worry into a guarantee: rather than reporting a circuit and hoping it is stable, report
a circuit whose *membership decisions* are proved invariant to any bounded edit of the concept dataset, and abstain on
the components where no such proof is available. The larger aim is to put circuit discovery on the same footing as
certified robustness — a claim that comes with a threat model, a radius and a confidence level.

## What they studied
ResNet-50 and ResNet-101 on ImageNet classification, with each class treated as a concept dataset and circuit vertices
taken as the feature channels at the output of the four residual blocks (256, 512, 1024 and 2048 channels). Circuits are
discovered on 100 randomly selected classes and evaluated in-distribution and on four shifts: OOD-CV (novel context and
background), ImageNet-A (natural adversarial examples), ImageNet-O (classes absent from ImageNet) and ImageNet-C
(defocus blur). The unit is the neuron/channel, with edges induced by the certified vertex set. The paper's abstract
also names a ViT and GPT-2 with IOI, IOI-Hard and Greater-Than, but this HTML version reports only the ResNet suite in
its tables — the language experiments are announced, not shown, and nothing in this entry draws on them.

## How they did it
The base discovery algorithm is the standard top-k-per-layer rule, scored three ways: mean activation magnitude over the
concept set, mean attribution relevance, or a per-image rank aggregated across the set. Around that black box they wrap
randomized deletion smoothing of the dataset: each example is independently kept with probability 1−p, the base algorithm
is run on the surviving sub-dataset, and this is repeated over many Monte Carlo draws. For every vertex they estimate the
*smooth evidence* of inclusion — the probability that the base algorithm selects it across draws — and then apply a
three-valued threshold rule inherited from certified segmentation: above a margin the vertex is certified *in*, below the
complement it is certified *out*, and in between the method abstains. The certified circuit is the subgraph induced by the
certified-in vertices, so abstention both shrinks the circuit and removes components whose inclusion is unstable. The
guarantee comes from combining an edit-distance smoothing certificate for sequence classifiers (deletions under a
Levenshtein metric covering insertions, deletions and substitutions) with the per-component abstention scheme: any
non-abstaining vertex keeps its decision for every concept dataset within the certified edit radius, with the stated
confidence. Since the smoothed probability is unknown, it is bounded by a lower confidence bound from the empirical
frequencies. Operating settings are p = 0.5 and a confidence threshold of 0.7, giving a certified radius of 5 edits, with
a stated failure probability and a union bound if the guarantee is wanted simultaneously for all vertices.

## What counts as an explanation here
A certified sparse subgraph: a set of feature channels, with the edges between them, each member carrying a proof
obligation rather than just a score. The explanation is explicitly partial and three-valued — included with a
certificate, excluded with a certificate, or abstained — and it is relative to a named threat model (edits of the
concept dataset) rather than to input perturbations. It predicts something specific: that re-running discovery on any
dataset within the radius yields the same decisions on the certified vertices. It says nothing about whether the circuit
is sufficient in a semantic sense beyond the sufficiency accuracy it is scored on.

## How they judged success
Sufficiency is class accuracy under sufficiency pruning — all non-circuit channels zeroed at each residual block, the
pruned model evaluated on the class's own images — reported as mean circuit class accuracy (cACC), with peak cACC over a
sweep of the retained fraction and the circuit size at which that peak occurs. Compactness is the same retained fraction,
reported as the effective fraction after abstention rather than the base algorithm's target. Class specificity is
measured as overall accuracy on images from *other* classes (oACC), where lower is better. OOD transfer reuses circuit
accuracy on the four shifted sets without re-discovery. Structural stability is IoU between the circuit discovered on
ImageNet and the circuit re-discovered on the shifted set, per class, summarised as a distribution over classes, at the
retained fraction that maximises the gap. Every certified decision carries the confidence and radius from the theory,
so the comparisons are between a certified object and uncertified top-k baselines at matched scoring functions.

## What they found
Certified circuits beat the uncertified baselines on every dataset and every scoring function. On ResNet-101 with
relevance scoring, peak cACC rises from 86 to 96 in-distribution while the circuit shrinks from 0.70 to 0.34 of the
channels, and on the shifts from 74 to 93 (OOD-CV), 62 to 94 (ImageNet-A), 93 to 98 (ImageNet-O) and 74 to 93
(ImageNet-C) at the same 0.34 fraction; activation and rank scoring show smaller but same-signed gains. Specificity
moves sharply too: on ImageNet the certified circuits' other-class accuracy falls to 1% (a 99% reduction, as the
appendix states it), i.e. the circuits become more class-specific rather than merely more accurate. The
accuracy-versus-size curves for both families are three-staged — too small omits class-critical channels, an
intermediate band peaks, too large admits competing features — and certification shifts the curve in both directions at
once, peaking smaller and higher, which the paper reads as the unstable channels being both unnecessary and harmful.
Structural stability moves the same way: median IoU with the re-discovered circuit is higher and the spread tighter for
certified circuits, with the largest gain on the two shifts that show the largest accuracy gain, and ImageNet-O the
exception where both methods are high-variance. The headline abstract figures are up to 91% higher accuracy at 45% fewer
neurons; the intro and conclusion quote up to 56% / 80%-fewer and 91% / 45%-fewer in different places, so the exact
pairing depends on the scoring function and dataset, and the table cells above are the numbers I would quote.

## Limits and open problems they name
Certification costs many runs of the base algorithm — 100 in these experiments — so runtime grows linearly, though the
runs parallelise and the aggregation step is cheap. Larger certified radii require larger deletion probabilities, which
starve the base algorithm on small concept sets; more samples mitigate rather than remove this. A single sparsity
fraction is used across all layers, and the authors expect layer-wise budgets from the pruning literature to allocate
better. The experiments are CNNs; ViTs and other modalities, language models included, are named as future work. The
impact statement adds the honest caveat that a certificate covers only the threat model and concept dataset it was
computed for, and could be misread as a complete account of the computation. Two further limits of my own reading: a
guarantee about stability is not a guarantee about sufficiency, and the certified radius of 5 edits has no argument for
being the right size relative to a class's concept set.

## What we take from it
A method to import wholesale, and the cheapest stability instrument in this folder. The lesson is that stability is a
property you can *buy* rather than measure: aggregate a black-box estimator over randomized resamples of its input, and
report the components whose decision survives with a confidence margin, abstaining elsewhere. The wrapper is agnostic
to the base algorithm, so our own predictor of a component's role can be replaced by a vote over resampled fitting sets
and the resulting set carries a radius, not a percentage. The measure to copy is the three-valued output: certified-in,
certified-out, abstained, with the certification rate published beside the accuracy, since the compactness gain here
is partly abstention doing the work and a reader cannot tell those apart from the accuracy column alone. The warning is
the one this folder keeps meeting — a stability certificate is silent on coverage, and the paper's own defence of
sufficiency is a pruning accuracy scored against a full model, which is exactly the denominator a stable-but-empty
circuit would also pass. Practical consequence for us: any claim of the form "this component implements this behaviour"
should come with a resample-abstention table, and the abstained set is the interesting object, because it is the
component list whose role depends on which examples you happened to look at.

## Tags
unit: neuron | feature | circuit | layer
form: graph | causal-abstraction
method: ablation | patching | attribution | other
eval: ablation | prediction/simulation | faithfulness | downstream-task | benchmark
goal: understanding | safety/auditing | evaluation-of-methods
