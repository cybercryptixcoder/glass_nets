# Controlling Neural Collapse Enhances Out-of-Distribution Detection and Transfer Learning

- **Key:** arxiv-2502.10691
- **Authors / organisation:** Md Yousuf Harun, Jhair Gallardo, Christopher Kanan; Rochester Institute of Technology and University of Rochester
- **Date:** 2025-02-15 (v3 2025-09-22); ICML 2025
- **Link:** https://arxiv.org/abs/2502.10691
- **Kind:** paper
- **Read from:** full text (v3 HTML and the abs page)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Detection of out-of-distribution inputs and generalisation to new distributions are usually studied by different
communities, and the paper sets out to show that the two pull the feature geometry in opposite directions. The
observation is that a layer whose class features are tightly collapsed makes ID-vs-OOD separation easy and makes
transfer to new classes hard, while a layer that keeps its features spread out does the reverse, so that a single
embedding cannot serve both ends. They argue this is a real trade-off and not an artefact of tuning, and the
constructive half of the paper is a way to hold two different degrees of collapse in two different parts of the same
network, one for detection and one for transfer. For our purposes this is the closest thing in the folder to a
controlled experiment on the collapse axis: the collapse level is a hyperparameter the authors can push up or down,
and there is a behavioural metric (a detection error and a transfer error) standing next to it at every setting.

## What they studied

ImageNet-100 as the in-distribution training set for the main results, with eight semantic-shift OOD sets (NINCO,
ImageNet-R, CIFAR-100, Flowers, CUB-200, Aircrafts, Pets, STL-10) used for both tasks, and ImageNet-10 for the small
coefficient sweep and the per-epoch dynamics runs. Five backbones: VGG17, ResNet18, ResNet34, ViT-Tiny and ViT-Small.
The unit of analysis is the neural-collapse geometry of a layer's activations — the four standard metrics NC1 to NC4,
with NC1 (the trace of within-class covariance times the pseudo-inverse between-class covariance, divided by the
class count) treated as the dominant scalar — together with the classifier weights of a protection head and, on the
encoder side, the effective rank of the feature matrix.

## How they did it

The architecture is encoder, then a two-layer MLP projector, then a linear classifier. Everything hangs on which
degree of collapse is wanted where. In the encoder, an entropy term is added to the usual classification loss; the
differential entropy of the layer's mixture density goes to minus infinity as the class-conditional covariances
shrink to deltas, so maximising it is maximising the distance to perfect collapse. Since the density is not available,
it is replaced by the Kozachenko-Leonenko nearest-neighbour estimate, and because affine constants drop out the
implemented penalty is the batch mean of the log distance to each sample's nearest neighbour, computed on
unit-normalised embeddings. In the projector the opposite is done: both MLP layers are set to a rank-(D-1) simplex
equiangular tight frame and frozen for the whole of training, so the projector is an architectural inductive bias
toward collapse that the classifier is then trained through, and the projector output is L2-normalised to push
features toward equal norm. Batch normalisation is also replaced by group normalisation plus weight standardisation
in the encoder, on the argument that batch statistics estimated on ID data are a bad match for OOD inputs. Detection
uses the energy score on the full network's logits and is reported as FPR95 averaged over the eight OOD sets; transfer
is a linear probe trained and evaluated on OOD embeddings taken from the encoder, never the projector. The comparison
that carries the causal claim is the encoder-versus-projector contrast inside one trained network, since the two
embeddings are produced by the same weights up to the projector and differ only in which mechanisms have acted on
them.

## What counts as an explanation here

A causal-engineering account rather than a description: the authors do not try to say what the network's collapse
means, they show that a scalar geometry statistic can be driven up or down by an explicit term, and that a behavioural
metric moves with it in the predicted direction. The "explanation" is the pair (mechanism, direction of effect) for
each task, plus one proposition formalising why entropy maximisation opposes collapse. It predicts a sign, not a
value: strengthen collapse and detection error falls while transfer error rises, and nothing in the paper claims a
functional form or a per-sample prediction.

## How they judged success

Three errors, all on held-out OOD data: ID classification error, OOD generalisation error (top-1 error of the linear
probe on each OOD set, averaged), and OOD detection error (FPR95 averaged). Every table prints the NC1-NC4 scalars
beside the errors it is supposed to move, and the tables are matched comparisons where they are readable. Controls
are within-network (encoder versus projector), within-configuration (entropy term on versus off, coefficient swept,
fixed ETF projector versus trainable projector, L2 normalisation on versus off, batch norm versus group norm, CE
versus MSE) and against one external baseline built on collapse theory. There is no seed reporting anywhere — Table 1
and Table 2 are single numbers per cell, and the word standard deviation does not appear — so every delta below is a
point difference whose noise floor is unstated.

## What they found

Inside one network, the more collapsed embedding detects better and the less collapsed one transfers better, on all
three architectures. Table 1: VGG17 projector NC1 0.393 with detection error 65.10 and transfer error 66.36, encoder
NC1 2.175 with detection 87.62 and transfer 41.85 — changing embedding costs 25.70 points of detection error and buys
24.51 points of transfer error. ResNet18 goes 0.341 / 69.70 / 63.08 against 1.762 / 86.17 / 47.72 (detection down
19.11, transfer up 32.19); ViT-Tiny goes 2.748 / 83.16 / 63.53 against 5.769 / 90.89 / 52.63. Against plain baselines
without either mechanism, transfer improves by 1.70 to 7.69 points absolute and detection by 7.01 to 29.82, with ID
error roughly unchanged (VGG17 12.18 to 12.62, ResNet18 15.38 to 16.14). The entropy term on its own, with the ETF
projector held fixed, gives NC1 2.18 against 1.31, transfer 41.85 against 44.56 and detection 65.10 against 67.46 —
2.71 points of transfer and 2.36 of detection. The frozen ETF projector against a trainable one gives detection 65.10
against 74.00 with transfer 2.47 points worse and ID error 2.48 points better. Every layer's NC1 is positively
correlated with its detection performance (Pearson 0.77) and negatively with transfer (−0.60) on VGG17; NC1 and
entropy across layers correlate at 0.88. The load-bearing ablation is Table 14, the only place the collapse axis is
swept: on ImageNet-10 without a projector, the entropy coefficient runs 0 to 1.0, effective rank runs 2211.99 / 2964.39
/ 3170.92 / 3761.33 / 4815.32, and transfer error falls 77.78 / 68.87 / 67.94 / 65.15 / 64.96 while ID error rises 9.20
/ 9.80 / 10.20 / 12.00 / 12.80 — the transfer gain is largest between 0 and 0.1 and nearly flat after 0.2, while the
ID cost is roughly linear in the coefficient. Two reversals are worth printing beside the trend: at coefficient 1.0
transfer error is slightly *worse* than at 0.6 (64.96 against 65.15), and for the fixed-ETF versus trainable-projector
contrast the lower-NC1 configuration wins on ID error (13.56 against 15.10) and loses on transfer, i.e. collapse does
not simply degrade transfer — it degrades transfer *and* helps in-distribution accuracy in that pair.

## Limits and open problems they name

The paper has no Limitations section; the Discussion names future work instead (open-world continual learning,
optimisation-driven routes to ETF alignment, other modalities, tasks beyond classification, parametric instead of
nearest-neighbour entropy estimation, and a deeper theory). Read off the text: the "trade-off" is established by
correlations across layers and across configurations, with no seed variance, on one ID dataset for the main table and
a ten-class ID dataset for the coefficient sweep; the entropy side relies on a nearest-neighbour estimate of a
quantity that is being driven to minus infinity, so the regulariser is measuring the density of a mixture the
classifier is simultaneously sharpening; the OOD sets are all semantic shifts at a fixed resolution, so the paper's
own claim that batch statistics break under covariate shift is never tested on a covariate shift; and the
proposition proves only that collapse implies unboundedly negative entropy, which licenses the sign of the penalty and
nothing about how much collapse it removes.

## What we take from it

Three concrete things. First, a method: the collapse scalar of a layer can be pushed at training time by a single
coefficient, and where that is possible a behavioural metric can be plotted against it, which is exactly the shape
our white-box model needs — geometry as an independent variable, task error as the dependent one. Second, the warning
that the direction of the effect is per-task and per-pair: collapse helped detection in every configuration here,
helped in-distribution accuracy in one pair, and hurt transfer, and the coefficient sweep shows the marginal benefit
turning negative at the top of its range; a sign claim built from one comparison would have been wrong. Third, a
measurement gap to exploit: the paper's central comparison is a single point per configuration with no variance and no
sweep, so the curve in Table 14 is the only evidence that the relation is smooth, and it is on a ten-class task with a
different architecture from the main results. For whatever probe we build, the practical reading is that a geometry
statistic read off a checkpoint inherits the sign of the training term that produced it, so any prediction about
behaviour should name that term — the same accounting run 736 extracted from an open-set loss, arriving here as a
training knob rather than an algebraic identity.

## Tags

unit: layer
form: causal-abstraction
method: training-interpretable
eval: ablation | downstream-task
goal: understanding | control/editing
