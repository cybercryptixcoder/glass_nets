# Orthogonalizing Convolutional Layers with the Cayley Transform

- **Key:** arxiv-2104.07167
- **Authors / organisation:** Asher Trockman; J. Zico Kolter (Carnegie Mellon University; Kolter also Bosch Center for AI)
- **Date:** 2021-04-14
- **Link:** https://arxiv.org/abs/2104.07167
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Orthogonal weight layers are a known source of three desirable properties: activation and gradient-norm stability, better
conditioning for very deep networks, and a hard cap of one on the layer's Lipschitz constant. For fully connected layers
several exact parameterisations existed; for convolutions the field had only penalties, upper bounds obtained by
reshaping the kernel into a matrix, or (BCOP) a parameterisation restricted to fixed-size kernels. The authors want a
parameterisation that is direct (no projection step, no penalty), expressive (kernel size not fixed in advance), and
scalable (works for large channel counts). The larger goal is a deterministic robustness certificate: if every layer is
1-Lipschitz then the network is, and a margin test yields an ℓ2 certificate without any sampling.

## What they studied

Convolutional layers of the form ℝ^(c_in×n×n) → ℝ^(c_out×n×n) for circular (wrap-around) convolutions, with unit
analysis on the layer's Fourier-domain representation: the unit is the linear map, not the kernel and not a neuron.
Models are CIFAR-10 networks — the "KWLarge" architecture taken from prior work, plus ResNet9 and WideResNet10-10
made 1-Lipschitz. Comparisons are against BCOP, reshaped-kernel orthogonalisation (RKO and a Cayley variant CRKO),
one-sided spectral normalisation (OSSN) and singular-value clipping with alternating projections (SVCM).

## How they did it

A multi-channel circular convolution is block-diagonalised by a Fourier transform into n² blocks of size
c_out × c_in. The Cayley transform maps a skew-Hermitian matrix to a unitary one; applied block-wise in the Fourier
domain to a skew-Hermitised weight it produces a convolution that is exactly real in the spatial domain (they prove
this with a unitary-conjugation lemma). Because the blocks are small, the inverse the Cayley transform needs costs
O(n²·min(c_in,c_out)³) rather than inverting the whole c·n² × c·n² operator, and the rectangular case is handled with
a Schur-complement identity. Training adds weight normalisation on the raw tensor, uses a multi-class hinge loss whose
margin is the certificate radius, GroupSort (MaxMin) activations, no batch norm, and residual connections made
non-expansive by a convex combination with a sigmoid-gated coefficient. Striding is replaced by invertible downsampling
or by scaled average pooling. The whole network's Lipschitz bound can be retuned by multiplying each of m layers by
L^(1/m), and the authors use that knob to walk the accuracy/robustness curve.

## What counts as an explanation here

Not a description of mechanism and not a probe. The object produced is a *parameterisation* — a constraint on the
weight space such that every point it contains has a known, exact spectral property, and therefore a known contribution
to a compositional bound. The claim about the network is the product of its layers' constants and a margin; that is a
proof about the whole model, complete for the adversary it names and silent about everything else in the network. It
predicts exactly one thing: which points are certified.

## How they judged success

Three separate measurements. (1) Orthogonality in practice: the ratio of output to input norm of sampled activations
under 32-bit arithmetic, checked before and after training, expected within about 1e-5 of one. (2) Deterministic
certified accuracy at ℓ2 radius 36/255 on CIFAR-10, using the Lipschitz-margin test, reported as the fraction of test
points whose margin exceeds √2·L·ε. (3) Empirical robustness under PGD and AutoAttack at the same radius, plus an
estimated *local* Lipschitz constant computed with a PGD-style method. Averages carry standard deviations over 3 or 5
seeds. Runtimes are profiled per epoch.

## What they found

The layer is orthogonal to within roughly 0.99999–1.00001 of unit gain in 32-bit, at any channel count, before and after
training. On KWLarge it reaches 75.33% clean and 59.16% certified accuracy, against BCOP's 75.11% and 58.29%; widening
the network threefold and using 5×5 kernels lifts the certified figure to 61.13% against 60.55% for BCOP. Reshaped-kernel
methods are "a percent or two" behind on certification. In the empirical-robustness tables on ResNet9 the ordering is
muddled: Cayley has the best clean accuracy (81.70%) and best AutoAttack accuracy (71.17%) but RKO is ahead on PGD
(73.37% vs 73.77% — Cayley ahead there too), and on KWLarge the reshaped-kernel methods are about two points more robust
for a marginal clean-accuracy cost. Their empirical local Lipschitz constants (Table 2) are 2.245 for Cayley, 2.272 for
BCOP, 2.891 for RKO, around 1.7–1.9 for CRKO/OSSN/SVCM — i.e. the layers with the strongest hard constraint are not the
ones with the smallest measured local constant. Runtime: the Cayley layer is roughly 0.5×–4× BCOP depending on shape,
much faster at small spatial size and unequal channel counts, but slower than BCOP on WideResNet (210.6 vs 109.4 s per
epoch), and orthogonal layers generally cost several times a plain convolution.

## Limits and open problems they name

The parameterisation is incomplete: Cayley maps onto orthogonal matrices without eigenvalue −1, so a fixed ±1 diagonal
factor would be needed to cover the rest and the authors could not optimise over that discrete set. Strided convolutions
are incompatible and are emulated. Convolutions are circular, so padding is wrap-around rather than zero. Cost is
substantially higher than plain convolution. They could not improve certified accuracy with ResNets, and they say the
orthogonality benefit may only show up in deeper networks than were affordable. Their own closing hypothesis is that the
gain in robustness comes not from orthogonality but from the ability to *tune* the Lipschitz constant — which is a
hypothesis about their own headline contribution.

## What we take from it

A **method and a warning**. Method: this is the cleanest worked example we have of buying a guarantee by
reparameterisation rather than by measurement — the Cauchy/Cayley construction is a bijection, so the property holds at
every point of the weight space and needs no penalty term or projection, and the Fourier block-diagonalisation is what
makes the inverse cheap. Warning: their own fixed-Lipschitz comparison is the strongest evidence in the paper *against*
the framing, and it is the reason this entry is relevance 2 rather than 3 — once each layer family is rescaled to the
same certified Lipschitz bound, the four top methods land within about a point of each other, so "orthogonal" was never
the operative variable. For our white-box account the rule is: a structural property we attribute a capability to must be
separated from the hyperparameter that usually accompanies it, because the property and the knob travel together and only
the equal-knob comparison tells them apart.

## Tags

unit: weights | layer
form: replacement-model
method: weight-analysis | other
eval: prediction/simulation | ablation | downstream-task
goal: control/editing | understanding | science
