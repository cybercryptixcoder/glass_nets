# Lipschitz-bounded 1D convolutional neural networks using the Cayley transform and the controllability Gramian

- **Key:** arxiv-2303.11835
- **Authors / organisation:** Patricia Pauli, Ruigang Wang, Ian R. Manchester, Frank Allgöwer; University of Stuttgart (Institute for Systems Theory and Automatic Control) and The University of Sydney (Australian Centre for Robotics)
- **Date:** 2023-03-20
- **Link:** https://arxiv.org/abs/2303.11835
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The robustness of a classifier is often reduced to the Lipschitz constant of its input-output map, and the field had two ways of dealing with it: certify a bound after the fact by solving a semidefinite program, or train with the constraint in the loop and pay for projections and penalty terms. The direct-parameterisation line had already removed that cost for fully connected layers — write each weight as the image of free variables so the certificate holds by construction and training is unconstrained. That trick had not been carried to convolutions, where the existing parameterisations lived in the Fourier domain and needed a matrix inverse whose size grows with the input length. The authors want the same unconstrained training story for 1D convolutional networks, and they frame 1D as a stepping stone toward the 2D case. The larger goal is a layer-wise construction that imposes dissipativity on each layer while only promising a bound on the composition.

## What they studied

1D convolutional networks built from convolution-plus-activation stages, optional max or average pooling, and dense layers, trained on the MIT-BIH arrhythmia benchmark in its preprocessed five-class form (26,490 samples split evenly into train and test). The studied network is small: two 3-kernel convolutions with front padding and average pooling, flatten, a 60-unit dense layer and a 5-way read-out. The unit of analysis is the layer as a linear system — each convolution is written as a finite-impulse-response system in state space, with its kernel taps sitting in the read-out matrix — plus per-layer multiplier and metric matrices inherited from the quadratic-constraint relaxation of slope-restricted activations.

## How they did it

Step one: restate each convolution as a discrete-time state-space system whose state is the rolling window of previous inputs, with dimension equal to kernel length minus one times the input channels, so nothing in the description depends on signal length. Step two: take the layer-wise linear matrix inequality conditions that certify that such a CNN is bounded (the ones this group published earlier for Lipschitz estimation of 1D CNNs) and read them as a recipe for constructing weights rather than for checking them. Step three: parameterise the dense layers with the Cayley transform, in the form already published: weights are a scaled product of the Cayley image of free variables with a lower-triangular factor carrying the incoming metric, with a per-unit diagonal scaling between them. Step four — the new contribution — make the convolutional version work by noticing that the upper-left block of the convolutional inequality is a Lyapunov inequality, so it has a unique closed-form solution, and that solution is the controllability Gramian of the layer's state-space pair. Parameterising the block's inverse as that Gramian (plus an extra freely-trained square term and a small regulariser) makes the block positive definite by construction, and the kernel is then recovered as a scaled Cayley image times the Cholesky factor of that block. Step five: handle pooling. Average pooling is linear, so it is folded out by dividing the target bound by the product of the pooling constants and leaving full metric matrices in place; max pooling is not, and the relaxation only supports it with diagonal metrics, so those layers get a separate parameterisation using a single Cayley image with the metric fixed from two diagonals. Step six: train everything unconstrained with cross-entropy, then compare against a vanilla CNN and two L2-regularised CNNs at fixation strengths 0.05 and 0.1, measuring clean test accuracy, the certificate from the semidefinite program, an empirical lower bound on the Lipschitz constant, and accuracy under L2 projected-gradient attacks at several radii.

## What counts as an explanation here

A parameterisation, not an explanation. The object produced is a map from free variables to weights that lands inside the set of weights whose layers satisfy a certain matrix inequality, and the claim is that everything in the image is safe by construction while the map is broad enough to cover the useful part of that set. For the dense layers the paper inherits a necessity-and-sufficiency result from its predecessor, so the map covers the set exactly. For the convolutional layers what is proved is sufficiency only: the constructed kernels satisfy the inequality, and no argument is given that every kernel satisfying it arises this way. Nothing in the paper is intended to predict what the network computes or to describe what individual units mean — the weights and the metric factors are training coordinates, consumed by the loss and never read.

## How they judged success

Test accuracy on the benchmark, the certified bound from the semidefinite program, an empirical lower bound on the Lipschitz constant, and accuracy under an L2 projected-gradient attack swept over perturbation magnitudes, all averaged over five trained networks. The comparisons are a vanilla CNN and two L2-regularised CNNs at different fixation strengths, and the trade-off plot is accuracy against the certified bound. There is no ablation of the Gramian term, no test of how much of the feasible set the parameterisation reaches, and no measurement of training wall-clock against the constrained alternatives — the paper asserts the cost saving from the earlier literature rather than measuring it on this task.

## What they found

On the arrhythmia task the bound-controlled models keep clean accuracy close to the unconstrained baseline while pushing the certified bound far down: the vanilla network certifies at a bound in the high hundreds, the regularised ones at tens, and the targeted models at roughly the target values they were trained for. The empirical lower bounds sit well under the certificates in every column, with the ratio varying by row rather than sitting at a constant — but the lower bounds are not close to the certificates, so the certificates are not obviously tight. Under attack, the bound-controlled models hold accuracy better as the perturbation grows, and accuracy falls as the target bound is squeezed, tracing the expected frontier. One practical observation: increasing the L2 penalty far enough made training fail outright, whereas the parameterised models trained at small bounds without difficulty.

## Limits and open problems they name

The work is 1D only, and the extension to 2D is named as future work with a pointer to reading convolutions as 2D systems. The convolutional parameterisation is not shown to be necessary, only sufficient, so the reachability question is open for every layer that is not dense. Max pooling forces diagonal metrics, which is a real restriction imposed by the relaxation rather than by the architecture. The appendix-level assumption that certain triangular factors are invertible is handled by an assertion that it held in all experiments plus a suggested exponential reparameterisation, not by a proof, and the authors say so.

## What we take from it

Three things. First, a method worth stealing outright: when a certificate's feasibility block is a Lyapunov inequality, its closed-form solution is a Gramian, and a Gramian is a *sum of outer products over the system's own state trajectory* — which means the certificate's metric is computable directly from the weights and the kernel taps, with no data and no solver. That is the same class of quantity run 668 asked for as a per-layer attribution instrument, obtained here for free. Second, a measure this paper prints and the neighbouring members do not: both the certificate and an empirical lower bound in the same table, which lets the ratio be read per row. Third, a warning about a gap in the family: the dense construction is proved necessary and sufficient while the convolutional construction is proved sufficient only, and the paper does not state the asymmetry. Anyone adopting the layer-wise account should check, per layer type, which half of the claim they actually have.

## Tags

unit: layer | weights
form: causal-abstraction | direction
method: weight-analysis | steering | training-interpretable
eval: ablation | patching | downstream-task
goal: control/editing | safety/auditing
