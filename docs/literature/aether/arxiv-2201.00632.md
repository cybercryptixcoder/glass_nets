# Neural network training under semidefinite constraints

- **Key:** arxiv-2201.00632
- **Authors / organisation:** Patricia Pauli, Niklas Funcke, Dennis Gramlich, Mohamed Amine Msalmi, Frank Allgöwer (University of Stuttgart IST; RWTH Aachen)
- **Date:** 2022-01-03 (v1); v3 2022-09-19
- **Link:** https://arxiv.org/abs/2201.00632
- **Kind:** paper
- **Read from:** full text (arXiv HTML v3)
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Violating a robustness or stability certificate is easy when the constraint is imposed as a penalty term, and the existing exact alternatives for imposing a semidefinite constraint during training — ADMM with an inner SDP per step — made constrained training far slower than unconstrained training. The authors want a general recipe for "train a network while a matrix inequality stays true", and they want it cheap enough to use on networks with millions of parameters, including convolutional ones. The concrete goal is twofold: beat ADMM on the standard Lipschitz-bounded classifier benchmark, and be the first to train a Wasserstein GAN whose discriminator is certified 1-Lipschitz by a semidefinite constraint rather than by weight clipping or a gradient penalty.

## What they studied
Feedforward networks with slope-restricted activations (tanh in the classifiers, leaky-ReLU and sigmoid in the discriminator), from a three-class 2-D toy problem with two hidden layers of 10 units, through MNIST classifiers with two hidden layers (100 and 30 units, 14×14 inputs) and a scaling sweep on the toy problem, up to a fully convolutional WGAN with a 3-hidden-layer discriminator and 5-hidden-layer generator on 18×18 MNIST. The unit of analysis is the whole network's certificate matrix — a block-tridiagonal symmetric matrix whose blocks interleave one layer's weights with the next layer's slack — plus the slack multipliers themselves, one nonnegative diagonal entry per unit.

## How they did it
They frame training as bi-level: minimise the task loss over weights subject to a set of matrix inequalities that depend on weights and extra slack variables. The inequalities come from summing an incremental quadratic constraint on the input-output difference (what the network should satisfy) with a slope-restriction constraint on every activation's input-output difference (always true for the network); the result is positive semidefinite exactly when the network property follows. Rather than solving each subproblem exactly with ADMM, they replace the hard constraint with a negative log-determinant barrier and simply backpropagate through the modified loss, so ordinary first-order optimisers work and the slack variables can be learned jointly. Two feasibility devices keep the iterate inside the cone: weights start small enough that the network is trivially Lipschitz-bounded, and after each update a Cholesky factorisation of the certificate matrix is attempted — if it succeeds the iterate is feasible, and the gradient step is shrunk until it does. The engineering contribution is doing that Cholesky block by block along the tridiagonal band and then reconstructing only the banded blocks of the inverse that the gradient of the barrier actually touches. They compare against the same constrained problem solved with ADMM (using the reference authors' code) and against projected gradient descent, and ablate whether the slack multipliers are fixed hyperparameters (linear constraint) or learned (bilinear).

## What counts as an explanation here
Not an explanation of behaviour — the artefact is a *certificate*: a positive-semidefinite matrix that proves a norm bound holds for all inputs, plus the multipliers that make it true. The certificate is partial in the sense that it is sufficient, not necessary: it pins an upper bound of a prescribed size, never the network's true Lipschitz constant. Its practical content is a number (the certified bound) attached to a trained network, and a set of per-unit slack weights the certificate needed to reach it.

## How they judged success
Test accuracy on held-out data; the certified Lipschitz upper bound reported as mean and maximum over 20 differently initialised networks (5 for the GAN); normalised training time relative to nominal unconstrained training in the same framework; a scalability curve of mean training time against number of decision variables. For the GAN, the certified bound is compared with the bounds obtained under weight clipping and gradient penalty, with generated images shown qualitatively.

## What they found
On the 2-D example with a prescribed bound of 50, the nominal network's reported way of measuring its own sensitivity lands near 10,780 while all constrained variants sit near 44–48 with no accuracy loss (0.90 against 0.868 nominal); the bilinear barrier gives the tightest bound (43.94) and is the fastest constrained method (1.15 normalised against 78.72 for ADMM). On MNIST with a prescribed bound of 20, the bilinear barrier reaches 16.7 with 0.982 accuracy at 1.74 normalised training time, against 7.80 and 0.883 for the ADMM baseline at 104.4 — i.e. the barrier method is about two orders of magnitude faster and, in this run, more accurate than both. For the WGAN, weight clipping keeps the certified bound absurdly low (0.005), gradient penalty overshoots the unit bound by three orders of magnitude (1,234 mean), while the barrier method lands just above one (1.38–1.40) at roughly 2.4× the unconstrained training time, and only the barrier-trained generator's samples resemble digits. They also note that the constraint is bilinear in weights and multipliers but becomes linear under a change of variables to the product of the two.

## Limits and open problems they name
The exact certified bound could not be computed for the convolutional GAN — the certificate for that architecture is intractable to solve, so Table II's numbers come from a cruder scalar-multiplier relaxation with the network split in two and the parts' bounds multiplied, which they flag as conservative. Feasibility after each step is maintained by shrinking the step size rather than by a principled line search, so the schedule is a hyperparameter. Convolutional layers are handled only by expanding them into doubly block-circulant dense matrices, which discards the structure that made the dense case cheap. The nonlinear (bilinear) constraint needs a feasible initialisation heuristic; the linear one is convex but conservative.

## What we take from it
Two things. First, a **framing**: the certificate's slack variables are themselves a per-unit read-out — one nonnegative weight per hidden unit, learned jointly with the weights, which is exactly the "which units is the guarantee allocating budget to" quantity the constraint head keeps circling, and here it comes free with training rather than from a post-hoc SDP. Second, a **warning with a number attached**: presetting the multipliers (the convex variant) versus learning them buys a modestly better bound at the cost of a non-convex constraint, but the paper never reports what the learned multipliers look like — so our predictive model can claim a genuinely unclaimed measurement simply by printing the fitted per-unit slack vector against ReLU/tanh firing statistics on a trained checkpoint. Third, a **method worth copying**: checking PSD by attempting the factored Cholesky and reconstructing only the banded blocks of the inverse whose entries interact with the parameter's position in the matrix — a cheap "did this edit keep the certificate alive" test that any interventional experiment on a certified network could use as a guard rail.

## Tags
unit: weights
form: direction
method: training-interpretable
eval: downstream-task
goal: control/editing
