# Semialgebraic Optimization for Lipschitz Constants of ReLU Networks

- **Key:** arxiv-2002.03657
- **Authors / organisation:** Tong Chen, Jean-Bernard Lasserre, Victor Magron, Edouard Pauwels; LAAS-CNRS Toulouse, IMT Université Toulouse 3, IRIT/IRT Saint-Exupéry Toulouse
- **Date:** 2020-02-10
- **Link:** https://arxiv.org/abs/2002.03657 (read as https://arxiv.org/html/2002.03657v1)
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The Lipschitz constant of a network is the quantity behind robustness certificates, Wasserstein-GAN training and any
stability argument, and the field's cheap answer — multiply the layer norms — is admitted in the introduction to be far
too loose to use. The paper therefore asks how to get a *valid upper bound* on the constant (global, over all inputs, or
local, over a small ball around one input) with an optimisation method that is both sharper and cheaper than the
incumbent LP-based hierarchy. The larger goal is a computational pipeline that a practitioner can actually run on a
trained classifier, so the authors care as much about the solver's memory and runtime as about the tightness of the number.
The setting is fully connected ReLU classifiers with a linear scoring head, and the norm of interest is L∞ throughout,
chosen explicitly because real applications want it even though the SDP literature had mostly done L2.

## What they studied

Random fully connected networks of sizes (80, 80), (20, 20, 10) and a family of small 1- and 2-hidden-layer networks with
a symmetric banded sparsity mask on the weight matrices; plus one real trained MNIST classifier of size (784, 500) taken
from a robustness-certification worksheet. The unit of analysis is the whole network's Jacobian as a product, and the
quantity is one scalar per network (one scalar per label, computed for one label in the experiments). There is no feature,
neuron or circuit object anywhere in the paper — the witness objects are the multipliers of a moment relaxation, and they
are used only as solver variables.

## How they did it

They start from the observation that for a piecewise-linear network the gradient is a product of the weight matrices
transposed and the diagonal activation-derivative matrices, and that the Lipschitz constant equals the supremum over the
input region of the dual norm of that product (Theorem 1 of the earlier SDP paper, re-stated here). Because the constant
is a supremum rather than a difference quotient over pairs of points, only one input point is needed, which is what makes
a polynomial formulation tractable. The second ingredient is the key technical move: the ReLU and its derivative are both
*semi-algebraic*, so instead of relaxing the derivative to a variable in [0,1], as the LP predecessor did, the authors write
the graph of ReLU exactly as a polynomial system — the product of the output and its own difference from the input equals
zero, together with two linear inequalities — and likewise write the derivative as a binary variable with a signed
condition that ties it to the sign of the pre-activation. Adding a squared unit-norm constraint on the dual direction and
a quadratic box constraint for the local case yields an exact polynomially constrained problem whose objective is a product
of m activation derivatives, i.e. degree m+1 in the variables.

They then solve that problem by a moment (Lasserre) hierarchy, exploiting the block structure of the constraint graph:
the derivative variables and the pre-activations form small independent blocks, so a sparse hierarchy is applicable in
principle. It is not applicable in practice, and the paper is unusually explicit about why — the linear pre-activation
constraint of each unit joins the input block to the hidden block, and the standard repair for nearly sparse problems
(augment each block with two artificial variables per offending constraint) would blow the block size up past any gain.
Their remedy is a heuristic they label HR-d: keep the original small blocks and their localising matrices at order d, but
add back a single *dense first-order* moment matrix and one linear moment inequality for the troublesome constraint. For
two or more hidden layers the objective becomes cubic, so it is either lifted by a new product variable (p1·p2 of them,
rejected as memory-prohibitive) or replaced by a quadratic over-estimator on the unit box, obtained off-line by solving a
small SOS program that minimises the worst-case L∞ error. The over-estimator they get for a triple product is the
symmetric three-way average minus its marginals plus a quarter, with error exactly 1/4 on the box.

## What counts as an explanation here

Not an explanation of the network in any mechanistic sense — the output is a number, or a bracket on a number. What the
method contributes in the way of understanding is a *certificate*: an SOS decomposition of the gap between the claimed
bound and the network's gradient-norm function on the constraint set, whose existence proves the bound. The bound is
one-sided and partial by construction (an upper bound on a supremum), it is exact only for the one-hidden-layer case, and
it predicts nothing about behaviour except that no input pair can separate the outputs by more than L times their
separation. The paper never inspects the certificate's factors.

## How they judged success

Four arms compared on the same networks: Shor's first-order relaxation of their exact formulation, their HR-2, the LP
hierarchy of Latorre–Rolland–Cevher at degree 3 (and degree 4 where affordable), and LBS, a sampled *lower* bound obtained
by evaluating the gradient's dual norm at 50,000 random points. LBS is used as a consistency floor rather than as a target:
they state explicitly that every method must return a number no lower than it, which is the minimum sanity check that
no bound is invalid. Metrics are the bound value and wall-clock time, averaged over ten draws of random networks with
medians and quartiles shown; the real-network table is a single run. Global bounds use an input ball of radius 10 for
numerical stability, local bounds a ball of radius 0.1, and both are computed only for the second label of the MNIST
classifier. Hardware is stated (a 4-core laptop CPU, 8 GB), and implementations are YALMIP/MOSEK in MATLAB for random
networks and Julia/MOSEK for the trained one.

## What they found

On (80, 80) random networks the SDP arm is worse than the LP arm only at the loosest mask they test and wins steadily as
the mask is relaxed: at the loosest band the LP gets 1.42 in 22.4 s and HR-2 gets 1.45 in 3.14 s, while at the widest band
the LP gives 4.23 in 415 s against HR-2's 2.68 in 9.82 s — roughly a factor of 1.6 in bound and 42 in time at the dense end.
On (20, 20, 10) networks HR-2 is within 0.02 of the degree-4 LP arm at the sparsity where that arm is affordable (0.42
vs 0.41) while costing 0.65 s against 1261 s. Global bounds from Shor and HR-2 nearly coincide on the small random
networks; on the real MNIST net (784, 500) they separate — HR-2 14.56 global and 12.70 local against Shor 17.85 and 16.07,
with HR-2 costing about four to five times the time (12246 s vs 2869 s global, 20596 s vs 4217 s local). The LP method
runs out of memory on that network. The sampled lower bound for the MNIST net is 9.69 global and 8.20 local, so even the
better certificate is a factor 1.5 above the observed value globally and 1.55 locally. On two-hidden-layer networks the
cubic-term approximation still leaves HR-2 beating the degree-3 LP everywhere and costing a hundredth to a thousandth of
the degree-4 LP's time.

## Limits and open problems they name

The one-hidden-layer case is the only one the hierarchy is exact for; beyond that the objective's degree rises with depth
and must be reduced by lifting (memory-prohibitive) or by the quadratic over-estimator whose error they never propagate
into the reported bound. They do not state whether the numbers in the two-hidden-layer figures use the over-estimator
without correcting for its error, which would make those bounds not strictly valid. The heuristic HR-d has no convergence
guarantee and is named as heuristic throughout; its whole justification is empirical comparison with Shor. Local bounds
for the LP arm could not be compared at all because the predecessor did not publish code for that case. Runtime is minutes
to hours on a laptop-class machine for a single 784×500 network, which they acknowledge only by contrast with the LP arm.
They name one open direction: cheap approximate sparse certificates for higher-degree objectives, so that deeper networks
and non-ReLU activations (semi-algebraic ones only, by construction) become reachable.

## What we take from it

A **framing** and a **measure**, plus one **warning**. The framing: represent the activation and its derivative as an
exact polynomial system rather than a relaxation, and let the derivative variable be a genuine binary that carries the
sign of the pre-activation — this is the cleanest statement in the survey's constraint head of why the exact formulation
is worth paying for, and it is a one-line change to the relaxation the rest of the head uses. The measure: report every
certified constant beside a *sampled* lower bound computed from the same weights with no retraining, and report the ratio;
this paper's own ratio of about 1.5 on a real trained network is the most useful number in it and is the discipline the
head has been missing since run 679. The warning: a cheaper relaxation that adds a single dense first-order moment
constraint is a heuristic without a convergence proof, so a bound from it is only as good as its agreement with the LP arm
on the same networks — which is exactly the isolation discipline of run 682. Also directly usable: the block-size
arithmetic here (block size grows as input width plus d for the input block) is the same computation run 683 named as the
binding constraint, reached independently, and it is what makes a one-layer 784-input network take hours.

## Tags

unit: whole-model
form: replacement-model
method: decomposition
eval: prediction/simulation
goal: evaluation-of-methods
