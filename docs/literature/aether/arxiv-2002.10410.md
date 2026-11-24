# Lagrangian Decomposition for Neural Network Verification

- **Key:** arxiv-2002.10410
- **Authors / organisation:** Rudy Bunel, Alessandro De Palma, Alban Desmaison, Philip H. S. Torr, M. Pawan Kumar (University of Oxford); Krishnamurthy Dvijotham, Pushmeet Kohli (DeepMind). Bunel and De Palma listed as equal contribution.
- **Date:** 2020-02-24 (arXiv v1; v2 2020-04-05)
- **Link:** https://arxiv.org/abs/2002.10410
- **Kind:** paper
- **Read from:** full text (v2, all appendices A-G)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Getting sound bounds on a network's outputs is the bottleneck of every verifier, and the two existing routes both waste something: hand the relaxed problem to an off-the-shelf LP or MIP solver, which ignores that the constraint matrix is a stack of layer-wise blocks, or relax the problem further until it can be propagated in closed form, which gives up tightness. Their target is a solver that keeps the tightness of the standard relaxation and still costs only forward and backward passes. The vehicle is Lagrangian decomposition, the variable-splitting cousin of Lagrangian relaxation, which in general produces no weaker duals. The larger goal is the one this head shares: a bound routine cheap enough to sit inside branch-and-bound and inside adversarial training without over-regularising the network.

## What they studied

Feedforward networks with piecewise-linear (and, in an appendix, sigmoid) activations, on CIFAR-10. Two settings: incomplete verification on the small convolutional model of Wong and Kolter with two convolutions and two linear layers, and complete verification inside branch-and-bound on the base, wide and deep models of the branching dataset with 3,172, 6,244 and 6,756 ReLU units respectively. The unit of analysis is the layer boundary: the dual variable they carry is one vector per layer boundary, indexed by neurons, which is exactly a per-neuron, per-layer "worth" array.

## How they did it

They start from the convex problem obtained by swapping each activation equality for its convex hull (the PLANET relaxation for ReLU), then split the constraints into groups, one group per layer plus one group for the input domain and first linear layer. Variables that touch two groups are duplicated so each group owns its own copy, and agreement between the copies is enforced by a multiplier, so the multipliers are the only coupling. This yields a dual whose inner minimisation decouples completely: the input subproblem is either a coordinate choice when the input set is a box or a boundary point when it is a ball; the layer subproblem is a scalar-per-neuron convex piecewise-linear minimisation that they solve by evaluating three candidate vertices. The primal operations are a forward pass and a transposed weight multiply, so a whole batch of bounds runs on one GPU. For the dual ascent they give two solvers. One is supergradient ascent with Adam, where the supergradient is the difference between the two copies of the boundary variable. The other is proximal: an augmented Lagrangian with a quadratic consistency penalty per layer, the primal step taken by Frank-Wolfe with a closed-form conditional gradient reused from the inner minimisation and a closed-form optimal step size, the primal accelerated by block-coordinate (Gauss-Seidel style) updates and the dual by momentum, with a per-layer penalty weight as the only remaining hyperparameter. They prove their dual dominates the earlier Lagrangian-relaxation dual for ReLU at the same multipliers, and show in an appendix that initialising from the Wong-Kolter solution reproduces that method's bound exactly, which is how the iterative solvers are warm-started.

## What counts as an explanation here

Not an explanation of network behaviour at all, in the survey's sense: the output is one number per queried output unit, a certified lower or upper bound, plus the dual vector that produced it. What is interesting for us is that the dual vector is the interpretable object in disguise: one vector per layer boundary, each of which measures how much the relaxation is fighting the consistency constraint at that layer, and the sign of a transformed copy of it ("the multiplier pushed through the next weight matrix") selects which vertex of each neuron's hull is optimal, i.e. which of the three relaxation cases is active. The paper uses that quantity purely as a descent direction.

## How they judged success

Optimality gap to the best bound found, plotted as a distribution against runtime and pointwise per image, with Gurobi solved to optimality as the gold standard and a time-limited Gurobi arm stopped at the first dual-simplex iteration that exceeded the cost of 400 proximal iterations. Iteration counts per method are tuned so each arm runs in roughly equal time. For complete verification: cactus curves of percentage of properties verified versus seconds, plus average time, average number of sub-problems solved in the search, and percentage of timed-out properties, reported over all properties and split into easy, medium and hard by how long Gurobi took. Theorem 1 is the formal claim and the equal-time comparison is the experimental one.

## What they found

Incomplete verification: both their solvers beat the supergradient baseline on the same relaxation and approximately the same wall-clock, and the proximal arm in turn beats their own supergradient arm, with a heavier tail toward tighter bounds; the earlier relaxation-based dual is described as returning loose bounds even after 1,040 iterations on some properties of the normally trained network. Complete verification on the base model: average time 368.9 s for proximal versus 377.1 s for supergradient, 426.6 s for the earlier dual, and 1,588.7 s for Gurobi inside branch-and-bound, with sub-problems solved falling from 5,312.7 to 4,569.4 across the same ordering; timeouts 5.82%, 5.76%, 6.54% and 10.56%. On wide and deep models the advantage widens and the gap is visible mainly in the number of search nodes rather than in per-node cost, with the earlier dual timing out on at least two percentage points more properties. The Gurobi arm is the slowest everywhere despite optimal per-problem bounds, and the time-limited Gurobi arm worsens bounds substantially without much saving, which they attribute to the cost of a dual-simplex iteration.

## Limits and open problems they name

The theory guarantees their dual is at least as tight as the earlier one only at matched multipliers, for ReLU; other activations require redoing the hull and the per-neuron subproblem, which the sigmoid appendix does by hand and which tanh is explicitly left out. Intermediate bounds, which the whole method needs, are not computed with the same tight relaxation: doing so would cost two relaxations per neuron and a sequential layer sweep, so they fall back to the best of interval propagation and the Wong-Kolter layer-wise bounds, i.e. the bounds that shape every later per-neuron decision are from a different, looser method. They flag combining the supergradient and proximal solvers as future work, and note the proximal penalty weight has to be scheduled rather than derived. The per-layer multiplier profile is never printed anywhere: only aggregate time, node counts and gap distributions are reported.

## What we take from it

Three things. (1) A reusable framing: duplicate the boundary variables so every layer owns a copy, and the multiplier on the disagreement is a per-layer, per-neuron reading of where the relaxation binds — the same object the head's earlier verifier entries optimise and discard, here made the only unknown of the problem, which makes it the cheapest per-layer reading a bound routine can print. (2) A measure: sign and magnitude of the multiplier pushed through the next weight matrix decides which relaxation case each neuron sits in, so a per-neuron case map is already computed at every iteration and could be logged as the reading rather than as an intermediate. (3) A warning that this head has now met repeatedly: the intermediate bounds that condition all of this come from a looser method for cost reasons, so any per-layer reading we take inherits the quality of whichever cheaper pass produced its boxes, and must be reported with that provenance. Also a concrete calibration item: the search-node count is the quantity that moves when the bound improves, and they report it, which is the reporting habit the head wants carried over to per-unit objects.

## Tags

unit: layer, neuron
form: direction, probe
method: decomposition, weight-analysis, ablation
eval: prediction/simulation, ablation, benchmark, downstream-task
goal: understanding, control/editing, evaluation-of-methods
