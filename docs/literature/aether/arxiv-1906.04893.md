# Efficient and Accurate Estimation of Lipschitz Constants for Deep Neural Networks

- **Key:** arxiv-1906.04893
- **Authors / organisation:** Mahyar Fazlyab, Alexander Robey, Hamed Hassani, Manfred Morari, George J. Pappas; Department of Electrical and Systems Engineering, University of Pennsylvania
- **Date:** 2019-06-12
- **Link:** https://arxiv.org/abs/1906.04893
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

The Lipschitz constant of a network is the sensitivity dial that decides whether a robustness certificate means anything: divide a margin by it and you can lower-bound the perturbation needed to flip a decision, and pass it into a feedback loop and you get a stability certificate for a learned controller. The field had two usable answers in 2019 and neither was good. Multiplying per-layer spectral norms is trivially scalable and in practice vacuous, while the two non-trivial alternatives run out of capacity — one blows up exponentially in depth, the other uses a greedy search that may not even be an upper bound. The paper asks whether the exact quantity can be *certified* (a guaranteed over-estimate, unlike sampling methods) without paying an impractical price, by borrowing a certificate formalism from control theory.

## What they studied

Fully connected feed-forward networks of arbitrary depth, in the ℓ2 norm, on the map from input to logit vector. The unit of analysis is the whole network treated as one operator, but the certificate is carried by a matrix over the *hidden neurons* — the decision object is a symmetric matrix T whose entries couple pairs of neurons, so the same formulation can be read at three granularities: one scalar per layer, one scalar per neuron, or a full pairwise interaction matrix. Experiments use randomly generated networks plus small classifiers trained on MNIST (five hidden layers, dropping to three for the robustness sweeps) and Iris.

## How they did it

The premise is an observation about activations rather than about networks: any activation a deep model actually uses is the derivative of a convex potential, which makes it slope-restricted on some interval — ReLU, tanh and sigmoid all sit in the [0,1] sector. Slope restriction can be written as a single scalar quadratic inequality in the difference of two input-output pairs, so the activation is *abstracted* by a family of quadratic constraints on (input difference, output difference) that hold for every pair of points. The key step generalises this to a whole layer: because every neuron in a layer uses the same activation with the same sector, one may take any nonnegative conic combination of the per-pair constraints across all neuron pairs, which yields one matrix inequality parametrised by a symmetric matrix T in a cone (diagonal plus pairwise-difference terms). They then lift this to the network: stack all pre-activation and activation values into one long vector, express the network as an affine map plus the componentwise activation in compact block form, and compose the quadratic constraint with the Lipschitz inequality itself. The result is a linear matrix inequality in the scalar ρ = L² and the multiplier T, so a single semidefinite program minimises ρ over both. That program has three natural restrictions — full pairwise T (n² variables), diagonal T (one per neuron), and one scalar per layer (block-identity T) — and the paper evaluates all three, plus a splitting strategy that chops deep networks into short sub-networks, bounds each, and multiplies.

## What counts as an explanation here

A certificate, not an account. The output is a number L, produced by a convex program whose feasibility is a proof that the number cannot be exceeded. There is no claim about what the network computes, which features cause the sensitivity, or which layer dominates — T is a dual object whose entries weight neuron pairs, and the authors never attach meaning to them. It is deliberately partial: a sufficient condition, so the gap between the certificate and the true constant is not analysed.

## How they judged success

Three measured things, plus two baselines they define themselves. Tightness: the certificate compared against the *naive lower bound* — the spectral norm of the product of weight matrices, which is always a valid lower bound on the true constant — and against CPLip and SeqLip, the only two competitors that scale past two hidden layers. Cost: wall-clock SDP solve time as a function of hidden units (500 to 3,000) and of depth (5 to 500 layers at 100 units each), plus the pairing sweep that shows the full pairwise variant growing quadratically. Utility: a robustness proposition that turns the bound into a certified ℓ2 radius per test point, applied to LP-trained and PGD-trained networks and compared against standard training.

## What they found

The headline empirical claim is that the certified bounds nearly touch the true value on the MNIST networks, and that the truth is very close to the naive lower bound — i.e. the certificate is tight *because the networks' Lipschitz behaviour collapses to the product of weights*. Across depths the neuron variant beats both competitors on tightness; the layer variant trades some tightness for a large cost reduction, with the deepest setting (500 layers, 50,000 hidden neurons) solved in roughly four minutes against roughly twelve for the neuron variant. The full pairwise variant is the most accurate and is reported as intractable past modest size. Robust training lowers the certified constant, and the accuracy collapse as the training perturbation budget grows tracks a matching collapse in the constant. Leaky ReLU networks are certified looser than ReLU networks of identical architecture.

## Limits and open problems they name

Convolutional networks are handled only by unrolling them into a very large dense network — the SDP sees no structure from weight sharing — and exploiting that structure is named as future work. Time grows sharply with width (the neuron variant crossing 700 seconds at 3,000 units in one hidden layer), so the largest experiments rely on splitting, which multiplies per-block bounds and therefore loosens as blocks are made shorter. Only the ℓ2 case is developed beyond a norm-conversion remark, and only one application (robustness) is pursued; they say the deeper question of *why* a trained network lands near its product bound is untouched.

## What we take from it

The method: slope restriction is a two-line property that turns any activation into a conic constraint, and a conic combination across neurons is what lets one program describe an entire layer — this is the cheapest route from "weights on disk" to "a guarantee about the map" we have in the head, and it needs no data and no forward pass. The measure: the paper's own two-sided comparison (certificate against the product-of-norms lower bound) is the discipline run 668 demanded, and it is printed here a year before the convolutional members of the family adopted it. The warning, and the sharpest thing in the paper: the certificate is tight *because the lower bound is tight*, and the authors state in passing that the full pairwise relaxation nearly coincides with that lower bound — so on the networks measured, the SDP's advantage over "multiply the spectral norms" is a few percent, and the entire field's tightness narrative lives inside the gap between the two columns of this one plot. For our account: any structural claim about a layer is a claim about *which* quadratic constraints its activation satisfies, so the constraint set is the detector we should be reading, and the certificate's tightness must always be reported relative to the free lower bound rather than in absolute terms.

## Tags

unit: whole-model | feature
form: replacement-model | probe
method: attribution | probing | other
eval: prediction/simulation | benchmark | downstream-task
goal: safety/auditing | evaluation-of-methods | understanding
