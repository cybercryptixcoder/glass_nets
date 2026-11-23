# State space representations of the Roesser type for convolutional layers

- **Key:** arxiv-2403.11938
- **Authors / organisation:** Patricia Pauli, Dennis Gramlich, Frank Allgöwer; Institute for Systems Theory and Automatic Control, University of Stuttgart, and Chair of Intelligent Control Systems, RWTH Aachen
- **Date:** 2024-03-18
- **Link:** https://arxiv.org/abs/2403.11938
- **Kind:** paper
- **Read from:** full text (arXiv HTML, v1)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The group that recast CNNs as 2-D dynamical systems needs, for every convolutional layer, an explicit
state-space model of the Roesser form, because that is the object their LMI machinery consumes. Their earlier
work built such models with a symbolic-robust-control toolbox, which is correct but slow and does not reach the
sizes of modern layers. This paper asks whether the model can simply be written down from the kernel. The larger
goal is scalability: LMI-based robustness analysis and Lipschitz-bounded training only become practical for
convolutional nets if turning a layer into a system costs nothing. They also want structure, not just speed, so
they ask when the construction is the smallest possible one.

## What they studied

The unit of analysis is a single convolutional layer treated as a linear time-invariant (or affine, once the bias
is present) N-D system, with state split per propagation direction. The objects are the kernel tensor and the
four system blocks: the state matrix, input matrix, output matrix and the direct feedthrough. Concrete examples
are 3x3, 2x3, 3x2, 5x5 and 2x2x2 kernels; no trained network and no dataset appear. The paper is a construction
and minimality proof, not an empirical study.

## How they did it

They state the convolution as a finite sum of shifted kernel slices times shifted inputs, define the N-D Roesser
model with one state vector per signal dimension and zero initial conditions, and then propose an explicit
mapping from the kernel tensor into the five blocks. The construction is a memory shift register: the state along
the second dimension stores recent pixel values, exactly as a 1-D FIR filter does, while the state along the first
dimension stores partially weighted sums of those values, so they push the kernel slices into the matrix
positions where the shifting states will pick them up. The state matrix comes out block lower-triangular by
design. They prove the realisation correct by computing the 2-D transfer function of the proposed system and
showing it equals the transfer function of the convolution. Minimality is handled by a rank argument: after
matching coefficients of the Neumann series, one row-block times one column-block of the reachability and
observability matrices must have a certain block-triangular form with the leading kernel slice repeated down its
diagonal, and full column rank of that slice forces a lower bound on the state dimension that the construction
meets. They then extend the recipe to 1-D, N-D, dilated and strided convolutions, and treat padding as a choice of
boundary conditions.

## What counts as an explanation here

A compact, exact, non-unique algebraic surrogate: the layer's input-output behaviour is reproduced by a sparse
linear system whose entries are literally the kernel entries. It is complete for that layer's mapping and
predictive in the strongest sense, since a linear system simulates the layer exactly, but it is a re-description
of the weights rather than a statement about what the layer computes. Nothing is compressed away: the state
dimension is of the same order as the kernel's number of parameters.

## How they judged success

Two criteria only. Correctness is an exact transfer-function match, proved symbolically. Compactness is measured
as the state dimension, benchmarked against the obvious alternative of unrolling the convolution into a sparse
fully-connected layer, and for one channel counts it is proved minimal against every possible realisation.
There is no accuracy metric, no empirical comparison of analysis runtime, and no benchmark network; the
scalability claim rests on the construction being closed-form while the ancestor's is symbolic.

## What they found

For a standard 2-D layer the realisation has only a few states per direction, growing with the kernel size times
the channel counts, with the kernel slices sitting in the state matrix, the input matrix and the output matrix
rather than in one huge stored matrix. The realisation is proved minimal whenever the number of input and output
channels agree and the last kernel slice has full column rank; for unequal channel counts the construction may be
larger than necessary and the paper says so. N-D layers generalise by keeping an (N-1)-dimensional hyperplane in
memory and accumulating weighted sums along the last direction. Dilated convolutions are handled by expanding the
kernel with zeros and reusing the same recipe, and stride greater than one by first vectorising patches of pixels
into one input vector, with the input size scaling by the stride product and the state size correspondingly
reduced. When the stride equals the kernel size the system loses all memory and the state blocks vanish
entirely.

## Limits and open problems they name

Minimality is available only for the equal-channel-count case, and finding minimal realisations for general
channel sizes is listed as the immediate next step. The strided construction is presented only for two dimensions.
The realisation requires choosing a propagation direction and an origin corner, which is a modelling decision for
images but is fixed by time for recurrent layers. The paper never runs the analysis its construction is for:
Lipschitz estimation and Lipschitz-bounded synthesis for convolutional nets are named as future work rather than
performed.

## What we take from it

A method, plus a useful warning. The method: if our account wants a trained convolutional layer re-expressed in a
form where control-theoretic certificates apply, this gives a closed-form, exact, kernel-initialised system with
no fitting step, and its state is partitioned by direction, which means any per-block certificate slack has a
spatial interpretation (per row offset versus per column offset) rather than an arbitrary one. This is precisely
the missing piece behind the previous entry's per-layer-slack question: the slack blocks index offsets in the
kernel, so a layer whose row-offset state carries all the gain is a real finding. The warning: this is a
re-description, not a reduction. The state dimension scales with the kernel's parameter count, so an equivalence
argument of this kind buys formalism and certifiability but no compression whatsoever, and a "we converted the
network into a dynamical system" claim should always be accompanied by the state-size bill. Also worth keeping:
the paper's own honest admission that the minimal realisation is only known for square channel counts, which
mirrors the broader pattern that clean theoretical ground is repeatedly the case where the architecture happens
to be symmetric.

## Tags

unit: layer
form: replacement-model
method: weight-analysis
eval: prediction/simulation
goal: understanding
