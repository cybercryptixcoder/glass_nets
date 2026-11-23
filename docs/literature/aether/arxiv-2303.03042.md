# Convolutional Neural Networks as 2-D systems

- **Key:** arxiv-2303.03042
- **Authors / organisation:** Dennis Gramlich, Patricia Pauli, Carsten W. Scherer, Frank Allgöwer, Christian Ebenbauer; Chair of Intelligent Control Systems, RWTH Aachen, and Institute for Systems Theory and Automatic Control plus the Chair of Mathematical Systems Theory, University of Stuttgart
- **Date:** 2023-03-06
- **Link:** https://arxiv.org/abs/2303.03042
- **Kind:** paper
- **Read from:** full text (arXiv HTML, v1)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The robust-control line that treats a network as a linear system plus sector-bounded nonlinearities had a
convolutional gap: to get a Lipschitz bound for a CNN by that route, one first flattened the image and wrote the
convolution as an enormous sparse Toeplitz matrix, which turns every certificate into a problem sized by the
number of pixels. This paper asks whether a convolutional layer can instead be written as a proper state-space
object whose size depends on the kernel and the channel counts but not on the image. If that works, the same
robust-control machinery applies to an operator on whole signals, so one certificate covers every input
resolution simultaneously. The stated goal is exactly that: a compact realisation, and then a bound that holds
for images of any size.

## What they studied

Individual convolutional layers and stacks of them, with a fully convolutional net modelled as an operator on
two-indexed signals over the integer plane with finitely-supported inputs; a later section glues a flattened
convolutional trunk to ordinary dense layers. The unit of analysis is the layer as a linear shift-invariant filter
whose impulse response is the kernel, lifted into a two-directional state-space form with one state vector per
propagation direction. The nonlinearity stays outside the linear part as a slope-restricted scalar map applied
pointwise. Experiments use randomly sampled small kernels and one trained MNIST classifier; there is no
activation-level analysis and nothing about features.

## How they did it

They define the layer as a kernel-sum convolution on the signal space, then take the transfer function of that
impulse response and turn it into a linear fractional representation in the two shift operators, which reads off
directly as a two-directional state-space model of the Roesser kind. Affine terms carry the bias, and a shift is
introduced to make the non-causal image convolution into a causal transfer function; because all layers and
activations commute with shifts, the per-layer shifts can be gathered at one end. They then write all layers'
equations in one vectorised system, block-diagonal in the state matrices with the activation outputs feeding the
next layer, which yields a linear system in feedback with the pointwise nonlinearity — a two-dimensional analogue
of the classical Lur'e interconnection — and show by the block structure that it can be solved layer by layer,
hence well posed. For a worked single layer they simply take as state every input value in the causal cone ahead
of the current pixel, i.e. the recent pixel neighbourhood, with the state matrices as selection/shift matrices and
the read-out applying the kernel entries. To get from a CNN to a Lipschitz statement they form the system obeyed
by the difference of two trajectories, which keeps the same linear part and makes the nonlinearity a
point-dependent difference of the activation, and then bound that error system's energy gain. Bounds come from
dissipativity for such two-indexed systems: quadratic storage functions per direction and quadratic supply rates,
with the dissipation inequality replaced by a linear matrix inequality in the storage matrices under a
slope-restriction supply rate for the activation. A separate theorem handles the hybrid case, chaining the
convolutional certificate to a standard dense-layer LMI and optimising over the shared supply matrix.

## What counts as an explanation here

A replacement model in the systems sense: an exact linear state-space surrogate for each layer, plus a certificate
about the whole network. It is complete for the layer's mapping and exactly simulable — the point of the paper is
that the surrogate is the same operator, not an approximation — but it is a re-parameterisation of the kernel, not
a statement about what the layer represents. The only predictive object is the bound: a single number that
constrains output differences for all images at once, for all inputs of any size.

## How they judged success

Certificates, not benchmarks. Correctness of the realisation is an algebraic identity between transfer functions.
Lipschitz success means a solved semidefinite program yielding an upper bound, compared against four alternatives:
two other state-space routes (a toolbox-generated small realisation, and the redundant one just described, the
latter reduced to its reachable subspace), a Fornasini–Marchesini realisation with its own bounded-real lemma, the
published dense-network Lipschitz toolbox, and the operator norm of the Toeplitz matrix (the last two give the
exact value for one layer and serve as a lower reference). Measures are the bound value and solver wall-clock time,
averaged over one hundred randomly sampled single layers as a function of input size, and then reported once for a
trained classifier at two resolutions.

## What they found

Every state-space route gives a bound that is flat in input size with roughly flat solve time, while the
Toeplitz-and-dense routes grow and become intractable: for the single-layer sweep the dense-network solver and the
Toeplitz norm agree with the exact value at small sizes but the dense route's cost blows up with input side (the
authors' own fitted growth exponents are steep), so from roughly a few tens of pixels per side the convolutional
routes win on cost, and the Toeplitz/dense routes stay tighter for a fixed small size. For the trained MNIST
network with two convolutional and two dense layers, at the reduced resolution the two-dimensional route solved in
under two minutes and reported a bound about twice the dense-network solver's, which took several minutes and
needed far less memory per neuron; at full resolution the dense solver ran out of memory while the two-dimensional
route still returned a (looser) bound in around twenty-five minutes, and the cheap Toeplitz norm gave a much
larger number in milliseconds. Two qualitative findings matter more than any of these numbers: different
realisations of the same layer give different bounds, with the redundant neighbourhood-state realisation
apparently hitting the exact value while its minimal counterpart does not, and the Fornasini–Marchesini
realisation was consistently the most conservative.

## Limits and open problems they name

Only fully convolutional stacks plus a dense tail are covered — normalisation and pooling are excluded, with
pooling named as an extension. The bound is loose where kernels are large relative to the image and the
convolution-to-flattening transition is suspected of adding conservatism. Batching, stride and dilation are not
treated here. The exactness of the redundant realisation is reported as an observation, explicitly flagged as
something that should be studied rather than explained, which leaves the reader unable to choose a realisation on
principle — only empirically. The paper also notes that ReLU is a poor choice for genuinely Lipschitz-constrained
training and points to group-sort activations instead.

## What we take from it

Three things, all methodological. First, a framing we can reuse: the *state* of a layer is the causal cone of
pixels ahead of the output, so if our account ever wants to say what a convolutional unit "knows at a position",
this paper's state is the honest, exact answer — the recent receptive-field history — and any interpretability
claim about a layer's internals that exceeds that window needs separate evidence. Second, a usable warning about
realisation choice: bound quality here depends on a modelling decision (how many redundant states to keep), not on
the network, so any certificate-style number we report about a layer must name the realisation that produced it,
in the same way we insist on naming the comparator for other metrics. Third, and most useful to us as a measure:
the difference-of-trajectories move. It reduces "how does the network respond to a perturbation" to a fresh
autonomous system with a signal-dependent nonlinearity, which is exactly the object we would need if we wanted a
predictive account of intervention response rather than a description of weights — here it is used only to
integrate a gain, but the construction itself is free and does not assume linearity of the network.

## Tags

unit: layer
form: replacement-model
method: weight-analysis
eval: prediction/simulation
goal: understanding
