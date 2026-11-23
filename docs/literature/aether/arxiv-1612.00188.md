# Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections

- **Key:** arxiv-1612.00188
- **Authors / organisation:** Zakaria Mhammedi; Andrew Hellicar; Ashfaqur Rahman; James Bailey (Data61, CSIRO; University of Melbourne)
- **Date:** 2016-12-01 (v1); v5 revision 2017-06-13
- **Link:** https://arxiv.org/abs/1612.00188
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Recurrent nets fail on long sequences because the hidden-to-hidden transition matrix either amplifies error
across steps or damps the signal out; a spectral norm of exactly one is the sweet spot, and the unitary-RNN line
of work got it by forcing the transition matrix to be unitary throughout training. Those methods trade
expressiveness against cost: the original factorised unitary net cannot reach the whole unitary group above a
small hidden width, and the full-capacity versions pay a cubic-in-width term per gradient step from a matrix
exponential or a Cayley-based projection. The authors' goal is a parameterisation that is exactly orthogonal by
construction, reaches the entire orthogonal group, and costs no more per step than a plain RNN even at a batch
size of one.

## What they studied

The hidden-to-hidden transition matrix of a simple RNN, as a unit of analysis: not neurons, not features, but one
weight matrix constrained to lie on a manifold. Models are dense RNNs of 28 to 512 hidden units. Data are five
long-range tasks - song-sequence prediction, the addition problem with lags of 400 and 800, pixel-permuted MNIST,
character-level Penn Treebank, and the copy task - chosen because they are the standard proving ground for the
unitary-RNN family. Baselines are a plain RNN, an LSTM, and published numbers from the factorised unitary net,
its restricted- and full-capacity variants, the Lie-algebra exponential parameterisation, and a soft-constraint
Cayley/Stiefel method.

## How they did it

First a translation lemma: an RNN over complex numbers with a unitary transition matrix is exactly an RNN over
reals with twice the hidden width, whose block transition matrix embeds the real and imaginary parts; the
activation function needs a matching real-space rewrite (they supply one for modReLU). That lemma is the
justification for working on real orthogonal matrices at all. The parameterisation is then a product of
Householder-type factors, one vector per reflection, where the last factor is a plain scalar instead of a
reflection. Each partial Householder acts only on a trailing block of coordinates, which means the product has a
compact representation (a rectangular parameter matrix, plus an upper-triangular correction matrix) so that
neither the forward pass nor the backward pass ever materialises an n-by-n matrix. They differentiate through
that compact form to get closed-form gradients with respect to the parameter matrix and the hidden state, then
turn the derivation into a short iterative algorithm whose per-step cost sums a handful of vector operations per
reflection. Training is Adam with uniform initialisation; the scalar factor is clipped to ±1 after every update,
and the structurally-zero upper triangle of the parameter matrix is re-zeroed after every update, so the orthogonality
holds under any number of iterations rather than being restored by a projection.

## What counts as an explanation here

None in the interpretability sense - this is a numerical-parameterisation paper, and what it produces is a
guarantee and a gradient formula. The "explanation" is constructive: a map from unconstrained parameters to the
orthogonal group, proved surjective onto that group when the number of reflection vectors equals the width, plus
an argument that the map is differentiable everywhere off a measure-zero set. It predicts that a network built
this way cannot have exploding gradients through the recurrent path, and constrains nothing at all about what the
network computes - the parameters are not meant to be read.

## How they judged success

Four measures: the realised spectral behaviour (does the product actually stay orthogonal, checked by
construction rather than by a test statistic); predictive error on the five tasks, mostly as normalised or plain
mean squared error and bits-per-character, with medians over five seeds and five excerpts and 25th-75th and
9th-91st bands for the song task; accuracy against published unitary-RNN numbers on permuted MNIST at roughly
matched parameter counts and, in one comparison, against a baseline with three times the parameters; and flop
counts per forward and backward step as a function of width and number of reflections, with and without storing
the intermediate factor values. Baselines are the plain RNN and LSTM run on the same data by the authors, plus
numbers lifted from the literature for networks they did not retrain.

## What they found

The parameterisation does what it claims mechanically: with fewer than the hidden width in reflection vectors the
product is orthogonal as long as the vectors are nonzero, and at full count the map covers the whole orthogonal
group with n(n+1)/2 parameters, which the authors note is only n more than the manifold's own dimension. Per-step
flops stay linear in width times the number of reflections, and at full count the global backward pass is about
five thirds of a plain RNN's while the forward is about three halves; the price is storage, since caching all
intermediate factors costs a factor of the reflection count more memory than caching hidden states. On the tasks,
the song experiment improved on the plain RNN and matched or beat the LSTM on average; the addition task beat the
MSE of the trivial predictor within 5,000 iterations at both lags for two seeds, which the authors report the
plain RNN and LSTM failed to do; permuted MNIST reached 97.2% test accuracy with about 11K parameters, above three
of the four published unitary baselines; and on character-level Penn Treebank the method tied the LSTM at one step
ahead (1.68 bits per character against 1.68) and trailed it at five (3.85 against 3.80). On the copy task it
failed outright with the leaky-ReLU it used elsewhere and only became comparable after switching to a
norm-preserving activation - and swapping in the unitary net's own activation did not fix it, which the authors
read as evidence that the block structure of the real-space unitary matrix is doing something their reflection
product cannot.

## Limits and open problems they name

The authors say plainly that further experiments are needed to explain the difference between the unitary and
the orthogonal constraint, i.e. they do not claim equivalence, only similar benefit. The copy-task failure is
attributed to activation choice on the strength of two attempts, and the counter-attempt (the unitary net's
activation on their architecture) leaves the gap unexplained rather than closed. Storing the factor matrices is
called impractical at large reflection counts, and the alternative recomputation is offered without a comparison
of the two at scale. The soft-constraint extension is a one-sentence suggestion - let the scalar factor vary in
the closed interval - and is never run. The literature comparisons on permuted MNIST are against published
numbers rather than retrained baselines, and no variance is reported for any comparison except the song task.

## What we take from it

Two reusable pieces and one warning. The reusable mechanism is the transcendental-free surjective map onto the
orthogonal group: because the scalar factor is a sign rather than a reflection, the construction reaches the
matrices a reflection-only product provably cannot (the authors give the two-dimensional example of a reflection
composed with a flip that no product of two pure reflections produces) - which is exactly the class of matrices
the Cayley parameterisation of the convolutional case also cannot reach, so this is the missing-coverage answer
that head asked for, arriving five years earlier and in a different modality. The gradient derivation is a
template worth copying: differentiate through the compact factorisation rather than the factors, and the cost
collapses from cubic to linear-in-width-times-reflections. The warning is about how the guard rails are enforced:
orthogonality here is kept true not only by the parameterisation but by two discrete repairs after every gradient
step - clipping the scalar and re-zeroing the structural zeros - and neither repair is ablated, so a reader cannot
tell how much of the stability is the map and how much is the hand at the elbow. For our own probing work the
direct use is on the evaluation side: this is a concrete case where the same guarantee (norm one) is delivered by
two parameterisations whose expressiveness differs only on a measure-zero-ish subset, and the reported task
differences between them are large enough (and uncontrolled enough) to be a caution against attributing
capability to the constraint's name rather than to the parameterisation's coverage.

## Tags

unit: weights
form: probe
method: weight-analysis
eval: downstream-task
goal: understanding
