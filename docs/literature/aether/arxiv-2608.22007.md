# The Communication Map of a Transformer

- **Key:** arxiv-2608.22007
- **Authors / organisation:** Richard Zhe Wang; St. John Fisher University
- **Date:** 2026-08-22 (v1); v3 2026-11-07
- **Link:** https://arxiv.org/abs/2608.22007
- **Kind:** paper
- **Read from:** full text (v3 HTML, including Appendices A-E)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Mechanistic interpretability has mapped one circuit at a time, starting from a known
behaviour and working backwards to a few components. The author asks the opposite
question: for a whole transformer, who can talk to whom, over which subspaces, and how
strongly, read off the weights alone with no behaviour chosen in advance. The premise is
that the residual stream is a shared medium of many near-orthogonal channels, so the
weight matrices already contain a complete directory of plausible connections; the
paper's job is to compile that directory and see whether known circuits fall out of it.
The stated motivation is that questions such as how far communication reaches across
layers, whether heads form communities, and whether pairs avoid as well as couple cannot
be posed one circuit at a time.

## What they studied

Seven language models in three families: GPT-2 small, medium and large (Radford et al.),
GPT-Neo-125M, and Pythia-160m, 2.8B and 6.9B. The unit of analysis is the writer-reader
pair, i.e. every causally eligible ordered pair of components: attention heads (through
three composition channels), individual MLP neurons as rank-one readers and writers, and
three interface matrices (token embedding, positional embedding, unembedding). The paper
counts 18 connection classes, 6.3 x 10^8 candidate pairs for GPT-2 small and 1.3 x 10^11
for Pythia-6.9B.

## How they did it

Each pair gets a coupling coefficient: the Frobenius norm of the composed read-write
product, normalised by the product of the two norms, which reduces to the Elhage et al.
composition score for head pairs and to |cos| for rank-one neuron pairs. The paper proves
that this quantity is rotation invariant, that it collapses to a weighted average of
squared cosines between the reader's principal input directions and the writer's
principal output directions, and that its expectation under a Haar rotation of either
subspace is exactly 1/d for any singular values, with an exact rank-one null law
(Beta(1/2,(d-1)/2)) and closed-form variance. Every pair is standardised by that rotation
null to a z-score. Because multiplying d x d matrices per pair would be infeasible, exact
factored forms reduce every class to matrices of head width; the census then runs in 15
seconds for GPT-2 small and 11 minutes for Pythia-6.9B on one consumer GPU. For edge
selection the paper does not use the theoretical null (47% of GPT-2 head pairs clear
z >= 2 against it), but an empirical null fitted per stratum, where a stratum is one
connection class at one writer-to-reader layer separation, using median and MAD, with
Benjamini-Hochberg FDR control at q = 0.05. Heads are then partitioned by Louvain, and
the communities are validated by mean-ablating them against five random sets matched in
size and per-layer head counts. The application builds a "reader-writer PCA": pool the
trace-normalised Gram matrices of every head factor and interface matrix into one
matrix, eigendecompose it, take the ten leading bands, then pick as the induction-critical
pair the two bands with the highest ratio of positional to token coupling, and delete
that two-dimensional plane from the stream at every layer input and before the
unembedding.

## What counts as an explanation here

A graph, and a quantitative one: every potential channel in the model carries a score
with a calibrated chance level, plus a null distribution per class, plus a standardised
z. The map is explicitly a map of permitted channels rather than of traffic, and its
validation is a behavioural ablation, so what it claims to explain is the wiring that the
weights allow, not what any prompt exercises. The paper is careful that "potential
connectivity" and "communication on an input" are different objects.

## How they judged success

Four criteria. (i) Calibration: the census of a randomly initialised model must sit at
chance; the exact chance law must predict neuron-pair exceedance counts. (ii) Recovery
without supervision: known circuits (five behavioural induction heads, IOI name-movers)
must fall into the graph blind, and the previous-token head must top the K-composition
writer ranking into every induction head. (iii) Instability across seeds: the Louvain
partition is re-run over ten seeds and reported with modularity range and adjusted Rand
index. (iv) Ablation: an "induction gain" readout, the loss on a repeated second block
minus the loss on the first, averaged over 32 prompts of 128 random tokens plus an exact
copy, always with five random control sets matched in size and per-layer counts, and a
Pile NLL readout as a check on non-specific damage. The application is judged by the
fraction of the clean induction gain destroyed, by the Pile loss delta, by principal
cosines against three competing subspaces, and by a dose check.

## What they found

- Census: averaged over the three head channels, 70-89% of head pairs in each model sit
  beyond two standard deviations from chance, with both super-coupled and avoidant
  populations. In GPT-2 small, 55% of K-composition pairs are super-coupled against 34%
  avoidant, while V-composition inverts the pattern (25% super, 65% avoidant) in every
  model mapped.
- Layer-resolved map (GPT-2): layer-0 heads avoid heads of layers 1-7 and their own
  layer's neurons, but couple to heads of layers 10-11 and the unembedding; the
  positional matrix is read by heads of layers 1-2 and avoided by everyone else; neurons
  read the embedding more strongly with depth (8% net super-coupling in layer 0 against
  88% in layer 11).
- Neuron wires: neuron-to-neuron pairs whose |cos| >= 0.5 number 2,066 in GPT-2 despite a
  chance expectation of about 4 x 10^-41 for a single pair at that level; 140,927 pairs
  exceed |cos| = 0.2 where chance predicts about 14. Wired pairs co-activate on natural
  text at median |corr| 0.31 against 0.03 for random pairs.
- Training trajectory (Pythia-160m and 2.8B): at initialisation 2-3% of pairs clear
  |z| = 2. Super-coupling is already present at step 1,000 (61% K, 73% Q for 160m; 79-84%
  for 2.8B), with none below chance; avoidance appears between steps 1,000 and 4,000 and
  grows to the end of training (V-composition avoidant share 0% -> 63% in 160m).
- Head graph: 1,051 of 28,512 candidate GPT-2 head edges selected at FDR 5%; 133 heads in
  the giant component; five Louvain communities (sizes 38, 30, 25, 21, 19; modularity
  0.283; adjusted Rand index 0.84 average across ten seeds). Communities span depth, no
  community spans fewer than eight of twelve layers. Median writer-to-reader separation
  is five layers and the linked rate rises with distance, from about 6% at one to three
  layers to 11-14% at seven or more.
- Blind recovery: the 30-head community holds all five induction heads and two of three
  IOI name-movers in ten of ten seeds; previous-token head L4H11 tops the K-composition
  writer ranking into every induction head.
- Community ablation (GPT-2): freezing the community's attention-weighted value vectors
  at their corpus mean destroys 94.7% of induction gain against a matched-control median
  of 1.9% (max 4.9%), at +0.51 nats Pile. The five behavioural induction heads alone
  account for 60.8%. The nested decomposition is strongly non-additive: 60.8% plus 1.9%
  (previous-token head) plus 13.5% (remaining 24 heads) sums far below 94.7%, which the
  paper reads as redundancy inside a boundary drawn from weights alone.
- Cross-model: whole-community ablation destroys 86.2-108.5% of induction gain in five
  further models, but the matched layer controls are themselves elevated where the
  community is large (20.3% at GPT-2-large, 90.5% at Pythia-2.8B), and GPT-Neo's random
  controls reach 97.1% layer-matched, which the paper reports as a caveat rather than as
  evidence.
- Neuron writers into induction heads: no small set is critical (top 100 destroys under
  1%); the top 1,000 neurons (2.7% of neurons) destroy 19.5% on GPT-2 against 0.1% for
  matched random sets (max 0.9%), and 37.1% on Pythia-160m against 5.1%, at a Pile cost
  of +1.42 nats. Induction heads are among the least neuron-fed heads in their layer.
- Application: deleting the two-dimensional reader-writer PCA plane destroys 82-96% of
  induction gain on six models from 124M to 6.9B. At Pythia-6.9B, 2 of 4,096 dimensions
  (0.05% of width) remove 89.5%, against 84.5% for activation PCA and 36.5% for outlier
  dimensions; at Pythia-160m, 95.8% against 42.0% and 35.4%. Random two-dimensional
  deletions destroy 0-3%. Eigenvalue-only selection on Pythia-160m reaches 46%, so the
  positional-to-token ratio does the work, and the selected pair is identical from 4 to
  128 estimation sequences at three seeds.

## Limits and open problems they name

Five stated: the map is potential connectivity, not measured traffic on any input;
validation and application both rest on one capability (in-context copying) and the other
communities are unexamined; interventions are head-level except one neuron ablation, so
the neuron-level structure is largely untested; experiments stop below 10B parameters; and
nodes are heads and neurons rather than learned features, with a feature-level map left to
future work. Additional limitations visible in the tables: Pythia-6.9B could not be
ablated for a 55 GB vs 45 GB memory reason, so its map is validated only through the
subspace application; GPT-Neo is excluded from the subspace comparison because random
deletion already removes most of its induction gain; the robust-null choice is load
bearing, since refitting with mean and SD removes 61% of the map's head edges; and the
"wire" threshold 0.5 for neuron pairs is a stated convention while the chance-facing
numbers are at 0.2 and 0.23.

## What we take from it

A method and a warning. The method is the cheapest whole-model object in this folder: an
exact, non-sampled null per connection class, standardised scores computed from weights
only, in minutes, with no dictionary trained and no forward pass on a target corpus. For
our white-box model this is the natural *prior graph* over the network, computed before
any activation is seen, against which any activation-level reading can be compared: a
channel our model claims is used but that sits at z ~ 0 in the map is a claim that needs
explaining, and a channel the map scores highly but that no attribution ever touches is a
candidate for the "available but unused" category this folder keeps finding. The second
gift is the empirical-null machinery: strata by class and layer separation, robust centre
and spread, BH control. Every ranking our model emits over millions of candidate pairs
needs exactly that, because the paper measures that the theoretical null selects 47% of
pairs and a non-robust refit lets signal inflate its own null by up to 5x and deletes
61% of the graph. The warning is the community result: an ablation of a set that
overshoots 100% of the effect, with controls that reach 90%, is what a set chosen by
graph clustering on a fragile stream looks like, and the paper says so. Also useful: the
"avoidant" half of the census is a real measurable object that most of this literature
ignores, and it is learned later than coupling, which gives the availability-versus-use
thread a developmental signature to look for.

## Tags

unit: circuit, head, neuron, weights, whole-model
form: graph, description
method: weight-analysis, decomposition, ablation, attribution
eval: ablation, prediction/simulation, faithfulness, downstream-task
goal: understanding, evaluation-of-methods
