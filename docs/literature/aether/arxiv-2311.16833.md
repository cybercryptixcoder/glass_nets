# 1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness

- **Key:** arxiv-2311.16833
- **Authors / organisation:** Bernd Prach, Fabio Brau, Giorgio Buttazzo, Christoph H. Lampert; ISTA (Klosterneuburg, Austria) and Scuola Superiore Sant'Anna (Pisa, Italy)
- **Date:** 2023-11
- **Link:** https://arxiv.org/abs/2311.16833
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Seven published ways of building a convolution whose Lipschitz constant is at most one are all
claimed to work, and the field reports them on incomparable footing: different architectures,
different training lengths, different hardware. The authors' complaint is that a reader cannot tell
from the literature which method is actually better, because the papers each enlarge the model or the
training budget until their own number looks good. The paper is a re-implementation and controlled
comparison of those layer constructions under a common architecture, a common loss, a common
optimiser, and - the design decision that matters - a **common wall-clock training budget** instead
of a common epoch count. The larger goal is a purchasing decision rather than a scientific one: given
a GPU with a fixed memory ceiling and a fixed number of hours, which layer family buys the most
certified robustness, and where does the time and memory actually go.

## What they studied

Seven convolutional 1-Lipschitz constructions - AOL, BCOP, Cayley, CPL, LOT, SLL, SOC - each wrapped
in one fixed architecture (a VGG-style net that doubles channels on every resolution halving, with
MaxMin activations, no skip connections, no batch normalisation) and swept over four widths from
roughly 1.5M to 100M parameters. Three datasets (CIFAR-10, CIFAR-100, 200-class Tiny ImageNet at
64x64), three training budgets (2h, 10h, 24h on one A100), and four metrics per cell: training time,
inference throughput, measured peak GPU memory, and certified robust accuracy at radius 36/255. The
unit of analysis is the **layer construction**, not the trained function - the paper is about which
parameterisation of a convolution to use, and it explicitly excludes methods that only penalise the
Lipschitz constant in the loss (Parseval, orthogonal regularisation, spectral normalisation) because
those come with no guarantee. Four more methods are discussed in an appendix and deliberately
excluded, three of them for defects the authors found while re-implementing them.

## How they did it

The paper has two halves that are then checked against each other.

The theoretical half writes every layer as two pieces - a **transformation of the input** and a
**transformation of the parameters** that enforces the constraint - and prices each in
multiply-accumulate operations and in memory, using three symbols: C for one standard convolution,
M for the number of input elements, P for one kernel's worth of parameters. The resulting table is
the organising device of the whole paper, and the ASYMMETRY it exposes is what the experiments then
test: input-side costs are broadly similar across methods, so inference should be similar (and it
is, except where a method applies more than one convolution), while parameter-side costs differ by
whole orders - some methods rescale a P-sized object, others perform matrix orthogonalisations that
scale cubically in channels, and the Fourier-domain methods (Cayley, LOT) additionally carry terms
proportional to the squared spatial size. At inference, parameter transformations can be cached, so
only the input side is paid per sample; at training, both sides are paid on every forward pass and
the parameter-side intermediates must be held for the backward pass.

The empirical half then trains every method to the same *time*, not the same number of epochs,
having first measured each method's epoch time and converted the budget into an epoch count. A
random hyper-parameter search (16 runs per setting, learning rate and weight decay drawn log-uniformly,
selected on validation certified accuracy) is run for each method and dataset before the reported
runs. Certified accuracy is computed by the standard margin test for this family: an input counts as
certified at radius epsilon if it is classified correctly and the gap between the top two logits
exceeds epsilon times root two. Because a fixed budget would otherwise let wider models win by simply
being wider, the fixed-time design deliberately lets bigger models see fewer epochs, and this shows up
as the middle width usually beating the largest.

A third, smaller piece is a **post-hoc audit of the trained models**: two cheap checks that the
constraint actually holds. One is that the variance of a validation batch across its examples can
never increase from layer to layer in a 1-Lipschitz network - an argument the authors prove in an
appendix from the fact that a batch mean minimises the summed squared distance - so any increase
flags a layer that is not what it claims to be. The other is a convolutional power iteration applied
to every trained layer, with the per-layer results multiplied. These two checks are what catch the
excluded methods, and what catches implementation bugs in the included ones.

## What counts as an explanation here

Not an explanation of a model's behaviour - the paper is engineering economics. What it produces is a
**cost model plus a leaderboard**: an analytic expression for the time and memory of each
construction, verified against measurement, and a ranking of methods per metric with a written
selection rule at the end (CPL by default; SOC when compute is plentiful and latency is not
binding; AOL or BCOP when inference latency dominates). The explanatory content is the causal chain
from a complexity term to a measured wall-clock behaviour - for example, why SOC's five convolutions
cost only about three times a standard convolution while its input-side operation count suggests
five, or why Cayley and CPL end up with the same measured inference throughput despite CPL doing
twice the convolutions (the answer offered is that the standard implementation uses a fast
convolution algorithm which a plain FFT does not get). Fully predictive at the level of resource use,
and not at all predictive at the level of accuracy.

## How they judged success

Throughput (examples per second, averaged over 100 mini-batches, forward plus backward plus update
for training and forward only, with parameter transforms cached, for inference); peak allocated GPU
memory read off the framework's own counter; clean and certified robust accuracy at a single radius
of 36/255, on three datasets and four widths, at three budgets, with the full per-cell tables
appearing in an appendix; and a composite ranking built by counting top-1, top-3 and top-10 finishes
across datasets, which is plotted as a five-point radar chart. Baselines are the other six methods
plus, separately, a table of state-of-the-art numbers quoted from the literature at much larger sizes
and budgets (up to 236M parameters) - printed specifically to show that those numbers are not
comparable to the controlled setting. The theoretical half is judged by whether measured time and
memory scale the way the table says they should.

## What they found

SOC takes the most top-1 certified finishes (twice) and six top-3 finishes; CPL is the most stable,
twice top-3 and nine top-10 finishes; LOT wins on Tiny ImageNet; Cayley is consistently last on
robustness though competitive on clean accuracy; SLL is good on CIFAR-10 and poor on the two
many-class datasets. At 24h on CIFAR-10 the best certified accuracies cluster around 65-66% for SOC
and CPL against 61-62% for AOL, BCOP, Cayley and SLL. The resource picture is much less ambiguous than
the accuracy picture: at inference, AOL and BCOP add essentially nothing over a standard
convolution, CPL and SLL pay two convolutions, SOC pays roughly three times, and Cayley and LOT are
competitive through the FFT path with Cayley about twice as fast as LOT. At training, Cayley, LOT and
BCOP are far slower per epoch, and LOT's large model does not fit in 80GB of memory at all - the
authors had to cut its inner iterations to two to train it all, and then threw those results away as
unrepresentative.

The audit results are the sharpest numbers in the paper. Variance growth plus power iteration show
that one method's released implementation (ECO) **does not produce 1-Lipschitz layers** with this
architecture, and it is excluded on that evidence rather than on performance grounds. The Sandwich
layer is excluded for a vanishing-gradient pathology that the authors quantify layer by layer, from
around 1e-7 at the layer nearest the input to around 1e-1 at the layer nearest the output. And two of
the seven included methods are found to be numerically fragile in ways their papers do not mention:
CPL produces NaNs in long trainings and has no stated initialisation (the authors borrow SLL's), and
SLL's released code divides by quantities that can approach zero and works only for 3x3 kernels. In
the per-cell tables the failures are visible as degenerate rows - a method that reaches 1% accuracy
at a 24h budget because the run diverged.

## Limits and open problems they name

The comparison is scoped hard: one architecture, no residual connections, no batch normalisation,
one radius, image classification only, and a single GPU generation. The fixed-time design is fair
across methods but it couples size to epochs, so the paper cannot say how each method would behave
with an unlimited budget - and it says so. The exclusion list is itself a limitation, since four
methods are dropped for reasons ranging from "no guarantee" to "does not fit in memory", which means
the leaderboard is over the survivors. Several of the reported differences are within a point or two
and the ranking is built by counting finishes rather than by a significance test. The appendix notes
that the parameterisation of at least one method contains a redundant degree of freedom (a learnable
diagonal matrix whose effect can be absorbed into the bias), which the authors point out but do not
exploit, and that one method's scaling matrix can be reparameterised away entirely - the paper leaves
these implementation-level issues open.

## What we take from it

Three things, all method rather than result. First, a **budget-matched comparison protocol**: when
comparing designs whose per-unit cost differs by orders of magnitude, fix the resource (time, memory,
flops) and let the amount of training or the size float, because fixing epochs silently rewards the
cheapest-per-epoch design and fixing size silently rewards the most expensive one. This is directly
transferable to how we compare our own analysis methods against baselines - our cost per explanation
is not the baselines' cost, so the fair axis is a shared budget and the reported quantity is the best
result reachable inside it. Second, a **derived invariant used as a cheap monitor**: the
non-increasing batch-variance property is a two-line proof that turns an expensive per-layer
post-training check into a quantity you can log during training, and it immediately caught a method
whose constraint did not hold. Any guarantee we publish whose premises can be checked by a cheap
monotone statistic should ship with that statistic. Third, the honest and instructive finding that
several published implementations do not deliver the property their paper proves - one does not
produce the layers at all, two are numerically fragile, and the fragility shows up as a single
degenerate row in a large results table - which means that when we cite a guarantee from the
literature, the object to check is the code, not the theorem.

## Tags

unit: layer | whole-model
form: cost model; leaderboard; per-layer certificate (1-Lipschitz by construction)
method: Lipschitz; spectral-norm bound; orthogonal parameterisation; weight-analysis; benchmark; ablation
eval: certified rate; benchmark; downstream-task; throughput and peak memory as primary metrics; post-hoc layer audit (batch-variance monotonicity, power iteration)
goal: evaluation-of-methods; safety/auditing; control/editing (architecture choice under a resource budget)
