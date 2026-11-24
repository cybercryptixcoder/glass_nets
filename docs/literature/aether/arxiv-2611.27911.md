# Functional Depth Is Learned: Isolating the Computation That Layers Produce Only Together

- **Key:** arxiv-2611.27911
- **Authors / organisation:** JaeSeong Kim, KyungTae Lim; Semyung University (Jecheon, Republic of Korea) and KAIST
- **Date:** 2026-11-20
- **Link:** https://arxiv.org/abs/2611.27911
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why
A transformer's depth is usually treated as an architectural fact: the model has as many chances for
sequential computation as it has blocks. The authors point out that having the opportunity is not the same
as using it — a trained stack could still be a sequence of nearly independent refinements to a residual
stream. They want a quantity that asks, for a given set of layers, whether anything is computed that
appears *only* when all of those layers act together, and then they want to watch that quantity form
across pretraining checkpoints of open models. The larger goal is to replace "number of layers" with a
learned measure of functional depth, with consequences for pruning, compression and how scaling is
reasoned about.

## What they studied
Pythia (410M / 1B / 1.4B / 2.8B / 6.9B, 8 checkpoints from step 0 to 64k), OLMo-1B (7 checkpoints from
4B to 3T tokens) and Amber-7B (8 checkpoints, up to 1.26T tokens), all evaluated on WikiText-103. The unit
of analysis is a *tuple of adjacent layers* — each order-k adjacent window of the depth. Inside such a tuple
the objects are whole-layer updates to the residual stream (each block treated as identity plus one vector
write), not neurons, heads or features.

## How they did it
Each block is written as a residual identity plus an update. For an ordered subset S of the layers, they
define a "subset forward pass" that applies only the blocks in S, in their original depth order, with all
other blocks replaced by the identity map. The *connected component* of a tuple T is then the alternating
(inclusion–exclusion / Möbius) sum of these subset passes over all subsets of T, with sign given by the
subset size. For two layers this reduces exactly to the change one layer's write causes in the later
layer's update; for k layers it removes every contribution attributable to a proper sub-tuple, so what is
left exists only when all k layers participate. Because the last layer's update can be factored out of the
alternating sum, the quantity is a combination of *that* layer's writes under different subset-conditioned
inputs, which gives a natural yardstick: they normalise by the magnitude of the last layer's actual write
when all previous participants are present, and call the result the write-relative connected component.
Cross-checkpoint development is summarised as a ratio of that normalised quantity at the final checkpoint
to the same quantity at the baseline checkpoint (initialisation where released, else the earliest one),
taken as a geometric mean over windows. Behavioural consequence is tested by intervention: during a full
forward pass they subtract the computed connected component from the residual stream immediately after
the last participating block and continue, and they compare against subtracting a random Gaussian
direction rescaled per token to exactly the same norm at the same site. A third analysis correlates the
pairwise (order-2) connected component with pairwise CKA between the same two layers, and then compares
block-removal decisions made by each criterion on post-removal perplexity.

## What counts as an explanation here
Not a description, a dictionary or a circuit. The explanation is a *scalar (per window, per order) with an
exact definition*: a vector-valued conditional-computation term, normalised, measured per tuple, plus a
training trajectory of it. It is complete in the algebraic sense — nothing is approximated or fitted, and
the alternating sum is exact for any k — but partial in scope: only adjacent windows are measured, so
coordination between distant layers is not covered, and the paper says so. It predicts one thing directly:
that deleting this component should hurt more than deleting an equal-norm random direction at the same
site, which is exactly what the intervention section tests.

## How they judged success
Three checks. (1) The coordination ratio, reported per model per order over all adjacent windows, with
95% bootstrap confidence intervals over windows for the trajectory figure. (2) Intervention damage
reported as post-intervention perplexity divided by clean perplexity, for component deletion versus the
norm-matched random control, in all 35 model×order cells; the norm matching is the control that makes the
comparison about direction rather than magnitude. (3) A ranking comparison: Spearman correlation between
pairwise CKA and pairwise connected component (model-level and per order), Spearman correlation between
each metric's block score and post-removal perplexity, and matched block-removal experiments where the
same number of consecutive layers is deleted from different positions. No downstream benchmark, no
fine-tuning, no human rating.

## What they found
Every one of the 35 model×order cells shows the ratio above 1 — the full table is printed. The weakest is
Pythia-1B at order 2 (1.62) and the strongest are Pythia-1.4B (60.8) and Pythia-6.9B (64.4) at order 6.
The baseline decays roughly by half per order (e.g. 0.30 / 0.11 / 0.05 / 0.03 / 0.02 for several models)
and the final checkpoint no longer decays, hovering around 0.5-0.8 at order 2 and rising with order in the
Pythia and OLMo families; OLMo-1B is the highest at order 2 (ratio 2.05). For Pythia-1.4B the ratios stay
near 1 at every order until roughly 2.1 billion tokens, after which the bootstrap bands over windows stop
containing 1 — coordination is organised in a specific phase rather than drifting with the weights. On
behaviour: deleting Pythia-1.4B's order-6 component raises perplexity 224× while the norm-matched random
direction raises it 19.8×; Pythia-6.9B shows 1472× versus 162× at order 5 and 1253× versus 898× at order 6;
OLMo-1B shows 218× versus 18.6× at order 2. Amber-7B is the narrowest case throughout (order 4: 1.1×
versus 1.04×; order 6: 1.8× versus 1.20×) but still sign-consistent in every cell. Pairwise CKA is
negatively correlated with pairwise connected component on average in all seven models, but the strength
varies widely: Pythia-410M averages about -0.52 to -0.71 across orders while Pythia-6.9B is as weak as
-0.19 at order 2, and Amber-7B's CKA-based block ranking correlation is near zero or positive (+0.18 at order 2, +0.01 at
order 3) even though the coordination criterion stays positive there (+0.57, +0.49). The block-removal table makes the divergence concrete on Amber at
order 2: CKA's "high similarity" block costs 55.7 perplexity and its "low similarity" block only 26.9,
i.e. the similarity heuristic points at the cheap block. The agreement is much better in Pythia, where
the low-coordination and high-CKA block perplexities agree within 8% in 10 of 15 cells.

## Limits and open problems they name
Only adjacent windows are measured, so distant-layer coordination is largely unmeasured; identity ablation
also creates counterfactual states inside each window, which is an assumption rather than a neutral
intervention. The normalisation is relative to the last participating layer's write and does not
separately normalise the preceding layers' writes, and the authors explicitly warn that it is *not* a
fraction of the update — the components are not orthogonal and the normalised value can exceed 1. Exact
computation needs subset forward passes, whose cost grows exponentially in k, so orders beyond 6 will
need approximation. Causality is explicitly left open: they cannot say whether coordination enables a
capability or is a signature that the capability has been learned, and they name two experiments that
would separate the directions (manipulating a task's sequential complexity, or tracking capability onset
against coordination onset jointly). The paper is 9 pages and the intervention table has no variance
reported per cell — the sign claim is stronger than any magnitude claim in it.

## What we take from it
- **Method (take):** a reusable, exact way to define "what only these layers do together" — subset forward
  passes plus the Möbius / alternating sum, with a normaliser drawn from the same algebra rather than
  chosen by hand. This is directly transferable to our units: heads, MLP blocks, SAE features, channels.
- **Framing (take):** "architectural depth is only *potential* depth; functional depth is learned" is the
  cleanest statement the folder has of why a predictive white-box model must report *training-time*
  structure and not only checkpoint geometry.
- **Measure (take):** the coordination ratio against a baseline checkpoint is the cheapest developmental
  statistic in the folder — one number per model per order, comparable across families, and it is a
  trajectory rather than a single-checkpoint score.
- **Control design (take, strongest):** the intervention's control is a *norm-matched random direction at
  the same site*. That removes magnitude as an explanation and asks only about direction. This is the
  control our own intervention experiments should use by default whenever we claim a direction matters.
- **Warning:** an "exact decomposition" is exact as algebra and still partial as measurement — the paper's
  own limitation is that only adjacent windows are sampled, so an exact estimand can be measured on a
  narrow subset of the tuples it is defined for. We should print the tuple family we enumerated, not just
  the fact that the decomposition is exact.
- **Warning:** a large intervention gap and a correct sign can coexist with an ambiguous effect — Pythia-6.9B
  order 6 is 1253× versus 898×, a factor 1.4 apart, in the same column where Pythia-1.4B shows 224× versus
  19.8×. The paper reports both without commentary. We must report the effect size and not only the count
  of sign-consistent cells.
- **Warning:** similarity of representations is negatively correlated with joint computation, and the
  correlation is weak enough in the strongest model (about -0.19) that a redundancy-pruning pipeline built
  on CKA would have pointed the wrong way on Amber — a concrete failure of a cheap proxy that the folder
  should cite whenever it uses representational similarity as a stand-in for redundancy.

## Tags
unit: layer
form: causal-abstraction
method: ablation
eval: ablation
goal: understanding
