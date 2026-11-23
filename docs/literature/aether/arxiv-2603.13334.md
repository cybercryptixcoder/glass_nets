# Lipschitz-Based Robustness Certification Under Floating-Point Execution

- **Key:** arxiv-2603.13334
- **Authors / organisation:** Toby Murray; University of Melbourne, Australia
- **Date:** 2026-11
- **Link:** https://arxiv.org/abs/2603.13334
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Lipschitz-based certification is the lightest form of robustness checking available: compute a
margin Lipschitz constant once per model, then certify an input by one comparison against
epsilon. It is millisecond-cheap and small enough that its implementation has been formally
verified. The gap the author targets is that this whole line, like most of the verification field,
reasons about a network that executes in exact real arithmetic, while deployed networks execute
in IEEE floating point. The paper asks whether the *conservatism* of the Lipschitz bound - the fact
that it over-estimates sensitivity and therefore under-estimates robustness - is large enough to
absorb rounding error, so that a real-arithmetic certifier remains trustworthy in practice. The
larger goal is a certificate that is still worth having after the arithmetic gap is closed: sound
against the execution the model actually performs, but keeping the per-input cost that made the
method attractive.

## What they studied

Dense feed-forward ReLU classifiers with identity output activations, run in float32 (the
deployment format), float64 and float16. Six models: three image classifiers taken from a prior
verified-certifier paper (MNIST, Fashion MNIST, CIFAR-10 with inputs of 784 or 3072 dimensions
and 8-12 hidden layers of 128-512 units), a tabular high-energy-physics binary task (HIGGS, 28
features) trained as a *width sweep* at 128/256/512/1024, and two many-class handwritten-character
tasks (EMNIST ByClass, 62 classes, and Balanced, 47 classes) all trained with globally robust
objectives. The unit of analysis is the whole model's input-to-margin sensitivity, plus a
per-layer quantity that is new to this folder: the deviation between the floating-point and
real-arithmetic activation at each layer.

## How they did it

Three steps, in order.

First, the paper attacks the assumption. It searches for input pairs where a formally verified
real-arithmetic certifier says an input is robust but the floating-point network disagrees. A
continuous search crosses the decision boundary with an adversarial attack and bisects back
toward the test image; a second, "grid", search runs a projected-gradient attack restricted to
8-bit pixel grids inside the standard epsilon-ball of certified test images. Every returned triple
is re-checked independently in exact rational arithmetic. Then a stronger test: since biases do
not affect Lipschitz constants, the author adds a large constant bias to every unit of the last
hidden layer and cancels it with a compensating bias at the output. In real arithmetic the model
is unchanged, so every certificate and every constant is unchanged; in float32 the inflated
activations make the output layer's rounding error huge relative to the residual signal, and the
same certifier then issues false certificates on ordinary test images.

Second, the theory. The floating-point execution is not a Lipschitz function (rounding can move
the output by a fixed amount at arbitrarily small input distance), so the method does not try to
bound it that way. It keeps the real-arithmetic margin Lipschitz constant, and *separately* bounds
how far floating-point outputs can depart from real outputs - uniformly over the whole
epsilon-ball, not just at the centre. That deviation is carried by a two-coefficient linear
recursion through the layers: an amplification factor (the layer's spectral norm plus a
rounding correction built from the spectral norm of the entrywise-absolute-value matrix, since
the perturbation acts on the input vector's magnitude) and a fresh-error term proportional to how
large the activations were when they were rounded. The recursion is interleaved with a separate
check that no layer overflows, expressed as two scalar inequalities per layer on the sum of
absolute products and the largest single product; the interleaving is not circular because the
input is exact and so the base case needs no deviation bound.

Third, two reusable tricks to make this practical. Pre-deployment, a single float64 forward pass
*measures* the last hidden layer's deviation from the high-precision execution, and the measured
value plus that execution's own (negligible) deviation bound replaces the worst-case number;
the same pass supplies tighter per-layer activation radii, each being the measured norm plus
the ball's reach under the layer's Lipschitz product plus the measurement's error. And for the
spectral norms themselves, the dominant computation (Gram iteration, which raises the matrix to
successive Gram products and recovers the norm by square roots) is run in float64 with its
rounding error propagated as a term in the bound, so the result is a rational upper bound
obtained orders of magnitude faster than the exact-rational version. Everything is mechanised in
a proof assistant on top of a floating-point linear-algebra library; the executable certifier is
separate, unverified Python, and the paper states that trust boundary explicitly, noting that
generative AI was used throughout and audited by the author.

## What counts as an explanation here

A certificate, not an explanation: a per-input decision that the floating-point execution
classifies the whole epsilon-ball identically, obtained by comparing an observed floating-point
margin against the sum of the Lipschitz term and a floating-point margin-degradation term. It is
sound and partial - it can only ever say "safe", never why, and never "unsafe". The interesting
object for us is the intermediate one: the per-layer deviation bound, a scalar that measures the
semantic gap between two arithmetics of the same weights, computed compositionally from the
weights and the format constants alone. It predicts something about the model that no earlier
paper in this folder predicts: how much of its behaviour survives being run in the precision it
was deployed in. The paper is also careful about a distinction the rest of the thread tends to
blur - the theorem is real-arithmetic, the check is floating-point, and each is charged
separately.

## How they judged success

Three questions. (1) Does the certifier reject every certificate the counterexample search
refutes? (2) What does floating-point soundness cost, measured two ways on each model's *full*
test set: certified rate (the fraction of points certified robust, which isolates the
certifier's conservatism from accuracy) and verified robust accuracy (certified and correct),
with the cost defined as the drop from the real-arithmetic ceiling; plus a conservatism ratio -
mean margin degradation divided by mean real-arithmetic certification margin - indexed against
the input-radius-to-epsilon ratio. (3) How does it compare with the only other public
floating-point-sound verifier of this scale, on the same rows? That baseline is a per-neuron
relaxation tool, converted from l2 to l-infinity radius so that its box lies inside the paper's
ball, and run in a deliberately generous configuration in which its products are charged at the
target format's roundoff bound but its own additions are not charged at all - under-charging its
rounding in the baseline's favour. Three certification modes are reported side by side
(worst-case, measured centre, measured radii), which makes the reclaim from measurement an
ablation rather than a claim.

## What they found

- Conservatism does **not** absorb rounding. Counterexamples exist for all three naturally trained
  float32 models; their radii are minuscule (order 1e-8 to 1e-6), many orders of magnitude below
  any perturbation of interest, and they scale with the format (float16 gives radii around 1e-4
  to 1e-2, float64 around 1e-16 to 1e-15).
- On naturally trained models at the standard radius, the grid search finds nothing - a
  real-arithmetic certifier is sound there in practice, but only because the model's rounding
  errors happen to be tiny, which is a property of the arithmetic the certifier never inspects.
- Under the bias trick, false certificates appear on ordinary test images at standard radii. The
  MNIST variant keeps 98.33% accuracy (against 98.40%), the certifier still certifies 9,549 of
  10,000 test images, and 24 of those (0.25%) are refuted; Fashion MNIST refutes 204 (2.5%) and
  CIFAR-10 370 (7.9%). Because the search is incomplete these are lower bounds. Both images in
  every refuted pair lie on the 8-bit grid, so restricting certification to grid inputs would not
  help.
- Cost of floating-point soundness is small on five of six models. The MNIST certified rate drops
  from 95.74% (real-arithmetic ceiling) to 95.50% worst-case, recovering to 95.65% with measured
  radii; Fashion MNIST 83.65 -> 82.16 -> 82.98; EMNIST-ByClass and Balanced follow the same
  shape. CIFAR-10 is the exception: 46.12 -> 18.90, recovered only to 32.95.
- The CIFAR-10 cost is governed by the task, not by model scale: EMNIST-ByClass reuses CIFAR-10's
  architecture and costs 1.16 points. The conservatism ratio tracks input-radius over epsilon
  (0.08 at ratio 35, 0.17 at 52, 1.09 at 207); CIFAR-10's high-dimensional input gives a much
  larger input norm, and even before floating point enters, attacks refute at least 33.4% of its
  test points while 20.5% are neither certified nor refuted - evidence that the *global* Lipschitz
  bound itself, not the rounding, is the binding slack there.
- Measurement reclaims most of the conservatism where there is any: measured radii closes over
  half the gap on CIFAR-10 and EMNIST-Balanced and four fifths on HIGGS. On the models where the
  worst-case mode is already near-tight there is nearly nothing to reclaim.
- Speed: the one-off norm computation is seconds to a few minutes per model (19.5 s for CIFAR-10,
  against tens of hours in exact rationals, matching the exact bounds to within 3.5e-11 relative
  per layer); certification is a few milliseconds per instance in every mode and nearly
  insensitive to class count, so 62-class EMNIST-ByClass is faster per instance than 2-class
  HIGGS-1024. Over the HIGGS width sweep, the worst-case cost climbs from 0.42 to 5.93 points and
  measured radii holds it to at most 1.4 points; the one-off norm cost grows about 4x per
  doubling of width (2.8 s to 192 s).
- Against the per-neuron baseline at float32: the measured-radii mode retains 98.3% or more of its
  own real-arithmetic verified robust accuracy on four models and 76.6% on CIFAR-10, while the
  baseline retains between 1.3% and 70.0% on four of five; and the paper is 11-26x faster per
  instance on a laptop CPU against the baseline on a GPU, even after amortising the norm
  precomputation. On CIFAR-10 the baseline's tighter abstraction verifies more in real arithmetic
  (41.26% against 35.95%) but loses 38.3 points once rounding is charged, at which point the order
  reverses. At float64 neither tool pays anything; at float16 all are vacuous.

## Limits and open problems they name

The theory needs activations that are pointwise, zero-preserving, Lipschitz and *exactly*
computed in floating point - which excludes sigmoid and tanh. It needs no overflow, obtained via
sufficient conditions that a failing check refuses to certify. The certifier refuses any model
whose layer width times unit roundoff reaches 1, which excludes CIFAR-10 at float16. Batch
normalisation, residual connections and rounding activations each introduce error at a point the
theory does not currently charge. The flush-to-zero variant (which real TensorFlow enables at
float32 and float64 with no supported way to disable, found by validating the inference path
against the standard model) is worked out on paper - only absolute-error constants change - but
not mechanised. The theorem instances a *global* constant; local constants would suffice and
would be tighter, but are recomputed per input. The certifier is not verified itself, only
validated against the reference outputs of the verified real-arithmetic kernel. Everything is l2
only; an l-infinity certificate would need operator norms in place of spectral norms. Fusion,
hardware-specific kernels and reduced-precision accelerators remain outside any assumption.

## What we take from it

1. **A validation design worth copying verbatim: attack the verifier's own assumption rather than
   its accuracy.** The bias trick is the strongest methodological move in this paper and it
   transfers with no modification: if a claim is invariant to some parameter by construction,
   move that parameter to an extreme and see whether the claim still holds. Biases do not enter
   the Lipschitz constant, so they are the free degree of freedom an adversary may use without
   touching the certificate's own arithmetic. For our account, whose "constants" are all computed
   from weights, we should name which weight transformations are invisible to our account and
   check whether behaviour is in fact invariant to them.
2. **A warning about the domain of every claim: real arithmetic is a different network.** The
   paper's framing applies directly to us - a white-box account computed in exact arithmetic
   describes a network nobody deploys, and the gap is a computable function of *how big the
   activations got*, not of anything semantic. The per-layer deviation recursion is the cheapest
   possible instrument for this and is our recommended default: one scalar per layer, computed
   forward from the weights and the format, no search, no reference set. It answers "how much of
   this account survives being run in float32" without running anything.
3. **The most reusable number here is not the certificate, it is the lived-versus-worst-case gap
   measured twice.** The measured-radii mode converts a worst-case radius into an observed one
   plus the reach of the ball plus the measurement error - a three-term bracket where the middle
   term is the only part that scales with epsilon. Run 648 and 649 both demanded that analytic
   constants be printed beside measured ones; this paper is the first in the folder to print
   *both* and to show the ratio moving with scale (the HIGGS width sweep: worst case 0.42 -> 5.93
   points, measured at most 1.4, one-off cost 4x per doubling). It shows the two curves have
   different shapes, which is exactly what the earlier rule was designed to expose.
4. **An optimistic estimator inside a guarantee, done right - and it is the mirror image of the
   truncated power iteration.** Run 648 caught a power method that could only under-estimate a
   norm, used inside a constraint, leaving the constraint non-binding and the bound wrong in the
   optimistic direction. This paper's Gram iteration is the opposite direction by construction: it
   can only *over*-estimate, and when it is run in cheap arithmetic the truncation and product
   errors are added back into the bound as terms. So the estimator is permitted to be fast because
   its error direction is known and is charged. That is the general pattern: an approximate
   estimator is legitimate inside a guarantee when its error is bounded *in the safe direction and
   paid for*; the failure in run 648 was not the approximation but the unpaid, wrong-way error.
5. **A cost model to keep beside run 649's.** There, the *training* line's Lipschitz constant was
   parameter-space and global to the model; the verification line's is input-space and computed
   per model. Here both appear as different objects in the same proof: the sensitivity bound is
   per-model (computed once, amortised over millions of inputs) and the deviation bound is per
   layer and per format. The amortisation argument - a constant is affordable when it is a
   function of the model rather than of the query - is the same one run 644 made for a reduced
   controller, appearing for the fourth time in the folder.
6. **The honest boundary statement is the model here.** The paper separates what is proved
   (mechanised theorems), what is validated (an unverified implementation reproducing a verified
   reference), what is measured (the pre-deployment readings), and what is assumed (no fused
   kernels, no sigmoid). Our account should be presented in exactly those four registers, and the
   refusal conditions of the certifier (width times roundoff at 1, non-finite intermediates)
   belong in the output, not in a footnote.

## Tags

unit: whole-model | layer
form: certificate; per-layer deviation bound; upper-bounded spectral norms (a numeric object rather than a description)
method: Lipschitz; spectral-norm bound; formal mechanisation; error analysis; ablation
eval: certified rate; verified robust accuracy; counterexample search; benchmark; ablation (worst-case vs measured)
goal: safety/auditing; understanding; evaluation-of-methods
