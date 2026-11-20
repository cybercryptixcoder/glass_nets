# Efficient multivariate entropy estimation via k-nearest neighbour distances

- **Key:** arxiv-1606.00304
- **Authors / organisation:** Thomas B. Berrett (University of Cambridge), Richard J. Samworth (University of Cambridge), Ming Yuan (University of Wisconsin–Madison)
- **Date:** 2017-06-22 (v3; v1 2016-06-01)
- **Link:** https://arxiv.org/abs/1606.00304
- **Kind:** paper
- **Read from:** full text (main body pp. 1–~20 in the v3 HTML; the *appendices* live in a separate supplement document that this fetch did not contain)
- **Relevance to us:** 3 (central)

## What they wanted, and why
Differential entropy is the quantity that sits underneath every information-theoretic diagnostic an interpretability account reaches for — a probe's measure of how much of a layer's variance a feature dictionary leaves unexplained, an MI-based independence test's joint term, an SAE loss written as a KL. The classical estimator of choice is the nearest-neighbour estimator of Kozachenko and Leonenko, which is cheap and generalises to any dimension. The authors' starting observation is that this estimator is known to be asymptotically efficient only up to three dimensions; above that a bias term of order (k/n)^{2/d} does not vanish fast enough for the parametric n^{-1/2} rate, so in any realistic embedding dimension the estimator a practitioner actually uses is *not* the estimator the theory was proved for. They therefore want an estimator of the same family that stays efficient in arbitrary dimension, with a normal limit and hence honest confidence intervals, and they want the result to hold for densities with unbounded support — the regime that matters for embeddings, whose cloud is not a compact box.

## What they studied
Mathematical statistics, not networks: no model, no data set. The unit of analysis is a single distribution f on R^d and the functional H(f). The object manipulated is the vector of k-th-nearest-neighbour distances of each of n i.i.d. sample points, and the estimator is a weighted average of the log-distances over several k. Dimensions are treated as arbitrary natural numbers; the results are stated uniformly over a nonparametric class of densities with a moment bound, a sup-norm bound, and a smoothness condition that is deliberately relaxed where the density is small.

## How they did it
1. Define the Kozachenko–Leonenko estimator as the average over sample points of a log of the k-NN distance, normalised by the digamma function so that the plug-in approximation is exactly unbiased under a local uniformity assumption; with k = 1 this is the original 1987 estimator.
2. Observe through a Taylor expansion of entropy around the density that a bias survives in high dimension, and that at d ≥ 3 it can be written as an integral of the Laplacian of f divided by a power of f — a functional of curvature relative to density level.
3. Derive the exact bias expansion for fixed k and then for k growing, by conditioning on the point x and passing from the binomial distribution of "how many points fall in the ball" to its Poisson limit; the leading bias weight is a ratio of Gamma functions in k and n.
4. Build the weights that kill that family of terms: choose w supported only on j ∈ {floor(k/d), floor(2k/d), …, k} and impose moment conditions sum_j w_j Γ(j + 2l/d)/Γ(j) = 0 for l = 1..floor(d/4). This is the paper's core move — the bias terms at different k are made to cancel linearly instead of shrinking individually.
5. Prove a variance expansion: n times the variance converges to Var(log f(X)) — the efficient-influence-function variance — provided k diverges at least like log^5 n and not too fast.
6. Establish asymptotic normality with an explicit variance estimator, and turn it into a Wald interval whose width shrinks at the parametric rate.
7. Prove the matching lower bound in the local asymptotic minimax sense by constructing a local contamination of the density along the score direction log f + H(f) and bounding any estimator's worst-case risk over a finite set of alternatives.

## What counts as an explanation here
Not an explanation of a model — an *estimator with a proof*, plus an explicit asymptotic statement of what is and is not attainable. The form taken is a bias-corrected estimator with three coupled certificates: the expansion of the bias, the limit of the normalised variance, and a lower bound showing the limit cannot be beaten. It is deliberately not a full finite-sample theory in most regimes; the authors say where the expansions hold, give explicit examples of densities in the class (Gaussian, multivariate-t, and a compactly supported one vanishing at the boundary), and give a fallback condition for densities outside the class with a worked Gamma example. Predictive content: the expansions predict the actual bias of a k-NN entropy estimate as a function of d, k, n and the density's curvature, which is testable on real embedding clouds.

## How they judged success
Three criteria, in this order. (i) Efficiency: the estimator's mean squared error must converge to that of the oracle estimator that plugs the true density in. (ii) Local asymptotic minimax optimality against a lower bound (Theorem 8) — the comparison is not against a competing estimator but against the best possible estimator over local perturbations. (iii) Coverage of the confidence interval, uniformly over the density class and over the nominal level q. Uniformity in k is treated as a separate virtue and is exploited to permit data-driven choices of k: the conclusions survive as long as the selected k lies outside the extremes with vanishing probability. The fixed-k argument is a convergence-of-variance computation rather than a theorem, and the paper flags it as suggestive.

## What they found
- The unweighted estimator is asymptotically normal and efficient only for d ≤ 3; at d ≥ 4 the Laplacian bias term precludes efficiency whenever that integral is nonzero.
- With weights supported on multiples of approximately k/d and satisfying the moment cancellation conditions, the weighted estimator's normalised risk converges to Var(log f(X)) uniformly over the density class, for arbitrary dimension, requiring smoothness above d/2 and a moment above order d — provided k grows faster than a fifth power of log n and stays below a small positive power of n.
- The estimator is asymptotically normal with an estimator of the variance obtained by the same weighted trick applied to squared log-distances, giving (1 − q)-level intervals of width of order n^{-1/2} that are uniformly asymptotically correct.
- The choice k = ceiling(log^6 n) works without knowing the smoothness or moment parameters, which is the practical content: no tuning constants are required for the guarantee to apply.
- For fixed k, the authors derive the variance inflation explicitly and tabulate it: the inflation is about 2.14 for d = 1, k = 1 and falls steeply as k rises (roughly 0.97 at k = 2, 0.64 at k = 3, around 0.4 at k = 5) and also as d rises, and it is distribution-free. This is the concrete cost of using a data-driven fixed k, and it is the number a practitioner would want before choosing k by a heuristic.
- The confidence interval construction has a self-contained warning: the variance estimate is truncated at zero, so the interval degenerates exactly when the estimate is badly low.

## Limits and open problems they name
The proof of the variance lemma needs k to diverge and relies on a normal approximation to a multinomial, so the fixed-k regime has no theorem, only the limiting-variance computation. The authors note that under stronger conditions on k the same variance conclusion should follow from weaker density assumptions (essentially those of their fallback proposition) but that they do not pursue it. The smoothness required in high dimension is real, not nominal: they state their β > d/2 assumption exceeds what the compact-support two-stage estimator needs, and they explain that the non-differentiability of the entropy functional where the density vanishes is what forces it. All results are asymptotic; no finite-sample guarantee is offered, and there is no simulation study, so nothing here prices the estimator at the sample sizes an interpretability study actually has.

## What we take from it
- **A measure, and the reason our informal entropies are biased.** If we report entropy of a residual cloud in high dimension, the bias is not noise, it is a first-order functional of the density's curvature and it does not vanish at n^{-1/2}. The paper's bias formula names exactly which quantity is being misestimated, and the companion entry on the MI independence test (arxiv-1711.06642) uses the unweighted estimator — so that entry's statistic inherits this bias and its size, and the fix here (weighted k with moment cancellation) is transferable to it.
- **A method for the bias problem generally.** Cancelling a family of error terms by imposing moment conditions on a weight vector is not special to entropy; it is the same move any white-box account needs when it summarises a layer by a low-order statistic and wants the summary to be first-order-unbiased. Say which moment conditions a summary satisfies, or say the summary has a known bias of a known order.
- **A warning about validation-by-consistency.** The whole paper is about an estimator that is consistent at every d but efficient only below 4. Consistency is what most probing papers report; this paper is the folder's clearest demonstration that consistency can coexist with a bias that dominates the error budget. Whenever a claim rests on "our estimator converges", read the rate.
- **A usable number.** The fixed-k variance inflation table (2.14 at k = 1, d = 1, down to ~0.4 by k = 5) is a ready-made correction factor when a k-NN entropy or MI estimate is computed with a small fixed k, which is the common default.

## Tags
unit: feature
form: description
method: other
eval: benchmark
goal: evaluation-of-methods
