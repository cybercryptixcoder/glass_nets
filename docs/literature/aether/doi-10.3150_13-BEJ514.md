# A consistent test of independence based on a sign covariance related to Kendall's tau

- **Key:** doi-10.3150_13-BEJ514
- **Authors / organisation:** Wicher Bergsma, Angelos Dassios; London School of Economics and Political Science, Houghton Street, London WC2A 2AE, United Kingdom
- **Date:** 2014-05 (Bernoulli Vol. 20, No. 2, pp. 1006-1028; arXiv v6 dated 2014-03-14)
- **Link:** https://doi.org/10.3150/13-BEJ514 (arXiv preprint: https://arxiv.org/abs/1007.4259)
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

The two workhorses of ordinal association, Kendall's tau and Spearman's rho, are zero for any association that is not monotone, so tests built on them can never detect a general dependence: they are inconsistent. Hoeffding's coefficient and the Blum-Kiefer-Rosenblatt coefficient do detect general dependence, but the authors find their formulas opaque and, in Hoeffding's case, actually unsound for discrete data. The goal is a coefficient that is (a) simple enough to be described the way tau is described, as a comparison of orderings among a small number of replications, and (b) still zero exactly when the two variables are independent, so a permutation test on it is consistent. The result is tau-star: a sign covariance that is tau squared with squared distances replaced by absolute differences. The larger goal is a rank-based, tuning-free, exhaustive screen for dependence, which is what any probe-versus-label battery is trying to be.

## What they studied

Two ordinal random variables, continuous, categorical or a mixture of both; the unit of analysis is a quadruple of independent replications (four points in the plane), and the population object is a functional of the underlying copula. Data used are small: an artificial 5x7 contingency table, a 2x4 gastric-ulcer treatment table from Agresti, six constructed copulas supported on line segments inside the unit square, a random-walk process, and bivariate normal. No neural network appears anywhere; this is a statistics paper about a measure.

## How they did it

They start from the observation that tau is a probability of concordance minus a probability of discordance for a *pair*, and rho the same for a *triple* (two points on one axis, one on the other), and that tau is the simpler object for that reason. Squaring tau gives a sum of products of a pairwise sign kernel over quadruples; they then replace the squared differences inside that kernel with absolute differences, producing a kernel `a` that asks whether two of the four points are separated from the other two on *both* axes at once. The population coefficient is the expectation of the product of `a` applied to the x's and to the y's. They prove it is non-negative with equality only under independence, for any law with both a mass part and a density part, via two lemmas about monotone-likelihood structure (a conditional CDF difference decomposition, and a second-moment lemma for conditional CDFs below a common threshold); the proof repeatedly conditions on the two inner x-order statistics and pushes all remaining terms into non-negative sums. They then reinterpret Hoeffding's coefficient as the same concordance-through-five-points idea (with the fifth point at the axis crossing), which shows tau-star is the simpler object: four replications instead of five. Finally they test by Monte Carlo permutation of the y values, and they explicitly decline to build an asymptotic null. They compare against D and H on the artificial table and the ulcer table and by average p-value over six copulas plus a random-walk Bayesian simulation.

## What counts as an explanation here

A single scalar coefficient plus a theorem: the understanding is a *probabilistic interpretation* (four configurations, weighed as concordant or discordant, with tau-star equal to (2*Pi_C4 - Pi_D4)/3), not a model of the data. It is complete in the narrow sense that the coefficient is exactly zero only under independence, so a permutation test on it is consistent against every alternative. It predicts nothing about a particular sample beyond the p-value, and the authors are explicit that no sampling distribution theory is needed or offered.

## How they judged success

Theorem 1 (non-negativity with equality iff independence, for discrete, continuous or mixed laws) is the headline. Empirically: Monte Carlo permutation p-values with stated resample counts (10^6 resamples for tau-star, D and BKR; 4000 for Hoeffding) on the two real/artificial tables, and average p-values over at least 40,000 simulations per cell (200,000 for D) on nine settings in Table 3. There are no size or power curves in the modern sense, no calibration against a nominal level, and the comparison metric is deliberately the average p-value rather than rejection rate.

## What they found

On the artificial 5x7 table, Kendall's tau gives p = 0.99 and chi-square p = 0.252 (0.25 in the table caption) while tau-star gives p = 0.032, D gives 0.047 and H gives 0.028 — the point being that a consistent ordinal test finds a visibly non-monotone association that both the nominal-data chi-square test and the rank test miss. On the ulcer table tau-star gives 0.028, D 0.026, tau 0.019, chi-square 0.118, H 0.006. In Table 3's average p-values tau-star beats D in seven of nine settings and beats H in four of nine; the losses are informative — H is much stronger on the Zig-zag, Double-bump, Cross and Box copulas (0.011, 0.005, 0.003, 0.008 against tau-star's 0.036, 0.019, 0.021, 0.019), and D is slightly stronger on bivariate normal with correlation 0.5 (0.055 against 0.073), i.e. on the plain monotone alternative. Hoeffding fails badly on the parallel-lines copula (0.710 against tau-star's 0.076), which they attribute to the geometry that five-point concordance there requires collinearity. They also report that tau-star on a binary variable is equivalent to the Cramer-von Mises test, and note that tau-star is *not* an HSIC and does not appear expressible as a kernel or as the norm of a random element, which is why the proof is hard where the distance-covariance proof is easy.

## Limits and open problems they name

The theorem's distributional hypothesis is conjectured rather than proved to be necessary, and the extension to arbitrary metric spaces is shown to *fail* — they give an explicit eight-point configuration in R^8 where tau-star is negative (minus 1/64), so the sign property is a property of the real line, not of metric spaces. Evaluation of the estimator is O(n^4) and they can only suggest approximating it by a random subsample of quadruples. No asymptotic null is derived and they do not seem to want one. There are ties: the coefficient is undefined on quadruples that are neither concordant nor discordant, and the paper handles ties by exclusion rather than by a correction, which is exactly the gap the 2016 algorithmic note closes. The power comparison is over six hand-chosen copulas and an average p-value, not a minimax analysis.

## What we take from it

A **measure** with a usable shape: tau-star is monotone-invariant, distribution-free, zero exactly under independence on the real line, and — unlike distance covariance — has a four-point combinatorial interpretation, which is what makes it a candidate first-pass screen for "does this read-out direction know anything about this label at all, including non-monotone structure". A **warning** about the failure direction of the competitors: a rank test on activations can read 0.99 on data where a consistent test reads 0.032, so an all-clear from Kendall or Spearman on a parity/zig-zag-shaped relation is worthless. A **framing**: the paper's whole argument for simplicity is that a coefficient's definitional complexity is the number of replications it needs, and that argument is worth copying when we defend one screen over another in a paper. A **shortcut**: binary labels reduce this to the Cramer-von Mises test, and binary labels are most of what a probe battery has, so the classical test is available for free in that case. A **gap to remember**: ties are excluded rather than corrected here, which is precisely where the follow-up algorithmic papers had to do real work.

## Tags

unit: whole-model
form: probe
method: probing
eval: prediction/simulation
goal: evaluation-of-methods
