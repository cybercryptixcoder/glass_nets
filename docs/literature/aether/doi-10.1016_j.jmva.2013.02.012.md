# The distance correlation t-test of independence in high dimension

- **Key:** doi-10.1016_j.jmva.2013.02.012
- **Authors / organisation:** Gábor J. Székely (National Science Foundation, Arlington VA; also Rényi Institute of Mathematics, Hungarian Academy of Sciences) and Maria L. Rizzo (Department of Mathematics and Statistics, Bowling Green State University)
- **Date:** 2013-05 (received 2010-08-12, available online 2013-02-27)
- **Link:** https://doi.org/10.1016/j.jmva.2013.02.012 — PDF read from a course mirror at pages.stat.wisc.edu
- **Kind:** paper
- **Read from:** full text (12-page PDF, main text plus the appendix proof sketches and the reference list)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Distance correlation characterises independence in any dimension and gives a consistent test for it, so in principle one statistic answers "are these two random vectors dependent?" without choosing a model. But the sample coefficient is a plug-in estimator with a bias that grows with dimension, and in the regime where the number of coordinates of each vector is large compared with the number of observations that bias dominates: under independence with identically distributed coordinates the sample distance correlation climbs to one as dimension grows, for every fixed n. A number that reads near one when the vectors are independent is useless to a practitioner who wants to quote it, and the permutation null is still available but tells them nothing about how strong the dependence is. The authors therefore want a corrected coefficient whose null law is known in closed form, so that a single number, not a resampling loop, can be read against a table.

## What they studied

Pairs of random vectors X in R^p and Y in R^q with independent, identically distributed coordinates with positive finite variance — the high-dimension, fixed-n regime, p and q taken to infinity with n fixed. The unit of analysis is the vector pair and, inside it, the n-by-n matrix of pairwise Euclidean distances. They also treat exchangeable (not iid) coordinates and, in an application section, strongly stationary time series that are turned into exchangeable vector samples by taking random contiguous windows. Real data: daily log returns of the 30 Dow Jones Industrial Average stocks, 1993 to 2003.

## How they did it

They start from the classical double-centred distance matrices A_ij and B_ij (distance minus row mean minus column mean plus grand mean) whose elementwise product averages to the squared sample distance covariance. The observation that drives everything: for fixed n, the off-diagonal entries of A have expectation alpha/n while the diagonal entries have expectation alpha/n minus alpha, and alpha is the mean distance, which grows without bound with dimension — so the diagonal is the wrong scale and the bias it creates is exactly what drives the coefficient to one. They define corrected entries A*_ij by rescaling off-diagonal entries by n/(n-1) after subtracting a_ij/n, and diagonal entries by n/(n-1) times (row mean minus grand mean), a correction that makes every entry mean zero. Summing the elementwise products of the corrected matrices, with a doubled diagonal correction subtracted, and dividing by n(n-3) gives an unbiased estimator of the squared population distance covariance; dividing that by the geometric mean of the two corrected variances gives the modified correlation R*. They then show that in the infinite-dimension limit each corrected self-term converges to a scaled chi-square and the cross-term to a sum of products of independent normals, so that the correlation becomes the cosine between two random vectors in n(n-3)/2-dimensional Gaussian space — whose t-transform is exactly Student t. Inference is cheap: a plug-in R estimated from the sample, a variance that happens to cancel in the ratio, and a t quantile. For time series they append a proposition that random contiguous windows of a stationary series are exchangeable and hence conditionally iid given the symmetric-event sigma algebra, which is enough for the same test.

## What counts as an explanation here

A distributional guarantee rather than a decomposition: the "understanding" is that the empirical coefficient is a cosinus between two independent Gaussian vectors in a space whose dimension is set by n, from which the null law follows arithmetically. Nothing about where the dependence lives, which coordinates carry it, or what function of the inputs produces it — the object is a scalar and its null. It is a complete explanation of the statistic's high-dimension behaviour, and no explanation at all of the dependence it detects.

## How they judged success

Three things. (1) A theoretical claim: dimension-asymptotic convergence of the transformed statistic to Student t with a stated degrees-of-freedom formula, proved for iid coordinates, then extended to exchangeable coordinates by a conditional-CLT argument. (2) Unbiasedness of the test at every n of four or more and every level — a property no permutation test needs and no plug-in test usually has. (3) Empirical calibration: a table of the corrected and uncorrected coefficients along one sample as dimension climbs from 1 to 16,384 at n = 30; 1,000 replications to check the null tail (0.100 at the 10 per cent level, 0.046 at 5 per cent); a linear-dependence example where all 1,000 replications reject; a power curve over the error correlation of a bivariate AR(1) pair; a type-1-error sweep over window length and series length against white noise; and a dendrogram of the 435 stock-pair coefficients. Baselines are the uncorrected distance correlation and, in the recycling exercises, the Z transform of the same statistic.

## What they found

The uncorrected sample coefficient against the corrected one, on independent uniform coordinates at n = 30, is the paper's cleanest number: the uncorrected value moves from 0.430 at one dimension to 0.9995 at 4,096 dimensions and 0.99990 at 16,384, while the corrected coefficient stays within about plus or minus 0.12 of zero across the whole sweep and the t statistic stays between minus 2.25 and plus 2.39. The null tail comes out close to nominal (0.100 and 0.046 empirically at 10 and 5 per cent), the linear-dependence example rejects in every one of 1,000 draws, and the degrees of freedom are large enough (404 at n = 30) that the t is effectively normal. In the stock example most pairs are declared dependent at the 5 per cent level, with the largest t statistics around 15 and 16 for pairs of financial and of technology stocks and near-zero values for pairs that are visibly unrelated; the dendrogram built from the corrected coefficients recovers recognisable sector groupings. The time-series window construction holds its nominal level except when the total number of sampled values approaches the series length.

## Limits and open problems they name

The guarantee is a limit in dimension, not in sample size: the test is justified when p and q are large relative to n, and the paper offers no guidance on the crossover at which the t approximation becomes honest (elsewhere in this folder a companion result cites the same asymptotic as requiring large dimensions in a way that rules out small p in practice). The iid-coordinate assumption is doing real work in the proofs, and the exchangeable and stationary versions are paid for by conditional-CLT arguments the authors attribute to known results rather than reprove. The corrected statistic can be negative, which a coefficient is not supposed to be, and the authors exclude the exactly-uncorrelated-after-linear-transform case where the denominator vanishes. Finally, and most relevant to us, the single number still does not localise anything: the JDIA application recovers clusters only by building a 435-entry dissimilarity matrix and clustering it, not from any one coefficient.

## What we take from it

Three things. (1) **A bias diagnosis that transfers verbatim to probe batteries.** The failure was that the estimator's diagonal and off-diagonal entries sit at different scales, and the scale gap grows with the number of coordinates; the repair was to make every entry mean-zero by an explicit rescaling whose constant is known. Any per-direction statistic we aggregate over a large bundle — a correlation between a probe read-out and a label, averaged over directions — has the same structure, and the same check applies: compare the individual entries' expectations under the null before and after aggregation, and if the two differ in a dimension-dependent way the aggregate is not measuring what the individual numbers measure. The companion entry `arxiv-1902.03291` proves the asymptotic version of this failure; this paper supplies the finite-n version and the correction, and the two should be read as one document. (2) **A null in closed form instead of a permutation loop is available when the statistic is a cosine of random vectors**, which makes a per-direction scan free — the objection every earlier entry in this folder had to concede. (3) **A reporting convention worth stealing for our own high-dimensional screens:** print the corrected and uncorrected statistic side by side along the dimension axis, as Table 1 does, so a reader sees the estimator's own pathology on the same page as the fix, and print the empirical null tail next to the nominal one, as the 1,000-replication check does.

## Tags

unit: whole-model  
form: probe  
method: other (unbiased U-statistic modification; dimension-asymptotic limit theory)  
eval: benchmark, downstream-task  
goal: evaluation-of-methods
