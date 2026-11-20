# A new coefficient of correlation

- **Key:** arxiv-1909.10140
- **Authors / organisation:** Sourav Chatterjee — Department of Statistics, Stanford University (address and email printed in the v4 HTML; the abs page prints no affiliation)
- **Date:** 2020-04-28 (v1 2019-09-23; to appear in J. Amer. Statist. Assoc.)
- **Link:** https://arxiv.org/abs/1909.10140
- **Kind:** paper
- **Read from:** full text — v4 HTML, ~112.7k chars, read end to end (abstract, address, §1-§9 and the reference list); the proof sections were skimmed for their strategy, not followed line by line
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Pearson, Spearman and Kendall detect linear or monotone association and have cheap asymptotic tests, but they are blind to wiggly relations even when those relations are completely noiseless. The many later coefficients built to fix that (maximal correlation, kernel measures, distance correlation, copula and information-theoretic measures) mostly serve as tests of independence rather than as measures of how strong a relation is, and most of them have no simple null distribution, so practitioners fall back on permutation tests. Chatterjee asks whether one coefficient can have all three properties at once: a formula as simple as Pearson's, a limit that is a genuine strength measure — zero exactly under independence, one exactly when the target is a measurable function of the predictor — and a simple asymptotic law under independence. The larger goal is a dependence instrument with no tuning constants and no distributional assumptions.

## What they studied

Two random variables from an i.i.d. sample of size n, with no assumption on their joint or marginal laws beyond the target not being almost surely constant. The unit of analysis is the pair of variables and the statistic computed on the sample; there is no model and no network. Real datasets used as examples: Galton's 1875 sweet-pea data (700 observations), and the Spellman et al. yeast cell-cycle expression data (4381 genes after the curated filtering, 23 time points).

## How they did it

Sort the sample by the predictor, write down the rank of the target in that sorted order, and add up how far the target's rank moves between consecutive predictor values; rescale so that the sum sits in the unit interval. The formula is a single normalised sum of absolute rank differences, so it is a function of ranks only and costs n log n. With ties, predictor ties are broken at random and the denominator is replaced by an expression in the target's own tie counts, which reduces to the tie-free form when the target is continuous. The deterministic limit is derived as a ratio of integrated variances of threshold indicators: the conditional variance of the indicator that the target exceeds a level, given the predictor, averaged over levels, divided by the indicator's unconditional variance averaged over the same levels. Under independence the scaled statistic is shown to converge to a centred Gaussian whose variance is 2/5 for a continuous target and an explicitly estimable expression otherwise; the paper also gives an O(n log n) estimator for that variance. Independence testing is compared against MIC, distance correlation, HHG and HSIC at n = 100 with 500 simulations per alternative across six noise-graded shapes; run times are compared for permutation tests with 200 permutations. A separate section constructs a noisy mixture-of-two-normals example and proves that maximal correlation and MIC both saturate there. A gene-selection experiment applies Benjamini-Hochberg at a false-discovery level of 0.05 to each test's p-values and looks at the genes only one test selects.

## What counts as an explanation here

A scalar statistic together with its population limit and the limiting distribution — a measurement instrument, not a description of a mechanism. The limit itself is an explanation of what the number means: it is the fraction of the target's threshold-indicator variance that survives conditioning on the predictor, averaged over thresholds. It is exact at both ends (independence, determination) and is meant to be a graded strength in between, not a decision. It predicts: given a joint law, the limit is computable, and given a sample, the null distribution is known well enough to attach a p-value without resampling.

## How they judged success

Four criteria, each with its own check. (1) The limit's endpoints: proved to be zero iff independent and one iff the target is a measurable function of the predictor, with proof in §9 and a sketch in §8. (2) Null calibration: histograms of ten thousand simulations of the scaled statistic against the predicted density, at n = 20 and n = 1000, for uniform and for binomial(3, 0.5) marginals, plus a dependent Bernoulli product model whose limit is computed in closed form and whose simulated mean matches it. (3) Power: comparison against MIC, distance correlation, HHG and HSIC across linear, step, W-shaped, sinusoidal, circular and heteroskedastic alternatives with a noise parameter. (4) Run time: wall-clock permutation-test times at sample sizes from 100 to 10000. Equitability and generality are assessed qualitatively, using the definitions from the MIC paper.

## What they found

The limit's endpoints and the null theory hold as stated, and the null approximation is described as already close at n = 20. The finite-sample ceiling is not one: with no ties the largest attainable value is (n-2)/(n+1), about 0.86 at n = 20, and the smallest is near -1/2 plus a small correction; a large negative value can only mean the data are not an i.i.d. sample, which the paper flags as a usable diagnostic. The coefficient is explicitly asymmetric: on Galton's peas the predictor-on-target direction reads about 0.11 (p below 0.0001) while the reverse reads about 0.92, revealing that each daughter-plant diameter value maps to a single mother-plant value in that table — a data-collection artifact the coefficient surfaces numerically. Power is the honest weak point: it beats the competitors on oscillatory signals (W-shape, sinusoid) and is reasonable on step functions, but loses on smooth ones (linear, circular, heteroskedastic). Run time is the decisive advantage — at n = 10000 the permutation-based competitors take from about 89 seconds to over 30 minutes while this statistic's asymptotic test takes about 0.011 seconds, and it is already about 5x faster than the nearest competitor at n = 500. On the yeast data it selects 586 genes, of which 215 are selected by no other test and which inspection shows to be strongly oscillatory, while genes the other tests alone select show only slight trends. In the two-cluster normal mixture, MIC reads 1.00 and maximal correlation 0.99 while this coefficient reads 0.48; a proposition proves the saturation is not an artifact — any law supported on two disjoint rectangles with equal mass makes maximal correlation exactly 1 and MIC tend to 1, even though the target is far from determined by the predictor. This is the paper's sharpest negative result and it targets the coefficients most often used as "strength of relation" measures.

## Limits and open problems they name

No asymptotic theory is given when the variables are dependent; simulations suggest normality of the scaled error but nothing is proved. The null distribution of the symmetrised statistic is not derived, and the paper says it has neither a proof nor a conjecture for its parameters, recommending a permutation test instead. The coefficient is less powerful than competitors on smooth alternatives, which the author concedes is the majority of relations met in practice, and blames the n log n budget. It measures how well the target is predicted by the predictor, so it is deliberately not equitable for non-functional relations. The formula is so simple that the author expects better variants to exist.

## What we take from it

Three things, in order of use. First, a free and near-universal null test: any pair of activation statistics — a direction's projection against a label, a probe score against a residual, two feature activations — can be cleared for dependence in n log n with an asymptotic p-value instead of a permutation loop, and the same statistic with conditioning gives the parent paper's `arxiv-1910.12327` coefficient, so the two form a matched pair (unconditional and conditional) built on the same threshold-indicator identity. Second, a warning with teeth about the strength measures a probing account is likely to reach for: MIC and maximal correlation can read 1 on a relation that is nearly pure cluster membership, so any "the model represents X with strength 1.0" claim resting on them should be replaced by this coefficient or by a determination claim that the reverse direction actually licenses. Third, the asymmetry is the feature, not an inconvenience: reporting both directions on the same pair distinguishes "the predictor determines the target" from "the target determines the predictor", which is exactly the difference between a direction that carries information about a property and a property that is merely correlated with it — and the peas example shows the two can differ by nearly an order of magnitude on real data. Method to reuse: the threshold-indicator variance ratio as a general way to define a dependence strength, since it needs no density estimate and its denominator is computable in closed form under any marginal. Warning to carry: a negative estimate on real data is not a weak relation but evidence of non-i.i.d. structure.

## Tags

unit: whole-model
form: probe
method: probing
eval: benchmark
goal: evaluation-of-methods
