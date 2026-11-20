# Efficient Computation of the Bergsma-Dassios Sign Covariance

- **Key:** arxiv-1504.00964
- **Authors / organisation:** Luca Weihs, Mathias Drton, Dennis Leung — all three at the Department of Statistics, University of Washington, Seattle, WA, USA (per-author email addresses in the byline; corresponding author Weihs)
- **Date:** 2015-12-28 (v2; published in *Computational Statistics* 31(1):315-328, 2016, online 2016-01-04)
- **Link:** https://arxiv.org/abs/1504.00964
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The Bergsma–Dassios coefficient tau-star is the rank measure whose population value is exactly zero when and only when two ordinal variables are independent, a property Kendall's tau and Spearman's rho both lack. The price of that property is combinatorial: the empirical statistic t-star is a U-statistic averaged over ordered quadruples, so a literal evaluation inspects every distinct four-tuple of the sample, which is quartic in n. The authors of the *original* proposal had themselves stopped at samples of fifty points and recommended approximating the statistic on subsamples for anything larger. This paper is the engineering step that makes the exact statistic an ordinary tool: it asks whether the quartic cost is intrinsic to the statistic, argues that it is not because the kernel depends only on relative order, and delivers an exact algorithm at two factors of n times a logarithm. No new theory of dependence is claimed; the deliverable is a runtime.

## What they studied

Pairs of ordinal observations (x_i, y_i) in the plane, i indexing the sample of size n. The unit of analysis is a *quadruple*: four distinct sample points, classified once they are relabelled so that their x coordinates are non-decreasing. A quadruple is separable if the two points with the middle x values have distinct coordinates and one pair of y values sits strictly on one side of the other; it is *inseparable* when a tie in x, or a pattern of ties in y, prevents any axis-parallel line from splitting the four points into two groups of two; and a separable quadruple is *concordant* when the pairs' y-intervals are disjoint and *discordant* when the intervals overlap. The statistic is then a fixed combination of the counts of concordant and discordant quadruples. Nothing is trained and no model is fitted — the object under study is an exact counting problem over order patterns, and the sole empirical content is a runtime benchmark.

## How they did it

Three steps, in increasing generality. First, a **combinatorial lemma**: because the original sign kernel is unchanged by permuting its four arguments, the sum over all ordered tuples inside a fixed four-element subset is a single number b, and a case analysis — the one genuinely long proof in the paper, relegated to an appendix — shows b takes only three values: sixteen when the four points are concordant, minus eight when they are discordant, and zero when they are inseparable. This collapses t-star to a rescaled count: with no ties every quadruple is one of the first two, so the statistic becomes sixteen times the concordant count minus eight times the rest, i.e. it is an affine function of the concordant count alone, and only that count must be computed.

Second, the **tie-free algorithm**: sort on x and reorder y to match, so the problem becomes counting quadruples whose y-intervals separate. Sweeping the sample in x order, the count contributed by a fixed pair of later points (k, l) is the number of earlier points *below* both of them, choose two, plus the number *above* both, choose two — separability means both earlier points fall on the same side. Maintain the earlier y values in a balanced binary search tree (the implementation uses a red-black tree C library); then a rank query returns how many stored values fall below a threshold or above it in logarithmic time, and the insert of the next point is logarithmic too. That is two nested loops of logarithmic queries, hence n squared times log n.

Third, the **general algorithm for tied data**: a tied sample can produce inseparable quadruples that must not be counted as either concordant or discordant, and the two-pass structure is what handles it. The tree is forbidden from containing y values belonging to points whose x equals the current x, so insertion is deferred until x changes. A first forward sweep counts the separable contribution using a five-way partition of the earlier points — above both, below both, strictly between, and equal to the lower or upper of the pair — and deliberately over-counts discordant quadruples by pairing up earlier points that share a tied y value inside the straddled interval. A second sweep runs in reverse order to compute the exact size of that over-count, using a mirrored tree, and subtracts it. Both sweeps are n squared times log n, so the general algorithm keeps the complexity of the special case. An appendix repeats the whole construction for the biased V-statistic form used by the original authors, which needs correction coefficients of one half on the triple-with-one-repeat patterns and one quarter on the doubled-pair pattern, because a multiset of four indices with a repeated element is fixed by fewer permutations than a set of four. Code is the TauStar R package, with a C++ backend, and the tables' generator script is linked.

## What counts as an explanation here

A program plus an asymptotic statement: the understanding offered is the *reduction* — the claim that an order statistic over quadruples can always be expressed through range counts over a one-dimensional structure, and that the reduction is tight enough to be implemented. It is complete in the sense the paper claims (the algorithm computes the exact statistic, not an approximation, for every input including tied and degenerate ones) and it predicts runtime, not data. There is no surrogate model, no feature dictionary, and no claim that tau-star is *better* than any competitor; the paper is explicit that usefulness is untouched.

## How they judged success

Wall-clock runtime, in seconds, averaged over ten samples per configuration, from two C++ implementations reached through R — Algorithm 2 (with the red-black tree library) against a naive quadruple-enumerating implementation, at sample sizes 100 to 300, then Algorithm 2 alone up to ten thousand. A second benchmark quantifies the alternative it replaces: relative sample variance, meaning the variance of a subsampling estimate divided by the variance of the full exact statistic, estimated from a thousand samples of size a thousand drawn as independent standard-normal pairs, for resample counts from 200 to 12,800 and subset sizes four and thirty. There is no synthetic correctness test against brute force reported, and no accuracy metric — correctness rests on the proof, and the numbers are all cost.

## What they found

The separation is total. At a hundred points the exact algorithm takes nine ten-thousandths of a second against 0.287 for enumeration; at three hundred, one hundredth of a second against 28.95. That is roughly a factor of three thousand at the top of the compared range, and the ratio grows with n. Scaling out, the exact statistic costs 0.1265 seconds at a thousand points, 1.74 at 3,250, 5.27 at 5,500, 11.18 at 7,750 and 19.1 seconds at ten thousand — one second per five hundred points at the large end, i.e. the practical wall for the exact statistic on a single core is somewhere in the tens of thousands. The subsampling benchmark is the more consequential number: with subset size four the relative variance is 3,932 at 200 resamples and still 57 at 12,800, so the subsampling route is hopeless at the subset size the original authors suggested; with subset size thirty the relative variance falls to 8.24 at 200 resamples, 1.21 at 3,200 and 1.04 at 12,800. The paper's own reading is that a larger subset size, not a larger resample count, is what rescues approximation — a point it states and does not develop.

## Limits and open problems they name

The exact algorithm is conceded to be insufficient at the largest scales, and the authors hand those cases back to subsampling without saying where the boundary is. Two threads are left hanging: the observation that a subset size above four is preferable, with no guidance on choosing it or on how the resulting estimator's bias depends on it; and every question about usefulness, since the paper says nothing about the power of tau-star relative to finer-partition tests such as the quadrant or distance-covariance families. The tree-based approach is also confined to the bivariate case, which is the only case the statistic is defined for. There is no discussion of parallelisation, of the space cost of the tree, or of behaviour on heavily tied data beyond the correctness argument.

## What we take from it

Three things. (1) **A method, and the missing half of the arc.** This is the algorithm every cheap rank screen in the folder assumes exists; it makes an exact four-point independence statistic affordable at probe-battery sample sizes, so a screen over many candidate read-out directions no longer has to choose between permutations and subsampling. (2) **A warning about the approximation alternative.** The subsampling table generalises past this statistic: an average of statistics computed on random small subsets is not a mild approximation when the subsets are too small, and the correct knob is the subset size rather than the resample count — relative variance falls by a factor of a thousand when the subset grows from four to thirty at fixed resample count, and the resample count alone is fighting a losing battle. Applied to our own batteries, any "sample a small tuple of examples and average" shortcut should be benchmarked this way before it is trusted. (3) **A framing for the reporting of engineering claims.** The paper says plainly that it establishes a cost and not a power; that separation is exactly the discipline we need when we say a probing procedure got faster — those are two sentences, and only one of them is a claim about interpreting the network.

## Tags

unit: whole-model (statistic over sample order patterns; no model internals)
form: program
method: weight-analysis (order-statistic counting; algorithmic, not a network method) — closest fit: other
eval: benchmark
goal: understanding (methodological enabler)
