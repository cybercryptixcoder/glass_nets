# Treatment of Statistical Estimation Problems in Randomized Smoothing for Adversarial Robustness

- **Key:** arxiv-2406.17830
- **Authors / organisation:** Vaclav Voracek; University of Tübingen
- **Date:** 2024-06-25
- **Link:** https://arxiv.org/abs/2406.17830
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Randomised smoothing certifies a classifier by estimating, from noisy forward passes, the probability that the smoothed classifier returns its top class — and the field does this with a fixed, large sample budget (on the order of a hundred thousand passes per point) plus a Clopper–Pearson (so-called exact) confidence interval. The author asks whether that budget is necessary, and separates two questions the practice conflates. First, is the interval used to bound the class probability as short as it can be? Second, is a fixed sample count the right way to spend samples at all, given that most points either certify or fail well before the budget is exhausted? The larger goal is to put randomised-smoothing certification on an optimal statistical footing, so that the reported certificate is limited by the network and the theorem rather than by how many samples someone chose to draw. The paper is a statistics paper that happens to be about a robustness certificate: no training method, no architecture, and no weight enters any of its claims.

## What they studied

The unit of analysis is not a model but the *estimator* used inside a certification wrapper. The objects are: a Bernoulli/binomial mean estimated from samples (the probability that the base classifier returns the top class at a noisy input), two kinds of confidence bound over that mean (a fixed-sample interval, and a confidence sequence that holds simultaneously at every sample count), and the decision problem of declaring a point robust at radius r with a type-1 error budget α and a type-2 error budget β. The networks appear only as sampled black boxes that produce the Bernoulli draws; the experiments use a smoothed ResNet-110 and a WideResNet-40-2 on CIFAR-10 under Gaussian and uniform-box smoothing, but nothing about either architecture is measured.

## How they did it

Step one: the standard Clopper–Pearson upper interval is shown to be conservative for a structural reason — the binomial has only n+1 outcomes, so exact coverage 1−α can hold for at most n+1 values of p, and for large p the interval is wide regardless of the observation (the author's figure puts the crossover near the n-th root of α for the case shown). The remedy introduces one uniform random draw per interval and interpolates between the intervals for x and x+1; coverage then equals exactly 1−α at every p rather than at a handful, which the author proves and demonstrates on a two-trial example worked by hand. Step two: replace the fixed sample count by a confidence sequence — a bound valid simultaneously for every t — so sampling can stop the moment the decision is settled. Two constructions are given: a union-bound sequence that recomputes an interval only at exponentially spaced sample counts with a schedule of per-check failure budgets summing to α, and a betting sequence in which, for each candidate mean p, a fair-game martingale's wealth is tracked and p is eliminated once the wealth crosses 1/α (Ville's inequality makes the elimination sound), with the surviving set shown to be an interval so that its endpoints are found by binary search. Step three: the sequential decision problem is posed abstractly (decide whether the true mean exceeds a known threshold, choosing when to stop), lower bounds from the law of the iterated logarithm and from Wald's sequential probability ratio test are used as the ideal, and the two sequences are compared against a prior adaptive method (Horváth et al.) on real CIFAR-10 certificates. The author also runs binary versus multiclass certification and reports the average certified radius each way.

## What counts as an explanation here

A statistical guarantee, and nothing about the network. The deliverable is: (a) an interval mapping each observed sample count to a bound on the class probability with a stated coverage, (b) a stopping rule with a proved sample complexity, and (c) a proof that these are optimal up to constants — the sequence widths match the law-of-iterated-logarithm rate, and the sequential procedure's sample complexity matches the ideal sequential test up to a constant. It is a complete account of the estimation layer and says nothing predictive about the model; the certificate's radius formula is imported wholesale from earlier work and never re-derived.

## How they judged success

Two axes, kept apart deliberately. Theory: exact coverage of the randomised interval (proved, via a decomposition over the observed count), and width of the confidence sequences against the optimal rates — square root of (log(1/α) + log log t)/t for the union-bound sequence over powers of two, and square root of (log(1/α) + log t)/t for the betting sequence, the latter candidly described as not asymptotically optimal. Practice: samples needed to decide robust-or-not at three fixed radii, averaged over a thousand trials at 51 evenly spaced values of the mean, and reported as a multiple of the ideal (SPRT with both probabilities known); plus a wall-clock column beside every sample column, and error bars over repeats. The empirical comparison is against the Horváth et al. adaptive method only, with its finite grid of sample budgets named as the reason it must abstain on hard points.

## What they found

The randomised interval attains exact rather than conservative coverage, and the visible effect on the robustness curve is concentrated at the top of the probability range — which is where the Gaussian radius formula diverges — so the sharp drop at the end of a robustness curve is partly an artefact of the interval rather than of the classifier. On the sequential task the saving is large and grows with radius: at radius 0.5 the adaptive baseline needs about 1,976 samples against 531 for betting and 635 for union-bound; at radius 2 the figures are about 4,623 against 2,130 and 2,670, i.e. roughly a halving to a quarter, with wall-clock times falling from 0.13 s to 0.05 s at the easiest radius and from 0.30 s to 0.17 s at the hardest. A second model (retrained at a larger smoothing σ) shows the same ordering with smaller ratios, and the ℓ1 setting shows betting ahead again at the smallest radius. Multiclass certification, which budgets the failure probability across the top class and the runner-up, raises the average certified radius from 0.50 to 0.61 in the comparison the author shows, at essentially no statistical cost. The betting sequence has the larger lead at low sample counts; the union-bound sequence is faster at the smallest radius in one table (0.016 s against 0.19 s) despite using more samples, which the paper does not explain.

## Limits and open problems they name

The betting sequence is not asymptotically optimal and no log-log-rate version is built here; the analysis hides the dependence on the true mean by assuming it stays away from 0 and 1, so the constants are only order-level. Empiricism is thin: one dataset, one or two models, no comparison on larger radii or other smoothing families, and no measurement of how the certificate changes when the interval is randomised — the claim that the improved bound is "strictly stronger" is argued through the interval, not shown as a per-image certified-radius table. The hyperparameters of the union-bound schedule (step factor, decay exponent) trade asymptotic rate against small-sample behaviour, so the method has a dial that the paper says should be set for the regime of interest without saying how to choose it in advance. The paper also notes that comparing robustness curves across methods of this kind is vacuous, since the adaptive task does not spend samples to improve the curve.

## What we take from it

Three things. (1) **A measurement budget is a design decision whose optimum is data-dependent, and a fixed budget wastes samples in both directions** — the point that certifies after ten passes is paid for at a hundred thousand, and the point that needs more than the budget is declared uncertifiable for lack of budget rather than lack of robustness. Any probe of ours with a fixed query allowance inherits the same two failure modes, and a confidence sequence is the general fix: it needs no prior guess of the cost and it can stop early. (2) **A monkey-patchable conservativeness fix exists in the interval rather than in the method**, taking a certificate from conservative to exact with one random draw — the same class of move as run 655's sigma-and-probability surface, but here it moves the number in the direction that helps, and the honest statement of its size is a per-image radius delta, which the paper does not print. (3) **The paper is the folder's cleanest example of a metric whose entire remaining gap is Monte Carlo error, and it tells us what to do about it**: the radius is capped by sampling noise, so instead of a fixed budget the account should carry a stopping rule and a coverage level, and report the budget actually spent per query as a distribution — because a mean sample count over test points hides exactly the tail that the budget was chosen to protect.

## Tags

unit: whole-model
form: probe
method: other
eval: prediction/simulation
goal: evaluation-of-methods
