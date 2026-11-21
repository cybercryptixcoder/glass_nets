# Efficient estimation and correction of selection-induced bias with order statistics

- **Key:** arxiv-2309.03742
- **Authors / organisation:** Yann McLatchie, Aki Vehtari; Department of Computer Science, Aalto University, Finland
- **Date:** 2023-09 (v1); v3 revised 2024-08; journal version Stat Comput 34, 132 (2024)
- **Link:** https://arxiv.org/abs/2309.03742
- **Kind:** paper
- **Read from:** full text (v2 HTML as served; abs page for metadata)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Choosing the best of many candidate models by a noisy held-out score silently over-credits whichever one happened to win: the reported score for the winner is inflated even though each individual score is nearly unbiased. The authors call this selection-induced bias and treat it as a magnitude to be measured, not a yes/no property to be tested — their stated aim is to tell the practitioner when the bias is negligible and when it is large enough to invalidate the decision, rather than to eliminate it. They want a tool that costs nothing beyond the numbers a standard cross-validation run already produces, because the two established repairs (nested cross-validation and the bootstrap) are both expensive. The larger goal is to make cheap score-based model selection usable with a stated safety condition attached.

## What they studied

Nested and one-predictor linear regression models fitted to simulated data (n = 100, up to K = 100 candidates), then a 100-predictor block-correlated regression with forward search over 20 steps, then four public datasets: Crime (regression, 1992 rows, 102 predictors), Heart/SPECT, Ionosphere (351, 33) and Sonar (208, 60) for binary classification. A second real-data demonstration builds 10 generalised linear models of differing observation families and predictors on an infant EEG cash-transfer dataset. The unit of analysis is the *model*, and the quantity being modelled is the collection of leave-one-out score differences between each candidate and a baseline.

## How they did it

The procedure rests on one observation: when no candidate is genuinely better than the baseline, the K score differences are noise, and the best of them is an extreme order statistic of that noise. They approximate the expected maximum of K standard normal draws by the inverse normal CDF at a rank fraction with Blom's correction, using the inside-the-interval choice of the approximation constant for safety. The scale is not the pooled standard deviation but a maximum-likelihood half-normal fit to the *upper* half of the observed differences above their median — deliberately asymmetric, because the right tail is what the maximum lives in and the elbow of near-tied accounts is the risk case. Multiplying the two gives a threshold: if the best observed difference sits under it, all candidates are declared practically equivalent and no selection is made. For forward search they go further and *subtract* an estimated bias at each step, multiplied by a factor of about one and a half, justified by a reported strong negative correlation between leave-one-out scores and average Bayes generalisation loss, so the correction is a reflection across the median rather than a downward shift. The corrected increments are summed along the path, and the stopping rule picks the model size where the corrected curve peaks. They then compare that rule against stopping at the raw maximum, against a two-standard-error rule, and against projection predictive inference, under both independent Gaussian and sparsity-inducing priors.

## What counts as an explanation here

Not an explanation of a model, but a stateable condition on a decision: a threshold value computed from the candidate set's own score dispersion, above which selection is licensed and below which it is not, plus a corrected score curve along a search path. It is partial by construction — they say explicitly they cannot state a formal boundary between negligible and non-negligible bias, and their correction rests on a correlation constant taken from a single published example. It predicts one thing well: the height that chance alone will produce among K equivalent candidates.

## How they judged success

Simulated ground truth where the true coefficient is set to zero (so the true bias is known and every observed difference is noise); mean best observed difference across 100 data realisations compared against the predicted threshold; test-set mean log predictive density as the oracle curve, with the gap to the leave-one-out curve read as bias and the gap to the oracle read as over-fitting; selected model size across cross-validation folds as a stability measure; and head-to-head model sizes and test scores against the raw-maximum, two-sigma and projection-predictive rules on four real datasets. Priors are varied as an ablation, and the correction magnitude is repeated at values corresponding to zero, moderate and full correlation.

## What they found

The threshold tracks the simulation closely, growing steeply up to about ten candidates and then flattening. With two candidates the bias is negligible in their experience; at K = 100 they report that leave-one-out differences are unsafe until the true effect exceeds roughly 0.3 in their standardised design. In forward search the correction recovers the shape of the test curve and locates saturation near the true point in all six simulated regimes, while under-correcting slightly when the sample size equals the predictor count; they note bias is negligible once n exceeds about four times p. Sparsity-inducing priors shrink the bias before any correction is applied, and the corrected rule selects slightly smaller models than projection predictive inference at similar test performance, with stability comparable to it. On real data the correction fixes the raw maximum's most visible failure — a sub-model scoring better than the full reference model — with one exception, the heart dataset, where it still does not, and on sonar it over-corrects. The incremental two-sigma stopping rules pick much smaller models but at a real cost in predictive performance, and are described as unusable for the classification tasks as written.

## Limits and open problems they name

The assumptions are that the differences are independent and not-too-skewed; correlated predictions (which regularising priors produce) break independence, and the paper says so. The correction magnitude is the weakest link: it is calibrated on one example, the authors offer no principled way to derive it, and they invite future work. The distribution may not be centred exactly at zero in practice. They note that more flexible tail fits would trade bias for variance, and they set the maximum-order-statistic approximation constant conservatively, which biases the threshold upward. Projection predictive inference is recommended first wherever the observation family permits, with their method standing in only when it does not.

## What we take from it

A **measure** and a **stopping rule**. (1) The dispersion that governs whether an account search has found anything is the spread of *score differences among candidates*, fitted symmetrically-in-kind but asymmetrically-in-support (half-normal on the upper tail) — not any single account's own error bar, and not a standard error computed over the winner alone. (2) The threshold scales like the expected maximum of K draws from that dispersion, so doubling the account set raises the bar by a multiplicative factor rather than leaving it fixed, and any report of "account X is best" from a set of K must print K. (3) A search over accounts that makes decisions step by step compounds the bias and needs a *cumulative* corrected curve with a peak-based stopping rule, because per-step evidence is not path-level evidence. (4) The warning to carry: the correction constant here was borrowed from a single published example and the authors say plainly they cannot justify it — a correction of unknown size dressed as a precise number is worse than the named refusal to decide that their two-candidate recommendation actually is.

## Tags

unit: whole-model
form: replacement-model
method: other (score-difference distribution modelling; order statistics)
eval: prediction/simulation | benchmark
goal: evaluation-of-methods
