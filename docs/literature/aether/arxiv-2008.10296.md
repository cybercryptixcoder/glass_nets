# Uncertainty in Bayesian Leave-One-Out Cross-Validation Based Model Comparison

- **Key:** arxiv-2008.10296
- **Authors / organisation:** Tuomas Sivula, Måns Magnusson, Asael Alonzo Matamoros, Aki Vehtari; Aalto University (Finland), with Magnusson at Uppsala University (Sweden) and a printed note that most of his work was done while at Aalto
- **Date:** 2020-08 (v1); this entry reads v6, 2025-10-28
- **Link:** https://arxiv.org/abs/2008.10296
- **Kind:** paper
- **Read from:** full text (v6 HTML, 372,348 chars after tag strip)
- **Relevance to us:** 3 (central)

## What they wanted, and why

The field's default way to say "model A beats model B" is a point difference in leave-one-out predictive score with a symmetric error bar around it, and a probability of superiority read off that bar. The authors ask whether that error bar is actually calibrated — and, if it is not, in which situations it fails and whether anything can be done from inside the data. Their motivation is that the point estimate itself has been studied to death while the uncertainty around the *difference* has been assumed to inherit the good behaviour of the estimate. The larger goal is a trustworthy report of predictive-performance comparisons: not just a number, but a number whose stated confidence is honest. They are also careful to say what they are not doing — they do not assume one of the compared objects is the truth, so "did we select the right one" is never the question; only "how much better is each, and how sure are we" is.

## What they studied

Bayesian models compared pairwise on the expected log pointwise predictive density (elpd): a normal linear regression data-generating process with known noise structure, against nested and non-nested linear regression models with flattened priors. The analytic work hinges on a single scalar control parameter — the true coefficient on the covariate(s) that belong to one account and not the other. Setting it to zero makes the two accounts predict identically; growing it separates them. Data-driven magnitude is the object of study, not neurons or weights: they measure the *sampling distribution* of the score difference, the *true* score difference for each dataset, and the *error* between them, as functions of three things — account similarity, a deliberately corrupted observation, and sample size. Empirical arms extend the same measurement to hierarchical linear, latent-Poisson-generalised-linear and penalised-spline models, plus fixed-covariate designs and ten-fold rather than leave-one-out splits.

## How they did it

1. Define the estimand: the difference in expected predictive density between two accounts for the *observed* dataset, and separately for the population of datasets. These are different quantities and the paper keeps them apart.
2. Write the error of the leave-one-out estimate as an explicit expression, and in the linear-regression case prove it collapses to a quadratic form in the residual noise — a matrix times εε plus a linear term plus a constant. That form makes its mean, variance and third moment derivable in closed form given the design.
3. Derive those moments, then vary the single similarity parameter to see how the moments move. Prove the shared-covariate effects cancel out of the comparison entirely, and derive limits for growing similarity, growing outlier magnitude, growing residual variance, and growing n.
4. Simulate: draw thousands of datasets per setting from a known generator, compute both the estimate and the true target (the latter against an independent large test sample), and compare the true error distribution with the symmetric approximation built from the pointwise differences — including its calibration via the probability integral transform.
5. Repeat for the extra model families and for ten-fold splits, to check the conclusions are not an artefact of the simplest linear case.
6. Three real datasets as worked examples — 17 primate species, a sleep-restriction reaction-time panel, and a roach-count regression — each chosen to sit in one of the three problem conditions, with an explicit statement of which condition applies and what the reported probability should be read as.

## What counts as an explanation here

Not an explanation of a model's internals at all — an explanation of the *reliability of a comparison*. The form is a distributional characterisation: for a given comparison, the error of the score difference has a shape (mean, spread, skew), and the claim is that the symmetric approximation is good exactly when the two accounts are well separated, and misleading when they are close. It is deliberately partial: the paper does not offer a corrected estimator, it offers a diagnosis — a short list of conditions under which the uncertainty is not to be trusted, plus the statement that no repair available from the pointwise terms can fix it. Prediction is limited to predicting the *direction of error*: the approximation understates spread and, near the mode, can report the wrong sign of the difference with high apparent confidence.

## How they judged success

Calibration of the approximate error distribution against the true one, measured by probability-integral-transform uniformity across simulated datasets, with a band for the variation expected under perfect uniformity. Supporting quantities: the ratio of the reported standard error to the true standard error of the error (target 1), the skewness of the error against the skewness of the sampling distribution, and the correlation between the estimate and the target. Head-to-head baseline: the Bayesian bootstrap, which the theory says should capture higher moments better; it is compared directly in every calibration figure. Analytic moments are compared to simulated ones as a correctness check on the derivation.

## What they found

- **The error is a quadratic form in the noise, and depends on no shared coefficient** — anything both accounts agree on drops out, so a comparison is governed entirely by what they disagree about.
- **Skewness is worst when accounts are similar, and for nested pairs it does not shrink as data grows.** In the one-covariate case the limit of the error's skewness is a fixed negative constant when the distinguishing effect is exactly zero, and zero otherwise; the relative mean of the estimate converges to a fixed non-zero value in the degenerate case too, so the simpler account is the asymptotically better one on average.
- **Skewness of the point estimates is not a usable proxy for skewness of the error.** The paper states this directly: the connection between the two distributions is weak, so a bootstrap or moment-correction applied to the pointwise terms cannot recover the shape of the error.
- **Both approximations — symmetric and bootstrap — are badly calibrated in the same settings**, and the bootstrap gives no practical improvement in any experiment run. Near the mode of the difference, the reported interval can be too narrow and the probability of superiority can point the wrong way; in the tail, too wide, which is at least harmless.
- **A single corrupted observation biases the estimate**, and no symmetric interval can be well calibrated against a biased centre — the paper notes that refining the variance estimate in that setting would make calibration worse, not better.
- **Small samples hurt in both directions** — sometimes overstating and sometimes understating the certainty of the sign, with the reservation that the direction of failure is not knowable in advance.
- **The practical recipe they land on**: treat a difference under a small absolute threshold as "no practical difference, use either"; above the threshold, do model checking before trusting any interval; for very small samples, multiply the reported standard error by a factor of two as a deliberately conservative gesture, and say plainly that the real fix is more data.
- **The three worked examples show the recipe in use**, each naming its own condition: a 17-observation comparison where the interval was widened and the probability of superiority dropped from 0.96 to about 0.81; a panel where two outliers made symmetric calibration implausible and the better-fitting distribution changed the estimate materially; and a count model where the difference was large enough that a misspecified competitor did not matter.

## Limits and open problems they name

The analysis is exact only for normal linear regression, and the other model families are simulation evidence rather than proof — the authors say so and point to singular learning theory as a speculative route to the general case. Other scoring rules are expected to behave similarly but are not tested. Only pairwise comparisons are analysed; many-account comparisons are deferred to the order-statistics line of work the authors cite. Leave-one-group-out and leave-future-out splits change the exchangeability structure and are explicitly left open. The advice about small samples is a heuristic inherited from an earlier bias bound, not a derived correction; the paper is candid that no fix from the pointwise terms exists, and that the only real remedy is more data or a better-specified comparison.

## What we take from it

- **A measure, and a warning that goes with it.** Any comparison we publish between two accounts — two probes, two dictionaries, two circuit hypotheses — is a comparison of *differences*, and this paper's core result is that the error of a difference is not the difference of the errors. Precisely: the shared part cancels, the unshared part does not, and the residual error is asymmetric in a way that does not average out as data grows when the accounts are close. That is the mechanism behind the thread this folder has been circling: a per-account criterion errs asymmetrically when the difference is latent.
- **A falsification test anyone can run.** Report the *ratio* of your stated uncertainty to the actual error of your estimate over resampled datasets, not just the estimate. If that ratio's median sits below one in the near-tie regime, your confidence numbers are decoration. The paper's own figure for this is the model to copy.
- **A rule for hedging.** When two accounts are within a small absolute difference, the honest report is "no practical difference, either will do" — not a probability of superiority with three decimals. When they are far apart, the comparison is safe regardless of calibration. The dangerous middle is where the interval must be inflated and labelled.
- **A baseline to beat before believing any fancier uncertainty machinery.** The bootstrap is the intuitive upgrade and it bought nothing here, because the quantity being resampled is the wrong one. Before adding a more elaborate resampling scheme to any of our reports, check it against this result: the thing that needs estimating is the *error*, whose distribution is not recoverable from the pointwise contributions alone.
- **A framing for the folder's next question.** If the difference between two accounts is latent — present in structure but nearly invisible in prediction — then this paper says the predictive criterion will systematically understate the evidence against the simpler account, and no amount of resampling will reveal it. Detecting latent structural difference needs a criterion that is *not* a prediction score, which is the gap to read into next.

## Tags

unit: whole-model
form: description
method: other
eval: prediction/simulation | benchmark
goal: evaluation-of-methods | understanding
