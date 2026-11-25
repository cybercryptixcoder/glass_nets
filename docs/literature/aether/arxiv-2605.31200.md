# Beyond Additive Decompositions: Interpretability Through Separability

- **Key:** arxiv-2605.31200
- **Authors / organisation:** Jinyang Liu, Munir Eberhardt Hiabu; Department of Mathematical Sciences, University of Copenhagen
- **Date:** 2026-05-29 (v1); v2 2026-06-01; ICML 2026
- **Link:** https://arxiv.org/abs/2605.31200
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Nearly every interpretability method in wide use expresses a prediction as a sum of per-feature terms —
additive models, SHAP, functional ANOVA, partial dependence. The authors argue this is a modelling choice
with a specific cost: when a function contains a strong interaction, projecting it onto per-feature
marginals destroys the interaction's signature, and the resulting chart can be flat or misleading on exactly
the region where the feature matters most. Their goal is a *glass-box* predictor that is not additive in
the features, but is still exactly readable — a model whose fitted internal structure can be recovered from
the same one-dimensional curves a practitioner already knows how to read. The larger aim is to make
interaction structure first-class in interpretability rather than a residual that post-hoc methods try to
recover and generally fail to.

## What they studied
Tabular regression only: the OpenML Curated Regression 23 suite (27 datasets after filtering on
samples-times-features, excluding one for missing values), plus two interpretability case studies
(California Housing with its latitude/longitude spatial interaction, and Bike Sharing with an hour
by working-day interaction) and one synthetic problem built to contain a masked third-order interaction.
The unit of analysis is a **stage** — a separable rank-1 product of univariate per-feature functions, or
rather the difference of two such products. Nothing here is a neural network; the models compared are
tree ensembles (XGBoost, LightGBM, Random Forest), an Explainable Boosting Machine, an unconstrained
separable alternating-least-squares baseline, and their own estimator.

## How they did it
Each stage of the proposed model is a difference of two *positive* rank-1 products, so every component
function is a non-negative univariate function of one feature and the stage can only gain or lose
magnitude in one direction depending on which branch dominates. All components are piecewise constant on
a per-feature grid that is refined by CART-style greedy splitting: for each candidate split the model solves
a small ridge-regularised least-squares system for the multiplicative update to each branch, scores the
split by the error reduction *at the clamped update it will actually apply*, and keeps the best one.
Stages are fit forward, each against the residual of the previous ones, and after every stage all stage
scalars are refit jointly by least squares over the residual — the orthogonal-greedy pattern. Bagging is
used for variance reduction, but because averaging products is not a product, they average the per-feature
factors instead, and because factors fitted independently need not live in a comparable gauge they
normalise each bag, pick the bag nearest the centroid as a reference, discard bags whose backbone and
tilt profiles fall below a cosine threshold, and average the survivors. A backbone/tilt reparameterisation
splits each stage into a shared positive magnitude and a signed imbalance, so a stage can be read as
"how strongly is this stage switched on" separately from "which way does it push". The claimed training
cost is linear-to-modest in each of: stages, bagged grids per stage, split iterations per grid, samples
and features — the bagged grids are embarrassingly parallel and histogram binning cuts the per-iteration
sample dependence.

## What counts as an explanation here
A fitted separable model plus an identity, not a post-hoc surrogate. Because marginalising a product over
all but one coordinate rescales it by a constant, the one-dimensional partial dependence function of a
stage on one coordinate is exactly that coordinate's factor shape times a constant. So the 1D partial
dependence curves — the standard diagnostic plots — are, for this model class, *exact* pictures of the
fitted components up to one scalar per stage per branch, and the stage is reconstructed from them. The same
holds for backbone and tilt. The explanation is a set of per-stage curves with a magnitude summary that
cannot cancel and a direction summary that can, plus an additive decomposition over stages that sums
exactly to the prediction. It is complete with respect to the model, and partial with respect to the
function: the stage count is capped, and a low stage count is a claim about the data, not a theorem.

## How they judged success
Two separate yardsticks, and the paper is explicit that they are separate. Predictive: test RMSE on a
single 80/20 split after 200-trial Optuna tuning with 10-fold cross-validated model selection, against
tree ensembles at two depth budgets and an EBM, with rank counts within the interpretable group and within
the black-box group. Interpretive: whether the plots say something true. The synthetic problem is the
sharp instrument — the data-generating function contains a quadratic term reached only through a
third-order interaction whose expectation is exactly zero, so the *population* 1D partial dependence for
that feature is identically flat by construction; any fitted model's partial dependence plot should
therefore be flat, and the question is which model can still show the effect. An ablation at matched
total separation rank (same number of products, positivity and the ordered difference disabled) isolates
how much of the representational power comes from those two constraints rather than from the product
form itself. Approximation theory is judged separately: an orthogonal-greedy approximation rate for
functions with a bounded highest-order mixed derivative, in a dictionary of normalised non-negative
rank-1 products.

## What they found
The estimator is competitive rather than dominant, and the paper says so. Over the 27 datasets it places
in the interpretable top three on 17 and is the best interpretable model on 5; the unconstrained separable
baseline wins 13 and the EBM 5, so the separable family as a whole beats their variant on aggregate
accuracy — their claimed advantage is the per-stage readability, not the error. The pattern of wins is
legible: the largest margin is on a dataset with known log-additive structure, where the estimator posts
an RMSE around 9.5 against roughly 17.7 to 21.7 for the tree ensembles and the EBM, and it beats the
unconstrained baseline on California Housing at matched rank budgets (about 49.4 thousand against 62.2
thousand, which they quote as roughly a 21 percent reduction). At matched separation rank the ablation
without positivity and without the ordered difference is clearly worse on several datasets, so the
positive-factor pair structure buys real flexibility rather than only interpretability. On the synthetic
masked interaction every model's signed 1D partial dependence is near zero, matching the population
identity, and only the backbone — the shared magnitude rather than the signed difference — retains a
readable quadratic shape. In the California case the second stage's multiplicative backbone gates onto
the inland regions where the first stage's product over-predicts, and supplies the negative correction;
the 2D partial dependence surface makes the mechanism legible as gating rather than as a subtraction.

## Limits and open problems they name
The approximation guarantee is an approximation rate only — no finite-sample statistical rate, no
consistency. Identifiability is the stated principal limitation and it is threefold: the usual tensor
scaling and permutation ambiguities, an ambiguity arising specifically from non-rectangular support
(region-dependent rescaling of two factors that leaves every observed prediction unchanged), and an
ambiguity from fitting against a noisy objective, so that different random bags converge to genuinely
different factorisations of the same stage. Positivity removes sign ambiguity and stabilises averaging but
does not remove the other two. The aggregation therefore discards bags that disagree with the reference,
so only a fraction of the fitted bags contribute, and across-seed variance is high; aligning all bags on
the positive rank-1 manifold is named as future work. On largely additive data the EBM remains stronger.
The theory also carries a caveat the authors make explicit: the dimension-free approximation rate is not
an escape from the curse, because the target class tightens with dimension — the required order of mixed
differentiability grows with the number of features.

## What we take from it
A **framing and a warning**, and the warning is the transferable part. The synthetic construction is a
ready-made unit test for any decomposition we build: choose a function whose interesting coefficient sits
behind an interaction whose expectation is zero, and then check that the quantity we design to carry
magnitude does not vanish where the standard marginal does. Concretely for us: **when our white-box model
summarises a component by a signed net number, it inherits an arbitrary zero — several positive components
here have no absolute sign at all, only a joint one, and the paper shows a rank-1 fit in which each
individual factor looks negative while the prediction is positive because four signs cancel.** The
mechanical rule we take is that every component we report carries a *magnitude* summary that cannot cancel
by construction (the analogue of the backbone) alongside the signed one, and we say explicitly which
gauge fixed the sign. Second, from the identifiability section: **a decomposition fitted independently
several times is not one decomposition**, and averaging the pieces without a gauge-alignment step produces
a curve that is neither the fit nor the truth — a direct caution for any bagged or seeded decomposition we
report. Third, from the cost statement: the separable form is the cheap escape from exponential
subset enumeration that run 744 left open on the theory side, but it buys that at the price of a *cap on
the number of products*, and the paper's own results show the cap costing accuracy — so an exact
decomposition and a cheap one are not interchangeable just because both are additive sums.

## Tags
unit: feature
form: replacement-model
method: decomposition
eval: benchmark
goal: understanding
