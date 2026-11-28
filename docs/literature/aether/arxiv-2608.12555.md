# CAS: A Causal Attribution Score for Local and Global Explainable Artificial Intelligence

- **Key:** arxiv-2608.12555
- **Authors / organisation:** Michael Georgiades (Department of Computer Science, Neapolis University Pafos, Cyprus); Charalambia Varnava (CaSToRC, The Cyprus Institute)
- **Date:** 2026-08-12
- **Link:** https://arxiv.org/abs/2608.12555
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The paper's opening wedge is that predictive explanations answer a different question from the one a decision-maker usually cares about. A feature can top a SHAP or LIME ranking because it is a proxy for selection, a confounder, or a downstream marker, while an intervention on it may change nothing; conversely an intervention with a real effect can sit near the bottom of a predictive ranking. They want a score that keeps the object of explanation explicit — an identified intervention contrast — and that can be read both for one individual and for a population. The larger goal is a small, honest reporting vocabulary for causal attribution, not a new attribution formula: they state repeatedly that the Shapley value itself is unchanged and that the contribution is the layer built around it.

## What they studied

Unit of analysis is a declared **action set Q** (the interventions with a baseline and a target level each) for the core score, and a **pre-treatment covariate** for the empirical variant. Core score: q = 3 binary actions, so all 2^q = 8 coalitions are enumerated exactly, on a synthetic known-truth structural benchmark with six standard-normal covariates, three non-linear treatment-effect functions, three pairwise interaction functions scaled by s in {0, 1, 1.8}, an outcome noise SD of 0.8, and propensity-dependent (not randomised) treatment assignment. Empirical variant: two double-machine-learning datasets shipped with DoubleML — 401(k) eligibility versus net financial assets (n = 9,915) and a Pennsylvania reemployment bonus versus log unemployment duration (n = 5,099) — with 9 and 15 pre-treatment covariates respectively. No neural networks are studied; the outcome models are random forests.

## How they did it

1. Declare the interventions, their baseline and target levels, and the identification assumptions (consistency, conditional exchangeability, positivity, interference specification).
2. Define the interventional game v_x(S) as the expected potential outcome at profile x with target levels active for the coalitions in S, and the joint contrast as v_x(Q) − v_x(∅).
3. Apply the classical Shapley formula to that game to get raw contributions in outcome units, which sum exactly to the joint contrast.
4. Normalise absolute raw contributions by their total to get Local CAS (magnitude only), and keep direction in a Signed Local CAS whose entries sum to the joint contrast over the total absolute mass — a ratio in [−1, 1] that shows reinforcement versus cancellation rather than behaving as a probability.
5. Aggregate two different ways: an instance-balanced global score (mean of local scores, "which action is typically prominent") and an effect-mass global score (mean absolute raw contribution over its total, "which action carries the most causal mass in the population").
6. For the empirical work, estimate the effect with cross-fitting and an orthogonal AIPW score, and get an individual effect from a fold-specific learner; then explain the **estimated effect surface** rather than the outcome, by TreeSHAP-ing the exact fold-specific effect learner and normalising the resulting contributions (Feature-CAS).
7. Fairness device: pick the local example row using only the covariates (the observed row closest to the covariate median in robust IQR-scaled distance) so no explainer's output can pick the row; explain that same row with SHAP, LIME and Feature-CAS.
8. Stability: rerun the whole pipeline over five split/model seeds holding the row fixed.

## What counts as an explanation here

A **score architecture**, in two layers. The first layer is a raw allocation of a declared intervention contrast across declared actions, kept in original outcome units and summing exactly to the contrast. The second layer is a normalised report: percent of absolute mass per action, a signed version that preserves direction, and two population summaries that deliberately answer different questions. For covariates, the same allocation logic is applied to an already-estimated individual effect, yielding "effect-modifier attributions" which the authors explicitly refuse to read as "manipulating this covariate would do that". It is partial by construction: it explains one declared contrast at one profile, it is not an optimisation objective, and they say raw signed effects and outcome direction must be kept for any action choice.

## How they judged success

Core score: known-truth structural benchmark where the true allocation is computable, with raw contribution MAE, Local CAS MAE, and top-action accuracy (whether the arg-max contribution matches the truth), averaged over R = 8 seeds of n = 2,200 instances with descriptive 95% Monte Carlo intervals across seeds. Three comparisons: coalition-aware CAS against one-at-a-time normalisation and against a globally normalised |ATE| vector. An interaction stress test — a **separate repeated experiment**, not a subtraction of table entries — reports the paired MAE reduction at interaction strengths s = 0, 1 and 1.8, and the authors pre-commit to the requirement that the advantage vanish under additivity. Empirical: agreement of the transform (exact reconstruction of the stored effect to numerical tolerance), conditional row-bootstrap intervals with B = 500 over 301 displayed profiles, split-seed stability over 5 reruns, and qualitative comparison of rankings. There is no benchmark of whether the causal rankings are *correct* on the empirical data — by construction there is no truth there.

## What they found

Known truth: coalition-aware CAS has Local CAS MAE 0.107 [0.100, 0.114], against 0.173 [0.165, 0.182] for one-at-a-time normalisation and 0.213 [0.196, 0.230] for the globally normalised |ATE| vector; raw contribution MAE 0.321 / 0.500 / 0.617; top-action accuracy 0.803 / 0.775 / 0.762. The paired advantage over one-at-a-time normalisation is 0.066 [0.057, 0.075] at primary interaction strength and 0.091 [0.080, 0.103] under strong interactions, and −0.003 [−0.010, 0.003] with interactions removed — indistinguishable from zero, which is the control the design intended. The authors also show analytically that coalition-aware CAS reduces to the normalised individual effect under additivity, so the null is arithmetic rather than empirical.

Empirical: rankings diverge between predictive and effect-modifier explanations on both datasets. In 401(k), predictive SHAP's top three by mass are inc 0.315, pira 0.302 and age 0.138, while Feature-CAS puts inc 0.395, age 0.157 and educ 0.104 first, dropping pira from rank 2 to rank 7. In the Pennsylvania bonus data, predictive SHAP's top three are agelt35 0.207, black 0.157 and agegt54 0.131, while Feature-CAS ranks female 0.123, dep1 0.121 and lusd 0.108 highest — dep1 moves from rank 13 to rank 2 and is also the leading local modifier on the typical row, where SHAP ranks agelt35, lusd and q2 highest and LIME ranks black, agegt54 and agelt35. Feature-CAS reconstructs the stored effect essentially exactly on both rows (absolute error 3.71×10⁻⁵ with relative error 5.4×10⁻⁹ for 401(k); 1.10×10⁻⁹ and 1.46×10⁻⁸ for Pennsylvania), whereas LIME's surrogate misses the outcome model by 9.4% with local R² 0.417 on the 401(k) row and by 0.31% with local R² 0.808 on the Pennsylvania row — the authors present the gap as surface-dependent fidelity rather than as a defect. Split-seed stability: the 401(k) ATE is 7,915 ± 130 across five reruns but the typical-row individual effect is 7,218 ± 1,821; Pennsylvania is −0.07427 ± 0.00296 population against −0.1548 ± 0.1198 at the row, i.e. the population number is stable and the individual number is not.

## Limits and open problems they name

Ten pages, and the limits are stated rather than elaborated. The empirical comparison has no ground truth, so the demonstration is that two methods *disagree*, not that the causal one is right; the intervals are conditional on the fitted models and do not propagate refitting uncertainty; Feature-CAS is delivered by TreeSHAP on a random-forest effect learner, so it is exact for the fit but the fit's own validity rests on the DML identification assumptions, and the authors say so in a paragraph distinguishing "exact additive reconstruction" from exhaustive subset enumeration. Two datasets from one package's built-in examples, both binary-treatment and observational, both from US survey/experiment data of an earlier era. There is no experiment where an estimated causal attribution is checked against a real intervention rolled out — the only place truth exists is the simulation. The paper does not compare against causal-Shapley baselines from prior work beyond a citation, and does not test its own stability statement against the global rank ordering.

## What we take from it

Three things, in order of usefulness for us.

**A naming discipline for what a score is allowed to be used for.** The paper's contribution is not the formula but the refusal to let a normalised report stand in for the quantity it was normalised from: raw outcome-unit allocations are kept, the two global summaries are given separate names because they answer separate questions, and the signed version is explicitly *not* a probability. This is the same failure the folder has now recorded at least four times from the other direction — run 942's reversal rates, run 941's capture-then-collapse, run 938's three columns — and CAS is the cleanest statement of the fix at the level of a single score's documentation.

**A pre-committed null that the method is required to fail.** The additive scenario is designed so the coalition-aware method must show no advantage (and it shows −0.003 against a 0.066 / 0.091 advantage elsewhere), and the paper states the algebraic reason in advance: under additivity the coalition-aware allocation equals the normalised individual effect. That is a stronger control than a shuffled-label null because the null value is derived, not estimated, and the method's advantage is measured as a distance from a value the authors had to hit rather than as an improvement over a baseline. It is also the exact complement of run 942's no-effect arm scoring exactly 1.0: there, an ablation's honesty was checked by a structural null; here, a method's honesty is.

**A cheap fairness device worth copying verbatim.** Select the example to explain using only the inputs (the row nearest the covariate median in robust scaled distance), then explain it with every method under comparison — the paper is explicit that no outcome, prediction, attribution, ATE, CATE or score value enters the selection. Their motivation is that picking a row because one explainer finds it extreme is a free win for that explainer, and they report that the earlier version of the paper did exactly this. For us this is the local-explanation analogue of the standing rule about not selecting test artifacts: when we show a per-item attribution, the item must be chosen before the attribution is computed.

What we do **not** take: their empirical evidence says nothing about whether causal rankings are right, only that they differ from predictive ones; and the individual-effect instability they report themselves (7,218 ± 1,821 against a population 7,915 ± 130) is the strongest argument in the paper that individual-level attributions should be reported with their own spread rather than as point numbers.

## Tags

unit: feature
form: causal-abstraction | description
method: attribution | other
eval: benchmark | prediction/simulation
goal: understanding | evaluation-of-methods
