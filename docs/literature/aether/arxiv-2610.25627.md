# Conditional Average Treatment Effect Distillation for Hypothetical State-Intervention Targeting Under Budget Constraints

- **Key:** arxiv-2610.25627
- **Authors / organisation:** Vasundhara Acharya, Bülent Yener; Rensselaer Polytechnic Institute
- **Date:** 2026-10-19
- **Link:** https://arxiv.org/abs/2610.25627
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Public-health programmes that target modifiable risk states (obesity, elevated glucose, lifetime smoking
history) have to spend a fixed amount of intervention capacity, so they need a rule that ranks eligible people by
how much they stand to gain. Flexible causal estimators — causal forests, Bayesian causal forests — produce good
individual benefit estimates but are awkward to deploy and to describe; coarse subgroups are easy to deploy and
audit but may bury the individual-level variation the intervention is supposed to exploit. The paper asks whether
a *high-capacity causal teacher can be compressed into a simple student* and, crucially, whether the compression
preserves the *decisions* the teacher would make, not merely its predicted surface. The larger goal is a
white-box account of a black-box causal model that is adequate for the downstream use of the model.

## What they studied

Three observational analytics: PIMA Indian Diabetes (outcome: diabetes) used twice, for a BMI-state policy and a
glucose-state policy, and NHANES 2015–2020 pre-pandemic data (outcome: sleep disturbance) for a lifetime-smoking
policy; plus the semi-synthetic IHDP benchmark in 10 realisations, where true potential outcomes exist. The unit
is the individual decision (who gets shifted from the adverse state to the lower-risk state) and, secondarily, the
student model's approximation of the teacher's CATE surface. Teachers are a DML-orthogonalised causal forest
(EconML) and a Bayesian Causal Forest with a separate prognostic and treatment-effect forest (StochTree); students
are a single decision tree and a small random forest regressing on out-of-fold teacher CATE targets.

## How they did it

Pretreatment covariates were partitioned into adjustment variables and candidate effect modifiers using a
consensus causal graph discovered by an ensemble procedure in earlier work, then fixed before any policy learning;
the graph is a design input and its uncertainty is not propagated. Each analytic cohort was split 60% discovery /
40% held-out evaluation. On the discovery cohort the teacher is cross-fitted in five treatment-stratified folds
to produce out-of-fold CATE targets, a final teacher is fitted on all of it, and the students learn only the
deployable covariates mapped to those out-of-fold targets — so a student never sees treatment assignment or
outcome. Every policy, teacher or student, ranks eligible individuals (those already in the adverse state) by
predicted benefit, keeps only positive-benefit individuals, and allocates up to a budget of 70% of the eligible
population, with a fractional shift probability applied where the budget boundary cuts through a tie. Two
comparators use the same ranking and budget machinery but learn their partition differently: K-means on
pretreatment covariates alone (clusters ranked afterwards by estimated gain) and a causal tree with its leaf
count matched to the number of K-means clusters. All policies are frozen before evaluation. Two reference
policies are carried through every table: shift nobody and shift every eligible individual.

## What counts as an explanation here

A compressed decision rule plus a fidelity budget: a shallow tree or a small forest whose output *is* the policy,
readable as if–then paths over a handful of covariates, together with a set of numbers saying how far the
compressed object has drifted from the object it replaces. Notably the paper does not treat the student as a
causal estimator at all — the students inherit whatever causal content the teacher had — so the explanation is
explicitly partial: it is an approximation of a mechanism, not a claim about one. Its predictive content is about
*allocation*: which individuals the teacher would pick under a budget. The paper is explicit that the deepest
approximation errors sit where they matter most, right at the operating threshold.

## How they judged success

Policy risk and utility on the held-out cohort, from an intervention-specific cross-fitted doubly robust score
(propensities trimmed at 0.05, five-fold treatment-stratified cross-fitting), with 95% bootstrap intervals;
selected-set agreement by Jaccard; global rank fidelity by Spearman and Kendall tau-b plus prediction fidelity
R² of student against teacher CATE; and a **boundary-restricted Kendall tau-b computed within a ±0.05 quantile
neighbourhood of the teacher's budget boundary**. Eight paired bootstrap comparisons are pre-specified per
experiment and Holm-adjusted. Structural complexity is counted in terminal leaves, with the BCF measured on its
treatment-effect forest only. Four ablations: swapping the logistic outcome nuisance in the evaluator for
HistGradientBoosting, five independent discovery/evaluation splits, re-encoding the policy as a score threshold
extended over the ineligible, and the IHDP run with oracle comparison (true policy value, regret, PEHE, and
Jaccard overlap with the oracle top-q set).

## What they found

Over the three observational experiments, every distilled student landed within 0.011 absolute utility of its
own teacher, with allocation Jaccard between 0.738 and 0.937 — but no pre-specified paired comparison survived
Holm adjustment, so the ordering across policy families is descriptive only. The point-estimate leader moved
between applications: the causal tree was top for BMI (utility 0.7923 against the BCF teacher's 0.7901 and the
no-shift reference's 0.6733) and for glucose (0.7348), while on smoking K-means was highest at 0.7754 and the
causal-forest teacher followed at 0.7739. The global-fidelity numbers look better than the local ones: BMI
students reach Spearman 0.8315–0.8977 and Jaccard 0.7500–0.8472, while **boundary tau-b is weak or negative for
three of the four BMI students (0.1188, −0.1534, −0.0259, −0.1056)**; on smoking the sign of the boundary score
splits by teacher family exactly as the global scores split — the BCF students keep positive local ordering
(0.5985 and 0.6301, with Jaccard up to 0.9370) while both causal-forest students go negative (−0.3686, −0.2872)
despite positive global correlations near 0.77–0.78. The shift-all-eligible reference beats every learned policy
on utility in all three experiments (0.837, 0.7625, 0.7818), which is a property of the estimator under a
harmless intervention rather than a recommendation. On IHDP the BCF teacher is the best teacher (root-PEHE 1.8410,
true policy value 14.2940, regret 0.0727 against the causal forest's 3.2190 / 14.0385 / 0.3282) and the BCF
random-forest student is the best student (root-PEHE 2.4207, value 14.1724, regret 0.1943), with oracle top-q
Jaccard 0.8153 against the causal forest teacher's 0.7681. The two subgroup policies are clearly worse on IHDP —
regret 1.0980 for K-Prototypes and 1.2231 for the causal tree, oracle Jaccard 0.5421 and 0.6283 — and both fall
below the unrestricted treat-all reference's 14.0893. The ablations move utility by little: replacing the
evaluator's outcome nuisance shifts BMI utilities by between −1.69 and −0.30 points, and re-encoding the policy
threshold leaves allocation among the eligible unchanged with mean absolute utility changes of 0.0064 (BMI),
0.0044 (glucose) and 0.00004 (smoking).

## Limits and open problems they name

Observational, so everything is conditional on an assumed graph and the standard identification assumptions, and
the graph's own uncertainty is not propagated; NHANES weights are not applied, so the results describe the
analytic sample rather than the population; complete-case analysis drops people. The smoking exposure bundles
current and former smokers, which likely attenuates the effects. IHDP has degenerate overlap by construction, so it tests ranking and ground-truth value rather than
full-population identifiability. Subgroup admission uses point
estimates with no uncertainty gate, and K-means hard assignments carry no membership uncertainty. The students are
regressors that inherit any teacher misspecification. The authors also note that non-significance is not
equivalence, and that the random-allocation comparison is built from averages across runs rather than paired
per realisation.

## What we take from it

A **measure**, and the sharpest one in the folder for the sufficiency-versus-coverage problem: report agreement
restricted to the neighbourhood of the operating threshold, not only the global correlation. The paper's own
contrast shows why — a student with Spearman near 0.90 and boundary tau-b of −0.1534 is a faithful summary that
reverses the order of the two people who decide whether the budget runs out. A **framing** as well: the
compressed object's claim is that it preserves the decisions of the thing it replaces, and that claim is
falsifiable by a set of allocation-level numbers rather than by a surface-fit number; the leaf-count compression
figure alone is explicitly demoted, since a regularised BCF can have fewer leaves than its student. A **warning**
carried from elsewhere in the folder and printed here for free: the shift-everyone reference outscores every
learned policy, so a utility number is only meaningful against both references, and the paper's multivariate
criterion (utility, selected-set Jaccard, global rank, boundary rank, complexity) is the ordering we should use if
we ever nominate a cheaper surrogate for our own white-box model.

## Tags

unit: whole-model
form: replacement-model
method: decomposition | other
eval: prediction/simulation | benchmark
goal: understanding | control/editing | evaluation-of-methods
