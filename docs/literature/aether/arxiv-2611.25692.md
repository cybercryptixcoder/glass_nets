# Measurable Does Not Imply Transportable: A Prospective Audit of Model Explanations in Multi-Year Field Trials

- **Key:** arxiv-2611.25692
- **Authors / organisation:** Rayhaneh Shabani Nia, Ali Karkehabadi; University of California, Davis
- **Date:** 2026-11-19
- **Link:** https://arxiv.org/abs/2611.25692
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Scientists increasingly cite model explanations as evidence, but an attribution is usually validated
only against the model that produced it, or against resampled data from the same period. The authors ask
a different question: when an explanation was built from past data and is then confronted with a
genuinely later context, does its *structure* come back? They name this **prospective explanation
transportability** and separate it from three neighbouring quantities - how important a feature is for
predictions, how reliably the scientific target itself is measured, and how reproducible the
explanation is when the data are resampled inside one context. The motivation is the common inference
that a failure to replicate a fine-grained explanation must mean the target was noisy; if reliability
does not track transport, that inference is void and explanations used as scientific claims need a
validation axis that no faithfulness or stability check supplies.

## What they studied

Multi-year Genomes-to-Fields maize field trials: 107,399 hybrid-by-environment observations across
roughly 270 year-location environments with 2,425 SNP markers for the historical analysis, plus a
locked 2024 evaluation of 9,486 observations from 1,063 hybrids in 22 eligible environments. Inputs are
reduced to 16 prespecified agronomically interpretable environmental concepts (heat, water availability,
atmospheric dryness, thermal accumulation, soil-water context, canopy conditions). Two explanation
resolutions, i.e. two scientific targets: the **coarse** one is mean yield in an environment; the **fine**
one is a genotype-conditioned response - the within-environment slope of yield on genomic PC1, with the
PCA fitted on the historical reference and frozen. The unit of analysis is the environment (year plus
location), not the plot, because plots within an environment are not independent future contexts. A
second phenotype, Grain Moisture, is used as a boundary case.

## How they did it

For each historical evaluation they build an explanation vector from strictly past years and compare it
only with the next year: 2014-2018 against 2019, and so on through 2018-2022 against 2023. The vector is
16-dimensional and signed; its **directions** come from standardised environment-level regression on the
concepts and its **magnitudes** come from nonnegative concept-usage weights taken from an
XGBoost/TreeSHAP model fitted to the same target, so the vector is not simply an average of signed SHAP
attributions. Transport is the cosine between the reference-year vector and the independently estimated
target-year vector. Reliability is measured on the target alone: hybrids inside each environment are
split in half, the PC1 slope is estimated in each half, the two sets of slopes are correlated
(split-half), and Spearman-Brown correction gives the full-sample figure. Within-context
reproducibility is measured by resampling the data that build the explanation and taking the cosine
between two independently resampled explanations from the *same* context. Before the 2024 evaluation
the design was frozen: the 16 concepts, the PC1 target, the transport statistic, a bootstrap routine of
200 balanced resamples of environment-level rows with seed 42, the 16 expected main-effect signs, and
a numerical main-effect forecast of 0.831 with an 80% percentile interval of [0.755, 0.942]. No numerical
forecast was made for the fine resolution because historical transport was too heterogeneous. One 2024
run failed for a software reason before producing any preregistered metric; the repair changed no
estimand. Secondary, explicitly post-hoc tests then compare three validation protocols (random
environment splits ignoring time, leave-year-out, strict forward), swap the explanation family for
standardised Ridge coefficients, check five XGBoost weighting seeds, and add four alternative similarity
metrics.

## What counts as an explanation here

A vector over domain concepts, at a stated resolution - a main effect or a genotype-conditioned
interaction - and the validation object is not the vector's plausibility but the *angle between two
such vectors built in different time contexts*. The authors are explicit that the number is a measure
of preserved direction, not a causal mechanism, and that a high cosine licenses only "the explanation
recurred", not "the relationship is real". Transport is therefore treated as a claim-level property:
the same data support a coarse claim and a fine claim with different evidential weight.

## How they judged success

Cosine similarity per forward split against the immediately following year, reported as a median over
five splits; a locked 2024 value checked against a prespecified forecast and a bootstrap percentile
interval with an absolute forecast error; split-half reliability and its Spearman-Brown correction;
within-context reproducibility from independent resampling; a prespecified heat-direction control and
16 prespecified concept signs, with failures retained; and protocol-comparison "validation optimism
gaps" (random/leave-year-out minus strict-forward), which the authors deliberately describe as a gap
rather than as bias because the estimands are not identical.

## What they found

For grain yield the coarse explanation transported with a median cosine of 0.831 across the five
historical splits (range 0.736 to 0.962) while the genotype-conditioned one had a median of 0.295 and
ranged from 0.006 to 0.822; the median gap was 0.536. The locked 2024 run reproduced the ordering -
0.811 coarse and 0.271 fine, gap 0.540 - with the coarse value inside the frozen interval and an
absolute forecast error of 0.020; the heat-direction control passed and 14 of 16 frozen concept signs
matched (the two misses were grain-filling water, expected negative and observed positive at +0.892,
and midseason soil-water storage, expected negative and observed +0.247). Reliability did not order the
outcomes: the fine target's pooled split-half reliability was 0.554 (0.713 after correction), yet 2019
combined modest reliability (0.352) with strong transport (0.822) while 2020 combined higher
reliability (0.798) with transport of 0.006. Within-context resampling put reproducibility at 0.77-0.95 across the 2020-2023
transitions against 0.01-0.36 prospective alignment (year pairs: 0.87-0.95 versus 0.01, 0.86-0.91 versus
0.36, 0.87-0.91 versus 0.30, 0.77-0.81 versus 0.18). Non-temporal validation was optimistic for the fine
resolution only: random-environment splits gave 0.512 fine against 0.295 strict-forward, a gap of
+0.217 (leave-year-out +0.154), while for the coarse explanation random splits gave 0.793 against 0.831,
a gap of -0.039. Grain Moisture transported strongly at both resolutions (0.907 and 0.859, gap 0.048),
so the failure of fine-grained transport is phenotype-dependent rather than a consequence of
conditioning on genotype. Under Ridge the absolute values fell (0.588 and 0.227) but the ordering held
(gap 0.361), and the ordering survived five weighting seeds for 2020-2024 (2024 mean values 0.809 and
0.265, 5 of 5 seeds in each year) though not for 2019, where the fine explanation scored higher
(0.870 against 0.777, 0 of 5 seeds). Full-data alternative metrics in 2024 all preserved the ordering:
unweighted cosine 0.789 against 0.145, absolute-importance Spearman 0.388 against -0.697, top-five
Jaccard 0.429 against 0.000, and SHAP-weighted sign agreement 0.831 against 0.549.

## Limits and open problems they name

One crop resource, five historical future-year evaluations, and correlated environmental concepts;
genomic PC1 is a statistical axis, not a mechanism; small environment counts constrain precision; the
whole study is observational. The authors are explicit that the audit does not identify *why* transport
fails - covariate drift, model refitting, and estimation variability are all open - and that the
validation-gap comparisons are not perfectly matched estimands. The seed sensitivity check covers the
weighting stage only, not the full pipeline, and the appendix tables are transcribed from analysis
outputs rather than independently reproduced. Year 2019 is a standing counterexample: the fine
explanation transported better than the coarse one there and reversed under every weighting seed, which
is stated as a retention rather than smoothed away.

## What we take from it

A measurement protocol we can copy directly, a warning we need, and a vocabulary. (1) **Protocol:** when
our white-box model is supposed to predict behaviour in a setting it was built from, the honest test is
a *forward* split - find the parameters on one set of contexts and ask whether the same explanation
emerges in a context the fitting never saw - with the split and the metric frozen before the held-out
value is looked at, and a numerical forecast committed in advance with an interval. (2) **Warning:** the
paper's central result is that reliability, within-context reproducibility and faithfulness are all
satisfiable while transport fails, and that non-temporal resampling inflates the apparent transport of
*interaction-level* explanations by about a fifth while barely moving main effects - so any check that
resamples inside one context is, for us, a claim about instability of our estimator and not a claim about
the mechanism. (3) **Vocabulary:** coarse/main-effect versus interaction-level explanations have
separate evidential regimes; we should report transport per resolution instead of one aggregate number,
because a single "explanation stability" score would have averaged 0.831 and 0.295 into exactly the
number that hides the finding. (4) **Cheap boundary test:** the Grain Moisture row is the pattern to
copy when we want to know whether a failure is structural - pick a second target with the same pipeline,
choose it by a coverage rule fixed before its results are seen, and see whether it also fails.

## Tags

unit: feature
form: direction
method: attribution, decomposition, training-interpretable, other
eval: prediction/simulation, benchmark, faithfulness
goal: understanding, science, evaluation-of-methods
