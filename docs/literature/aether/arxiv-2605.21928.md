# CausalGuard: Conformal Inference under Graph Uncertainty

- **Key:** arxiv-2605.21928
- **Authors / organisation:** Vikash Singh; Weicong Chen; Debargha Ganguly; Yanyan Zhang; Nengbo Wang; Sreehari Sankar; Mohsen Hariri; Alexander Nemecek; Chaoda Song; Shouren Wang; Biyao Zhang; Van Yang; Erman Ayday; Jing Ma; Vipin Chaudhary — Case Western Reserve University (affiliation from the HTML byline; the abs page prints none)
- **Date:** 2026-05-21 (v1, single version)
- **Link:** https://arxiv.org/abs/2605.21928
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Treatment-effect estimation from observational data needs an adjustment set, and which covariates are admissible depends on a causal graph nobody observes. The authors name the two available failures directly: commit to one estimated graph and misspecification silently destroys coverage, or ignore the graph entirely, wrap the estimator in a conformal calibrator, and recover the nominal rate only by paying a large width penalty. They want an interval that carries structural uncertainty inside the calibration step rather than padding it afterwards, so that when several adjustment strategies remain plausible the answer is hedged across them instead of being collapsed to one. The larger goal is a bad-graph insurance policy: full coverage even if the model's graph prior is wrong, with the interval tightening as the data speak.

## What they studied

Observational samples in the potential-outcomes framework, with a binary treatment, a scalar outcome, and a covariate set that includes a temporal-status label (pre-treatment, post-treatment). The unit of analysis is not a graph or a feature but the **adjustment strategy**: an ensemble of candidate directed graphs is collapsed, before any weighting, into the distinct back-door adjustment sets they license, and each surviving set defines one doubly-robust estimator. Benchmarks are IHDP, Sachs, ACIC2019, the LaLonde Jobs data, and Twins; ground-truth effects exist only for IHDP and Twins, so coverage means two different things across the table.

## How they did it

Four stages, all fitted on a training split and calibrated on a disjoint one. First, an instruction-tuned language model is asked, per ordered pair, whether one variable is a direct cause of another; the probability is read off the model's yes/no tokens and clipped, giving a Bernoulli edge prior. Second, candidate DAGs are drawn by sampling a topological order under the temporal-status constraint and then sampling each admissible forward edge independently, which makes cycles impossible by construction; each drawn graph is then *pruned* by a single-pass greedy conditional-independence test — for each edge, test conditional independence given the remaining parents of the child at a fixed significance level, and delete on failure. Third, graphs are reweighted by a BIC score plus the log Bernoulli prior, including the term for edges *absent* (they note that without it dense graphs win arbitrarily); graphs that induce the same adjustment set are merged so repeated proposal draws cannot count the prior twice. Fourth, for each surviving strategy a doubly-robust pseudo-outcome and a per-graph interval are computed on the training split, then the aggregate pseudo-outcome and aggregate bounds are formed by weighting, and **only then** a single split-conformal threshold is calibrated on the held-out split; the final interval is the weighted lower bound minus the threshold to the weighted upper bound plus it.

## What counts as an explanation here

A hedged interval plus a *decomposition* of the uncertainty, not a description of anything. The object delivered is a prediction interval whose finite-sample guarantee attaches to the aggregated pseudo-outcome, which they are explicit is a surrogate — noisy even with perfect nuisance models — and not the effect itself. Alongside the interval they report a structural-dispersion term, the posterior weighted standard deviation of the per-graph point estimates, and they are careful to say it is dispersion of *estimates*, not of edges: many structurally different graphs that induce the same adjustment set give near-zero dispersion, so a visually diverse ensemble can look certain.

## How they judged success

Coverage against one of two targets, always with width at the same nominal level: true-effect coverage where ground truth exists (IHDP, Twins), pseudo-outcome coverage elsewhere (Sachs, ACIC2019, Jobs) — with an appendix that says plainly that the theorem gives the second kind and the corollary, under extra causal assumptions, makes its conditional mean converge to the first. Baselines are conformalized causal forest and conformalized naive doubly-robust (graph-agnostic, so the comparison is efficiency at equal nominal coverage); non-conformal learners appear only to show they under-cover. Controls include a collider-injection stress test, an adversarial/flipped-prior test, an anonymisation test that renames every covariate to strip semantic hints from the language model, model-size scaling, a conditional-independence-threshold sweep, an ensemble-size sweep, and a cross-fitting comparison. Ten seeds for headline numbers, five for small-sample cells; appendix tables report mean ± standard deviation and, for the headline, the per-seed range.

## What they found

Mean coverage clears the nominal level on all five benchmarks (roughly 92.9 to 99.9 percent against a 90 percent target), with the smallest dataset showing the widest seed spread (ACIC2019, down to about 84.8 percent on a bad split) — a shortfall the authors say does not contradict the theorem because the guarantee is marginal over calibration draws, not conditional on a split. The efficiency story is the point: on IHDP the width is about 10.9 against about 17.8-18.0 for the conformalized graph-agnostic arms at equal nominal level, and on Jobs about 36,936 against about 74,875-82,127, while on Sachs and Twins the graph-agnostic arms are competitive or better (0.15-0.19 versus 0.181 on Twins). Injected-collider stress: forcing a post-treatment collider into the adjustment set drops coverage to about 82 percent with narrower intervals and tripled error, while the ensemble's temporal filter excludes it and keeps coverage near nominal. Graph recovery is poor (Sachs F1 around 0.10, 33 structural differences) and it does not matter downstream — the ensemble beats a conformal doubly-robust estimator given the *oracle* graph on coverage (95.7 against 92.8) at wider width, because averaging over plausible strategies hedges what a single graph cannot. Prior washout: at moderate sample size random, flipped, and anonymised priors all give essentially the same interval; the language-model prior's contribution is only visible in very small samples and in a synthetic set-up where the prior is given the true variable roles.

## Limits and open problems they name

The guarantee is for a surrogate: no finite-sample coverage for the true effect anywhere in the paper, only asymptotic alignment of the surrogate's conditional mean under assumptions the paper lists (a target-aligned adjustment class, nuisance stability, and score separation of the BIC). The method assumes causal sufficiency — no hidden confounding — and scales only to moderate graph sizes because candidate generation and reweighting are combinatorial; with a dense uniform prior on 54 variables, back-door identification became intractable and two ablation cells were simply not run. They also concede that the reported gain is only partly theirs: overcoverage (their own intervals sit well above nominal on IHDP) is blamed on the absence of cross-fitting inside the training split, and they chose not to fix it because the cross-fitted run cost roughly twice the compute for unchanged coverage and width.

## What we take from it

Three things, one method and two warnings. The **method**: when the answer depends on an unresolved discrete structural choice, collapse candidates by the *decision they induce* (here, the adjustment set), then aggregate and calibrate once — their proposition is a short convexity argument showing that calibrating the aggregate gives a threshold no larger than averaging per-candidate thresholds, so aggregate-before-calibrate is free and the reverse order pays a gap. The first **warning** is the coverage-target switch: their headline table mixes intervals scored against truth with intervals scored against their own surrogate, which is the same target-honesty demand earlier entries in this folder made, and the appendix admits the theorem covers only the surrogate. The second **warning** is the dispersion diagnostic: structural uncertainty measured as spread of *estimates* goes to zero when the structure collapses to one strategy, so a near-zero value is not evidence of structural certainty — for us the analogue is an ensemble whose distinct accounts all induce the same probe behaviour. Also worth stealing: pruning by a test only ever *deletes* candidates before weighting, so a true graph can be removed by a single greedy pass with no theorem noticing; and their honest negative result that a heuristic graph prior, once the likelihood is informative, buys nothing — which is the right way to report a prior.

## Tags

unit: weights
form: replacement-model | dictionary
method: attribution | probing | decomposition | other
eval: prediction/simulation | downstream-task | benchmark
goal: understanding | evaluation-of-methods
