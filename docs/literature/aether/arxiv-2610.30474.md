# Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy

- **Key:** arxiv-2610.30474
- **Authors / organisation:** Riteng Zhang, Xiaoqian Wang (Purdue University)
- **Date:** 2026-10-21 (v1; v2 2026-10-23)
- **Link:** https://arxiv.org/abs/2610.30474
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

SAE evaluation grew by accretion: reconstruction error, dead-feature rate, L0, disentanglement,
auto-interpretability scoring and downstream task scores were each added because the previous set looked
insufficient, and nobody stated the objective the whole set is supposed to serve. The authors' claim is that the
original objective (monosemantic features) was never a property of a *feature set*, and that the drift toward
global metrics is an unstated search for one. Their move is to name a single objective — the learned features
should form an atomic interpretable unit basis, a coordinate system in which attribution and intervention are
semantically meaningful — and then *derive* the necessary criteria from it rather than proposing more metrics.
That derivation is the paper's main product, because it converts "which metric should I use" into "which property
of the concept-to-unit mapping is this metric a proxy for". The larger goal is the same one we hold: a
decomposition over which a claim about an internal computation can be stated, and a way to know when it cannot.

## What they studied

Models in the primary benchmark are small-to-mid (TinyStories, Pythia-70M and 160M, GPT-2-Small, OPT-125M,
Gemma-2-2B), and a supplementary sweep covers public Qwen3-8B/14B/32B and Gemma-2-9B SAEs. The unit is the
**feature** inside a dictionary, and the object evaluated is the dictionary as a whole. Coverage is large: 600+
SAEs across seven architecture families (Standard ReLU, TopK, BatchTopK, JumpReLU, Gated, PAnneal, Matryoshka)
plus 162 SAEs trained for the study, from SAEBench, JBloom OpenAI-v5 and CanRager releases, with 28,000+ features
sampled for the semantic metrics. A second setting is fully synthetic: sparse ground-truth concept sets with
210 SAEs at widths 128–1024, where the concept set is constructed so that correctness is known without a judge.

## How they did it

The framework first argues that natural units fail an injectivity requirement — a token like a proper name
carries lexical identity, entity role, logical-variable role and positional cues at once, and a single neuron
mixes superposed features, so neither maps concepts to units one-to-one. From the objective they derive five
necessary criteria: non-redundancy (C1, each concept has one unit), completeness (C2, every concept has a unit),
monosemanticity (C3, each unit has one concept), no dead units (C4), and per-input sparsity (C5, few units active
per input, which they derive from human usability rather than from the mapping). C2, C4 and C5 get standard
proxies (reconstruction fidelity, dead-feature rate, mean active features plus an effective-sparsity variant);
the methodological work is on C1 and C3, which are *local* properties that existing benchmarks average uniformly
into a dictionary-level number.

Their fix is an explicit local-to-global aggregation principle built on **dominance**: a feature's share of the
total activation on one input. Features are stratified into dominance quartiles (Q1 = most dominant) for both
sampling and aggregation, and dominance never enters a feature's own score. For C3 they replace binary
fire/no-fire judging with **DAME**: for each feature, held-out inputs are binned into low/medium/high activation
by that feature's own quantiles, the explainer's text is shown one example from each bin *without* activation
values, and a judge must rank the three by activation strength; the score is perfect-ranking accuracy. For C1 they
justify measuring redundancy on **decoder** directions (cosine above a threshold) rather than encoder rows, on two
arguments: sparsity already makes encoder redundancy uneconomic, because two features with identical encoders would
be merged, while decoder redundancy is not penalised by reconstruction-plus-sparsity at all; and encoder rows are
forced nearly orthogonal to the dominant mean direction by the sparsity constraint, so they cannot point at
semantics in the way decoders do. They check the proxy directly by embedding explanations and showing semantic
similarity rises monotonically with decoder cosine.

The redundancy statistic is an **endpoint enrichment**: each dominance quartile's share of redundant-pair
endpoints normalised by its share of active features, so enrichment above one means the quartile has more
redundant partners per feature than the dictionary average. Controls: five ranking rules compared on the same 80
synthetic features with generator-fixed truth, explanation-length budgets, firing-frequency matching, an
explainer-by-judge crossed check, prevalence-corrected held-out support containment on the synthetic setting, and
one worked intervention on a real pair of features.

## What counts as an explanation here

Not a mechanism. The paper's object is the **quality of the coordinate system an explanation is written in** — a
set of five checkable properties that a dictionary must satisfy before attribution or intervention over it means
anything. It is deliberately partial and explicitly framed as *necessary, not sufficient*, for causal use: the
paper says plainly that whether a feature is recruited by downstream computation still depends on context,
intervention and outcome. The one predictive claim in the entry is local and interventional (the sharding case
below); everything else is a measurement of the dictionary, not of the model.

## How they judged success

Per-criterion proxies with declared thresholds and robustness sweeps. C3: DAME perfect-ranking accuracy, compared
Q1 against Q2–Q4 per SAE with a Wilcoxon signed-rank test and Cohen's d, plus the fraction of SAEs individually
showing a Q1 deficit; robustness at 15/30/60-word explanation budgets, at three firing-frequency matching levels,
across two explainers × two judges on six SAEs, and on 210 synthetic SAEs by one-sided exact sign test. C1:
endpoint enrichment at a primary decoder-cosine threshold of 0.7, swept over 0.6–0.9, with 95% SAE-level bootstrap
intervals and a null of one; encoder similarity as the contrast. C2/C4/C5: standard proxies from prior work. Score
rules were validated on synthetic features with generator-fixed ground truth and permutation-balanced triplets, with
tie credit and a family-stratified feature bootstrap.

## What they found

Dominant features are worse on both semantic criteria. On DAME, mean perfect-ranking accuracy is 61.8% for Q1
against 64.1% for Q2–Q4, 65.2% of SAEs show the deficit individually, and it is significant in 5 of 7
architectures — the exceptions being Gated and JumpReLU, which the authors attribute to their default
hyperparameters giving those architectures very high effective sparsity. The per-architecture Q1-minus-Q4 spread is
largest for BatchTopK (61.5 against 69.9), TopK (59.2 against 64.6) and Matryoshka (61.6 against 66.5). Without an
LLM judge, on 210 synthetic SAEs, Q1 features score lower prevalence-corrected held-out support containment at every
width (0.082–0.094 against 0.104–0.108), and a correlation-gap analysis finds a significant deficit in 71 of 105
SAEs.

Redundancy is concentrated in the same quartile: median Q1 endpoint enrichment 1.6 (95% CI 1.51–1.74) with 71.5% of
SAEs above the null. In Standard ReLU SAEs, 26.1% of top-10 features sit in Q1 redundant pairs, with 4.7 pairs
inside the top ten alone, and 20.5% of top-100 features with 296.3 pairs inside the top hundred. Encoder similarity
runs the other way — Q1 features have the *lowest* mean encoder similarity (0.056 against 0.109–0.135), and even
among redundant pairs Q1 pairs have 0.428 against 0.661 — so decoder-aligned duplicates are not encoder duplicates.

The named failure mode is **feature sharding**: features with nearly identical decoder directions and dissimilar
encoders that fire on complementary regions of activation space, i.e. one concept split into "concept + unrelated
gating context" shards so that sparsity and dead-feature penalties are met while the effective semantic dictionary
collapses to far fewer directions. Their worked case is a TopK-65k Pythia-160M pair keyed to definite versus
possessive determiners: encoder cosine 0.2066, decoder cosine 0.7405, 316 and 57 firings with 7 co-firings
(Jaccard 0.019); injected at positions where neither gate fires natively, both writes moved the same target logit
(15/18 and 16/18 prompts positive) and the two induced effect vectors had cosine 0.943. Across decoder-aligned
pairs, 21.5% fire at largely separate positions (sharding-compatible) and 15.3% have one support largely inside
the other (absorption-compatible). Higher sparsity correlates with higher dominance concentration — 17 of 18
GPT-2-Small width–architecture cells — which the paper raises as an open question about whether sparsity buys finer
units or merely concentrates computation into fewer, more dominant ones.

The scoring-rule comparison is the measurement-methods result: on 80 synthetic features where generator truth fixes
the answer, DAME ranks the correct explanation above an otherwise matched one with only the high-versus-middle
order reversed for all 80, discretised magnitude scoring for 67, and binary and fuzz-style rules for at most 22.
Dominance itself is not small: even in dictionaries of 4,000–128,000 features the ten most frequently firing
features carry a median 7.8% of per-input activation, and the top thirty carry 13.0%.

## Limits and open problems they name

Scale: the primary benchmark is 70M–2.7B and the larger public-SAE checks do not establish scale invariance,
monotonic scaling or replication of the redundancy failure mode. Every LLM-judged monosemanticity number inherits
judge noise, including DAME's — they say the relative ordering is more trustworthy than the absolute values and
that this too is a proxy property. They argue ground-truth concept sets are conditional rather than unavailable
(available if the dictionary width is near the true concept cardinality and granularity is human-verifiable), and
that neither condition is certifiable for a real model. Hyperparameters were adopted from SAEBench configs, giving
very different effective sparsity across architectures, so the architecture-specific patterns (and the two
non-significant ones) could sharpen or reverse under a matched-sparsity sweep — their synthetic setting, which
controls sparsity directly, gives JumpReLU the strongest deficit even though it is non-significant on real models.

## What we take from it

A **prior** we can adopt wholesale: an attribution or intervention result over a dictionary should be reported with
its dominance stratum, because the pooled number is a mixture of a small set of highly dominant units and a long
tail, and the two behave differently. Concretely, three instruments to lift: (1) the dominance quartile as a
mandatory stratifier next to any dictionary-level score — this is the same shape as the run-231 lesson that the
low-energy tail carries the between-seed disagreement, arrived at from the opposite end, so a score that is
stratified in *both* directions is now the standard we should hold ourselves to; (2) **perfect-ranking accuracy on
a three-bin contrast** as a replacement for binary fire/no-fire judging, since binary support cannot distinguish a
correct explanation from one that gets the ordering backwards and the ceiling it reaches (67 of 80 even for a
cruder rule) shows how much information the binary framing throws away; (3) endpoint normalisation — divide a
group's share of bad edges by its share of nodes — as the general form of "is this group's rate worse than the
population's", which our stability and redundancy write-ups currently do not do.

Two warnings we should carry. **An explanation's quality should be weighted by how much of the representation the
unit carries**, otherwise a dictionary can pass a semantic audit on the tail while its dominant units are the
polysemantic ones — the mechanism behind our own repeated finding that the features most likely to be named first
are the least stable. And **sharding is a different failure from absorption** with an opposite positional
signature (complementary supports versus nested supports), so a single redundancy or absorption check cannot
detect both; the interventional case here is the template for telling them apart, because two decoder-aligned
shards *look* like one concept on weights and like two on activations. For our battery: before claiming a feature
change across seeds or a circuit edge is real, print the pair's decoder cosine, encoder cosine and support
overlap, in that order — the three columns together separate a duplicate from a shard from a genuine neighbour.

## Tags

unit: feature | direction
form: dictionary | probe
method: SAE | decomposition | probing | other
eval: human-rating | benchmark | faithfulness | downstream-task
goal: evaluation-of-methods | understanding
