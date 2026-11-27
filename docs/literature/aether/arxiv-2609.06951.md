# Steering Interference Reflects the Model's Defaults, Not the Behavior Directions

- **Key:** arxiv-2609.06951
- **Authors / organisation:** Srikanth Malla, Chiho Choi, Joon Hee Choi; Samsung Semiconductor US (per the affiliation line, correspondence malla@samsung.com)
- **Date:** 2026-09-07 (v1); v2 2026-09-16
- **Link:** https://arxiv.org/abs/2609.06951
- **Kind:** paper
- **Read from:** full text (arXiv HTML v2)
- **Relevance to us:** 3 (central)

## What they wanted, and why
Activation steering is sold as modular control: add a behaviour's direction to the
residual stream and that behaviour turns on while nothing else moves. The authors ask
what actually decides which *other* behaviours move when one is steered, and how far. The
answer they argue for is that the model, not the steered direction, sets the side
effects: any perturbation of a given size relaxes the model toward a small set of
behaviours it already falls back on. That reframes a steer from an edit whose side
effects follow from what was edited into a perturbation whose endpoint the model chooses,
and it implies that better-behaved behaviour directions will not by themselves make
steering modular. The larger goal is safety-relevant: if a safety edit silently inflates
unrelated behaviours, the control surface cannot be trusted without a cheap way to
predict the collateral.

## What they studied
Twenty instruction-tuned models' worth of behaviour space in ten checkpoints across four
families — Qwen2.5 at 1.5B/3B/7B/32B/72B, Llama-3.1-8B and Llama-3.3-70B, Gemma-2 at
9B/27B, Mistral-7B — with Qwen2.5-7B-Instruct (28 layers, hidden 3584) carrying the
detailed map. The unit is a *behaviour direction* in the residual stream (difference of
class means over contrastive response pairs, twelve triples per behaviour, 24 behaviours
spanning register, affect, interpersonal stance and content style). Analysis unit is the
20x20 ordered-pair interference matrix, or 380 off-diagonal entries; the paper explicitly
rejects activations as the readout and judges every effect from the generated text.

## How they did it
For each behaviour they build a difference-of-means direction at each layer from twelve
contrastive triples that share the user turn, score it on held-out folds by projection
AUROC against a label-shuffled control, and keep the best layer inside a fixed band (25
to 85 percent of depth). Each behaviour then gets its own steering coefficient,
calibrated in a loop with a language-model judge: sweep coefficients, generate on 32
neutral prompts, have the judge rate every generation on all 24 behaviours from 0 to 10,
and take the coefficient whose self-effect is closest to a two-point target subject to a
distinct-bigram coherence floor. A behaviour with no coefficient that clears a one-point
self-effect floor is dropped from the map. The interference matrix M is then filled in:
entry M_ij is the judge-measured change in behaviour j when i is steered at its
coefficient, standardised by a per-column pooled standard deviation and called
significant when a prompt bootstrap excludes zero. They split M into its symmetric part
and its antisymmetric part and argue the antisymmetric part is *unreachable* by any
account in which interference is a function of direction overlap. They then fit
geometry: 13 features of the two directions (cosine overlap, cross-layer projections,
norms, self-effects) into ridge and gradient-boosted regressors under
leave-one-behaviour-out cross-validation that removes a behaviour and all 38 of its
pairs. Finally they repeat the whole pipeline with random directions matched in added
norm and held to the same coherence floor, and with the two modified steers the geometric
account prescribes (components along other behaviour directions removed; activation
length restored).

## What counts as an explanation here
A model-level default account rather than a feature dictionary or a circuit. The
explanation is a *property of the network*: a small set of attractor behaviours (sinks)
that any sufficiently large perturbation of the residual stream drifts toward, plus one
ranking of behaviours from sources to sinks. It is deliberately partial — the authors say
most pairwise interference is captured by neither geometry nor the ranking — and it is
predictive only at one level: it predicts the *ordering and rough size* of interference
across behaviours for a push of any content, not the specific effect of steering a named
behaviour. It is a replacement model in the weak sense (a one-number-per-behaviour
additive fit replacing 380 cells) and a causal abstraction in a loose sense (a
high-level "default set" variable standing in for the low-level steering operation).

## How they judged success
Held-out projection AUROC with label-shuffled controls for decodability; judge ratings of
generated text (never probe projections) for every intervention effect; a distinct-bigram
coherence ratio as a model-free degeneracy guard; prompt bootstrap 95 percent intervals
over 32 evaluation prompts with 2000 resamples for significance; Pearson and Spearman
between random-push susceptibility and real-steer susceptibility as the central positive
result; leave-one-behaviour-out held-out Spearman and R-squared against a label-shuffled
null for the geometric predictor, with the in-sample-versus-held-out gap as an
over-fitting check; singular-value energy fractions for the low-rank claim; an
alternating-least-squares additive fit with a gauge fixing, ranked by a
gauge-invariant source strength; and a judge ablation in which a fixed stronger model
re-reads three models' saved generations with generation and coefficient held fixed,
compared by Jaccard on sets and Spearman on orderings.

## What they found
Linear decodability badly overstates control: all 24 behaviours reach held-out AUROC 1.00
(shuffled controls 0.26 to 0.62, mean 0.50) but only 20 change the text; the four failures
are confidence, verbosity, humor and anthropomorphism. Of 380 off-diagonal entries, 149
are significant where chance at this test count would give about 19, and the mean side
effect is 0.37 pooled-standard-deviation units against 2.07 for the mean self-effect. The
asymmetry is the headline: steering profanity moves toxicity by 3.4 standard deviations
while steering toxicity moves profanity by 0.0, and across the matrix the antisymmetric
part carries 0.68 of total entry magnitude on the norm scale (the authors' printed figure;
Table 5 puts the antisymmetric energy fraction at 0.46, which they state is the square of
the norm fraction — 0.46 is 0.68 squared, so the two numbers are one measurement). Three
patterns carry 0.71 of the matrix energy, a single pattern carries 0.54 of what is inside
the antisymmetric part, and a two-numbers-per-behaviour additive fit explains 0.31 of the
off-diagonal magnitude. Ten behaviours are significant sinks, led by sycophancy, refusal
and poeticism; six significantly resist, led by sentiment, politeness and formality.
A random push reproduces the sink ranking at rank correlation 0.90 (Pearson 0.92 on
magnitudes), induces toxicity at 0.16 (interval 0.07 to 0.28) but profanity at exactly
zero, and so explains the one-way pair as drift rather than as overlap. Geometry fails the
out-of-sample test: best held-out Spearman 0.41, no variant reaching held-out R-squared
0.10, a directional feature built for the antisymmetric part at -0.03 over 190 pairs, and
a boosted model at in-sample 0.66 against held-out 0.09; a three-dimensional concept cone
does worse than cosine alone on ordering (held-out Spearman -0.21, R-squared -0.030, so
worse than guessing the mean) and buys nothing when combined with cosine (0.27, 0.088);
and acting on the geometry (orthogonalised or
length-anchored steers) leaves side effects unmoved (-0.03 and +0.02) while cutting
on-target effect from 2.2 judge points to 1.2 and 1.8. Unsteered behaviour predicts
*resistance* rather than susceptibility: the fourteen non-resistant behaviours average
0.04 judge points on neutral prompts against 3.78 for the six resistant ones (rank 0.75,
n = 20, loosely — fourteen sit at zero). Across models the asymmetry (0.63 to 0.69) and
low-rank structure hold everywhere and geometry fails everywhere (held-out R-squared
-0.11 to 0.13), while the attractor correlation weakens toward each family's largest —
Qwen2.5-72B 0.54, Llama-3.3-70B 0.38, Gemma-2-27B 0.15 — and steerable counts rise from
13 of 24 at 3B to all 24 at Llama-3.3-70B. Under the fixed stronger judge the steerable
boundary (Jaccard 0.81 to 1.00), self-effect ranking (0.77), susceptibility ranking (0.74
to 0.83) and asymmetry fraction all reproduce, but top-sink membership is unstable
(Jaccard 0.33 to 0.71).

## Limits and open problems they name
The default set is model-specific, so which behaviours to watch must be re-measured per
model; why the pull weakens at the largest scale is unexplained; most pairwise
interference is captured by neither account; all ten models share the same 24 behaviours,
which were generated by a language model to a specification and then read by the authors;
the detailed map is one model reproduced on nine; each model judges itself, with a fixed
judge ablation covering three models rather than ten; and every readout is a judge score
on 48 greedy tokens over 32 prompts.

## What we take from it
Three instruments and one warning. (1) The **random-push null as a mandatory control for
any intervention claim** — a direction with no semantic content, matched in norm and held
to the same coherence floor, is the cheapest way to ask whether an effect is content
specific or just a perturbation; here it reproduces the whole collateral ranking at 0.90
while producing none of the lexically specific behaviours, which is exactly the shape of a
real effect plus a generic drift. (2) The **antisymmetric split of an effect matrix** as a
cheap unreachability test: any account whose quantities are symmetric under swapping the
pair predicts a zero antisymmetric part, so the size of that part measures how much of the
phenomenon the account cannot express even in principle — and measurement noise lands in
the symmetric part, so the antisymmetric part cannot be inflated by it. (3) **Held-out by
group, not by pair** as the cross-validation rule: because every behaviour appears in 38
pairs, holding out one pair lets a predictor succeed by learning that the target is easily
induced; only removing a behaviour with all 38 pairs tests transfer. The warning is that
decodability is not controllability — a probe separating every behaviour perfectly would
have certified 24 of 24 as steerable when four never move the text.

## Tags
unit: direction | layer
form: description | direction | causal-abstraction | replacement-model
method: steering | probing | ablation | other
eval: prediction/simulation | ablation | benchmark
goal: understanding | safety/auditing | evaluation-of-methods
