# Confidence Without Cause: Predicting and Localizing Physics-Shortcut Sensitivity in Language Model Hidden States

- **Key:** arxiv-2611.19403
- **Authors / organisation:** Shubham Mishra, Atharv Sharma, Dhruv Kumar; Birla Institute of Technology and Science (BITS), Pilani
- **Date:** 2026-11-14
- **Link:** https://arxiv.org/abs/2611.19403
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1; figures not inspected)
- **Relevance to us:** 3 (central)

## What they wanted, and why

If a model's confidence on a physics problem falls when a physically irrelevant detail changes, that
confidence is not tracking the physics, and the failure is invisible in accuracy because the answer can
be right on all four versions. The authors want three things at once: a ground truth for that kind of
sensitivity that is certified rather than assumed, a way to detect it *before* the model generates
anything, and a component whose removal prevents it. Their complaint about the existing literature is
methodological and precise — perturbation benchmarks built by templates or heuristics cannot separate
instability in the model from an unverified change in the task, and a sensitivity label costs several
queries per problem to observe, which makes it useless as a deployment-time signal. Certification is the
move that resolves the first problem: prove with a computer algebra system that the cue symbol is absent
from the solved expression for the answer, and then any change in output across the family is the
model's. That is a statement about a *set* of prompts rather than a single prompt, and predicting the set
label from one prompt's hidden state is the bridge between the two.

## What they studied

Three open-weight chat models: **Qwen2.5-7B-Instruct** (28 layers, d = 3584; the primary), **Llama-3.1-8B-Instruct**
(32 layers, d = 4096) and **DeepSeek-R1-Distill-Qwen-32B** (64 layers, d = 5120; the reasoning-trained
comparison). The data is a purpose-built benchmark, **DECOY**: 155 certified *families* of four prompts
each (620 prompts), where a family is one symbolic physics template instantiated with four values of one
designated *cue* — an irrelevant detail — and a solver certifies that the correct answer is identical
across all four. Three cue types: an added physically irrelevant variable (Cue A, 35 families), a
distractor with the same units as the answer which the stated setup makes irrelevant (Cue B, 105), and the
same quantity in equivalent units (Cue C, 15). Three physics domains (mechanics 91, electrostatics and
circuits 43, thermodynamics 21). The unit of analysis is macroscopic and residual-stream level: for the
behaviour label, an attention-*family* of prompts; for the monitor, the residual-stream vector at a single
layer and position; for the causal claim, an attention **head**. No weights are read; nothing is described
neuron-by-neuron or feature-by-feature.

## How they did it

Every family begins as a symbolic template — relevant variables, stated assumptions, governing law, solver
answer — written *before* any text exists. The certificate is a machine-checked free-symbol test: solve the
governing law for the answer, collect the symbols that survive substitution of the assumptions, and assert
the cue symbol is not among them; for the equivalent-units cue the invariance is instead an exact identity
between the transformed and untransformed expressions. Three further checks complete each certificate (the
expression evaluated at the stated parameters matches the declared answer to a relative tolerance, every
parameter is finite and physically plausible, the declared answer passes the same parser later applied to
model output), and the released artefacts include verifier mutation testing: 330 mutation records over 30
families, 300 verified against expectation, with 30/30 correct acceptance of algebraically equivalent
rewrites and 30/30 correct rejection of cue-leakage mutations; three operators whose names misdescribed what
they executed were removed and the removal recorded. Language models are allowed to paraphrase prompts and
rename variables, never to write physics or answers.

The behaviour label is read from a single teacher-forced pass per rendering. For each family the authors
score the log-probability of the *certified answer* on all four prompts and take the largest drop relative
to the template's fixed base rendering, in nats; an ideal law-identifying solver has four equal
log-probabilities and a drop of exactly zero. A family is sensitive when that drop reaches **0.5 nats**, a
threshold registered on a date *before* any probe was trained, chosen from a 30-family pilot. Two
alternatives are defined and not used as the primary: the answer-flip rate (fraction of rendering pairs
whose parsed answers differ) and a Jensen–Shannon divergence between answer distributions; the authors say
why — the log-prob label needs no generation, is independent of temperature and sampling seed, is signed so
that a confidence *rise* under the edit is not counted, and is exact rather than approximated.

The monitor is L2-regularised logistic regression on the residual stream at the last prompt token, averaged
over a family's four renderings, swept over all layers, with regularisation and standardisation chosen
inside each fold, evaluated **leave-one-family-out** — all four renderings of a family, and every paraphrase
or renamed copy, stay on one side of every split. Eight non-activation comparators are fitted on the same
labels, and a generic hidden-state *correctness* probe is reported beside them because sensitive problems
are also often hard ones. Statistics are a family bootstrap (10,000 resamples, seed 42) resampling whole
families, and a **paired** bootstrap that scores both competing predictors on the identical resample and
puts the interval on their difference. Finally, on a panel of 20 strongly sensitive Cue B families the
authors knock out four attention heads at a chosen layer, across all prompt positions, and recompute the
family-level drop; the effect is a *normalised recovery* — change in drop divided by the original drop —
with six controls: twenty equal-norm random perturbations at the same site, the same knockout at another
layer, an unrelated-family donor, a same-answer donor, a cue-bearing but *stable* donor, and accuracy on 40
held-out families to detect damage. Instrumentation is gated before any experiment: a deliberately chosen
impossible intervention is run first and must actually move the logits by a large, measured amount.

## What counts as an explanation here

Three partially independent objects, explicitly separated by a "three-level" scheme (prompt condition →
observed behaviour → internal state) whose claim chain is stated as Level 3 predicts Level 2, Level 2
correlates with Level 1. (i) A *certificate*: a symbolic proof stored per family that the cue cannot change
the answer, which makes the behavioural label a property of the model rather than of the task. (ii) A
*probe*: one linear direction over an averaged residual-stream state that predicts, before generation,
whether the family will be sensitive. (iii) A *circuit fragment*: a small set of attention heads at one
layer whose knockout removes the behaviour. The paper is explicit that these are not the same object: the
probe direction is *partly sufficient* (injecting it raises the drop steadily but only a third of stable
families cross the threshold even at twice the probe's own scale), the heads are *necessary* (removal
recovers the drop fully, and generalises to a held-out panel of one cue type but not another), and the
authors say outright that the layer where the probe peaks does not by itself locate a circuit — the knockout
at the matching depth in DeepSeek recovers almost nothing. Prediction is claimed only for the probe, and only
at the level of a binary family label; nothing predicts a per-unit effect.

## How they judged success

AUROC with family-bootstrap intervals under leave-one-family-out; AUPRC, Brier score, expected calibration
error and the false-negative rate at a fixed 10% false-positive rate; a *paired* bootstrap difference
against every fitted comparator; calibration against threshold sweeps from 0.1 to 2.0 nats; a Level-1
control probe trained to predict the cue value instead of the behaviour (scored 0.532, i.e. the label has no
surface signature of the edit); a correctness-residualised probe; a frozen probe scored on 15 families built
*after* training; benchmark-integrity audits (surface-feature classifiers, domain and cue-type association
tests, an exact/near-duplicate leakage sweep over all 620 prompts); a permutation-shuffled null for
domain transfer; and for the causal claim, six controls plus 20/20-per-family specificity against the
random-direction control and an accuracy-damage check with a tolerance fixed in advance.

## What they found

Behaviour: Qwen2.5-7B-Instruct is sensitive on **70 of 140** labelled families (50.0%, interval
[42.1, 58.6]), mean drop **1.401 nats**, maximum **13.83**; Llama-3.1-8B-Instruct on 32 of the same 140
(22.9%), and only **25** families are sensitive for both — so the property belongs to a model–problem
pairing rather than to the problem. Cue B has the highest rate (53.0% against 34.3% for Cue A), and the
largest effects sit in a narrow near-match tail: a 6.9 N·m distractor beside a 6.928 N·m answer gives
10.03 nats, 1.6 Hz beside 1.592 Hz gives 5.56, and an irrelevant 18 °C wall temperature numerically equal
to an 18 N answer gives 2.30 nats — while the graded correlation between proximity and drop over the panel
is weak (Pearson 0.074 over 100 families). DeepSeek-R1-Distill is sensitive on 27.1%, and its suppression
is selective: 36 of 53 Cue B positives suppressed against 0 of 5 Cue C positives (Fisher p = 5.7e-3).
The worked family shows why the design matters: on one torque family Qwen answers wrongly on every
rendering, so it is sensitive without being correct — a case accuracy-based benchmarks cannot see.

Monitor: the layer-18 mean probe reaches **AUROC 0.760** [0.675, 0.838] with families held out; a
single-rendering probe 0.747; a correctness-residualised version 0.738 [0.650, 0.819]. It beats every
text or behavioural comparator by a wide margin (TF–IDF on the prompt 0.651, chain-of-thought classifier
0.599, zero-shot LLM judge 0.480, black-box two-query check 0.668) — but **a fitted token-entropy baseline
on the same forward pass reaches 0.727, and the paired difference +0.033 has interval [−0.074, +0.141]**,
i.e. statistically indistinguishable at this sample size. The authors print this as their headline
methodological result rather than an embarrassment, and recommend the paired fitted comparison as the
default standard for hidden-state monitors, because against surface or random-direction baselines the
same probe would appear to win by 0.11–0.19. The combination of hidden states with the nine raw
non-activation features gains +0.067 [−0.001, +0.137], also through zero.

Causal: knocking out heads {26, 24, 13, 11} at layer 16 gives mean normalised recovery **1.870**
[1.560, 2.198] on the 20-family Cue B panel (0.969 with each family capped at 1.0 — the uncapped number
exceeds 1 because the intervention *inverts* the drop). Head 11 alone carries 1.848 and the other three
combined add 0.022; the window is one layer wide (0.540 at layer 15, −0.049 at layer 17); every subset
containing head 11 recovers 1.71–2.01 and no subset without it exceeds 0.24. Controls separate it from
generic disruption: equal-norm random perturbations 0.021 (exceeded on 20 of 20 families), random-site at
another layer 0.374, unrelated-family donor −0.037, same-answer donor 0.405, stable cue-bearing donor
0.590 (from which they compute that ~68% of the effect is specific to sensitivity), and accuracy on 40
held-out families moves 0.700 → 0.713, inside a pre-set 5% tolerance.

## Limits and open problems they name

Sample size: 135 families with 70 positives resolve 0.11–0.28 margins but not 0.03–0.05 ones, so the
probe-versus-entropy comparison is unsettled. The layer-18 headline is *the best of a 28-layer sweep* and
the held-out intervals are not adjusted for that selection. The causal result is established in Qwen only;
normalised recovery divides by the original drop and is heavy-tailed when that drop is small, and in Llama
the multi-head arm's interval spans zero while in DeepSeek the matching-depth knockout gives 0.086.
Coverage: Cue C has five labelled families. Held-out families share a template cluster with training
families in **71%** of cases, so leave-one-family-out is not leave-one-cluster-out; holding out whole
domains gives 0.750 against a permutation-shuffled mean of 0.734 (p = 0.368), so domain generalisation is
not established — and on the frozen-probe transfer check the pooled 0.820 over 15 new families collapses to
0.667 and 0.6875 *within* cue type, which the authors diagnose exactly: a between-group score offset
aligned with a between-group base-rate skew inflates pooled AUROC while within-group ranking is weak.
One control is recorded as **unmeasured**: the renamed-donor specificity control produced no evaluable rows
because of a donor-selection defect, so the 0.681 specific fraction rests on the other two donor arms.
Human validation of cue plausibility has not been run. The appendix also reports later Llama sweeps that
looked successful (head 2 at layer 21, recovery 1.847) and explicitly refuses to let any claim rest on them
because their provenance is incomplete.

## What we take from it

Four transferable instruments and two warnings.

1. *Certified invariance as the denominator's guarantee.* A per-unit or per-condition effect is only
   meaningful when something independent proves the input edit cannot change the answer; here that is a
   symbolic free-symbol test with its own mutation audit. For us the analogue is cheap: any intervention
   study on our model pair should carry a machine-checked statement that the paired inputs must elicit the
   same behaviour in an ideal system, plus a mutation test of that checker.
2. *The paired-bootstrap comparison against the cheapest fitted baseline, as a standard rather than an
   option.* The instrument — score both predictors on the identical resample, put the interval on their
   difference — is what turns a 0.033 AUROC advantage into "indistinguishable" while the same advantage
   against a random-direction baseline would read as a finding. We should adopt it verbatim for any claim
   that a white-box read beats a black-box read.
3. *Leave-one-family-out with every rendering of a family on one side.* This is the resample design the last
   entry left open, stated at the level of the *group* rather than the example: the split unit is the
   container that shares its content, and the authors then measure the thing that is still leaky (71%
   cluster overlap) instead of assuming the split fixed it.
4. *Panel-level checkpointing and a damage control for a knockout.* Recovery divided by the original drop
   needs a fixed, and the fixed should be a pre-registered threshold plus an accuracy-on-held-out check
   with a pre-set tolerance — otherwise "the behaviour is gone" and "the model is broken" are the same
   number.

Warnings. First, **"necessary" and "sufficient" are separate instruments with separate failures**: the
heads pass necessity, the probe direction fails sufficiency, and the paper's own honest reading is that
the behaviour lives in more of the activation state than one direction — so a single read-out direction
must never be presented as the mechanism. Second, **a pooled metric over an unbalanced grouped holdout can
point the wrong way**: 0.820 pooled over two cue types, 0.667 and 0.6875 within them, because the model's
scores are offset between groups in the same direction as the base-rate skew. Any per-unit or per-condition
score our account reports is broken down by its grouping variable before it is pooled.

## Tags

unit: head | layer | whole-model
form: probe | direction | circuit
method: probing | ablation | patching | weight-analysis
eval: ablation | patching | prediction/simulation | faithfulness | benchmark
goal: safety/auditing | understanding | control/editing
