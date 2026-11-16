# Sparse Internal Control of Language Models

- **Key:** arxiv-2611.04349
- **Authors / organisation:** Xizhe Zhang (single author; no institutional affiliation on the record, ORCID 0000-0002-8684-4591)
- **Date:** 2026-11-03
- **Link:** https://arxiv.org/abs/2611.04349
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Activation steering shows that a single internal direction can move a frozen model's behaviour, but the field has no
shared answer to a prior question: if you are allowed a *small set* of intervention sites rather than one, how do you
choose them, and once you have chosen them, what exactly have you demonstrated about the control you now hold? The
author's complaint is not that steering fails but that a steering result is usually reported as one number — did the
target move? — and that this number is compatible with interventions that reach the target and damage everything else,
that cannot be turned down, that cannot be inverted, and that do not transfer to a new prompt. The paper therefore
offers three things in one package: a formal control problem with the collateral requirements written into the
constraints, two concrete selection algorithms that solve a linearised version of it, and a six-axis evaluation
protocol that measures properties of an intervention *beyond* whether the target readout changed. The larger goal is a
common basis on which different construction methods — SAE features, probes, mean-difference directions, hand-picked
heads — can be compared as controllers of the same behaviour, which is close to what we want when we claim that a
white-box object licenses an intervention.

## What they studied

Five task instances spanning four behaviours, on frozen models only, with everything computed from task-specific scalar
or vector readouts rather than from generation quality: indirect object identification (IOI) and a templated sentiment
task on **GPT-2 small**; MMLU Humanities and MMLU STEM multiple-choice QA on **Qwen3-4B**; refusal (an Arditi-style
refusal-versus-compliance lexical margin) on **Qwen3-1.7B**. The unit of analysis is a **candidate control node**: a
triple of a physical patched site (layer, component, token position or position-selection rule, and tensor slice), an
intervention direction that may be per-prompt or fixed, and a positive selection cost. One physical site can yield
several candidates through different directions. A *selection* is a coefficient vector over a finite candidate library,
and the model is patched additively at every site during the forward pass; the support of that vector is the "driver
set". So the object controlled is a small group of site–direction pairs acting together, not a feature dictionary and
not a circuit.

## How they did it

Everything is framed around three behavioural sets declared in advance: target inputs paired with desired readout values,
an off-target (preservation) prompt panel with a divergence, and an evaluation set that is the union. The desired
displacement for each target input is the target readout minus the model's own; for paired clean–corrupt tasks the target
is read off the clean partner's prompt, so the IOI task becomes "move the margin from the corrupt reading to the clean
reading".

The formal object is a **nonlinear control certificate**: an intervention passes only if its normalised target residual,
its off-target loss and its intervention energy all sit under tolerances fixed in advance (the paper writes this as three
inequalities, with a small numerical stabiliser in the denominator so that a near-zero displacement cannot divide the
score up). The optimisation then minimises weighted selection cost plus a multiple of energy, subject to that
certificate, under a selection-cost budget. Because support selection is combinatorial and the effects are nonlinear,
the paper solves it in three stages: **propose** a support from a local linear approximation, **refit** the strengths
jointly on that support, then **evaluate** the actual patch in the unmodified transformer and accept or reject it.

The local approximation is the piece worth following. For each candidate, the first-order effect on the target readout is
the Jacobian of the readout with respect to that site's activation, evaluated at the *unpatched* model, times the
candidate's direction; stacking these over candidates and over target inputs gives a matrix whose columns are candidate
effects, so "recover the requested displacement" becomes sparse regression of the displacement on that matrix, with
coefficients shared across inputs. A second dictionary, built the same way from a differentiable surrogate of the
off-target observable, and a quadratic form for intervention energy (mean squared perturbation magnitude, with the
cross-terms grouped by site) complete the proxy. **Driver-OMP** grows the support greedily by the best
cost-normalised residual alignment, refitting after every addition, stopping on residual, stall or budget — and the
paper states plainly that in the unregularised, unit-cost, unconstrained special case it reduces to ordinary orthogonal
matching pursuit. **Coverage-Driver** instead scores candidates by the new coverage they add across *rows*, with the
score saturating once a row's requested change is covered, so selections are pushed toward rows that remain uncovered;
it is meant for settings where one shared support must serve many prompts rather than reconstruct one displacement
exactly.

The six-axis protocol is then run as measurements of the accepted interventions, with the dose multiplier as the only
knob varied in three of the six axes: reachability (normalised target residual at nominal dose), **signed reversal**
(apply exactly the negated patch to the target-side example and ask where the readout lands relative to source and
target, with a pass at a ratio of two or more), dose response (target progress across a dose grid, summarised by a
signed rank correlation with dose), feedback controllability (fix support, base strengths and direction rule, adjust only
the scalar dose from observed error, and count endpoints inside a tolerance — with the readout-evaluation budget
recorded), off-target preservation (mean and per-prompt KL on a shared 32-prompt MMLU panel, plus a short generation
probe for refusal), and held-out reuse (either keep the selected *sites* and refit strengths on held-out targets, or
freeze the entire intervention).

## What counts as an explanation here

Not an explanation of the model — an **explanation of a control capability**. The form of understanding is a certified
intervention: a named support of site–direction pairs, a coefficient vector, and a triple of measured quantities
(target residual, off-target divergence, energy) that either jointly satisfy pre-declared tolerances or do not. It is
explicitly partial and local: the certificate is defined on the finite declared evaluation sets, says nothing off them,
and the paper is careful that separate axis results do not establish the joint certificate. In predictive terms the
object predicts one thing well — what a specified additive patch will do to a specified readout — through a first-order
proxy that is then checked nonlinearly, so the linearisation is a proposal mechanism rather than a claim about the
model's mechanism.

## How they judged success

Per-axis, with thresholds declared and then swept. Reachability: normalised target residual at dose one, pass at ε = 0.4,
with a reclassification sweep at ε ∈ {0.30, 0.35, 0.40, 0.45, 0.50}. Reversal: ratio of target-distance to
source-distance at least two, counted per seed. Dose: signed Spearman correlation with dose at least 0.95, plus
strongly negative curves counted separately (ρ ≤ −0.95). Feedback: terminal absolute error within 0.3 (IOI, MCQA) or 0.1
(refusal), with hits counted over seed–target endpoints and a reclassification of the *same* endpoints at five
tolerances. Preservation: mean and all-prompt KL at a 0.001 threshold. Reuse: signed mean-margin recovery, with the
paper noting that recovery can exceed one because it is a signed ratio rather than an accuracy. Baselines: random
supports, a Top-k arm (strongest predicted first-order effect, same budget), an average/nonlinear greedy reference that
is allowed extra model evaluations, and a single-site steering proxy. Controls that matter: the **direction-information
contrast** — per-evaluation-pair directions (P) versus directions averaged over the fitting split (F) — run on exactly
the same supports and evaluation panels; and the *information condition* of reuse, which separates "the sites come back"
from "the whole intervention transfers". Eight seeds on most series, ten for IOI reversal, with seeds varying the
fitting split and support.

## What they found

Reachability is reached on the same panel it was fitted on, and much less so held out. IOI Driver-OMP gives a residual
of 0.201 ± 0.058 with 10/10 seeds passing at ε = 0.4; module-level Coverage-Driver passes 3/3 seeds in both MMLU
categories (0.152 ± 0.154 Humanities, 0.061 ± 0.033 STEM). Random supports sit near or above 1 (0.747 ± 0.240, 1.209,
1.167), which is the scale of "predict no change"; the nonlinear greedy reference beats every selector in Table 2
(0.151, 0.073, 0.020) at the cost of extra evaluations. Held out, the result is a **direction-information split**:
Coverage-Driver and Top-k pass 8/8 seeds under per-pair directions and **0/8** under fixed training-mean directions on
both head-level MCQA (0.169–0.192 against 1.010–1.031) and sentiment (0.167–0.171 against 0.742–0.746), while
Driver-OMP manages 6/8 in MCQA-P and 3/8 in sentiment-P. Refusal is 0/8 in every selector and both modes, best residual
0.774–0.932.

Reversal follows the same split: IOI passes in 10/10 seeds under per-pair directions; sentiment reversal succeeds in 4/8,
8/8 and 8/8 seeds for the three selectors under P and 0/8 for all under F; across selectors 21/24 and 23/24 P-seeds pass
on Humanities and STEM against 0/24 in either F condition; refusal is 0/24 in each mode. Dose response is genuine but
weakly counted: Driver-OMP, Coverage-Driver and Top-k pass the ordering criterion in 3/8, 5/8 and 2/8 P-seeds on
Humanities and 4/8, 6/8 and 5/8 on STEM, and under fixed directions each STEM selector produces **four strongly positive
and four strongly negative** dose curves. Refusal is the clean dissociation: **all 48 of its dose curves have ρ = 1**
while none passes reversal at all. Feedback control with a fixed support tracks well when directions are per-pair
(Coverage-Driver 46/48 Humanities and 47/48 STEM endpoints, Top-k 45/48 and 47/48, Driver-OMP 38/48 and 39/48) and
collapses under fixed directions (7–8 of 48), with the paper noting that 6/48 and 10/40 of the endpoints were already
inside tolerance at zero dose and are included in the counts. Refusal reaches 12/40 to 22/40 across all conditions.

Preservation fails almost everywhere: at a KL threshold of 0.001 no seed satisfies either the mean or the all-prompt
criterion for Humanities, STEM or sentiment under any selector, with mean KL 0.110–0.497, 0.119–0.582 and
0.00553–0.01157 respectively; only refusal Driver-OMP comes close (mean KL 0.001080, 4/8 seeds under the mean criterion
and 2/8 under the all-prompt one). Reuse is the sharpest single contrast: retaining the selected *sites* and refitting
strengths recovers 0.992 ± 0.009 (Humanities) and 0.987 ± 0.034 (STEM) with Driver-OMP and at least 0.944 with both other
selectors, whereas freezing the entire intervention gives 0.021–0.041 — the same supports, and nothing transfers. A
support-recurrence analysis over five seeds and five budgets (25 configurations) finds two late head–position candidates
in 15/25 configurations each, with the paper explicitly refusing to read that as necessity.

## Limits and open problems they name

The evaluation-pair directions are computed from activation contrasts on the *evaluation* pairs, so the strong MCQA
results do not establish control without that information — this is the single largest restriction and the author states
it in the conclusion and the limitations section. The executed MCQA and refusal fits use plain ridge on raw contrasts
with no independent energy bound and no off-target penalty, so the runs are looser than the formulation. Separate axis
results do not establish the joint certificate. Preservation is 32 reused MMLU prompts measuring distribution change
rather than semantic correctness, and the refusal generation probe is sixteen greedy tokens in a window that never
reaches a final answer, so semantic refusal is unresolved. Sentiment feedback was not evaluated at all. Model panels and
candidate libraries differ across tasks, which precludes any scaling comparison, and the single-site proxy is
explicitly not a matched implementation of established steering methods. No multiplicity correction is applied across
the six axes. Thresholds are adjustable, and the paper's own sweep shows individual classifications move with them
(one MMLU cell passes 2/3 at ε = 0.4 and 3/3 at 0.45).

## What we take from it

Three transferable instruments and one warning. **Instruments.** (1) The *signed reversal* axis: negate the intervention
and see whether the state returns to the source, with the pass defined as "at least twice as far from the target as from
the source". It is nearly free (one extra forward pass over a panel already built) and it is the only axis here that
asks whether the intervention is a signed quantity rather than a push in a lucky direction — and it dissociates cleanly
from dose monotonicity, since refusal is perfectly monotone and never reversible. (2) The *direction-information
contrast* (per-example versus fitted-once direction, same supports, same panels) as a declared factor rather than a
caveat: it converts "this steering works" into "this steering works when the direction is recomputed from the example
being steered", which is what a practitioner would have to do at deployment and what most of these results require.
(3) *Held-out reuse with the information condition stated* — site-reuse with refit against fully frozen — which is the
cheapest test of whether a finding is about a location or about a coefficients-plus-direction artifact; the 0.99 against
0.02 gap here is the largest in the entry. **Warning.** The certificate's residual is normalised by the requested
displacement's own norm, so it is a relative measure with a near-zero denominator hazard — the paper adds a stabiliser
for exactly that reason and notes that recovery statistics can exceed one because they are signed ratio quantities. Read
with the folder's denominator lessons, this is the same object again: a recovery of 0.94 with a standard deviation of
0.53 (sentiment, fixed directions, k = 8) is a mean over a distribution with sign flips inside it, and the same series
has a 0/8 reversal count. Report an unbounded ratio with its sign distribution, or the mean will be read as a success.

## Tags

unit: head | circuit | layer
form: steering
method: steering | probing | other
eval: ablation | patching | prediction/simulation | benchmark
goal: control/editing
