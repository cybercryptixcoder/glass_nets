# Order Is Not Control: Driven-Dissipative Response Laws Across Artificial and Biological Systems

- **Key:** arxiv-2606.12923
- **Authors / organisation:** Gareth Seneque, Lap-Hang Ho, Nafise Erfanian Saeedi, Jeffrey Molendijk, Tim Elson; Australian Broadcasting Corporation
- **Date:** 2026-06-11 (v2 2026-06-12)
- **Link:** https://arxiv.org/abs/2606.12923
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

The authors argue that the interpretability and alignment literature repeatedly mistakes an
organised object for a lever. A prompt, a constitution, a LoRA adapter, an activation
direction or a neural perturbation can all impose structure — on hidden states, on output
form, on trajectory — without moving the thing anyone actually cares about, and the papers
reporting those objects routinely treat the structure as if it were control. Their target is
an applied one: organisations treat written standards and post-training recipes as alignment
controls, and the authors want a criterion that decides, empirically, when a drive has been
admitted by the system it is aimed at. The larger goal is a cross-substrate account of a
**response law** — a reproducible drive-to-readout relation — that would be the object a
white-box controller had to estimate before it could act. They state explicitly that a
controller is a separate policy over that object and is not established by the paper.

## What they studied

Four kinds of substrate, under their own denominators. Biologically: mouse
anterior-lateral-motor (ALM) population response vectors (86,811 held-out rows), *C. elegans*
perturbation propagation (a few hundred held-out receiver rows), and larval zebrafish
ROI/plane random-wave response operators (358,068 held-out rows). Artificially: frozen LLM
completions under fixed prompt/render/decode baths with rubric-assigned basin labels, a
predictive response-vector panel over four material states (each state 1,536 samples and
18,432 vector components), a held-out observer panel over hidden-delta source rows, and
constitution-conditioned LoRA adapters treated as frozen "prepared media". The unit of
analysis is never a neuron or a feature: it is the **denominator tuple** — material state,
drive, bath, receiver, response displacement, sink channels, declared effort, comparator.
One whole subsection is a stochastic response-operator panel whose job is to separate
measurable opportunity from a deployable action policy.

## How they did it

They formalise the object as a denominator-indexed stochastic response kernel mapping
material state, action, bath and receiver state to a distribution over response displacement,
sink channels, effort and basin projection; the reported "response laws" are finite
differences and summaries of that kernel. Every panel then has to declare its denominator
before any measurement counts, and the panels are deliberately heterogeneous — the paper
says a bath, prompt or adapter state is a condition, not an evidence domain in its own right.
For the LLM panels, action is a visible semantic field applied before generation and the
receiver is the saved completion, labelled into mutually exclusive basins (target, harmful,
null/evasive, invalid-format) by a rubric fixed before the text was read; adjudication is
done from text-only packets with intervention metadata hidden from the reviewer, and the
authors concede there is no multi-rater reliability statistic. The adapter arm trains
constitution-conditioned LoRAs and runs the same visible fields through base/adapter pairs
in a fixed response tensor, so the adapter is a *medium* whose susceptibility changed rather
than a controller picking actions. The response-vector arm predicts the sign of an
intervention-effect displacement from action and state, and the observer arm predicts
system-effect and target labels from non-endpoint features only (target-projection fields
excluded at capture time). The local-control arm assigns control only to cells where finite
effort moved the target basin while damage, null, invalid-format and overdrive stayed
bounded, and then tests a state-gated policy against a hold baseline and a retrospective
oracle. The stochastic arm measures how often an opportunity block exists and then asks
whether a held-out policy ever selects one.

## What counts as an explanation here

A **response surface**, not a description and not a circuit: a per-denominator table of
which drives are admitted into which basins with what sink routing and effort cost, plus
one predictive sub-object (the sign of the response displacement) and one observer
(held-out prediction of effect labels). It is explicitly partial — the authors give four
evidentiary levels (induced order; response evidence under a matched condition; observer or
candidate actuator; local admitted control) and refuse to promote evidence across them. It
can predict *the direction* of an intervention effect and *the family* of an outcome, but not
which action to choose.

## How they judged success

Basin labels from a frozen rubric with a documented adjudication contract; a composite
utility defined as target reward minus damage, null/evasion, format-invalid and semantic-effort
costs, with panel-specific weights declared as part of the denominator. Statistics: Wilson
intervals for capture rates, bootstrap intervals for utility lift, paired sign tests against
matched random controls, label and row-group shuffles, a random Gaussian score null,
wrong-action and axis-permutation controls for the sign predictor, bridge-class and
placebo-axis shuffles for the local-control cells, and a retrospective-oracle gap as the
headroom reference for the policy.

## What they found

Predictive structure is real but strictly directional: component-sign accuracy for response
displacement is 72.77–73.75% across the four material states (84.27–84.78% on nonzero
components), against sign-marginal baseline 67.99%, wrong-action 64.58%, axis permutation
64.77% and nonzero wrong-action 76.03% — so the margin over the strongest control is around
eight points on nonzero components, not the fifteen it looks like. Held-out observers separate
system-effect labels at 93.57% accuracy (weighted AUC 0.907, Brier 0.055) and target/oracle
labels at 91.74% (AUC 0.880, Brier 0.069) from non-endpoint features only. Local control
exists in pockets: 1,146 clean-positive admitted cells over 22,810 rows with mean target
delta +0.690, inside a tensor that also holds 706 sign-changing cells, 2,587 stiff/saturated,
1,262 mixed, 49 positive-but-leaky and 37 sink/overdrive cells. Effort does not scale
monotonically: a scheduler that picked visible fields from teacher-forced derivatives scored
composite utility 0.9125, but a larger-budget variant fell to 0.49375 with damage 0.1125 and
an always-on stronger-field variant to 0.31875 with damage 0.1500, and a dose sweep moved
format invalidity from +0.120 at dose 1.0 to +0.266 at dose 1.5. Composed response operators
do not commute (nonzero vector commutator rates 0.750–0.833). Adapters behave as prepared
media, not controllers: the standard editorial-principle condition beats the null-random
comparator by +0.453125 on matched cells, and the paper stresses that the null-random adapter
is itself an active trained medium rather than an inert control. The policy result is the
negative core: a state-gated policy beat the no-boundary baseline by only +0.014683 against
0.668542 retrospective best-action utility, selected the best action in 33.333% of complete
matrices, and in the stochastic panel there were 253 opportunity-positive blocks out of 1,248
with **0/104 held-out opportunity captures** and 0/516 selected nonidentity admissions. The
state-estimator decomposition explains the sign of the policy: correct baseline-state
estimates give mean utility delta +0.626 (95% lower +0.196) while wrong or unresolved
estimates give +0.056 (lower bound -0.101). Biological panels land where that framing
predicts — held-out sign accuracy 0.715969 on ALM, 0.596825 on worm, 0.576156 on zebrafish,
with the cross-substrate coordinate gate at 4/12.

## Limits and open problems they name

No coordinate identity between biological and artificial systems; no evidence for a
controller, only for local admitted control; hidden/logit causal sufficiency is left an open
validation test, so a patch or steering vector remains a candidate actuator; held-out
stochastic action admission fails and is named as the missing piece; adapters are not shown
to be rankable or to act autonomously, and wording-as-causal-alignment is rejected; the
thermodynamic vocabulary is mesoscopic — no heat, entropy production, Lyapunov exponents or
memory hysteresis is measured; manual labelling has no inter-rater statistic; and several
declared quantities (baths, weights, denominators) are set per panel rather than derived, so
the composite utility is not comparable across panels.

## What we take from it

Three things for a predictive white-box model. First, **a scoring protocol**: a read-out
becomes a control claim only when a policy built from it beats a hold baseline *and* is
scored against a retrospective oracle gap and a held-out capture rate — this paper prints a
0.0147 lift next to a 0.6685 oracle ceiling and a 0/104 capture, which is the cleanest
template on disk for reporting that a predictive surface has headroom while the policy over
it has none. Second, **the denominator as a first-class field in our output**: every claim our
model publishes should carry the material state, drive, bath, receiver and side-effect
channels it was measured under, because this paper's own results show the same drive has
different signs in different baths and the same amount of extra effort trades target gain
for format damage. Third, **the non-commutation and overdrive measurements** are direct
warnings for intervention-based evaluation: composed edits do not commute (commutator rates
0.75–0.83), and added effort is not monotone, so any "apply the direction harder" prediction
our model makes is falsifiable in one experiment and should be reported with its sink costs.

## Tags

unit: whole-model
form: causal-abstraction
method: patching | steering | probing | ablation | other
eval: prediction/simulation | faithfulness | benchmark | human-rating
goal: understanding | safety/auditing | evaluation-of-methods
