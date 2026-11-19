# How Reliable are Causal Probing Interventions?

- **Key:** arxiv-2408.15510
- **Authors / organisation:** Marc E. Canby, Adam Davies, Chirag Rastogi, Julia Hockenmaier; Siebel School of Computing and Data Science / The Grainger College of Engineering, University of Illinois Urbana-Champaign (all four on illinois.edu addresses; Canby and Davies marked equal contributors).
- **Date:** 2024-08-28 (v5 2025-12-22)
- **Link:** https://arxiv.org/abs/2408.15510
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Causal probing tries to say whether a model *uses* a property by intervening on that property inside its
activations and watching what the output does. A string of earlier critiques argued the instrument is broken —
either the intervention fails to remove what it meant to remove, or it smashes neighbouring properties on the way
through — but each critique was demonstrated on one method family and one use case (concept removal followed by
fine-tuning), leaving open whether the worry generalises. The authors want a way to *measure* whether a given
intervention did its job, so that methods and hyperparameters can be compared and chosen rather than assumed. The
larger goal is a precondition for everything else in this field: no causal claim about a model is worth more than
the intervention that produced it.

## What they studied

Six language models spanning encoder-only, decoder-only and instruction-tuned families: BERT, GPT-2,
Pythia-160M, Pythia-1.4B, Pythia-6.9B and Llama-3.2-3B-Instruct. The task is subject-verb agreement on LGD
sentences, with the number of the subject as the causal variable and the number of an intervening non-subject noun
as a non-causal environmental variable; the IOI task on GPT-2 is a secondary check. The unit of analysis is the
residual-stream embedding at a layer, and the object under evaluation is the *intervention method* rather than a
circuit or a feature: two linear concept-removal methods (INLP, RLACE), one linear counterfactual method
(AlterRep) and three nonlinear gradient-based counterfactual interventions (FGSM, PGD, AutoAttack).

## How they did it

For each layer and each property, they train a *validation probe* — a classifier whose only job is to decode what
the embedding currently encodes — on data that shares no examples with the data used to fit the interventional
probes, and with the two properties made independent by subsampling so that the arbiter cannot itself lean on the
spurious correlation. The interventional probe is trained on the ordinary, correlated data on purpose, so the
experiment reflects realistic conditions. An intervention then edits the embedding — projecting it into a
nullspace (removal) or pushing it across a hyperplane, or perturbing it adversarially inside an ℓ2 ball under a
possibly-nonlinear probe (counterfactual) — and the edited vector is handed to the validation probe *instead of*
being fed back into the model. Completeness is the total-variation distance between what the validation probe now
says and the target distribution the intervention intended (a point mass at the counterfactual value, or uniform
for removal); selectivity is the same distance for a non-targeted property, normalised and inverted so 1 means
untouched. Reliability is the harmonic mean of the two, described by analogy to F1 because doing nothing scores
perfect selectivity and zero completeness. Each method is swept over its single strength dial (perturbation radius,
rank, or the AlterRep multiplier), the best-reliability setting is taken per method, and separately the intervened
final-layer embedding is fed back into the model to measure the drop in task accuracy.

## What counts as an explanation here

Not an explanation of the model — an audit of the instrument. The deliverable is a two-number certificate per
intervention setting (completeness, selectivity) and a scalar summary (reliability), plus a calibrated operating
point chosen by maximising reliability. It is deliberately partial: it certifies that the edit did what it claimed
and touched nothing else, and says nothing about whether the property matters to the model; the accuracy-drop arm
is the separate, weaker evidence about use.

## How they judged success

Completeness and selectivity as defined above, using total-variation distance normalised to [0, 1]; reliability as
their harmonic mean; hyperparameter sweeps over the full strength range for every method; per-layer curves rather
than a single layer; task-accuracy drop ΔAcc as the behavioural correlate; standard errors reported for the BERT
table with all scores under that bound; and one explicit control for the framework's own bias — the whole layerwise
analysis is repeated with *linear* validation probes to check that the advantage of nonlinear interventions is not
an artefact of nonlinear arbiters. Validation-probe quality is reported first as a prerequisite: 97.3% and 94.4%
for the two properties.

## What they found

Every method trades completeness against selectivity as its strength dial goes up, in every model and every layer —
no setting buys both. At each model's best-reliability setting the nonlinear gradient interventions dominate:
0.934 (FGSM), 0.943 (PGD) and 0.938 (AutoAttack) on Pythia-160M, 0.920 and 0.951 on Pythia-1.4B and Pythia-6.9B,
0.960 on Llama-3.2-3B, 0.958 and 0.98 on GPT-2 — while the linear methods range from 0.106 (INLP on GPT-2) to
0.891 (AlterRep on Llama-3.2-3B). BERT is the one reversal: AlterRep is the most reliable method there at 0.835
and the best in its last three layers, which the authors read as a consequence of BERT's final layers being
linearly readable. Linear removal methods keep high selectivity but low completeness, consistent with their being
optimised for exactly that. Reliability correlates with accuracy drop: more complete counterfactual interventions
push subject-verb agreement to near-total failure. The nonlinear-over-linear gap survives the linear-validation-probe
control, with only AlterRep's late-BERT advantage shrinking. On IOI the same ordering holds (FGSM ahead of the
linear methods at all testable layers), with AlterRep beating INLP throughout.

## Limits and open problems they name

One task family, two binary properties — subject-verb agreement plus an IOI replication — chosen for parsimony and
because every model solves it, so nothing here speaks to tasks where the property is unclear or the model is
failing. Pythia-1.4B and 6.9B are restricted to three methods for compute. The linearity argument is offered as a
conjecture, not a result: they do not claim to refute or confirm the linear representation hypothesis, only to show
that conclusions drawn from final-layer geometry need not transfer to intermediate layers. They also concede that
the accuracy-drop relation could look different had they intervened in earlier layers.

## What we take from it

A *method*, and the cheapest one the folder now holds for making intervention claims trustworthy. Two moves
transfer directly to our own tables. First, an intervention is scored by a probe that never saw the intervention
data and is asked about *both* the property we meant to change and a property we did not — a completeness/selectivity
pair printed together, with a harmonic mean available when a single number is unavoidable, so that "the direction
did nothing" and "the direction did the right thing and a lot else" are distinguishable rows rather than identical
zero-effect results. Second, every strength knob is swept and the trade-off curve printed, because a method compared
at one hyperparameter is a comparison of two arbitrary points on two different curves — the same warning the
folder met on the attribution side. A *warning* travels with it: the audit's verdict flips by architecture (AlterRep
first on BERT, last on Pythia-1.4B and 6.9B), so the instrument's rating is a property of the pair, not of the
method. And a *framing*: the paper's own honest reading is that causal probing with linear interventions may be
measuring the final layer's linear readability rather than the model's use of a property, which is the precise
shape of the failure a weight-derived white-box model must avoid — a linear read that works at the output does not
license a linear claim about the middle.

## Tags

unit: direction | layer
form: probe | direction
method: probing | steering | ablation | other
eval: faithfulness | ablation | downstream-task | benchmark
goal: evaluation-of-methods | understanding
