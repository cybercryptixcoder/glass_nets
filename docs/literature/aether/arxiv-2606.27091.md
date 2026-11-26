# Inherited Circuits, Learned Semantics: How Fine-Tuning Creates Evasion Vulnerabilities Invisible to Standard Evaluation

- **Key:** arxiv-2606.27091
- **Authors / organisation:** Ryan Fetterman (Cisco Talos)
- **Date:** 2026-06-25 (v1)
- **Link:** https://arxiv.org/abs/2606.27091
- **Kind:** paper
- **Read from:** full text (full v1 HTML; abstract, §1-§7, Appendix A, Appendix B, references)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Security teams fine-tune a general model on a labelled corpus and then report accuracy on a held-out split of the same
distribution. The author's claim is that this number can hide a vulnerability created by the fine-tuning itself: if a
particular command token co-occurs with maliciousness often enough, the model may learn a rule about the token's
surface form rather than about the behaviour, and then pass canonical tests while failing on scripts that do the same
thing written differently. The larger goal is a pre-deployment check that flags which parts of a task became brittle,
using only clean canonical inputs, so red teams know where to look before adversaries do. A second goal is mechanistic:
to show *where* in the network the fine-tuned classifier differs from its base model, since the two share an
architecture and a tokenizer.

## What they studied

One natural model pair from the same lineage: Foundation-Sec-8B-Instruct, a cybersecurity fine-tune of
Llama-3.1-8B-Instruct, and the Llama base as control. Both are 32-layer, 32-head. The task is binary malicious-versus-
benign classification of PowerShell scripts drawn from a 2,556-script labelled corpus (from prior work), capping
scripts at 12,000 characters and building length-controlled, within-family matched pairs where benign and malicious
examples share the same suspicious indicators. The main evidentiary set is 293 pairs spanning 456 unique scripts across
seven indicator families (DownloadFile, DownloadString, FromBase64String, IEX, Invoke-Expression, Invoke-WebRequest,
-EncodedCommand), filtered down from 380 candidate pairs by requiring both members to be classification-correct at
baseline. The unit of analysis is the attention head and the residual-stream position, with interventions grouped over
a matched cohort rather than run per example.

## How they did it

Discovery first: scan for heads that attend to URLs, command spans and execution sites across matched pairs, score
candidate sites by whether they push the residual stream along a benign-to-malicious difference direction, then
decompose each site into individual head outputs to avoid attributing an effect to a whole layer. Validation second, in
two flavours on the 293 pairs: grouped path patching as a sufficiency test (copy a candidate route's activations from
the malicious member of a pair into the benign member at the same positions and measure the shift in the
malicious-versus-benign decision margin) and grouped head ablation as a necessity test (disable the candidate and
measure the drop). Effects are reported as a mean shift of the decision margin plus a flip rate, the fraction of pairs
whose predicted label actually changes. Third, the same battery is run on Llama so that structure shared between the
two models can be separated from structure the fine-tune produced. Fourth, an evasion benchmark is built only from
malicious seeds that the model already classifies correctly, and each variant is a behaviour-preserving rewrite of one
indicator, accepted only after a parse check and explicit invariants (preserved URLs, preserved path-like literals,
preserved arguments, encoded-command equivalence). Three tiers are used: direct rewrites (aliases, call-operator
indirection, method-name reconstruction), reconstructive rewrites (format strings, base64 or ASCII recovery,
subexpression strings, backticks, zero-width normalisation), and case mutation that changes only token casing. Finally
the authors fit a ridge probe at the pre-Layer-13 residual site predicting each core head's contribution, and run an
indicator-token sign test that replaces the indicator embeddings with the mean vocabulary embedding and records the
direction of the confidence change in each model.

## What counts as an explanation here

A causally validated circuit: a small set of attention heads and a specific residual site, shown to be both sufficient
(path patching recreates the behaviour) and necessary (ablation weakens it), with an explicit claim ladder in Appendix B
separating discovery, validation, refinement, cross-model comparison and evasion evidence. The explanation is partial
and local — it names a route and a boundary, not a full mechanism: the paper states outright that the reversal in
evasion cases comes from a distributed MLP contribution across Layers 0-12 that it does not decompose into components,
and its published artefact is a ranked list of command families to red-team rather than a per-script prediction of
evasion. It does predict something, but at family granularity.

## How they judged success

Mean shift in the malicious-versus-benign decision margin and flip rate over the 293-pair cohort; both a sufficiency
test (grouped path patching) and a necessity test (grouped head ablation) required for a route to count as validated;
recurrence counts for the early detector across discovery pairs; a matched 74-pair subset for the model pairing;
per-family breakdowns rather than pooled averages; cross-validated Pearson r from a ridge probe with 5-fold CV;
miss rates on accepted benchmark variants, always conditioned on baseline-correct seeds and always compared against the
same variants run on the base model; and a prompt-framing comparison (raw prompt versus an instruction to classify by
overall purpose rather than individual constructs) to test whether failures are an artefact of the prompt.

## What they found

A minimal direct branch runs from an early head at Layer 0 (L0H11) to three Layer-12 heads (L12H15, L12H5, L12H4):
mean patching effect about −3.614, taking the mean decision margin from about 4.71 to about 1.10, i.e. roughly 77% of
it removed, flipping 87/293 pairs (0.297). Adding L12H28 to the bundle gives −3.824, about 81% removed, same 87/293
flips. L12H2 turns out to be a genuine nuisance: removing it strengthens sufficiency (patching) but weakens necessity
(ablation, top-five bundle at −2.174 versus −1.893 for the H2-free carrier), so the paper files it as an auxiliary
family-sensitive helper rather than part of the stable core. The best route varies by family, and the -EncodedCommand
family (7 pairs) never flips under any route, which the paper flags as a weak cell rather than hiding.

The skeleton is inherited. Llama shows the same L0H11 entry point, recurrent in 14/18 discovery pairs, and the same
Layer-12 integration; what the fine-tune changes is which late heads dominate (L12H15 for Foundation-Sec versus L12H28
for Llama). Foundation-Sec is not better because its circuit is stronger: on the matched 74-pair baseline it gains 4.7
points of accuracy (100% versus 95.3%) driven by benign accuracy rather than malicious recall, while its patching
effect on a comparable route is smaller (−3.82 versus −5.45) and its mean malicious logit margin is lower (3.75 versus
8.15). Llama's route needs a broader head bundle to reach the same flip rate, which the paper reads as the fine-tune
having concentrated weight into fewer heads.

Evasion is confined and specific. Tier 1: 38/44 correct, the misses being 4/4 on replacing Invoke-WebRequest with its
alias `iwr` plus 2/6 on indirect DownloadString invocation (both misses on seeds that also contained FromBase64String).
Tier 2: 42/46 correct, all four misses on format-string reconstruction of Invoke-Expression; backticks, base64 and
ASCII recovery, subexpressions and zero-width normalisation were all clean on the accepted variants. Tier 3: 9/63
misses, made of 4/4 alternating-case Invoke-Expression, 1/4 lowercase Invoke-Expression and 4/4 case-mutated IEX alias,
with 51 other accepted case variants clean. Llama posts 0/44, 0/46 and 0/63 on exactly the same variants. Mechanically,
the evidence is not deletion: under the alias transformation the late bundle still responds to the malicious content
(all 4/4 format-string variants still flip under patching), but feed-forward computation before the Layer 13 boundary
reverses the sign of its contribution, so the same internal evidence now pushes toward benign. Since both models share
a tokenizer, the case-mutation misses cannot be blamed on tokenisation and must come from how the fine-tune changed the
MLP's treatment of those embeddings.

The monitoring signal is cheap and it works once. A ridge probe at the pre-Layer-13 site recovers each core head's
contribution to the contrastive direction at cross-validated r = 0.80-0.87 (n = 206, p < 10⁻⁴⁵), so the inherited
representation is linearly readable there. Transferred across models to predict Foundation-Sec's dependence on the
top-five ablation, the same probe reaches only r = 0.41 (n = 293, p < 10⁻¹²), which the paper explicitly calls the
ceiling and the reason the operational unit is a family average, not a script. The sign test is the sharper instrument:
ablating the canonical indicator tokens of Invoke-WebRequest raises Foundation-Sec's malicious confidence by +1.13 mean
logit-diff (73.7% of scripts positive — the token behaves like a suppressor, because this family is being classified
through URL and payload context instead), while the same ablation on Llama drops confidence by −1.60. Invoke-WebRequest
is the only family with that large sign divergence, and it is exactly the family where the alias variants all fail, so
the sign test predicted the red-team target from canonical inputs alone, before any variant was generated. Removing a
script's indicator tokens also produces a predictable shift in the representation, readable from the base model's
activation at r = 0.83 (n = 293, p < 10⁻⁷⁵).

## Limits and open problems they name

The strongest evasion groups are four variants each, so they are localised counterexamples and red-teaming priorities
rather than prevalence estimates; the author says this outright and lists cohort expansion as future work. The
mechanism behind the reversal is a distributed MLP effect across Layers 0-12 that is not decomposed, so "signal
inversion" is a description of where the effect lands, not an account of what computes it. -EncodedCommand has 7 pairs
and zero flips under every route. The probe's cross-model r = 0.41 means per-script prioritisation is noisy, and the
family-level recipe is validated on exactly one model pair with one fine-tuning run, with the extension to other
security fine-tunes left undone. Two transformation families the author considers important (network-object
indirection, staged payload flow) are named but untested. The cohort is also restricted to pairs where both members
were already classified correctly, so behaviour outside that clean subset is unmeasured. There is no test of whether
the flagged families can be repaired, and the prompt-level remediation attempt is described as uneven — fixing
Invoke-WebRequest with a purpose-oriented instruction brought 5/43 case-mutation misses up to 12/43.

## What we take from it

A method and a measure. The method is the base-versus-fine-tuned differential: when a capability is added by training,
the informative comparison is not the trained model against zero but the trained model against its own parent on
identical inputs, with the same intervention applied to both, so structural inheritance can be subtracted from learned
specialisation. That is the shape our white-box model of a black-box network needs, since we are always comparing a
surrogate to the thing it models. The measure is the sign test, and it is the specific transferable idea: ablate the
feature you believe the model is using, and check whether the model's confidence moves the way a user of that feature
implies; a feature whose ablation *raises* the target score has changed role, and a role change between two models
marks a rule that is being leaned on rather than a rule that is being read. This is a one-forward-pass, label-free
diagnostic that runs before you have any adversarial examples, which is precisely the regime our predictions operate
in. The framing to adopt: circuit sufficiency and necessity are reported as separate columns because they can
disagree in the same direction — here L12H2 is the least sufficient head in the bundle and one of the more necessary
ones — so our model should never publish a single "component importance" number. The warning: a fine-tune that
improves the metric can concentrate an inherited route into a smaller carrier and thereby make the model *less*
robust, so a headline accuracy gain paired with a shrinking intervention effect is a brittleness signature and not a
safety improvement.

## Tags

unit: head | circuit | layer | direction
form: circuit | causal-abstraction | probe
method: patching | ablation | probing | weight-analysis
eval: patching | ablation | downstream-task | benchmark
goal: safety/auditing | understanding | evaluation-of-methods
