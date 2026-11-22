# Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language

- **Key:** arxiv-2510.06378
- **Authors / organisation:** Angie Boggust, Donghao Ren, Yannick Assogba, Dominik Moritz, Arvind Satyanarayan, Fred Hohman; MIT CSAIL and Apple
- **Date:** 2025-10 (v1 2025-10-07; read v2, dated 2026-02-26)
- **Link:** https://arxiv.org/abs/2510.06378
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Automated interpretability normally asks a language model for a sentence describing each feature, and that sentence
is free-form: it drifts between runs, runs long, and can carry incidental detail that the reader then treats as
load-bearing. The authors argue this is a defect of the *interchange format*, not of the explanation step, and that
constraining the label to a small formal language — a "semantic regex" — costs no accuracy while buying consistency,
brevity, and a countable structure that can be aggregated over a whole model. The larger goal is to move from
one-feature-at-a-time stories to statistics over labels, where the label itself is a measurement that can be summed.
The paper is therefore less about a new explainer than about the description language as a design choice with its own
expressivity/affordance trade-off.

## What they studied

Subject models: GPT-2 Small with SAE residual-stream features at width 24,576, and Gemma-2-2B with Gemma Scope
residual features at widths 16K and 65K, all sampled through Neuronpedia's API (OpenWebText for GPT-2, Pile
Uncopyrighted for Gemma). The unit of analysis is a single dictionary feature; earlier pretraining neurons are
mentioned as in scope but are not the objects studied. Feature samples are 100 per layer for the benchmark, and up
to 1,000 per layer for the layer-wise complexity analysis. The explainer and the evaluator are the same model,
GPT-4o-mini, with a GPT-4o ablation.

## How they did it

They first built the language using a grounded-theory loop: survey thousands of Neuronpedia features across models
and layers, add a primitive or modifier only when a recurring activation pattern needs it, and stop when the
inventory saturates. The resulting grammar has three primitives — exact strings, word-form variants of a string,
and semantic-domain variants — plus three modifiers: a context wrapper that restricts a match to a domain, sequence
and alternation composition, and an optional-match quantifier. They then dropped this language into a standard
three-part pipeline (subject model, explainer, evaluator) and compared it against two established natural-language
prompts, one that shows token/activation pairs and asks the model to complete a fixed sentence stem, and one that
shows marked activating spans in running text. Every prompt difference between the new method and the marked-span
baseline is the instruction to emit a regex plus the grammar definition and four worked examples; the number of
demonstrations shown to the explainer is 10 for the regex method against 20 for the natural-language baseline, and
the activation highlighting threshold is lower (30 per cent of maximum rather than 60). Descriptions are scored with
six established autointerp metrics, then the regexes themselves are treated as data: component counts and primitive
types per layer, and a random-sample consistency test in which each feature is described five times from five
different draws of its activating examples and the descriptions compared for exact string equality. Finally,
24 in-house AI practitioners were asked, given a description, to write three phrases that should activate the
feature and one near-miss that should not.

## What counts as an explanation here

A structured expression over the feature's activating text, with a fixed vocabulary of matching primitives rather
than free prose. The claim is deliberately partial: the regex names the activation *pattern*, not the downstream
role, and the authors say outright it is not a solution to polysemanticity. The value added over a sentence is
compositional — because labels are drawn from one grammar, two labels can be compared, counted, and detected as
near-duplicates, which is what makes model-wide statements possible. It predicts input-side behaviour (what text
fires the feature) and says nothing about what the feature then does.

## How they judged success

Six metrics, three input-side families and one causal: a generation metric that asks an evaluator to write text
from the description and grades whether those generations activate the feature above random negatives; two
discrimination metrics (example-level balanced accuracy and a token-span variant); and two others scoring the
description's ranking of examples and its precision at separating activating from random examples; plus a
faithfulness metric computed by steering or ablating the feature during generation and asking whether the
description matches the steered text better than the ablated baseline. Statistical framing is explicit: a one-sided
paired t-test with a 5 per cent non-inferiority margin, Bonferroni-corrected for superiority claims. Controls are
the two natural-language baselines, a GPT-4o-for-GPT-4o-mini swap on 520 features, and cost accounting. For the
human study the outcome is not a rating but a behavioural measure — the activation gap between the participant's
best positive phrase and their counterfactual phrase — so a large gap means the participant located the decision
boundary.

## What they found

Regexes match natural language within the 5 per cent non-inferiority margin on the generation metric for all three
feature sets, and beat the weaker token/activation baseline on that metric everywhere; they clear the same margin
on one discrimination metric for GPT-2 and on the ranking metric for the 65K Gemma features, and each of these is
one of a handful of per-model comparisons rather than an across-the-board win. Descriptions are much shorter:
median 41 characters against 139 for the marked-span method and 55 for the token/activation method. Consistency
separates the methods sharply — 33.6 per cent of repeated descriptions are byte-identical for regexes against 12.2
per cent and 0.0 per cent for the two baselines — which is the property the redundancy-detection application needs.
Component counts and the share of abstract primitives rise with depth across all three feature sets, giving a
label-derived complexity curve. In the human study, participants' boundary measure was equal or better with the
regex on 9 of 12 features, with natural language winning the two cases where the regex was accurate but too terse
to prompt a full phrase. The GPT-4o swap moved absolute scores a little and left the method ordering intact.

## Limits and open problems they name

Terseness is the stated cost: a domain primitive such as a musician field over-fires on an instrument word, and in
the human study too-minimal regexes produced single-word positives that barely activated. The mapping from pattern
to description is not unique — several valid regexes can describe one feature — and the grammar leaves case
sensitivity underspecified, so the explainer sometimes commits to one convention and sometimes another. The authors
concede their own metrics are discrete judgments and therefore inherit both the ambiguity and the non-uniqueness as
false positives and negatives. Polysemantic features remain out of reach: composition can cover simple mixtures but
entanglement produces incoherent expressions, and they note the format is pipeline-agnostic and could be dropped
into a multi-label method instead. Last, models learn the language from a brief description only and make
grammatical errors that natural-language prompts do not elicit.

## What we take from it

A framing and a warning. The framing: when our account turns a feature into a nameable object, the format is a
measurable design choice with a cost curve we can chart — brevity versus over-firing, consistency versus coverage —
and the consistency measurement (describe the same feature five times from five different draws of evidence, count
byte-identical outputs) is cheap, is the honest way to say a label is stable, and is exactly the quantity our
prediction claims depend on. The warning: their own numbers show the method's advantage lives almost entirely in
the input-side metrics, the causal metric is near the noise floor for all three methods on the GPT-2 set
(0.37/0.46/0.52 with standard deviations around 0.45, i.e. a coin flip), and the comparison is asymmetric in
budget — ten demonstrations against twenty, a different highlight threshold — so "regexes are as accurate" is a
statement about a system, not about a language. Method to reuse: the layer-wise component count as a complexity
proxy read directly off the label, which turns labels into a model-wide statistic without any probing. Shortcut:
the released pipeline and prompts (Apple's repository and interactive interface) are directly reusable for our own
vocabulary design.

## Tags

unit: feature
form: description
method: SAE | probing | other
eval: downstream-task | human-rating | faithfulness | prediction/simulation
goal: understanding | evaluation-of-methods
