# Interpretable Cross-Lingual Alignment in Small Language Models: Probing Cultural and Pragmatic Reasoning in Japanese–English Bilingual LLMs

- **Key:** arxiv-2608.14896
- **Authors / organisation:** Florian Braun; Independent Researcher (single author)
- **Date:** 2026-08-14
- **Link:** https://arxiv.org/abs/2608.14896
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The author's complaint is that small Japanese–English models are judged by translation quality and reading-comprehension benchmarks, which blend vocabulary, grammar and social register into one number, so a model can look competent while picking the wrong honorific or the wrong perspective verb. He wants cultural and pragmatic competence treated as something representable — locatable inside the network, measurable apart from fluency, and editable — rather than only as an output score. Behind that sits the larger interpretive goal the whole folder shares: names for the internal variables a model uses, with a separate check that the model actually acts on them. The paper is a single-model, single-author case study that mostly stages the programme rather than completing it: benchmark plus probes plus behavioural check now, steering and cross-model transfer later.

## What they studied

One model, TinySwallow-1.5B-Instruct (Sakana AI / Institute of Science and Technology Tokyo; Qwen2.5-1.5B architecture, 28 layers, hidden width 1536), examined at residual-stream activations at two positions — the final prompt token ("pre") and the final token of prompt plus continuation ("post"). Llama-3.1-Swallow-8B-Instruct-v0.3 (32 layers, 4096 wide) is named as the intended cross-model reference and explicitly not run, along with Llama-3.1-8B and Mistral-7B-v0.3 as English-first baselines. The unit of analysis is a linear direction in the residual stream, and the four target features are Japanese pragmatic contrasts: honorific register, implicit subject resolution, in-group/out-group reference (the ageru/kureru, iku/kuru verb pairs), and indirect refusal. Stimuli are a new minimal-pair set, J-PragEval-v0, 56 author-written items per feature (224 total), filtered on an annotator-agreement tag with a 0.6 floor.

## How they did it

Each feature gets a labelled set of scenario-plus / scenario-minus prompt pairs, each carrying a pragmatically correct and a pragmatically incorrect continuation, built so the difference is register rather than content. Three readouts are fitted at every layer and both token positions on activations pooled from those items: logistic regression (regularisation 1.0, liblinear), a class-mean-difference direction thresholded at the midpoint between class means, and a random unit vector thresholded at the training median as the null. Each probe's balanced accuracy is reported at its peak layer, averaged over five stratified 70/30 splits. Behaviour is measured separately by teacher forcing: score the log-probability of the correct and incorrect continuation under both scenarios, divide each by its token count, and report the accuracy in each scenario plus the **scenario-flip rate** — the share of items whose preferred continuation changes when the scenario changes. Reading the two tables together is the design's whole point: a probe that is accurate while the flip rate stays near chance is diagnosed as reading a confound. Steering is specified but not run: a class-mean-difference vector added at a fixed layer every generation step, with the strength chosen on a held-out split and directions summed for multiple features.

## What counts as an explanation here

A localisation plus a behavioural predicate: "this pragmatic contrast is linearly present at layer L at the prompt-final token, and the model's preference tracks the scenario on X% of items." It is explicitly partial, and the author states the epistemics plainly — decodability is necessary but not sufficient for a causal role — with activation patching and path patching written into the method section as the causal instrument reserved for follow-on work. The steering method is offered as a specification with a feasibility argument, not as a result; no prediction of the model's behaviour under an edit is made or tested.

## How they judged success

Peak-layer balanced accuracy over five splits, with standard deviations, against two same-split baselines (the class-mean direction and the random vector); length-normalised teacher-forced accuracy under each scenario and the scenario-flip rate; a ceiling check on probe position (early-layer saturation treated as lexical separability of the continuation rather than representation); annotation agreement per item; and, in the design, a capability-preservation ratio of J-PragEval gain to change on MMLU and HellaSwag (never computed). Ablations run are the random-direction null, pre-versus-post position, and length normalisation itself; cross-layer transfer, cross-model transfer, patching, head attribution and the layer-by-strength grid for steering are all listed as future work.

## What they found

The four features fall into three regimes. **Honorifics is the clean case**: 0.959 ± 0.039 at layer 15 of 28, against a random-direction baseline of 0.641, with the class-mean direction at 0.947 ± 0.025 at layer 27, and a 0.929 scenario-flip rate — the only feature where a linear readout and a behavioural flip agree. **Implicit subject and in-group/out-group reference are behaviourally present but not linearly readable at the prompt-final token**: peak logistic accuracies of 0.476 ± 0.038 and 0.376 ± 0.067 (at or below chance, with the random null at 0.518 and 0.541), while the flip rates are 0.768 and 0.786. The author reads those two as computed during generation rather than staged in a static prompt representation, and notes post-position probing only lifts them to 0.606 and 0.524. **Indirect refusal is the negative case**: a probe of 0.953 ± 0.045 at layer 17 that survives, at best, a 0.429 flip rate, with the correct continuation preferred on none of the plus-scenario items (0.000) and 0.018 of the minus-scenario ones. His diagnosis is a confounded stimulus set — the indirect continuations are longer than the direct ones, so raw summed log-probability penalises the right answer (mean margin around −34 nats before normalisation), and length normalisation then leaves almost no scenario margin at all. An early-layer artefact is also reported honestly: honorifics and refusal probes hit 1.000 ± 0.000 at layers 3 and 1 in the post position, which the author attributes to the continuations differing in many tokens and the embedding layer alone separating them.

## Limits and open problems they name

One model, one language pair, 56 items per feature, single author, and no causal instrument actually run. The steering claim rests on an arithmetic precondition — the class-mean direction matches the logistic probe within one to two points wherever a signal exists (0.947 against 0.959 on honorifics, and 0.853 on refusal) — which the author is careful to call an indirect argument. The paper's own framing of the refusal result is that the fix is benchmark design, not modelling. Non-linear probes, sparse dictionaries and per-token positions are named as what the two undecodable features would need. Dialectal and generational bias in the labels, and dual-use of a steering method that can push a model toward inappropriate personas, are flagged.

## What we take from it

- **Method — the paired readout as a two-row entry ticket.** Probe accuracy and scenario-flip rate are the cheapest available version of the folder's "correlate versus mechanism" test, and this paper's refusal row is its sharpest published failure: 0.953 beside 0.429. Any of our probes whose target we can score behaviourally should be reported this way, and a probe whose behaviour row is at chance is filed as a confound detector, not a representation.
- **Method — the position ceiling.** Same model, same probes, one position later: 1.000 at layer 3 tells you the continuations are lexically separable and says nothing about mid-network computation. Print the earliest layer at which a probe saturates as a standard warning field, because that is a stimulus artefact that looks like a perfect answer.
- **Warning — length is a nuisance variable that binds probe and behaviour together.** The refusal set was confounded in two places at once: the stimulus (prompt leakage) and the metric (length bias), so the probe found a real axis that does not drive generation. Both arms of a minimal-pair design have to be length-matched before either number means anything.
- **Framing — three regimes, not one.** Cleanly readable and behaviourally used; behaviourally present and not linearly readable at the analysis position; and readable but behaviourally inert. Our white-box model needs the third bucket as a first-class category, because a dictionary of directions that treats every decodable axis as a feature will be full of it.

## Tags

unit: direction | layer
form: probe | direction | steering
method: probing | steering | ablation
eval: ablation | prediction/simulation | downstream-task
goal: understanding | control/editing | evaluation-of-methods
