# The System Prompt Illusion: How Instruction Preambles Modify Computation in Language Models

- **Key:** arxiv-2609.38205
- **Authors / organisation:** Muhammad Usama; Dong Eui Chang — Control Laboratory, School of Electrical Engineering, KAIST (Daejeon)
- **Date:** 2026-09-23
- **Link:** https://arxiv.org/abs/2609.38205
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1, licensed CC BY 4.0)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

System prompts are how almost every deployed assistant is steered, yet nobody had measured how far inside the network they actually reach. The authors frame three competing hypotheses: instructions touch only the first and last few layers (shallow), they touch every layer uniformly (deep), or the depth depends on what kind of instruction it is (layer-selective). The practical stake is safety — if a safety preamble only edits a thin band of layers, then a jailbreak is not an anomaly but the expected consequence of where the constraint was written. The paper's aim is therefore to convert "system prompts work" into a measurable quantity with a per-model number attached, across many architectures and sizes at once.

## What they studied

Seventeen instruction-tuned models from eight architecture families, 1.5B to 72B parameters (Qwen 2.5 at six sizes, SmolLM2, Gemma 1 and 2, LLaMA 3.2 and 3.1, Phi-3.5-Mini, Nemotron-Nano, Mistral, OLMo-2, InternLM 2.5). Twenty system prompts arranged in five categories of four — minimal, persona, safety, permissive ("no restrictions"), format, and domain — asked against 100 shared queries. The unit of analysis is the layer: one representational comparison per (prompt pair, layer), aggregated into a per-model and per-category profile.

## How they did it

For each query and each system prompt they extract hidden states at every non-embedding layer and compute linear CKA between the activation matrix under that prompt and under the no-prompt baseline, averaging across queries. From that they define *penetration depth* as the fraction of layers whose average CKA falls below 0.95, and they calibrate the threshold against a split-half null (CKA between two halves of the same condition), which they report at 0.997 plus or minus 0.003 — putting the cutoff roughly 300 standard deviations below chance agreement. The same pipeline is repeated for every pair of prompts, giving a full per-layer similarity matrix per model. They then run three converging checks: causal activation patching (replace the five lowest-CKA and the five highest-CKA layers with the baseline's activations and compare output change), a five-way logistic-regression probe on the difference vector to ask whether the prompt *category* is linearly decodable regardless of CKA, and a spectral analysis of the perturbation matrix (Frobenius norm, effective rank, a partial correlation controlling for norm). Behavioural effect size is defined as one minus the average pairwise ROUGE-L similarity of the generated answers across the twenty prompt conditions, so that "the prompt changed the representation" can be checked against "the prompt changed the reply".

## What counts as an explanation here

A per-model, per-category scalar — penetration depth — plus a layer profile, backed by a proof that CKA deviation grows quadratically in the perturbation norm with first-order terms cancelling. The explanation is explicitly *not* a feature-level account: the authors say CKA tells you the geometry moved, not which features moved, and they name sparse autoencoders as the natural next instrument. It is a compact, predictive summary of the form "this instruction type restructures roughly this fraction of the network in this architecture family", and it is used to predict behavioural effect size across the model cohort.

## How they judged success

Four checks: (i) the null-calibrated threshold, with rank stability of the whole ordering across thresholds from 0.90 to 0.99 (Spearman above 0.95 between any two); (ii) metric-substitution — CKA against Procrustes distance, which agrees at 0.9924, with cosine similarity of flattened vectors kept as a deliberately failing negative control at 0.0293; (iii) causal patching, where the affected-layer patch must move the output more than the unaffected-layer patch, with a stated 0.01 margin; (iv) bootstrap confidence intervals on penetration (1000 resamples). Significance across categories is a Bonferroni-corrected Wilcoxon signed-rank test. Behavioural prediction is a Spearman correlation over the seventeen-model cohort.

## What they found

Penetration spans roughly nineteen-fold across the cohort, from 3.6% of layers in Qwen-2.5-1.5B to 68.8% in OLMo-2-7B. The ordering by category is consistent and large: persona 50.3% (plus or minus 19.5), format 44.5%, domain 26.2%, safety 7.3%, minimal 4.9% on the fourteen sub-72B models, with persona and format significantly above minimal and safety (p = 0.010) while safety is statistically indistinguishable from the empty prompt (Wilcoxon p = 1.000). Prompt length is ruled out as the driver — the safety prompts average 15.0 tokens and the persona prompts 13.8, and the rank correlation between token count and penetration is 0.400 at p = 0.505. The sharpest result is the safety/permissive comparison: an instruction that says "you have no restrictions" produces a layer profile nearly identical to a careful refusal instruction, Pearson r averaging 0.997 across models (range 0.987 to 1.000) and 0.998 at 70B and 72B, so the two opposites move the same small set of layers. Causal patching passes in 13 of 14 small models and all three commercial-scale ones, where the affected patch moves the output by more than 0.42 while the unaffected patch stays below 0.03; the single failure is SmolLM2-1.7B, which at 4.2% penetration has almost no affected layers to contrast. The probe baseline is the interpretive key: category is decodable at every layer of every model (above 85% at minimum, mean 97.8% against a 21% chance rate) while CKA falls only in patches — the prompt is encoded throughout and acted upon locally. Finally, penetration predicts behavioural effect size across all seventeen models (Spearman 0.761, p below 0.001), and effective rank correlates with penetration at minus 0.793 only because it is a proxy for perturbation magnitude, with the partial correlation controlling for norm dropping to 0.17. Scaling does not repair the safety gap: within Qwen 2.5 the trajectory flattens after 14B (18.8%, 19.4%, 20.1%) and safety sits at 5.1% in the 72B model and 9.1% in LLaMA-3.1-70B.

## Limits and open problems they name

Nothing above 72B is measured, and they refuse to extrapolate to frontier scale. CKA is geometry-level only, so the paper cannot say which features were rewritten; SAEs are named as future work. Cross-model comparisons carry a chat-template confound, which they grant affects the between-model ranking but not the within-model category ordering. Only a single forward pass on prompt-plus-query is analysed, so multi-token generation dynamics are outside the design. The paper also confines itself to instruction-level overrides: attacks that exploit tokenisation or multi-turn social engineering are said to possibly travel different paths.

## What we take from it

Three reusable pieces. First, a *threshold with a null attached*: penetration is only a meaningful number because the split-half CKA distribution is measured and the cutoff placed far below it, and because the whole ranking is shown invariant across a wide threshold sweep — the template our own reports should follow for any "N% of layers/units/features are affected" claim. Second, the *encoding-versus-restructuring split*, which is the cleanest statement in the folder of why a probe and an intervention can disagree: a decodable-at-every-layer feature whose geometry barely moves is being read, not used, and the pair of numbers (probe accuracy, representational change) should always be published together. Third, a warning that a distance-based similarity is second-order in the perturbation norm (their Proposition 1), so a metric like CKA or RDM correlation will systematically under-report small-but-behaviourally-potent edits — exactly the blind spot we would inherit if we score our surrogate model on representational distance alone. The safety/permissive isomorphism is also a concrete adversarial example for any white-box model of ours: two opposite instructions mapping to one pathway means the model's internal representation of "the rule" does not encode the rule's sign.

## Tags
unit: layer
form: description
method: probing | patching | other
eval: patching | prediction/simulation | benchmark
goal: safety/auditing | understanding | evaluation-of-methods
