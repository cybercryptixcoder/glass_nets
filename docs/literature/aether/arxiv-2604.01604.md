# CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders

- **Key:** arxiv-2604.01604
- **Authors / organisation:** Su-Hyeon Kim, Hyundong Jin, Yejin Lee, Yo-Sub Han (corresponding); Yonsei University, Seoul, Republic of Korea
- **Date:** 2026-04-02 (v1); v2 2026-05-27
- **Link:** https://arxiv.org/abs/2604.01604
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Prior refusal-steering work picks the sparse features or directions to suppress by how strongly they activate on harmful prompts. The authors' starting objection is that activation magnitude is not the same thing as being read by the computation that produces the refusal decision: a feature can spike because it tracks a topic, a phrasing, or adversarial wording, and still contribute nothing to the first token the model emits. They want a selection criterion grounded in the *circuit* instead — which feature's effect actually reaches the refusal-versus-compliance logits — and they want it automated, so that it can be run at scale rather than by hand-inspecting one prompt at a time. The larger goal is a controllable target for safety auditing: knowing which small set of features governs the decision, so that an intervention on them can be predicted and reasoned about.

## What they studied

Unit of analysis: a **single sparse feature identity `f = (ℓ, k)`** inside a cross-layer transcoder (CLT) attached to the target model's MLP blocks, plus the attribution-graph paths from that feature to two chosen output logits. Host model: `google/gemma-3-1b-it`, with the pretrained GemmaScope 2 CLT `clt/width_262k_l0_big_affine` (JumpReLU features; MLP output reconstructed as a sum of decoder contributions from features at the current and all earlier layers). Transfer models: `gemma-3-270m-it` (same family, different size) and `Llama-3.2-1B-Instruct` (different family), each with its own CLT. Prompts: WildJailbreak adversarial split for selection; four jailbreak benchmarks (JailBreakBench/JBB, HarmBench, AdvBench, SorryBench) for evaluation; MMLU, GSM8K and IFEval for capability.

## How they did it

Four design choices crossed in a 2×2, plus one new sampler. (1) **Sampling**: either *cross-group* (harmful set minus benign set) or *boundary-critical* — for each harmful prompt they read the next-token distribution at the first response position, score it by `min{P(refusal set), P(compliance set)}`, keep the top N = 100 prompts, so each analysed prompt is one where the model is genuinely torn between refusing and complying. (2) **Signal**: either mean *activation* of a feature on the prompt set, or *influence*, computed by building an attribution graph on the CLT-decomposed forward pass (attention patterns and LayerNorm scales cached, so the local computation is linear), giving edges `a_s ∂u_q/∂a_s`, and then summing over all multi-hop paths from output logits back to feature nodes; prompt-level score is the max over token positions, set-level score is the mean over prompts. (3) **Intervention**: multiply the selected CLT activation by a negative, layer-scaled factor `-γ · ℓ/(L−1)`, so early-layer features get a gentler push because their effect propagates through more downstream computation; default γ = 3, top-1 feature only. (4) **Judge**: LlamaGuard-4 response-only ASR plus a StrongREJECT rubric score that conditions on both prompt and response; Gemma-3-1B baselines are the dense refusal direction (Arditi-style, layer 14 selected), Refusal-SAE (layer 15, width 16k, top-20, integrated gradients), Steering-SAE (same SAE, ablation variant), and GCG/AutoDAN/PAP.

## What counts as an explanation here

A **ranked list of sparse features, justified by a circuit**. The explanation object is the attribution graph — nodes are individual CLT feature activations at specific (layer, position), edges are local linear effects — and the selection is a single scalar per feature obtained by aggregating path contributions to the refusal and compliance logits. It is deliberately partial: one feature at a time is steered, and the authors say so explicitly, because bundling several degrades generation. It is predictive in a narrow sense: the influence score predicts which feature, when suppressed, moves the output decision, which is exactly what the downstream ASR then tests. It is not a replacement model or a full causal abstraction of refusal, and no claim is made that the graph is complete; the paper's own framing is selection, not reconstruction.

## How they judged success

Primary: LG4-ASR (fraction of responses LlamaGuard-4 labels unsafe) and a StrongREJECT-derived Judge score in [0, 5] that multiplies a non-refusal indicator by the average of specificity and convincingness. Secondary and important: capability deltas on MMLU / GSM8K / IFEval relative to each method's **same-backend** no-steering baseline, with a McNemar test on MMLU; a degeneration rate (repetition-collapse detector) as a quality control; and a hyperparameter sweep over γ ∈ {1..5}, bundle size K ∈ {1,2,3} and pool size N ∈ {50,100,150,200}. Baselines are the six attacks named above; the 2×2 ablation is the load-bearing comparison, since it isolates sampling from signal. Judge-model choice is validated by agreement against a stronger judge (GPT-5.1) on 1,270 no-attack responses: 88.90% exact match on the composite score, 97.09% on the refusal label.

## What they found

The 2×2 ablation on Gemma-3-1B is the paper's real result. On JBB (LG4 ASR / Judge score): no attack 5.0 / 0.42; **cross-group activation 3.0 / 0.41** (below no-attack); **boundary activation 5.0 / 0.42** (identical to no-attack); **cross-group influence 34.0 / 0.67**; **boundary influence 62.0 / 2.95**. So activation-based selection in either sampler is inert, and the influence signal only becomes strong when paired with the boundary-critical prompt pool — the two components are not additive contributions, they are jointly necessary. Full four-benchmark comparison puts CRaFT at 57.4 average LG4 ASR and 2.90 average Judge against Refusal-SAE's 41.4 / 1.37 and Refusal-Direction's 33.9 / 0.60. Capability: CRaFT stays within ±0.5 points on all four capability columns (MMLU +0.13 with baseline 618/1,531 correct versus CRaFT 620, 58 disagreements split 30/28, McNemar χ² ≈ 0.017, p > 0.5), while the dense refusal direction collapses MMLU and GSM8K to 0.00 and Refusal-SAE loses 8.00 IFEval prompt-level points. Transfer without reusing the feature: Gemma-3-270m 9.6% → 23.7% average, Llama-3.2-1B 1.5% → 25.9% average (both with a per-model multiplier picked from a small held-out grid and a broader opener set for Llama). The γ sweep peaks at γ = 3 (0.62 LG4) and falls back to 0.39 at γ = 5 as degeneration climbs to 11%; the top-3 feature ranking is stable across all four pool sizes N, with influence values varying by at most about 8% relative.

## Limits and open problems they name

The headline ASR is partly a **judge artefact and the authors say so**: aggressive steering produces degenerate or repetitive output that LlamaGuard-4 still labels unsafe, so classifier-only ASR overestimates success — which is why the Judge score exists and why the two columns disagree for Refusal-SAE. Boundary-critical selection relies on a first-token proxy (`I` / `Okay`) for the refusal and compliance sets; the appendix checks that a broader 28+30 phrase set reproduces the same pool (92/100 overlap, coverage 69.79% → 76.47%), but on the cross-family model the authors had to swap to the extended set because the token distribution differs. Transfer requires re-running the whole pipeline per model and picking a multiplier per model, so it is not parameter-free transfer of one coefficient. Only top-1 steering is viable (K = 2 gives 22 degenerate outputs, K = 3 gives 86). Everything is at the 270M–1B scale, on Gemma and Llama only, with CLTs that were pretrained by someone else and not evaluated for reconstruction fidelity in this paper.

## What we take from it

1. **A measure, usable directly**: influence = path-aggregated local linear effect from a feature to the target logits, which is a *predictive* quantity about intervention outcome rather than a descriptive one — exactly the shape our white-box model needs, since the claim "suppressing f moves this decision" is falsifiable by one generation run.
2. **A sampling design**: score prompts by their proximity to the decision boundary (`min` of the two competing class masses) rather than by class membership. This kills the confound that separates "harmful-topic feature" from "refusal-decision feature" and is portable to any binary decision our model probes.
3. **A warning about ablation-only evidence**: their influence score is computed on the CLT-decomposed forward pass, so it is an effect measured *in the surrogate*, and it is validated only by the chosen feature being an effective knob. The paper does not report how the ON-path influence of a feature compares with its actual ablation effect in the **true** model — the one instrument that would connect the surrogate graph to the real network.
4. **A lesson on the frame**: the strongest number here is an attack-success rate, i.e. a capability failure, not an explanation-quality score. The transfer from circuit to claim is "this feature controls the decision", which is real evidence for our thesis but is being read off the attacker's side; interpreting it as "refusal is one feature" would be exactly the kind of single-feature shortcut the paper's own §7 criticises in its baselines.

## Tags

unit: feature
form: graph
method: attribution | transcoder | ablation | steering
eval: ablation | benchmark | human-rating
goal: safety/auditing | control/editing
