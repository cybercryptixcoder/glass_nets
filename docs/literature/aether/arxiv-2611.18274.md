# Right Token, Right Time: From Per-Token Gating to Trajectory-Aware Activation Steering

- **Key:** arxiv-2611.18274
- **URL:** https://arxiv.org/abs/2611.18274
- **Authors:** Nima Eshraghi, Lovedeep Gondara, Sagarika Suresh, Xiaotong Xu, Jithin Pradeep, Fanny Chevalier (Vanguard / University of Toronto)
- **Submitted:** 13 Nov 2026
- **Read from:** full text (arXiv HTML v1) — §1–§5, Table 1, Appendix C.2 ablation, Appendix E limitations

## Problem

Standard activation steering pushes a fixed direction into *every* generated token. Plenty of those tokens already sit on target, so pushing them buys little behavior change while moving the residual stream off the pretraining distribution — the displacement compounds over the sequence, which is why steering strength and fluency trade off in practice. The authors reframe steering as a **budgeted allocation problem**: fix the fraction p of tokens that may be steered, fix the direction and magnitude, and let the only free variable be *which* tokens get it. That turns every method comparison into a clean ablation, because at matched p nothing else differs.

## Method

**Direction from SAE features.** Per-feature differential Δ_i between a target corpus D+ and a neutral D−, take the top-k features, sum their decoder rows, normalize. Suppression uses α<0, elicitation α>0. A task-agnostic probe u = sign(α)·v makes larger ⟨h_t, u⟩ always mean "more on target" regardless of task polarity. Reading the probe is one dot product — no SAE forward pass at inference.

**CFP (Contrastive Feature Projection)** — memoryless: s_t = ⟨h_t, u⟩, gate fires when s_t ≤ τ, with τ calibrated to hit budget p. This is the budget-aware general form of feature-triggered conditional steering (their citation of "Breaking Bad Tokens" reduces to CFP once calibrated). One dot product per token.

**CLABS (Closed-Loop Adaptive Bernoulli Steering)** — the contribution. Standardize s_t with calibration stats, pick setpoint s⋆, define error e_t = s⋆ − ŝ_t and a **leaky integral** ē_t as an EWMA with memory ~1/β. Steering probability p_t = clip(p₀ + κ_p·e_t + κ_i·ē_t), then sample the gate from Bernoulli(p_t). Bias p₀ is calibrated so realized rate ≈ target budget. Proportional term = same information CFP has; the integral term is what memoryless gates lack — it distinguishes a momentary dip inside an on-target span from sustained drift away from target. No rollouts, no backtracking, no learned scorer. Compute overhead ~10⁻⁶ of a forward pass.

**Theory.** Assumption 1 is a first-order linear steering-response model: ŝ_{t+1} = φŝ_t + (1−φ)β₀ + g·m_t + η_t, with φ activation persistence, g steering gain, β₀ unsteered equilibrium. That gives a 2×2 linear recursion on (e_t, ē_t) whose stability is a joint property of model and gate. Theorem 1: sufficient condition ρ(A)<1 is 0 ≤ a < 1 and 0 ≤ gκ_i < (2−β)(1+a)/β, giving geometric convergence of the mean, bounded asymptotic variance c(A)σ̄², and time-averaged squared error bounded by O(1/T)+O(σ̄²). Corollary 1: |e_ss| is non-increasing in κ_i, strictly decreasing when N≠0 — so the integral term reduces but never eliminates steady-state offset. Remark 1 makes this concrete: leaky integration trades exact tracking for bounded memory, and Eq. 16 gives a hard positive floor on |e_ss|. They claim this is the first stability certificate for a sparse steering gate.

**Setup.** Gemma 2 2B and Llama 3.1 8B, single mid-network residual layer, pretrained residual-stream SAE at that layer. Three targets: toxicity suppression on RealToxicityPrompts, fear and sadness elicitation on GoEmotions (Ekman-collapsed). AlpacaEval for competence. Three seeds. Norm preservation, activation clamping, and a repetition penalty applied identically to every method.

**Baselines.** Unbudgeted: No-steer, Full-steer, Prompting. Budgeted: SIA-PE and SIA-ER (the *selection signals* of Sparse Inference-time Alignment — predictive entropy and entropy rise — extracted and turned into single-pass gates, explicitly not a reproduction of the full SIA system); SRE (SAE reconstruction error, their own mechanistic counterpoint: steer only where the dictionary represents faithfully); STS random Bernoulli(p) as the allocation-agnostic control. They're careful that PID Steering (Nguyen et al.) is *not* a competitor — it controls magnitude, not allocation, and the two compose.

## Findings

1. **Allocation dominates.** CFP and CLABS get the largest behavioral shift among budgeted gates on every model–target panel. At the smallest budgets they reach 2–3× the shift of the entropy gates and random control while steering the same number of tokens. Gates converge as p→1, as expected.

2. **CLABS leads where it matters.** The gap is widest at tight budgets. At moderate budgets CLABS recovers 89–92% of the Full-steer shift on emotion targets — i.e. steering ~half the tokens gets most of the effect.

3. **Fluency holds.** On AlpacaEval under toxicity steering, CFP and CLABS are the only gates preferred to No-steer at every budget under both judges (gpt-5-mini and Claude Sonnet 4.6). PPL stays near 23–28 versus 41.2 for Full-steer at p=1.0. distinct-2 stays ~0.86–0.91 vs 0.68 for Full-steer.

4. **Ablation cleanly attributes the win (Table 3, Gemma 2B toxicity, p=0.5).** Toxicity reduction: open-loop 0.12, I-only 0.15, P-only 0.21, PI 0.24. Perplexity: 9.8 → 9.0 → 8.9 → 8.6. Both terms earn their keep; I-only has the largest seed variance, P-only reads the same signal as CFP and so can't tell dip from drift.

5. **Budget honesty.** Realized rates stay within 0.04 of nominal for all methods; CFP/CLABS ranges sit inside STS's, so they don't win by spending more. Re-reading every figure at realized rates leaves the ranking unchanged.

6. **SRE doesn't win.** Their own mechanistic baseline — steer where the SAE reconstructs well — is mid-pack, roughly tied with SIA-PE in places. Worth noting as a negative for "trust the reconstruction" intuition.

7. **Qualitative.** Table 11 shows CLABS/CFP answering a factual AlpacaEval question directly where baselines deflect into meta-commentary. Their read: sparse, well-placed steering leaves most of the trajectory undisturbed.

## Limitations (as stated)

Two open base models, one layer, one direction per target; no instruction-tuned chat models. Assumes an adequate pretrained SAE at the target layer — SAE-quality sensitivity is out of scope. English only, three targets chosen for clean classifier scorability. Automatic classifiers plus two LLM judges, no human eval. The stability result is a linearization around the interior operating point, explicitly not a worst-case guarantee for nonlinear decoding. Dual-use acknowledged.

## Why it matters here

The reframing is the useful part: "when to steer" is a separate control axis from "how hard", and once you fix the budget they don't confound. For Aether's steering layer this suggests the budget p should be a first-class user-facing knob alongside magnitude, with the probe readout (⟨h_t, u⟩ standardized) surfaced as a cheap live signal. CFP is a one-dot-product gate — trivially shippable. CLABS's leaky integral is ~5 lines on top. The stability box 0 ≤ a < 1, gκ_i < (2−β)(1+a)/β is a practical guardrail: given rough estimates of persistence and gain you can refuse to ship unstable gains instead of discovering oscillation live.
