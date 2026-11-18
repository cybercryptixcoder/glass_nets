# Token Commitment as a Control Surface for Steering Masked Diffusion Generation

- **Key:** arxiv-2611.21669
- **Authors / organisation:** Ambroise Aigueperse, Hugo Buurmeijer, Carmen Amo Alonso, Marco Pavone — Stanford University
- **Date:** 2026-11-11
- **Link:** https://arxiv.org/abs/2611.21669
- **Kind:** paper
- **Read from:** full text (HTML, §§1–7, Tables 1–3, 8–11, plus the appendix tables for the extreme-policy and best-of-N diagnostics)
- **Relevance to us:** 1 (background)

## What they wanted, and why

Inference-time steering of a frozen model is usually done by editing what the model predicts — activations, logits, guidance signals. The authors point out that masked diffusion models have a second decision surface that autoregressive models simply do not have: the sampler repeatedly proposes tokens at masked positions and then decides which of those proposals become permanent, and how many per step. That policy is normally treated as a fixed decoding detail tuned for speed or quality. The paper's goal is to make it an explicit, target-aware control axis and to characterise the operating curve it exposes, rather than to propose a better steering intervention. Their framing is that activation steering sets the base operating point and commitment control decides which of the resulting predictions survive.

## What they studied

Two masked-diffusion families: LLaDA-8B-Instruct and LLaDA-V for captioning, and Dream-7B (an autoregressive-initialised masked diffusion model with a different tokenizer, depth and confidence calibration) as a second family. Targets split into two kinds — *control targets* that name a shared output attribute (negative sentiment, subjectivity, negative caption sentiment) and *task-success targets* that are example-specific (NLI entailment between a SAMSum dialogue and its generated summary; CLIPScore between a caption and its image). Data: Amazon reviews (300 prompts), SAMSum (300), SentiCap (100 for caption sentiment, 100 for image alignment). Activation conditions: none, DiffMean, and diagonal Activation Transport (ACT), each held at a single strength. The unit of analysis is not a neuron or a direction but a **sampler-level policy**: a scoring map over unresolved positions plus a budget map for how many positions resolve per step.

## How they did it

They formalise a commitment policy as a pair of maps. The **scoring map** r ranks unresolved positions and induces a total order on them; the **budget map** u gives how many of the top-ranked positions become permanent at that step, with the committed set being the top-u under r. Standard confidence decoding is the special case where r is the model's max-softmax at the position and u is a fixed schedule. They then replace one component at a time, holding everything else fixed.

For order: at each step they build the full provisional completion (every masked position filled with its current argmax), decode it to text, and for each unresolved position i delete that position's decoded span and re-score with the frozen target scorer S. The score drop is the position's local target contribution; contributions are min-max normalised within the step and mixed with the model's confidence q as r = λ_order·q + (1−λ_order)·Δ̄. The budget stays at one token per step, so every point of the sweep uses identical denoising cost and every difference is attributable to which token was resolved first. Sweep λ_order ∈ {0.5, 0.75, 0.9, 1}.

For budget: ranking is fixed to confidence and only u changes. Target progress (the current provisional score) and a confidence summary (mean confidence of the top-3 unresolved candidates) are each converted to empirical quantiles within the current denoising phase, using a calibration set disjoint from evaluation (500 prompts for text, 125 for vision-language, quantiles estimated once from fixed-2 runs), mixed as z = (1−λ_budget)·ŝ + λ_budget·ĥ and mapped to a budget in {1, 2, 3}. Sweeps λ_budget ∈ {0, 0.5, 1}.

Controls: fixed budgets 1/2/3 show the constant-budget landscape; the decisive baseline is a **scorer-free stochastic schedule** with the same confidence ranking whose budget is sampled from a state-independent distribution, with its neighbouring integers and probability chosen post hoc from step counts alone to match the adaptive run's mean steps — so any remaining difference is state-dependent adaptation rather than average compute. Random commitment order is run as a diagnostic anchor, and all generations are re-scored with independent models (a different sentiment classifier, a larger NLI cross-encoder, a CLIP model on a different corpus). A best-of-N comparison at matched wall-clock budget prices the counterfactual overhead against spending the same compute on resampling. One seed, greedy decoding (temperature 0), 60-token generations for text and 35 for LLaDA-V.

## What counts as an explanation here

A **characterised control surface**, not a decomposition of the network. The "understanding" is a pair of scalar knobs with an empirically monotone target-versus-fluency curve attached to each, plus the claim that the same knobs work when the target is an example-specific proxy and therefore corresponds to no reusable direction. It is deliberately partial: the paper is explicit that it does not propose a new activation intervention and that the commitment controls act downstream of whatever produced the provisional predictions.

## How they judged success

Empirical monotonicity of the target score and of external-LM perplexity along each sweep (all four values non-increasing in both, in every configuration); reproduction of that ordering on a second model family; non-domination of the adaptive operating point by its compute-matched stochastic schedule in fifteen settings; resampling the evaluation set with replacement and reporting the fraction of resamples in which the ordering and the non-domination survive; re-scoring with independent scorers; a random-order control; and a matched-budget best-of-N comparison. There are no significance tests and no seed variance — the sweep is a monotone-checked curve, not an estimate with an interval.

## What they found

Counterfactual ordering moves the target and the perplexity together, at fixed denoising cost, in every tested configuration. LLaDA sentiment with no activation: 0.35 target / 9.5 PPL at confidence-only, 0.63 / 10.6 at λ_order = 0.5, 0.70 / 12.5 under pure counterfactual ordering. Random commitment order is the anchor that rules out "any departure from confidence works": it is the worst perplexity of any policy shown (13.1) while reaching only 0.41. DiffMean and ACT shift the curve but not its direction. Subjectivity moves 0.15–0.23 and caption sentiment 0.15–0.56 across the sweep.

Dream-7B traverses a wider range unsteered (0.30 → 0.71) for about a point of perplexity. NLI entailment on SAMSum moves from 0.31 to 0.54 (LLaDA, PPL 22.5 → 26.6) and 0.34 to 0.60 (Dream, 22.4 → 27.3); CLIPScore shows the same ordered behaviour. Adaptive budgets are ordered as predicted — target-progress-driven strongest on the target, confidence-driven lowest on PPL, mixed between — in all fifteen settings, never jointly dominated by the matched stochastic schedule, and in some settings outside the reach of any interpolation between fixed-1/2/3. Resampling reproduces the orderings in 86.7–95.3% of draws and the non-domination in at least 91.1%.

Costs are priced: adaptive budgeting adds 8.5% wall-clock at essentially identical average steps, counterfactual ordering adds 48.9% (scorer 18.4%, other 15.2%) at the same 60 steps. Against best-of-N at the matched expected generation count, λ_order = 0.75 wins on LLaDA sentiment (0.53/10.5 vs 0.483/10.92), Dream sentiment (0.52/7.8 vs 0.515/7.95) and LLaDA NLI (0.43/25.1 vs 0.404/25.51), and matches on Dream NLI at 1.1 lower perplexity — which the authors read as a match, not a win.

## Limits and open problems they name

One seed, one strength per activation method, so the natural question — how commitment control compares with simply steering harder — is explicitly left open, because a fair answer needs a joint sweep over intervention strength and commitment weight; the Dream strengths differ from LLaDA's, which the authors note confounds model and magnitude. The second family does not cover subjectivity or the vision-language settings. All targets are frozen external scorers, and NLI entailment and CLIPScore are proxies for task quality rather than task quality. Counterfactual ordering carries a ~49% latency overhead and the "counterfactual" in its name is a local span-deletion comparison, not the downstream causal effect of committing that token. The authors suggest amortising it with a surrogate that preserves the ranking — an admission that the on-the-fly measurement is too expensive to be the practical form.

## What we take from it

Three things, two of which are warnings. **First, a control surface is a legitimate object of explanation, and its proof obligation is monotonicity plus a compute-matched baseline**: if we ever expose a decoding-time or read-out-time knob of our own, the claim to make is "the curve is ordered and the movement is not bought with extra compute", and the harness is a stochastic schedule with the same average cost and no access to the target. **Second, and this is the warning the paper is most exposed to, a knob that needs the answer to choose the next step is not a read-out of the model.** Every rank in the counterfactual map is a call to the target scorer — the quantity being optimised is evaluated on the candidate completions themselves, ~one call per unresolved position per step batched, so 49% overhead — and the ordering is therefore partly a property of the scorer's preferences, which is why the independent-rescorer check matters more here than the monotonicity does: without it the curve shows only that an optimiser optimises its objective. **Third, one number worth copying as a control we rarely run**: random commitment order is a diagnostic anchor that separates "the mechanism works" from "leaving the default works", and it costs one extra sweep. Its result here (worst perplexity, almost no target movement) is exactly the shape a null should have. For our probing: a per-step decision like "which positions to probe next" is a commitment policy, and the same distinction applies — order of measurement versus rate of measurement are separate axes and should be swept separately.

## Tags

unit: whole-model
form: steering
method: steering | other
eval: prediction/simulation | benchmark | ablation
goal: control/editing | understanding | evaluation-of-methods
