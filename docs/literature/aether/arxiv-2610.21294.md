# Can Capability Enhancement be Localized?

- **Key:** arxiv-2610.21294
- **Authors / organisation:** Anh Tuan Nguyen, Anna Bair, Zico Kolter; Carnegie Mellon University
- **Date:** 2026-10-15
- **Link:** https://arxiv.org/abs/2610.21294
- **Kind:** paper
- **Read from:** full text (v1 HTML; figures not inspected)
- **Relevance to us:** 3 (central)

## What they wanted, and why

The head-ablation literature has spent years locating sets of heads that *matter*: remove them and the task breaks. The authors ask the opposite question and name it: given a model and a task, is there a small set of heads whose removal *improves* the task, with no fine-tuning? They call this Capability Enhancement Localization. The motivation is not efficiency but understanding: if a trained model's own internal layout can contain components that actively hurt one of the objectives it was trained on, then "this component is part of the mechanism for behaviour X" cannot be read off a ranking of how important the component looks. The paper pairs a small empirical search with a toy-model proof that such components can emerge from ordinary training, so the larger goal is a training-dynamics account of *when* a head's sign of contribution is negative, not just its magnitude.

## What they studied

Three instruction-tuned 3B models from different families (LLaMA-3.2-3B-Instruct with 28 layers and 24 heads per layer, 672 ablatable heads in total; Qwen2.5-3B-Instruct and SmolLM3-3B, 36 layers and 16 heads each, 576 heads), on four benchmarks covering arithmetic, code generation (MBPP), and language skills (HellaSwag, BoolQ). The unit of analysis throughout is the attention head, removed by zeroing its output at every token position before the output projection. The theory studies a three-head, one-layer attention model with diagonal projections over three hand-specified feature coordinates.

## How they did it

A two-stage routine, run per model–task pair on a discovery split kept disjoint from the test split. Stage one screens every head singly and keeps the top ten by a ranking that prefers higher gain, then lower damage (examples the intact model gets right and the ablated one gets wrong), then higher accuracy, then fewer heads, then lexicographic order. Stage two enumerates every subset of that ten-head pool of size up to three, evaluates each joint ablation on the same discovery split (reusing the singleton results), and picks the best-ranked subset. The chosen subset is then frozen and evaluated once on held-out data against an intact run under the same protocol. No test example touches selection. For the theory, they construct two equally weighted regression tasks on Rademacher inputs that disagree about a shared coordinate: task A wants that coordinate ignored, task B wants it at coefficient a. The attention model is given the feature directions in advance and a fixed paired memory bank so every head has identical access to every feature; training only decides how features are allocated to heads. Training is full-batch gradient descent from small Gaussian initialisation on the population loss, and the paper proves what allocation GD selects.

## What counts as an explanation here

Two different things, kept separate. Empirically, the explanation is a measured contrast: a specific small head set, a positive held-out accuracy difference against a fixed intact baseline. Theoretically, the explanation is a mechanism — an allocation event with a stated probability, under which a head carries a coordinate the two tasks disagree about, so deleting it moves the predictor off the *jointly optimal* compromise and improves one task at the other's expense. This is a causal abstraction of a training trajectory, partial by construction: it explains that removal can help and why, in a setting where the feature basis is supplied, and it does not predict which heads in a real LLM will be beneficial.

## How they judged success

Held-out accuracy with the subset fixed in advance on a disjoint discovery set; ablation gain in percentage points; a relative error-reduction figure; and, theoretically, the sign and size of the risk change per task together with the probability of the enabling allocation event. There is no matched-size random-head control, no permutation null, no seed or bootstrap reporting for the accuracy numbers, and no test of whether the discovery-selected subset is stable across discovery resamples — the paper names the last of these as an uncharacterised limitation. The theory's controls are structural: joint optimality of the aggregate predictor, and the explicit statement that no sign conditioning, parameter tying, or rebalancing was applied, so any localisation must come from dynamics.

## What they found

The empirical pattern is broad and the magnitudes are uneven. Every one of the twelve model–task pairs improves on held-out data, with SmolLM3 on arithmetic the largest at 70.40% to 83.50% (+13.10 points, 44.26% relative error reduction) and LLaMA-3.2 on arithmetic second at 68.00% to 74.30% (+6.30, 19.69%); the remaining ten gains sit between +1.29 and +5.20 points, and four of them (three Qwen rows and LLaMA on HellaSwag, plus LLaMA on BoolQ) are around +1.4 to +2.5. In the toy model, the aggregate optimum is unique — the shared coordinate's coefficient is pinned at a/2 by the conflict — but its decomposition across heads is not, and GD from small random initialisation provably concentrates each coordinate in one head while the other heads' coefficients stay tiny. When the head winning the conflicted coordinate is also not the head winning either task-specific coordinate (probability 4/9), deleting it drives task A's risk to zero while raising task B's risk in the small-initialisation limit, i.e. one task improves and the other degrades *at a global minimum of the joint objective*.

## Limits and open problems they name

The search is deliberately local: it can only find combinations whose members each help when ablated alone, so a beneficial set that works only jointly, with individually neutral or harmful members, is excluded by construction — and the authors are explicit that a failed search does not establish absence. The theory needs the feature basis supplied, diagonal projections, small initialisation, and a noiseless population objective; it demonstrates one possible mechanism and does not claim to explain the measured gains in the language models. Beyond what they name: the accuracy numbers carry no variance at all, so a +1.3-point gain cannot be distinguished from scoring noise in this paper; there is no random-set control showing that an equally sized arbitrary triple does worse; and the claim that selected heads *interfere* is never tested by identifying what the removed heads were computing.

## What we take from it

A framing and a control we were missing. Every entry in this folder that ranks units by the size of their ablation effect assumes the effect's *sign* is the useful part; this paper shows a trained model can allocate a conflicted coordinate to a head whose removal helps, which means an effect size with no sign is half a measurement and a negative effect is a fact about the objective, not noise. Concretely: (a) a measure — report per-task signed ablation gain, since one head can be positive on one benchmark and negative on another, and the paper's own theory predicts exactly that; (b) a warning — a "task-critical head" set and a "beneficial-to-remove" set come from the same screen with opposite intended signs, so our white-box model must predict sign, not rank; (c) a shortcut for the theory side — the toy construction is cheap to steal (supply the feature basis, let training allocate it, then ask which allocations make removal helpful) and gives a testable prediction: beneficial removals should concentrate where two training objectives disagree about one shared coordinate.

## Tags

unit: head
form: causal-abstraction
method: ablation, probing
eval: ablation, downstream-task
goal: understanding, evaluation-of-methods
