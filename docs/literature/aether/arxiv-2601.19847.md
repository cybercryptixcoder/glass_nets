# Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering

- **Key:** arxiv-2601.19847
- **Authors / organisation:** Fangan Dong, Zuming Yan, Xuri Ge, Zhiwei Xu, Mengqi Zhang, Xuanang Chen, Ben He, Xin Xin, Zhumin Chen, Ying Zhou; Shandong University; Institute of Software, Chinese Academy of Sciences; University of Chinese Academy of Sciences (EMNLP 2026 main)
- **Date:** 2026-01-27 (v3 revision 2026-09-12)
- **Link:** https://arxiv.org/abs/2601.19847
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Reasoning accuracy is usually bought with post-training or with extra inference compute — sampling several
trajectories, reranking with a verifier, self-consistency. The authors want the same reliability gain at single-pass
decoding cost, by editing the model's insides instead. Three questions structure the paper: can correctness of a
multi-step reasoning run be predicted from internal activations at all; is there a small set of MLP neurons whose
state carries that signal and can be pushed; and can such a set be found in one model and reused in another dataset,
another task or a larger model. Their stated frustration with black-box test-time methods is that those methods
spend compute without saying where the error came from; an intervention on named neurons is meant to be both
cheaper and more legible.

## What they studied

Qwen3 at 1.7B and 4B (plus Qwen3-4B-Thinking-2507 and Qwen3-14B in an appendix), on ten benchmarks: AIME-24,
AIME-25, an extended AIME set, MATH-500, GSM8K, AMC-12, HumanEval, HumanEval+, MBPP and MBPP+. The unit of analysis is
the **MLP intermediate neuron** — the post-SwiGLU activation — indexed by (layer, index), with a secondary unit at the
level of the whole reasoning trace, summarised by two geometric features over layers. Contrast traces are generated
with Qwen3-32B in the probing study and by self-sampling from the model under test in the steering study. The
dependent variable is end-task **accuracy**, and the account is evaluated as a detector of failure plus a causal edit,
not as a circuit.

## How they did it

A preliminary probe first: take last-token MLP activations across all layers, keep only questions that yield exactly
four correct and four incorrect sampled traces, and fit an L1 logistic regression on the 131,072 most discriminative
neurons by ANOVA F-statistic. Then the intervention pipeline. (1) *Identification*: for each neuron compute a
reasoning-critical score as the average over matched trace pairs of the mean activation in the correct member minus the
mean in the incorrect member, averaged over the whole trace rather than the last token — a mean-difference criterion
computable without training a probe at all. (2) *Selection*: keep only neurons whose mean activation has **opposite
sign** in correct and incorrect traces, then rank the survivors by |score| and take the top K per layer, which builds
one sparse steering vector per layer. (3) *Injection*: add alpha times that vector to the MLP intermediate activation
inside the block, uniformly at every decoding step. (4) *Gating*: because early trials showed steering fixes wrong
trajectories and spoils right ones, a small attention-pooled classifier over 256 F-selected features reads the prompt's
early activations and predicts failure; steering fires only when failure is predicted. Contrast data come from
high-temperature self-sampling, pairing traces of the same question with opposite correctness. Controls are a random-
neuron steering arm, a magnitude-only selection arm with the same K, a probing-weight selection arm, and arms that
drop the gating.

## What counts as an explanation here

A **sparse neuron set plus a steering vector**, i.e. a localization claim and a control handle rather than a mechanism.
The claim is of the form "this small group of MLP neurons, mostly in the middle-to-late layers, carries a signal that
separates correct from incorrect reasoning, and pushing them toward the correct-trace mean state raises accuracy". It
is explicitly partial: no circuit, no account of what the neurons compute, and the paper's own third limitation says
the mechanism of the set is left for SAE or vocabulary-projection work. It does predict things it then checks —
that the set is tiny, that it is polarity-structured, and that it transfers. The Geometric features (path magnitude and
path angle) are a secondary descriptive layer: they say the intervention straightens the latent trajectory without
rotating it, which is offered as evidence that the edit is narrow rather than as an explanation of the gain.

## How they judged success

Accuracy on the ten benchmarks under a single greedy decoding protocol against vanilla CoT on the same base model,
against three post-trained models of comparable size (R1-Distill-Qwen-1.5B, OpenThinker3-1.5B,
OpenReasoning-Nemotron-1.5B), and against a probing-based steering arm given the same number of neurons. Harder
controls follow in the appendix: majority voting over four samples and pass@4, verifier-based best-of-4 reranking,
three-shot CoT prompting, and the post-trained models under those same sampling metrics, so single-pass steering is
compared against arms that spend more compute. Cross-dataset transfer is measured by swapping in RCNs identified on
AIME for RCNs identified on the target dataset. Ablations remove mean-difference, polarity filtering and the gating one
at a time. Sensitivity sweeps over alpha and K, a prompt-robustness check with and without the "think step by step"
phrase, a sampling-seed reproducibility check over four runs, a response-length and dependency-depth analysis to rule
out formatting effects, and an AUROC for the failure predictor.

## What they found

The probe does separate the two outcomes from last-token activations alone, at AUROC 0.7639 and 0.7153 on AIME for the
1.7B and 4B models and 0.7091 and 0.6727 on AMC-12 — moderate, not decisive, and the gating classifier reaches 0.8347
on AIME with a global read of the prompt. Steering then moves accuracy on all ten benchmarks: the largest gains are
AIME-24 47.83 to 60.87 and AIME-25 40.91 to 54.55, with +13.04 and +13.64 attributed to the method, about 5% on
average, 9.11% on the hard set against 2.26% on the easy one, and the 4B model improves on AIME-25 from 59.09 to
72.73. The set is small: performance peaks near top-50 neurons, roughly 0.03% of all neurons, and adding more than that
hurts — top-2000 (about 1.20%) costs roughly 6% on AIME-24 relative to top-50. The polarity filter is the sharpest
single ingredient: keeping the same top-50 but selecting by magnitude alone gives 56.52 on AIME-24 and 45.45 on
AIME-25, against 60.87 and 54.55 with the sign-flip criterion. Transfer goes slightly *up* rather than down: RCNs
identified on AIME beat dataset-specific ones on every transferred benchmark, by under 1% each (MATH-500 87.40 vs
86.40, HumanEval 80.54 vs 79.19), including across the task boundary from maths to code. Two results cut against the
headline. The probing-based steering arm is not merely weaker but unstable in sign: it loses 4.35% on AIME-24 relative
to unsteered CoT while beating AdaRAS on AMC-12 (73.63 vs 70.33) — the paper reads this as sensitivity to neuron
selection. And the gating is not the source of the gain: adding the same gate to the probing baseline helps on AIME-24
(43.48 to 52.17) and hurts on AIME-25 (50.00 to 40.91). Alpha is peaked but forgiving, best at 0.3 on AIME-24 and 0.4
on AIME-25, returning to baseline only near 0.7.

## Limits and open problems they name

Two model families and STEM benchmarks only; spatial and multi-hop reasoning untested. Identification needs paired
correct and incorrect traces for the same question, which is hard exactly at the capability extremes where steering
would matter most. The neuron-level mean difference is the only identification criterion tried; SAE features and
vocabulary projection are named as the obvious next probes. Beyond those, the paper leaves unexamined: whether the
selected set is stable across retrains of the same model, since every arm here reuses one selection; what the alpha
sweep does to the *negative* direction, i.e. whether steering away from the correct-trace mean reliably hurts; whether
the transfer advantage is a property of AIME specifically (it is the hard source set, and the "hard tasks give better
neurons" reading is asserted from one source only); and what the neurons are — the case studies show the corrected
traces fixing a domain-constraint check, a log telescoping and a geometric relation, but no single neuron is
attributed to any of them.

## What we take from it

A **measure and a warning**, twice over. First the measure: the polarity criterion — retain a candidate unit only if its
mean state has opposite sign between the two outcome classes — is a one-line filter that here outperforms the same
top-K selected by effect magnitude by 4 to 9 points, and it costs nothing but the two class means we already compute.
For a white-box model whose units are signed, that is a cheap structural prior on which units are plausible handles,
and it is testable on a single contrast pair. Second, the **sign-instability of the probing arm** is the folder's
cleanest reminder that the ranking statistic and the intervention statistic are different objects: a decoder can rank
neurons well by weight and still produce a steering vector that pushes trajectories wrongly, which is why this paper's
contribution is the criterion shift, not the decoder. Third, the warning about the **gate doing the work**: the
adaptive-intervention arm is the most intuitively appealing part of the design and the ablation shows it is not what
carries the gain — a reminder to run our heads-off version of any auxiliary module before crediting it. Fourth, the
**sparsity ceiling**: peak at ~0.03% of neurons, harm at ~1.2%, is the sharpest dose-response curve in the folder for
"how many units should a handle touch", and it argues against the large dictionaries we habitually compute. What we
should not take: the transfer result is a single-source observation with sub-1% margins on 90-to-1,319-sample test
sets, and it should be cited as a hint about where reasoning-relevant units live, not as evidence that a set found on
one task generalises.

## Tags
unit: neuron | direction | whole-model
form: steering | probe | direction
method: probing | steering | other
eval: downstream-task | ablation | benchmark
goal: control/editing | understanding
