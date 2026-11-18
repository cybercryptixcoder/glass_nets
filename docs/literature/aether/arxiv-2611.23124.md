# Get Hint-Augmented GRPO Done Right Through Activation Steering

- **Key:** arxiv-2611.23124
- **Authors / organisation:** Khiem Le, Ting Hua, Nitesh V. Chawla (three authors; author institutions are not printed on the arXiv abstract page and the entry does not claim them)
- **Date:** 2026-11-17
- **Link:** https://arxiv.org/abs/2611.23124
- **Kind:** paper
- **Read from:** full text (abstract through §8, AI-use statement, Appendices A–G; all tables read as cells)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Group-relative policy optimisation stalls on any training question the base model cannot solve at all: every rollout in the group fails, all rewards are equal, the normalised advantages vanish, and the question contributes no gradient. A family of fixes appends a "hint" — usually a prefix of the reference solution — to the question during training so that some rollouts succeed and a signal appears. The catch the authors identify is that no hint exists at inference time, so the model learns to lean on an aid that is then withdrawn; they measure the size of that shortfall before proposing anything, then propose a method that keeps the aid's effect while removing the text. The larger goal is to convert what a hint *does* to the model into something that is carried in the model's activations rather than in its context, i.e. to make a training-time scaffold available at test time.

## What they studied

Four base models spanning two families and two scales: Qwen3-1.7B-Base and Qwen3-4B-Base, Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct. Training data is a 15k-question subset of OpenR1-MATH-220k, filtered through Math-Verify down from 94k and then restricted to questions whose reference solution fits in 4096 tokens. Evaluation is pass@8 on four competition-mathematics benchmarks (AMC, OlympiadBench, AIME24, AIME25) plus two out-of-distribution sets (Minerva, GPQA-Diamond), with two transfer benchmarks (LogiQA, LiveCodeBench-v5). The unit of analysis is a single **activation-level steering vector** — one vector, one layer, added to the residual at every token — whose coefficients are trained jointly with the policy. The layer is fixed to the model's middlemost block, chosen by convention rather than by a sweep.

## How they did it

Each training step is a two-stage co-training. First, for a sampled question, a hinted version is built by concatenating the question with the first half of the reference solution's tokens — the QUESTA recipe adopted unchanged, so the hint-extraction mechanism is deliberately not the contribution. The model then samples half its rollouts from the hinted question and half from the plain one; both halves are kept unfiltered by correctness, because the aim is to capture the behaviour change rather than the right answers. A DPO-shaped loss on the steering vector is then minimised: the log-probability ratio of each hinted rollout under the steered model versus the unsteered model, minus the same ratio for the matching plain rollout, passed through a sigmoid with a scaling constant of 0.1. Rollouts are paired index-wise rather than over all combinations, which the authors note is sufficient and avoids a quadratic blow-up. Because the reference model in that ratio is the same network being steered, the implicit reward is computed by parallel prefilling over rollouts already generated — no extra sampling.

Second, once the vector has been updated, the policy itself is updated by GRPO on *both* question variants, with advantages normalised within each variant's group separately and the two losses simply summed. The order matters and is stated as such: updating the vector after the policy would leave it fitted to a stale network, so the vector is re-optimised every step to track the model as it grows, and the final step deliberately skips the policy update so the returned vector and weights correspond. The generation budget is held at eight rollouts per question in every arm — four hinted plus four plain for the proposed method, eight plain for GRPO and eight hinted for the baselines — which is what makes the comparison like-for-like. The theory is a two-line chain: the expected reward gap between the hinted and steered-plain distributions is bounded by twice their total-variation distance, Pinsker bounds that by the square root of twice the KL divergence, and minimising the DPO loss is argued to be a surrogate for shrinking that KL.

## What counts as an explanation here

A single direction in activation space, trained online, and an explicit **claim about what that direction contains**: the question-agnostic component of the behavioural shift that hinting produces. It is not a description, a dictionary or a circuit, and it is not claimed to be interpretable — the vector is never inspected, projected onto vocabulary, or given a human-readable label. The explanation is functional and partial: the direction alone is asserted, on the strength of transfer evidence, to carry the part of the hint's effect that generalises to questions it was never fitted to. The paper's own framing — hinting as a form of context-based control, alongside in-context demonstrations and instructions — is the interpretive step that licenses treating the effect as separable from the text that produced it.

## How they judged success

pass@k for k up to 8, with n = 8 samples per question, temperature 0.6 and top-p 0.95, seed fixed; full pass@1/2/4/8 curves are given in an appendix figure and described as consistent with the headline. Baselines are GRPO, QUESTA, SAGE and HILL, each with its public implementation. The confirmatory controls matter more than the headline: (a) the same models scored **with and without the hint** at inference, which prices the crutch; (b) ASH-trained models scored **with and without the steering vector applied**, which isolates the vector's contribution from the joint-training regularisation; (c) two out-of-distribution benchmarks that were never trained on; (d) two transfer benchmarks in different reasoning domains, with the single maths-fitted vector applied unchanged. Wall-clock training time is reported per method per model. There is no human rating and no random-direction control.

## What they found

The mismatch is large and is measured before anything is proposed: a QUESTA-trained model loses 27.34% of pass@8 on Qwen3-1.7B-Base and 10.83% on Qwen3-4B-Base when the hint is removed at inference. Joint training on both variants shrinks those gaps to 21.33% and 8.13%, and applying the trained vector at inference buys back 11.11% and 4.51% — about half of the remaining gap, which is the paper's most honest number. Against GRPO, ASH gains 2.60% and 2.21% average pass@8 on the two Qwen models and 2.17% and 2.14% on the two Llama models; the absolute table shows, for example, Qwen3-1.7B-Base moving from 35.66 (GRPO) and 37.53 (HILL) to 39.50 on the maths average and from 55.05 to 60.84 on the out-of-distribution average, with every one of the 24 benchmark cells improved relative to GRPO. The vector carries roughly 69–83% of the gain over GRPO (68.9/83.1% for Qwen, 68.6/80.2% for Llama), so joint training's regularisation is the minority contributor. The maths-fitted vector transfers without retraining: +9.12/+7.23 pass@8 on LogiQA and +2.64/+3.00 on LiveCodeBench-v5 for the Qwen pair, +4.26/+3.74 and +2.46/+1.54 for the Llama pair. Cost is 1.5× GRPO's training time (15 hours against 10 for the 1.7B, 30 against 20 for the 4B) against roughly 2× for the two strongest baselines.

## Limits and open problems they name

The failure mode is reported as a case study rather than characterised: with the vector applied, the small Qwen models drift slightly off the input question — one solves for a perfect square when the problem asks for a fourth power, the other answers the value of the number when asked for the base — and the authors attribute this to the capacity of a single global vector. They name task- or question-specific vectors as the future work. The layer is fixed at the middlemost block and justified by the cost of a layer sweep rather than by evidence that it is the right layer, the scaling constant is inherited from DPO, and there is no ablation of either. The theory bounds the gap by a KL the loss only surrogate-minimises, so it establishes a direction of improvement rather than a guaranteed size. Nothing here inspects the vector: no projection onto output vocabulary, no comparison across seeds, no test of whether two runs trained on the same data give the same direction.

## What we take from it

- **A construction, and the one worth copying:** when an intervention has an effect, the effect has a *text-carried* part and a *state-carried* part, and the state-carried part is recoverable by fitting a direction to reproduce the intervention's preference — while holding the supervised target fixed so the direction cannot absorb the reward. The DPO framing with the unsteered model as the reference is what keeps the fitted quantity a *behavioural delta* rather than an answer.
- **A measure:** the explicit with-versus-without-hint evaluation of the *baseline*, before proposing anything. This is the ablation that tells you whether the help was real or remeetable, and it is the first table in most papers of this kind that is worth reproducing.
- **A warning about one-direction claims:** a single vector recovers about half of the gap and produces question-drift on small models. Whenever our own read-out reduces a behaviour to one direction, the honest report includes the residue it does not recover and a concrete failure case, both of which this paper does print.
- **A framing we can reuse:** treating a prompt-scaffold as a control signal rather than as extra difficulty reduction is the same move as reading a prompt-injected behaviour as a direction — it makes the scaffold's effect measurable *as a direction* and therefore predictable at inference.

## Tags

unit: direction | layer
form: direction, steering
method: steering, other
eval: ablation, benchmark, downstream-task
goal: control/editing, understanding
