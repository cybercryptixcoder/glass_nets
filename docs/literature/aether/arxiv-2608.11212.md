# Detecting a Route Flip Is Easier Than Knowing Whether to Fix It: Causal Route-Mediated Damage in Quantized Mixture-of-Experts

- **Key:** arxiv-2608.11212
- **Authors / organisation:** Parvel Gu (sole author, independent researcher)
- **Date:** 2026-07-14 (v1; HTML build stamped August 2026)
- **Link:** https://arxiv.org/abs/2608.11212
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Top-k Mixture-of-Experts routing is a hard argmax over expert logits, so it is discontinuous: a small numerical
perturbation of the router's input can move a token onto a different expert set. The paper studies one
deployment-realistic perturbation — the KV cache stored at simulated 4-bit precision while the router itself keeps
running in BF16 on the dequantized state — and asks two questions in sequence. First, how much of the resulting
loss damage travels through the routing decision rather than through the experts' arithmetic on corrupted inputs.
Second, whether a signal visible at inference can tell a harmful route flip from a helpful one, because any
selective repair scheme needs exactly that distinction. The authors state plainly that they propose no mitigation;
the goal is a measurement apparatus, an effect decomposition, and an honest negative result about what local router
statistics can and cannot see.

## What they studied

OLMoE-1B-7B (16 layers, 64 experts, top-k = 8, `norm_topk_prob = False`) is the primary model; cross-model probes
use DeepSeek-MoE-16B, Qwen1.5-MoE-A2.7B and Qwen3-30B-A3B. Text comes from a calibration split built over
WikiText-103 and C4; a reserved test split was read once, under a pre-registered protocol, at the end. Decoding is
teacher-forced: 64-token prefill, then decode, with per-token NLL as the primary metric. The pilot sample is 96
sequences (18,432 decode tokens), with an expanded nested re-estimate at 384 sequences where measured. The unit of
analysis is the token's expert set at every layer and the per-token loss difference between two counterfactual
executions.

## How they did it

Four forward runs per contrast give a 2×2 of compute state (clean vs. quantized) against route state (clean vs.
quantized): Y_CC clean compute with free route is the baseline, Y_QF is the deployment condition, Y_QP runs quantized
compute but pins the clean expert set and clean expert weights verbatim, and Y_CT runs clean compute with the
quantized route transplanted in. Pinning is an interchange intervention on the routing mediator, not an ablation.
The route-mediated fraction is a ratio of summed excess losses: (L(Y_QF) − L(Y_QP)) divided by (L(Y_QF) − L(Y_CC)),
with sequence-cluster bootstrap intervals. Because the clean-context path (Y_CT − Y_CC) turns out smaller than the
quantized-context path, the authors report the compute × route interaction as a measured quantity rather than
assuming the two channels add. Per-token attribution then splits the routing channel into three disjoint masks: a
*jump* at the layer whose output loss is read, a *nonlocal* change at any upstream layer, and a residual *pure flux*
label. For the detection question they take the router margin (weakest selected logit minus strongest unselected
logit), the most directly deployable single-layer statistic, and train a sequence-held-out classifier for flip
occurrence, for harm given a flip, and a probe battery (logistic, MLP, gradient-boosted trees, a temporal
route-history predictor, and an 80-dimensional cross-layer router vector) for the sign of the harm. Selective repair
is then *executed* rather than summed: an oracle gate, a rate-matched random gate and the deployable margin gate each
pin a chosen subset of tokens and the realized recovery is measured.

## What counts as an explanation here

A causal decomposition of a behavioural quantity (loss damage) into named channels of a mechanism (compute path vs.
routing path), plus a mechanism partition of the routing channel by which layer's route changed, plus a calibrated
statement of the limits of a feature family. It is deliberately partial: the paper claims no prediction of the damage
magnitude from observables, only a price for the channel and a barrier for detection. The negative result is scoped
explicitly to "local, inference-observable router statistics" and the authors name richer hidden-state or trained
decoder information as untested territory.

## How they judged success

Interventional contrast with sequence-cluster bootstrap CIs, a measured interaction term, and an INT8-activation
near-null control whose interaction shrinks toward zero; a token-level partition that sums exactly to the total
routing harm; AUCs on sequence-held-out splits for flip occurrence, harm-given-flip, and benefit-vs-all, each
compared against a rate-matched random selector with paired CIs; a pre-declared falsification band of
|AUC − 0.5| ≤ 0.05; a real int4 KV kernel as an independent-disturbance check; a five-process replication with an
intraclass correlation and a per-token between-process SD floor; and a held-out read of three pre-registered
endpoints reported with the one miss undiluted.

## What they found

The route channel carries a substantial minority of the damage: RMF ≈ 0.31 [0.20, 0.41] on OLMoE at the 1× dose
(five fresh processes mean 0.313 ± 0.020, range 0.288–0.339, pre-registered re-execution 0.231), with the
compute × route interaction positive at 0.0083 nats and excluding zero, against an INT8 control interaction of
−0.00006. The named mechanism per-token verdict is 45% jump, 55% nonlocal, ≈0.2% pure flux, and 99.8% of the net
signed contribution attaches to a route-set change somewhere, so a single scoring-layer indicator misses most of it.
Detection separates cleanly from usefulness: the deployable margin scores flip occurrence at AUC 0.772, harm given a
flip at 0.490 (with P(harmful | flip) = 0.572), and benefit-vs-all at 0.499, a mixture the paper reconstructs from
two negative groups; every richer local predictor, including the all-16-layer router vector, sits inside the
falsification band, and the executed selective policies put the deployable gate *below* rate-matched random.
Cross-model, DeepSeek-MoE-16B shows the same near-cancellation (harmful +0.2036, beneficial −0.1819 nats, 90.2%
cancellation) and the same sign-inseparability (0.507), while the oracle's advantage is null on OLMoE (+0.4%,
CI includes zero) and positive but small on DeepSeek (+2.3%). Clean-route pinning recovers a bounded, architecture-
modulated slice of the routed ceiling (OLMoE +0.231, DeepSeek 0.456, Qwen3 prefix-only 0.021 null), and a controlled
same-checkpoint flag swap shows the routing normalization convention moderates *damage magnitude* by roughly seven to
ten times rather than gating recoverability. The real int4 kernel gives 0.194 [−0.111, 0.394], compatible with the
fake-quant dose curve but underpowered. Out of sample, the partition replicates at 0.429 [0.330, 0.519] and the
near-cancelling tax holds at +0.0043 nats, while the strict impossibility exclusion misses by 0.0002.

## Limits and open problems they name

Pilot scale (96 sequences, with 384 where measured), one primary disturbance dose, NLL as the primary metric, and a
small numerator (route_fwd ≈ 0.018 nats) that makes the RMF a ratio of noisy quantities; the authors note fixed-process
re-estimates ranged ≈0.23–0.33. Task-level reach is unresolved: on ARC-Easy the tax replicates in NLL but produces no
resolvable accuracy drop. The real-kernel arm diverges from the fake-quant curve at aggressive bit widths (declining
0.194 → 0.129 → 0.090 for int4 → int3 → int2) and is left as future work, and the aggressive-dose peak in simulation
(58% near 5.4×) is flagged as probably over-stated. What modulates cross-architecture recovery magnitude is open at
n = 3, as is any predictor outside the local router family.

## What we take from it

A method and a severity measure. The four-run design — clean compute/free route, quantized compute/free route,
quantized compute/pinned route, clean compute/transplanted route — is the cleanest template in the folder for pricing
*which channel* of a black box carries a perturbation's damage, and the requirement that the interaction be measured
rather than assumed is a discipline our own decompositions should adopt. The severity measure is the detection-versus-
usefulness pair: any internal signal our white-box model proposes as a repair trigger must be scored twice, once for
whether it detects the event and once for whether it identifies the events worth acting on, with a rate-matched
random selector and an oracle as the two bounds. The barrier here is also a warning for intervention-based claims
generally: near-cancelling signed components mean a net number near zero can hide two large opposite effects, so an
entry that reports only the mean effect of a route change is hiding its own tax structure.

## Tags

unit: circuit, whole-model
form: graph, description
method: patching, ablation, probing, decomposition
eval: ablation, patching, prediction/simulation, benchmark
goal: understanding, debugging, evaluation-of-methods
