# The Integer Alibi: Localizing Cross-Kernel Divergence in INT8-Quantized LLM Inference

- **Key:** arxiv-2608.13756
- **Authors / organisation:** Teng-Ruei Chen; Krixvon (Taipei); preprint, preprint-pinned protocol with append-only amendments
- **Date:** 2026-08-13 (v2; v2 corrects the product-bound attribution from the activation side to the weight side)
- **Link:** https://arxiv.org/abs/2608.13756
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Serving engines carry several implementations of the same quantized linear layer and pick one at load time by
hardware check, environment variable or heuristic — a choice that appears in no model card and no benchmark
table. The working assumption is that the implementations are interchangeable: same operands in, same answer
out up to negligible noise. The author tests that assumption while changing exactly one thing, then asks a
sharper question than "do they disagree": *where* in the arithmetic the disagreement starts. The larger goal is
a conformance contract for quantized kernels — a check that can be run before deployment rather than inferred
from a benchmark discrepancy afterwards. Two intermediate ambitions carry the paper: a per-layer prediction of
where bitwise agreement is *required*, made before any comparison is run, and a counterfactual repair that must
remove the divergence if the localisation is right.

## What they studied

Qwen3 at 1.7B and 8B, quantised W8A8 INT8 (channel-wise symmetric weights, dynamic per-token activations). The
treatment is a single kernel substitution inside a pinned vLLM build: CUTLASS's scaled-INT8 GEMM versus the
Triton implementation, selected by one environment variable. Prompts are 64 deterministic 320-token WikiText-2
windows with greedy decoding. The unit of analysis is the **linear layer** — 196 for the 1.7B model, 252 for the
8B, 448 in the layer-level comparison — with a second unit at the level of the arithmetic stage (accumulation
versus scale application versus output rounding) and a third at the token, via teacher-forced replay. A
cross-implementation FP8 GEMM pair, and an SGLang-versus-vLLM engine comparison, act as format and level
contrasts.

## How they did it

The stack is pinned by container digest on one RTX 4090, and the protocol — estimators, the 1.7B per-layer
prediction list, acceptance criteria, analysis plan — was hash-locked before any backend-comparison measurement,
with later changes as append-only amendments; the 8B prediction list was pinned before the authoritative rerun
but after an earlier result on the same data had been seen, and the paper separates the two cases rather than
pooling them. The arithmetic argument is in three stages. First, for symmetric int8 operands under a verified
no-overflow bound the INT32 accumulator is exact and order-free, so tiling, split-K and reduction-tree choices
cannot change it — the reduction stage has a proof it could not have caused a difference, which is the "alibi".
Second, an integer accumulator is exactly representable in float32 while below the 2^24 threshold, so an fp64
emulation over the pinned prompts yields, per layer, a falsifiable prediction of where the epilogue receives an
unperturbed value and bitwise agreement is therefore mandatory. Third, multiplying by a power of two commutes
with rounding for finite normal values, so rewriting weight scales to powers of two removes the epilogue's
remaining freedom and turns "equality expected" into "equality required". Measurement follows: capture the int8
operands from every linear layer, feed the *identical* captured tensors to both kernels twice (once with scales
forced to powers of two, once at the checkpoint's real scales), and compare bit patterns. Then the same idea
end-to-end: rewrite all 573,440 weight scales of the 1.7B checkpoint to their nearest power of two, leave
weights, activation quantisation, prompts and both kernel arms untouched, and regenerate. Teacher-forced replay
on identical contexts separates single-position flips from autoregressive cascade, and the logit margin is
tested as a predictor of flips. Governance is part of the method: every teacher-forcing artifact carries a
manifest with prompt-list, checkpoint, quantisation-config, tokenizer and container digests plus the kernel
class read from the engine's own selection log, and the join fails closed on any mismatch — so "only the kernel
differs" is an audited property of the artifacts, while the earlier end-to-end token artifacts predate the
contract and are labelled unaudited on that dimension.

## What counts as an explanation here

An **exact control plus a counterfactual repair**, not a circuit and not a description. The explanation is a
three-way partition of the arithmetic (exact accumulation / epilogue scale application / output rounding), a
per-layer prediction list of where lossless FP32 entry is guaranteed, and a localisation claim of the form "the
difference can only originate in the epilogue, because the stage before it is exact by theorem and the stage
after it is one rounding". It is meant to be complete *within the INT8 GEMM stage and nowhere else* — the paper
says outright that attention, normalisation, rotary embeddings and the unquantised head are outside its
jurisdiction — and it predicts two things it then checks: that differences cannot grow with reduction depth on
the integer path, and that power-of-two scales make the difference vanish. The across-token part is a
distributional account rather than a mechanism: flips concentrate at small logit margins, and the margin is
calibrated into a flip-risk estimate.

## How they judged success

Pre-registered per-layer prediction accuracy (predicted bitwise-safe layers confirmed by measurement, reported
as exact counts, not a rate). Bitwise divergence as bit-pattern inequality, kept separate from numerical ulp
distance on finite pairs only, with signed zeros counting and near-zero elements sub-reported. Unit tests of
each arithmetic fact against an INT64 reference and against IEEE 754 behaviour at the 2^24 boundary, including
all-extreme operands so the product bound is exercised rather than assumed. A divergence-versus-reduction-depth
sweep from K = 512 to 32,768, with the FP8 series run over 20 seeds and four pre-listed growth models fitted
under one common objective with held-out folds, seed bootstrap, jackknife and leave-one-K-out. Token-level
statistics clustered by prompt (cluster bootstrap, leave-one-prompt-out influence), grouped-tie average
precision, and calibration by merged quantile bins with Beta(1,1) smoothing against a constant-prevalence
Brier baseline, with decreasing-isotonic regression as a pre-specified sensitivity check. An identity contract
that fails closed, and 72 regression tests drawn from counterexamples an independent automated reviewer raised.
A separate 200,000-sample experiment on the epilogue arithmetic alone with a binomial upper bound.

## What they found

The two arms are not noisy versions of each other: at 1.7B each reproduces itself bit-for-bit across two cold
restarts, yet they agree on **no** greedy sequence in any comparison run (0/8, 0/16, and 0/64 in the largest
rail run), and a third implementation (SGLang) gives a third distinct deterministic function. Every one of the
448 layers cleared the 2^24 threshold under the pinned prompts, with at least 2.82 bits of headroom at 1.7B and
3.36 at 8B, so the pre-registered prediction list marked all of them bitwise-safe — and under power-of-two
scales all 448 came out bit-identical (196/196 and 252/252). Under the checkpoints' real scales only 8 of 196
and 10 of 252 layers were bit-identical (the eight survivors at 1.7B are small-NN projections where rounding
coincidence holds), yet the largest difference anywhere was **one bfloat16 spacing**, with no element beyond one
ulp and no non-finite outputs; the per-element differing fraction is around 5×10⁻⁶. The power-of-two probe
checkpoint restores end-to-end bitwise agreement (8/8 at 1.7B, 16/16 at 8B), which is the paper's strongest
evidence that the divergence lives in scale application and output rounding. The format contrast goes the
expected way: across the same reduction-depth grid the INT8 arm produced 18 differing elements in total (1.9 to
7.6 parts per million, no trend detectable at that sample size, consistent with a constant-rate model), while
the FP8 pair grows from 8.2% to 52.7% of elements with a fitted saturation exponent near 0.518, and its tail
lengthens too — the share at ulp distance ≥ 2 rising from 8% to 33% among differing elements. Teacher-forced
replay over 16,384 positions shows 769 flips (4.7%), falling monotonically from 34% in the lowest-margin decile
to zero above a margin near 9, and the margin alone ranks flip risk at ROC-AUC 0.942 (prompt-cluster CI
[0.935, 0.949]) with held-out Brier 0.0352 against a 0.0440 constant-prevalence baseline. Three honest negatives
sit beside those numbers: position-exact prediction fails (the first teacher-forced flip position does not match
the first free-running divergence position); a same-arm prefill-versus-decode comparison flips 8.6% of rail
tokens under real scales and still 2.9% under the probe, unresolved and unattributed because the
first-divergence instrumentation stopped at the engine glue; and a separate 200,000-sample test of the two
epilogue multiplication orderings alone produced zero bf16-visible differences.

## Limits and open problems they name

No positive control — not one conformance check has been run against a deliberately faulty kernel, so their
sensitivity is unmeasured; every check reported is a check that passed, not one demonstrated to fire. The 2.9%
residual under the probe is reported without module attribution. The power-of-two probe is a conformance
instrument and explicitly not a validated mitigation: no accuracy, calibration or throughput consequence of
pow2-rounded scales was measured. Whole-model inference carries no bitwise guarantee, since the alibi covers the
integer GEMM stage only. One consumer GPU, one model family at two sizes, one engine version. Layer-level
localisation uses a single prefill-like tile (M=512) so the decode regime is untested there, and the isolated
operator probe at M=1 is too small to have power against a 5×10⁻⁶ per-element rate. A pre-registered batch sweep
{1,4,16}, the forward-and-reverse ordering rounds, and a task-level McNemar test were never executed and are
listed as deviations rather than dropped. Environment rebuild is partial (no wheel hashes, a fixtures directory
absent from the released tree), and the historical checkpoints lack calibration-dataset revisions. The repo is
private at the time of writing. The FP8 depth-dependence is offered as compatibility with a
square-root-shaped mechanism from seven grid points, not as a law.

## What we take from it

A **method and a measure**, and the cleanest instance the folder has of a control that narrows the hypothesis
space by argument rather than by ablation: if a stage's output is provably exact and order-free on the operands
in hand, then every downstream difference is attributable to the stages below it, and a pre-registered
prediction of *where equality is mandatory* is a stronger test than a post-hoc comparison of two runs. For us
the transferable object is the shape of the claim — a three-stage decomposition where one stage is acquitted by
a bound that can be verified on the actual inputs, one stage yields a per-unit falsifiable prediction list
committed before measurement, and the remaining freedom is removed by an intervention whose success is required
rather than hoped for. Second, the **audited-artifact discipline**: digests for every input plus the treatment
variable captured from the runtime's own log, joined with fail-closed semantics, and the split between what was
audited and what was not stated in the same paragraph as the result. Third, a **warning that applies directly to
our own harness**: the kernel or implementation that produced a number is an unlogged hyperparameter, so any
internal measurement we replicate across runs must pin and record it, since here two arms that each look
perfectly deterministic are different functions. Fourth, the **limitation list is the model of how to report a
control that has never failed** — we should say "this check passed" and never "this check works" until it has
been fired on a deliberately broken subject.

## Tags
unit: weights | layer | whole-model
form: causal-abstraction | description
method: weight-analysis | ablation | other
eval: prediction/simulation | benchmark | faithfulness
goal: debugging | evaluation-of-methods | science
