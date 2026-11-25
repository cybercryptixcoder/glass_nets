# Conditional Optimal Bridge for Riemannian Activation Steering

- **Key:** arxiv-2607.10517
- **Authors / organisation:** Seyed Arshan Dalili, Ajay Narayanan Sridhar, Vijaykrishnan Narayanan, Mehrdad Mahdavi; Pennsylvania State University / Georgia Institute of Technology
- **Date:** 2026-07-12 (v2 2026-10-21)
- **Link:** https://arxiv.org/abs/2607.10517
- **Kind:** paper
- **Read from:** full text (v2 HTML, main text + Appendices A, C, D, E, F, G; Appendix B read only where cited)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Activation steering is usually justified by the linear representation hypothesis and implemented as a
fixed direction added to every activation. The authors observe that this recipe has no principled
statement of *what objective it is optimising*, and that it has no way of knowing whether the
contrastive data support intervening on the prompt in front of it. They read the second failure as
the one that shows up as capability loss: in their own runs several strong baselines raise TruthfulQA
while visibly damaging MMLU and GSM8K. Their goal is therefore twofold — derive the commonly used
log-density-ratio steering objective from a well-posed transport problem, and get a criterion, from
the same fit, for deciding when an input should not be steered at all. This is a step toward the
folder's recurring question of scoring an edit before it is applied rather than after.

## What they studied

Four open decoder-only models at the 7-8B scale: Falcon-7B, Mistral-7B-v0.3, LLaMA3.1-8B and
Qwen2.5-7B. The unit of analysis is the last-token residual-stream activation at one middle layer,
treated as a point on a sphere — the authors measure that per-layer norms have a coefficient of
variation between 2.5% and 7.3% on the main four models, and their Appendix table over eight
architectures puts later-layer CoV at most 0.085 for the main four and 0.146 across all eight, so
they restore the norm after each step and intervene on direction only. Two contrastive populations
of activations (desired and undesired) are estimated empirically: pairs from TruthfulQA for
truthfulness, UltraFeedback-Binarized preferences for helpfulness, and a balanced Jigsaw set for
toxicity. The object being fitted is a map between those two empirical distributions, not a single
direction vector.

## How they did it

Steering is cast as a Schrödinger Bridge on the hypersphere: find the trajectory distribution closest,
in KL divergence, to Brownian motion on the sphere, subject to starting at the undesired activation
distribution and ending at the desired one. The solution factorises into two positive potentials whose
log-ratio gradient gives the drift of the probability-flow ODE, and whose product gives the density of
the bridge at each transport time. Because the endpoint distributions are only observed as finite
contrastive sets, the authors solve the static version on a matrix of sample pairs with Sinkhorn
iterations, using a geodesic-cost Gibbs kernel as the reference weights; five sweeps from zero
log-potentials, with the regulariser set to the squared median 5-nearest-neighbour geodesic distance
in the pooled contrastive set. To turn the discrete potentials into a field that can be evaluated at
a previously unseen activation, they build two kernel-weighted sums of the potentials over each
population and differentiate the log-ratio, which expands into an attraction term toward desired
samples minus a repulsion term from undesired samples, each tangent vector taken through the sphere's
log map. The intervention then integrates K geodesic Euler steps of a fixed arc length along the
normalised field's current direction, with the arc scaled by the activation's initial angular
separation from the mean-difference axis.

Two attenuation factors are applied. An alignment coefficient is a von Mises-Fisher sigmoid in the
cosine between the activation and the normalised mean-difference axis, so activations that already sit
on the desired side receive smaller steps. A support gate multiplies the two kernel sums of the fitted
potentials into an unnormalised density score at the prompt activation, converts it to a negative log,
and compares it with the same score computed over held-out reference prompts; the gate equals one
below a coverage percentile and decreases linearly above it, reaching zero when the prompt scores
lower than every reference prompt. No out-of-distribution prompts are used to calibrate it. The gate
is computed once on the last prompt token and reused for the whole continuation. A simplified
standalone variant is also given: a nearest-neighbour gate on geodesic distance to the k-th undesired
sample, calibrated on a quantile of leave-one-out neighbour distances within the contrastive set,
which wraps any existing method by scaling its displacement.

## What counts as an explanation here

A transport map, and from it two derived quantities: a query-dependent steering field and a support
score. Neither is a decomposition of the network; nothing is localised to heads, features or circuits.
The explanation is of the *intervention*, not of the model — what direction to move an activation for
this particular input, and whether moving it at all is warranted. The support score is the interesting
part for us because it is a one-number prediction about the edit, made before the edit: a scalar
readable at inference that says how much of the target concept's contrastive evidence sits near this
activation. The paper shows it is really a nearest-neighbour distance in the small-bandwidth limit,
which makes it a density score rather than a learned classifier, and shows the fitted transport
collapses to ordinary kernel density-ratio steering when the reference weights are flattened.

## How they judged success

Three research questions: in-distribution steering quality, capability retention under distribution
shift, and component ablations plus cost. In-distribution metrics are TruthfulQA truthfulness times
informativeness (judged by two fine-tuned LLaMA2-7B binary judges) with truthfulness and
informativeness reported separately; UltraFeedback win rate against the unsteered model under a
reward model, with mean and 90th-percentile reward; and Perspective API toxicity with GPT-2-XL
perplexity and Dist-N diversity. Distribution shift is tested two ways: within-task (fit on
TruthfulQA, evaluated on NQ and TriviaQA answer-ranking accuracy) and cross-task (MMLU and GSM8K
accuracy in a 5-shot format) — the latter being the escalation the paper is built around. Support-gate
quality is scored by AUROC for separating out-of-distribution prompts from TruthfulQA prompts and by
the mean gate value retained. Ablations replace the spherical geometry with Euclidean, replace the
transport sample weights with uniform weights, remove the alignment coefficient with a
step-length-matched control, and remove the gate. Every number is a mean over three seeds, with
standard deviations on the primary TruthfulQA metric.

## What they found

In-distribution, the method is best on the primary metric in essentially every model-task cell. Win
rate on UltraFeedback is highest for all four models, up 2.8, 7.1, 9.7 and 1.9 points over the
unsteered model for Falcon, Mistral, LLaMA and Qwen respectively. TruthfulQA truthfulness times
informativeness is highest on all four, with the largest gaps on Mistral (+29.5) and LLaMA (+25.1).
Toxicity drops 41-54% relative to unsteered, and is lowest on three of four models — the exception is
Falcon, where CAA reaches lower toxicity but at a perplexity of 75.55 against 15.30.

The support score is where the paper earns its place. AUROC for separating out-of-distribution prompts
from TruthfulQA ranges from 83.8 (NQ, LLaMA) up to 100.0 (MMLU, both models tested), averaging 93% on
LLaMA and 97% on Mistral. Mean retained gate value at a coverage of 0.7 tells the same story in
operational terms: about 85% of the step is kept on TruthfulQA, 49.9% on NQ, 30.5% on TriviaQA, 7.3%
on GSM8K and 0.0% on MMLU for LLaMA. The paper is candid that part of this separation is prompt
format rather than task: GSM8K and MMLU use different few-shot formats while NQ and TriviaQA reuse the
TruthfulQA instruction, so the scores reflect both shifts.

The ablations are the most quotable part. Removing the support gate raises LLaMA's TruthfulQA score
from 63.6% to 67.6% while GSM8K collapses from 49.5% to 0.6% and MMLU from 61.6% to 44.4% — so
essentially all of the ungated gain is bought with cross-task capability. Replacing spherical geometry
with Euclidean geometry, recalibrated at the same coverage, costs 2.1 points of TruthfulQA but drops
GSM8K to 15.1%: the geometry, not just the gate, is doing OOD work. Removing the alignment coefficient
raises truthfulness from 74.8% to 81.5% and cuts informativeness from 88.2% to 52.1%; a
step-matched control at a reduced horizon recovers only part of it, so the coefficient is not just a
step-size knob. Uniform sample weights change little (62.9% against 63.6%), which is the paper's own
admission that the transport weighting is a small effect next to the gate.

The standalone nearest-neighbour gate is a genuinely reusable result. Applied to CAA, ODESteer,
SphericalSteer and the proposed method on LLaMA, it restores GSM8K to 47.3-49.5% (unsteered 48.7%)
and MMLU to exactly 61.6% for all four, at a cost of 1.8-4.5 points on TruthfulQA — and with the
common gate in place the proposed method still leads on the target metric (63.6% against 61.7% for
SphericalSteer), so gating does not flatten the ranking between methods.

Sensitivity is reported honestly and is large. The coverage percentile moves LLaMA's TruthfulQA from
62.1% to 66.3% as it goes from 0.6 to 0.9, with NQ falling from 66.9% to 60.3% and GSM8K from 49.7%
to 43.6%. The neighbour gate's calibration quantile is sharper still: 65.5% TruthfulQA and 47.5%
GSM8K at 0.98, then 69.2% and 11.6% at 0.995. The authors explain the cliff with a diagnostic — median
neighbour distances for TruthfulQA and GSM8K prompts sit at the 96th and 99.2nd percentiles of the
contrastive reference distribution — and attribute the fragility to a calibration mismatch, because
the reference radius is computed between contrastive question-answer pairs while the gate is applied
to prompt activations. Their own fix is the support gate, which is calibrated on prompt activations.
Cost: 32% decoding overhead at full sets and K=10 on LLaMA layer 13; a quarter of each contrastive
set with K=2 brings it to 7% for 1.8 points of the target metric, and the pairwise cost matrix at
100,000 samples per side would need 37.3 GiB, which the paper names as the storage limit of the
approach.

## Limits and open problems they name

The log-density-ratio reduction is established for fixed bandwidth in the vanishing-transport-cost
limit, and does not establish optimality of the adaptive rule actually used. The sampling limit — that
the field converges to the population density-ratio gradient as sample sizes grow — is explicitly
outside the analysis. Pairwise distances and Sinkhorn updates scale as the product of the two
contrastive set sizes, which caps the sets at tens of thousands. The spherical assumption is measured
rather than proven and is weakest at early layers and on Gemma-2-2B and GPT-OSS-20B, where later-layer
CoV reaches 0.128 and 0.146. The gate depends on a coverage parameter whose right value differs across
models (0.7, 0.9, 0.8 in the reported settings), and on Qwen the whole sweep from 0.6 to 0.9 leaves
out-of-distribution accuracy untouched, so on that model nothing is being gated at all. The support
score is defined against *one* contrastive population at *one* layer of *one* prompt position, and the
gate value cached at the last prompt token is never revisited during generation.

## What we take from it

- **A measure.** The support score is a ready-made instance of the on-manifold check that run 797's
  generative prior supplied by reconstruction: an unnormalised density from the same fit that produced
  the intervention, turned into a percentile against held-out references, used as a multiplier on the
  edit. Ours should be able to produce the analogous quantity with a cheaper object, and the paper
  reports the small-capacity limit of that construction exactly — the score collapses to the squared
  nearest-neighbour distance to each population.
- **A rule.** The ablation table is the cleanest published evidence in this folder that an
  intervention's apparent gain can be almost entirely capability loss: 63.6% with the gate against
  67.6% without, with GSM8K going from 49.5% to 0.6%. Any effect our model reports on a target metric
  ships the cross-task number for the same run.
- **A warning about units.** This is effectively run 795's collateral-damage result reached from the
  other direction: the thing that predicts the damage is distance in the representation's own
  geometry, and the paper shows the *Euclidean* version of the same gate, calibrated to the same
  coverage, fails badly (GSM8K 15.1% versus 49.5%). A density score is only as good as the metric
  space it is computed in.
- **A warning about calibration sets.** The nearest-neighbour gate's cliff at the 0.995 quantile is a
  calibration mismatch of exactly the kind that would bite a learned monitor: the reference
  distribution was measured on a different kind of item than the one being scored. Our monitors must
  calibrate on the same activation population they will be applied to.
- **Framing.** `Table 2` of Appendix A is a genuinely useful piece of survey machinery for us — it
  reduces CAA, SphericalSteer, ITI, the Householder variant and the ODE method to a common tuple of
  manifold, number of steps, density model, and induced direction, which is the comparison table any
  intervention chapter of our paper should have.

## Tags

unit: whole-model
form: steering
method: steering
eval: downstream-task, benchmark
goal: control/editing, understanding
