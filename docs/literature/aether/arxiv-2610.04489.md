# DV-Lens: Revealing the Functional Organization of Language Model Parameters

- **Key:** arxiv-2610.04489
- **Authors / organisation:** Chenhang Cui, Jian Yu, Jiani Luo, Shuyi Miao, Dingrui Liu, Rui Huang, Xiaohao Liu, Fei Shen, An Zhang, Tat-Seng Chua; NUS / NJUST / Beihang / UESTC / HKUST / USTC (per arXiv affiliations)
- **Date:** 2026-10-03 (v2 updated 2026-11-18)
- **Link:** https://arxiv.org/abs/2610.04489
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Every weight-space readout this field has — projecting a column straight to the unembedding, or
reading a learned dictionary derived from weights — describes a parameter as if the layers after it
did not exist. A weight column in layer 3 interacts with the output through the whole remaining
stack, so its honest description is the *response* it produces at the end of the model, not its
cosine with the unembedding. The authors want two things at once: a parameter-level readout that
propagates through downstream computation before being read out in vocabulary space, and a
model-level statistic built from the *organisation* of that readout which might track capability.
The larger goal is a bridge between "what does this parameter do" and "what does this model look
like structurally", without running benchmarks.

## What they studied

Twenty-one small and mid-size open post-trained LLMs for the intervention study and nine for a
finite-difference validation (Llama-3.2 1B/3B, Llama-3.1 8B, Mistral-7B, OLMo-2-7B, Granite-3.0-2B
and 3.3-8B, Qwen2.5 0.5B/3B/14B, Qwen1.5 14B, Yi-1.5-9B, Falcon3 3B/7B, DeepSeek-R1-Distill-Qwen
1.5B/7B, SOLAR, StableLM-2 and others), and a fixed cohort of 48 post-trained checkpoints for the
capability analysis. The unit of analysis is the **native weight column** of the Q, K, V, O
projections and the FFN matrices — not neurons, not directions learned post hoc, not heads.

## How they did it

Write a module's output as a linear map applied to its input activation, so that a single column of
that matrix is a direction and the matching activation coefficient says how much of it participates.
Perturb one coefficient by epsilon and the logit change to first order is the unembedding applied
to the *downstream Jacobian* of the remaining stack, itself applied to that column (Eqs. 1–4). The
prompt-local response is then averaged over a fixed reference prompt set — a rank-128 approximation
of the mean Jacobian, estimated once per module with automatic differentiation — giving one signed
readout of length |V| per column, and the whole matrix's readouts by one multiplication (Eqs. 5–7,
23). Concept vectors are built from the unembedding rows of the concept's tokens and one-versus-rest
contrasted (Eq. 8); contracting the transposed mean Jacobian on that concept vector gives a
"parameter signal" in module space (Eqs. 9, 18–20, computed by vector–Jacobian products), and
weights are then ablated along it, steered along the target concept's signal, or swapped (Eq. 10),
always as nested prefixes of a fixed ranking at eight edited fractions from 0.05% to 5%. For the
model-level statistic, columns are clustered k=32 on their *normalised* readouts, each weight column
is reconstructed from its cluster mean, and the energy-normalised residual is averaged over all
eligible matrices (Eqs. 11–13); that single number per model and module scope is correlated with a
five-benchmark capability mean by Spearman over the 48 ranked models.

## What counts as an explanation here

A **dictionary over the final vocabulary, indexed by weight column**: each parameter gets a signed
vector saying which tokens it promotes and which it suppresses on average, plus a group label from
clustering those vectors. It is deliberately partial and explicitly not a causal claim about any
prompt — the authors separate the response direction (which can be read off the weights) from the
activation-weighted contribution actually realised on a given input, and say the readout is a
"footprint" rather than a contribution. The model-level object is a single structural scalar with
an honestly-stated interpretation: the weight energy that would be destroyed if each column were
replaced by its functional role's centroid, plus a theorem chain linking that energy to a bound on
cross-entropy increase.

## How they judged success

Four separate tests, and the paper is unusually clear about which one licenses what. (1) A
**matched-null** comparison: excess kurtosis and top-1% L2 energy of real readouts against Gaussian
directions matched in dimension, covariance treatment and norm — 12 concept-mean directions against
192 Gaussian draws per model-depth-module cell, 315 cells over 21 models. (2) A
**finite-difference** test: perturb the coefficient, measure the true logit change, and score
coordinate-orientation sign agreement, exact two-tail Top-32 overlap, and Top-32 Recall@128 (with its
F1 stated, since enlarging the reference tail raises recall by construction). (3) A **matched
random-column control** for the interventions: the same number of non-primary columns within ±5% of
the original norm rank, a random direction, rescaling so columnwise and total Frobenius norms match,
any cell with >2.5% norm mismatch discarded. (4) **Controls on the correlation**: a random-label
permutation with matched cluster sizes (128 draws, one-sided conditional p), a raw-weight k-means
baseline at the same k and seed, leave-one-family-out over 17 families, and a benchmark-exposure
experiment that fine-tunes six models on the full test split of all five benchmarks.

## What they found

Orientation agreement is 0.980 pooled over 720 frozen cases from nine models, but exact Top-32 tail
overlap is only 0.407 and Recall@128 0.613 (F1@128 0.245) — so the readout reliably predicts *which
way* a coordinate moves and much less reliably where it ranks inside the tail. Precision is
module-dependent: 0.947/0.209 for K up to 0.997/0.565 for FFN, with O close behind FFN. Depth helps
monotonically: from earliest to latest sampled layer, orientation 94.5% → 99.96%, exact overlap
21.4% → 67.2%, Recall@128 38.3% → 86.3%, improving in all nine models. Against matched random
directions, mean excess-kurtosis advantage is +2.550 over the eight highlighted models (95% CI
[1.626, 3.574]), positive in all 21 models, and it survives deleting any one of the 13 model
families. Interventions: at Joint scope, peak conditional decoded success is 61/66 cells for
ablation (5% of columns), 28/66 for steering (0.1–0.2%), 23/66 for swapping (5%); the wider
probability-direction event is near-saturated (210/210 for steering at 0.05–0.2%). The matched
random control averages 3.13% against 30.82% for the primary selection across 144 scope-operation-
fraction settings. DV-Complexity correlates 0.904 (Joint), 0.794 (Q), 0.756 (FFN), 0.743 (O), 0.701
(K), 0.596 (V) with the five-task capability rank, all six beating all 128 matched-size
random-label permutations (p at the 1/129 floor). Complete-benchmark exposure raises ability ranks
while leaving DV-Complexity ranks within one position.

## Limits and open problems they name

The readout is one of many views, shaped by architecture, optimisation, data and scale, and linking
internal structure to high-level behaviour "remains an open problem". Byte-level detail the paper is
candid about and which matters: the random-label null is on a *fixed* cohort with fixed ranks, so it
is not a significance test for a wider population; and — the number the abstract does not carry — the
raw-weight k-means baseline, which never computes a Jacobian, reaches 0.800 at Joint against the
method's 0.904, with every bootstrap interval for the difference including zero (+0.104, CI
[-0.023, 0.311]). The intervention comparison likewise rests on *one* matched random draw per cell,
which the appendix says outright is a reference control and not a null distribution, and the
success rate is computed only on concept cells whose clean continuation already contained the gold
answer. The theory proves monotone k-means improvement and a loss-increase certificate; it does not
prove DV-Lens clusters are better than raw-weight clusters, and the authors do not claim it does.

## What we take from it

A **method** worth stealing on structure: any readout of a weight we publish should be defined
through the downstream Jacobian rather than through the unembedding alone, and the identity readout
(native-space clustering, no Jacobian) is the correct cheap baseline to report next to it — this
paper does that and the honest result is that most of the capability correlation is available
without the expensive part. A **measure** worth adopting: coordinate-sign agreement as the headline
for a directional claim, with tail overlap and recall reported separately, because a 0.98 and a 0.41
describe the same readout and answer different questions. A **warning** for our own ablations: a matched
random-column control with matched norms and a single draw is a reference point rather than a null
distribution — we should either repeat the draw or label it as such. A **framing** to borrow: separating a parameter's
response direction (a property of the weights) from its realised contribution (a property of the
weights times the input) is exactly the distinction our predictive model needs, and the paper's own
admission that the averaged readout is a footprint rather than a contribution is the cleanest
statement of that gap in the folder.

## Tags

unit: weights
form: dictionary
method: weight-analysis
eval: ablation
goal: understanding
