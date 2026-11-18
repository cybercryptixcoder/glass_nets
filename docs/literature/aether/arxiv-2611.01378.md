# Information-Theoretic Design of Sparse Autoencoder-Based Steering

- **Key:** arxiv-2611.01378
- **Authors / organisation:** Hirofumi Morise (single author); Ricoh Co., Ltd.
- **Date:** 2026-11-01
- **Link:** https://arxiv.org/abs/2611.01378
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Activation steering works, but choosing the direction is still done by hand or by contrastive heuristics, and two
failure modes keep reappearing: the chosen components do not really move the behaviour, or they move it while
damaging fluency. The author's premise is that a *dataset* has a fingerprint in SAE feature space — a vector of
per-feature firing probabilities — and that the distance between two such fingerprints is a quantity one can
optimise rather than a score one can only rank features by. If steering is framed as minimising the divergence
between a target dataset's firing distribution and the distribution the model actually produces after an
intervention, the vector design becomes a small quadratic optimisation whose coefficients are read off the SAE
weights. The larger goal is a principled replacement for "pick the features that differ most between two
datasets", with the side benefit that the same divergence doubles as a diagnostic for what the intervention did
to later layers.

## What they studied

Llama 3.1-8B-Instruct as the main model, with a top-k SAE (k = 256, residual dimension 4,096, dictionary width
65,536) trained by the author on 7.7 × 10⁷ residual-stream vectors taken after block 16 (0-indexed), leaving
3.3 × 10⁴ features alive on the training set. A second configuration uses Gemma-2-9B-IT with the public
Gemma-Scope SAEs at layers 20 and 31 of the 42-layer stack. The unit of analysis is the **feature's firing
probability over a dataset**, not the feature's meaning: every dataset becomes a point in the H-dimensional cube
of firing rates, and the intervention is a vector in residual space decoded from a sparse combination of SAE
decoder columns. Two task families are used — a seven-way question-mode generation task and a safety
target/contrast task on jailbreak prompts.

## How they did it

The construction has three pieces. (1) **Featurewise divergence.** For each feature, the two datasets A and B
give Bernoulli firing probabilities, and the mutual information the feature's binary firing state carries about
which dataset an activation came from is written in closed form; it equals the Jensen–Shannon divergence between
the two Bernoullis, summed over features to give a single dataset-level objective. (2) **Analytic response.** The
firing probability is smoothed by a sigmoid with an inverse-temperature β, and a second-order Taylor expansion in
the steering coefficient α gives the change in firing rates as a function of the SAE encoder applied to the
steering vector, with diagonal weightings built from the firing probabilities themselves; substituting yields a
quadratic form whose linear and quadratic coefficients depend only on SAE weights and the two datasets'
empirical firing rates. (3) **Greedy support selection.** Since the steering vector is decoded from a restricted
set of features, the author solves the quadratic problem on the current support, then scores each candidate
feature by the objective decrease it would buy if added (with a one-dimensional minimiser whose second-order
coefficient can go negative, in which case the subproblem is unbounded), and adds the best until a budget is
reached. A second divergence against a generic-language reference distribution is computed and used as a
*knock-out filter* rather than a penalty — candidates whose implied deviation cost exceeds a budget are rejected.
Dropping the quadratic term gives a cheaper first-order variant. Baselines are a reimplementation of a
standardised-contrastive feature-scoring method and an SAE-space contrastive-addition variant; a prompt-instruction
condition that names the target mode explicitly is the non-intervention control. Steering vectors are built from
10,000–16,000 model-generated texts per mode, and evaluation uses a held-out 200-item subset with a separate
LLM judge that assigns the detected mode and a 0–5 fluency/format score. A separate pulse experiment attaches
SAEs to layers 16, 20, 24, 28 and 31, injects the vector at one position in the teacher-forced stream, and reads
the per-layer divergence from the target firing distribution at every downstream layer, under both
teacher-forced and teacher-then-free continuations.

## What counts as an explanation here

A **design rule plus a distributional monitor**, not a description of any feature. The understanding offered is a
derivation: "this vector moves the firing distribution toward the target by this much, to second order, given
these SAE weights", together with a scalar divergence per layer and per generation step. It is explicitly partial
— the author does not claim the features are interpretable, and the accompanying qualitative analysis of the ten
selected features is offered as illustration, with a stated disclaimer that these features are not claimed to be
monosemantic. Nothing in the framework predicts what a feature *means*; it predicts how its firing rate responds
to a shift, and which shifts are cheap.

## How they judged success

Behavioural congruence: the fraction of 200 held-out generations that the judge files under the intended mode,
reported per mode and under a fluency/format threshold (≥ 4.5 for Llama, ≥ 4.9 for Gemma) so that a high
congruence bought by degenerate text is not counted. Fluency/format is a 0–5 rubric scored by the same judge. The
safety task replaces congruence with eleven 0–5 judge scores (non-deception, harassment, harmlessness, hate speech,
legality, privacy, sensitive disclosure, self-harm, sexual content, ethics, violence) plus fluency, averaged over
100 jailbreak prompts. The pulse experiment uses the same divergence as its dependent quantity, plotted against α
per layer, with the explicit caveat that absolute values are not comparable across layers because each layer has
its own SAE. No confidence intervals, no repeated seeds, no significance tests, and no human evaluation are
reported; the baselines are reimplementations by the author.

## What they found

Mode steering works in most modes but is not uniformly better than simpler methods. On Llama, four of seven modes
exceed 0.9 congruence at the best operating point (implementation 0.995, consequences 0.970, alternatives 0.895,
evidence 0.850), while criteria sits at 0.700, causes/background at 0.440 and scope/applicability at 0.495 — and
the contrastive baseline beats it on scope/applicability (0.725 vs 0.495) and is close on consequences (0.900 vs
0.970). Explicit prompt instruction is the strongest method on the two modes steering handles worst: 0.895 on
criteria and 0.855 on causes/background against 0.700 and 0.440. On Gemma the picture is flatter and mostly
favourable: the layer-31 configuration reaches 1.000 on causes/background and implementation, 0.995 on evidence,
0.965 on alternatives and 0.960 on consequences, with the low cell at 0.800 on criteria; the baseline's worst cell
is 0.595 on scope/applicability and prompt instruction's is 0.255 on the same mode, where steering is far ahead.
Across the two models the proposal beats the contrastive baseline in most modes and beats prompt instruction in
five of seven on Gemma. Congruence and fluency trade off, and the trade-off point is mode-dependent: fluency in
the two hard modes degrades at smaller α, which the author reads as those modes being less likely under the
unsteered model. On the safety task the first-order variant is the best of the five conditions on nine of eleven
metrics and ties on privacy, with fluency improving rather than degrading (4.99 against 4.76 unsteered), though
sensitive disclosure moves least (4.11 from 4.07) and no condition lifts that cell. The pulse experiment shows a
smooth, roughly parabolic divergence response at the injection layer, the same shape persisting at layers 20
through 31 with the minimising α staying in a similar range, and — the result that matters most for practice — the
next-step response is almost identical between teacher-forced and teacher-then-free, so one token's changed
identity barely alters the next step's internal state, which the author reads as requiring repeated intervention
during generation. The qualitative feature analysis is the most interesting descriptive finding: of the ten
features the first-order rule ranks highest in the safety setting, only four fire on safety-relevant content or
explicit compliance/refusal stance, while six fire on context-management phrases (scope control, interpersonal
openers); and in the (p_A, p_B) plane the selected features are spread through the interior rather than lined up
off the diagonal the way a contrastive score selects them.

## Limits and open problems they name

Two models, two SAEs, one intervention layer per model (16 for Llama, 20 and 31 for Gemma); transfer to other
layers, dictionary widths and sparsity settings is untested, and so are other steering objectives. No exhaustive
search over extraction points, SAE training settings, feature counts or budget, so the reported operating points
are not tuned optima. Evaluation is entirely LLM-as-a-judge with custom rubrics, which the author flags as
model- and prompt-dependent; the reference datasets are model-generated and auto-filtered, which may shape the
steering behaviour. The budget parameter does not behave monotonically, and the author explains why (it filters
one candidate at a time and is not a global constraint on the final vector). The dual-use risk is acknowledged
explicitly. Not stated but visible in the numbers: the two modes where steering underperforms are exactly the two
where a sentence of prompt does the job, and no comparison at matched fluency threshold is reported for the
first-order variant on the mode task, only for the second-order one.

## What we take from it

Three transferable things. **A dataset as a firing-rate vector, and a divergence between two of them as the
objective** — this is a read-out of a *distribution over features* rather than of a single direction, and it is
the cheapest instrument in the folder for asking "how far did this intervention move the model's feature
economy?", with the honest caveat that it is layer-relative and cannot be compared across layers. **The knock-out
filter as an off-target control**: instead of a penalty weight, the author rejects candidates whose implied cost
against a generic-text reference exceeds a budget, which is a way to keep the off-target cost bounded by
construction rather than traded against the objective — directly comparable to the paired-rubric audit in
arxiv-2610.41661, and cheaper. **The pulse experiment as an intervention-locality measurement**: inject once,
read the divergence at every downstream layer, and separate the same-step response from the next-step response —
the finding that the next-step response is essentially unchanged is a needed check for anyone who assumes a
single-token edit compounds, and the recipe for it (teacher-forced versus teacher-then-free at the same position,
same samples, only the fed-back token differing) is a matched ladder we can copy. The warning to keep: the method
optimises a *proxy* for the behaviour (firing-distribution distance) and the behaviour numbers then show it
losing to a sentence of prompt on the modes where the proxy's target distribution is hardest to reach — a
reminder that the side of the ledger that is optimised is not the side that is scored.

## Tags

unit: feature
form: steering
method: SAE | steering | decomposition
eval: downstream-task | ablation | faithfulness
goal: control/editing | understanding
