# Circuit Claims Depend on What Is Extracted and How It Is Compared

- **Key:** arxiv-2607.18921
- **Authors / organisation:** Yang Sheng, Jie Fu; Fudan University / Shanghai Innovation Institute / IQuest Research
- **Date:** 2026-07-21
- **Link:** https://arxiv.org/abs/2607.18921
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why
Circuit extraction is normally read as recovering *the* mechanism for a behaviour, but preservation under
ablation does not pick out one subgraph: several objects survive the same test, and several plausible ways of
comparing them exist. The authors ask a narrower and more checkable question than "is this circuit right" —
when the benchmark, the checkpoints and the task set are all held fixed, which circuit-level claims change if
you change which graph you report, how finely you model an attention head, or the level at which two circuits
are compared. Their larger goal is a reporting discipline for the field: a circuit claim is meant to be treated
as incomplete unless the extraction rule, the pruning budget and the comparison level are stated. Their setting
is deliberately artificial so that a difference between two extracted circuits can be attributed to those
choices rather than to an uncontrolled change in the task.

## What they studied
One 8-layer transformer (d_model 2048, 128 heads of width 16, d_mlp 8192, context 128, ~415.7M parameters)
trained on a synthetic Lean tactic-prediction benchmark: nine generated tasks, four atomic and five
compositional, where proposition and hypothesis names plus distractor premises are randomized but the proof
rules are fixed, and the label is the exact tactic string. Checkpoints span dense plus 25/50/75/90%
weight-sparse training, three seeds per level, with the dense and 75%-sparse checkpoints used as the fixed
comparison points for graph-level work. On top of this sit grid factors for training distribution (baseline,
moderately expanded, extensively expanded surface forms) and three reinforcement-learning initializations that
differ only in their upstream supervised checkpoint (curriculum, atomic-only, composition-only) while the RL
prompt mix stays composition-only. Unit of analysis: nodes and edges of a residual-stream circuit vocabulary —
residual reads/writes, attention q/k/v/write channels, MLP read/neuron/write — and the coarser set of selected
attention heads.

## How they did it
Three circuit objects are extracted from the same checkpoint and task set, differing in selection rule: a
*compact prediction-preserving circuit* from learned-mask selection with exact-match tactic accuracy after
ablation (masked cross-entropy plus an ℓ0 penalty used only as a tiebreaker), a deliberately *wider support
graph* that keeps surrounding read/write/routing structure on the same machinery, and a *fixed-loss pruning
graph* — the smallest top-k subgraph whose ablated loss stays under a threshold τ, with τ = 0.12 as the main
reporting point and a sweep over 0.04–0.20. They then introduce a second axis: each attention head's
query-side and key-side support can be *coupled* (one shared qk node family) or *factorized* (separate q and k
families), with everything else held fixed, so the same rough read-to-write route acquires a different node and
edge set. Comparisons are made at two levels: exact component-to-component edges and the attention-head set
implied by the top routing edges, both scored with Jaccard@10 over n = 16 matched task-object entries per split
(success prompts solved by both checkpoints; near-miss prompts closest to the right tactic). Two independently
sampled 100-example test suites per task carry the behavioural claims; uncertainty is 95% bootstrap intervals
over matched entries, plus a bootstrap over the 15 sparsity–seed units per cell for the ordering claim.
Negative controls are matched-atomic unions, mismatched-atomic unions from the same checkpoint, and
size-, layer- and family-matched random top-k draws.

## What counts as an explanation here
A circuit — a node-and-edge subgraph, plus the extracted-object type and threshold that produced it — with the
explicit claim that such a description is *under-determined* until you say which graph is reported, under which
selection rule and loss budget, at what query/key granularity, and compared at what level. The paper's
contribution is not a new explanatory object but a demonstration that the object is not readable off behaviour
alone; its constructive output is a five-item reporting checklist rather than a dictionary or a predictor.

## How they judged success
Whether a *qualitative takeaway* (an ordering between conditions, or which summary overlaps more) stays the
same when the reported graph, the q/k granularity or the loss budget changes; whether that takeaway clears its
own random top-k baseline; and whether behavioural gains line up with a named structural quantity (bridge
fraction — the share of compositional-task circuit nodes outside the matched atomic circuits). Metrics: Jaccard@10
at exact-edge (structural and semantic bundling), routing-head level and random baseline; node fractions
normalized per object's own universe; mean absolute coupled-vs-factorized fraction gaps; pairwise ordering
agreement (3/3); bootstrap flip probabilities; exact-match accuracy over two suites.

## What they found
Overlap depends on the comparison level rather than on the checkpoint pair: exact-edge Jaccard@10 between the
dense and 75%-sparse RL checkpoints is 0.163 on the success split and 0.142 on the near-miss split, while the
attention-head sets implied by top routing edges score 0.666 and 0.554. The head-level numbers sit far above
their own matched random draws (0.199 and 0.185), whereas near-miss exact-edge overlap is statistically
indistinguishable from chance (0.142 against a 0.117 baseline, p = 0.106) — so on that split the literal edge
list carries no reproducible information while the coarse head set does. The q/k split and the pruning budget
both turn out to be near-inert for the claims these authors make: coupled-vs-factorized fraction gaps stay at or
below 0.024 at τ = 0.12 (0.008/0.010 core, 0.003/0.013 support, 0.015/0.024 pruning for the curriculum cell in
the two expanded families), all three RL-condition orderings by graph size agree 3/3 for every object and
family, the largest pairwise flip probability in the sparsity–seed bootstrap is 0.232, and ordering agreement
stays 3/3 across the whole τ sweep with gaps at most 0.055. Behaviour: curriculum-initialized RL lifts
compositional accuracy by 24.64 points in the moderately expanded family (43.83 → 68.47) and 31.27 in the
extensively expanded family (46.73 → 78.00) on Suite 1, against at most 14.20–18.11 points for composition-only
initialization and 4.27–5.59 for atomic-only, which drops overall accuracy by 21.83–25.48 points. The two
curriculum cells also have the largest core bridge fractions (0.343 and 0.161, versus ≤ 0.054 elsewhere). The
negative controls behave as designed but not cleanly: keeping matched atomic circuits retains 0.920–0.926 of
target predictions after ablation for the core circuit against 0.722–0.774 for mismatched unions and 0.298–0.332
for random controls, and mismatched unions stay well above random (0.390–0.476), which they read as evidence that
the correct source matters *and* that shared format or task-family cues contribute.

## Limits and open problems they name
The strongest in-family checkpoint (95.79% accuracy) collapses under every held-out shift — 0.51% on distractor
relabelling, 1.30% on names, 2.92% on goal-first ordering, 0.42% on deeper compositional extrapolation — so the
extraction findings describe the generator's own family and are not evidence about transfer. The bridge-fraction
result rests on two cells and is called a directional association, not a measured effect. Edge-overlap evidence
covers 16 + 16 matched entries with representative exemplars rather than frequency estimates over many extracted
graphs; there is one model family and one benchmark; the shared proof-state rules make format cues easier to see
than in heterogeneous families; and accuracy is not monotonic in weight sparsity under their fixed recipe. The
paper gives no rule for *choosing* the comparison level that matters, only the instruction to report it.

## What we take from it
A warning and a measurement recipe. The warning is that "we recovered the circuit" and "these two checkpoints
share a circuit" are level-dependent claims, and this folder's circuit evidence is mostly reported at whichever
level happened to be convenient. The recipe is the negative-control ladder this paper actually prints —
matched / mismatched / random — which is the honest version of the matched-random arm the folder keeps asking
for, and whose residual mismatched-above-random gap is exactly the shared-surface-cue confound a predictive
model must not absorb. Two concrete imports for us: (i) report our predictor's agreement with a reference
explanation at two granularities in the same table, because here one granularity is above chance and the other
is not; (ii) treat the "inert choice" result as a pre-flight — if a design choice moves nothing measurable in
any conclusion, that is worth one sentence, and if it only moves the headline number while ordering is stable,
the ordering is the real finding. Also note the single number that would have been most useful and is missing:
interpretive compression is not estimated anywhere, so the stability claims are about *agreement*, not about
how much of the mechanism the reported object covers.

## Tags
unit: circuit
form: graph
method: attribution | patching | ablation
eval: ablation | prediction/simulation | benchmark
goal: evaluation-of-methods | understanding
