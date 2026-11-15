# Tracking Equivalent Mechanistic Interpretations Across Neural Networks

- **Key:** arxiv-2603.30002
- **Authors / organisation:** Alan Sun, Mariya Toneva; Carnegie Mellon University / Max Planck Institute for Software Systems
- **Date:** 2026-03-31
- **Link:** https://arxiv.org/abs/2603.30002
- **Kind:** paper
- **Read from:** full text
- **Version note:** arXiv HTML v1 read; the body ends at the Conclusion and the appendices here run A–H (the text cites a later appendix letter in one cross-reference but the page's appendix list stops at H).
- **Relevance to us:** 3 (central)

## What they wanted, and why
Mechanistic interpretability is hard to scale partly because it has no precise notion of a valid interpretation
and because interpretations are produced by ad hoc search. The authors carve off a subproblem they can make
rigorous: decide whether two networks share a common interpretation *without* ever writing that interpretation
down. Their guiding principle is that two interpretations are equivalent when the sets of models implementing
them are equivalent — and that this can be measured by comparing representations. The motivation is practical
compression: if a large model is interpretively equivalent to a small one, or a hard task to an easier one,
then interpretability work done on the cheap member transfers. They position this against the known
many-to-many relation between circuits and interpretations, which makes direct circuit comparison ill-defined.

## What they studied
Three settings, all Transformer-based. (1) A toy *n*-permutation detection task at n = 10 with six hand-written
interpretations — four sorting-based, two pigeonhole/counting-based — each compiled from RASP into a
bidirectional Transformer and then expanded into 100 architecture/weight variants from an existing variant
generator, giving 6 × 100 models all above 96% accuracy. (2) Indirect-object identification on GPT-2
small/medium and the Pythia family, spanning roughly 85M to 2.9B parameters, with 10 parallel implementations
per model built by intervening on components not in the known IOI circuit. (3) GPT-2 next-token prediction over
100 sampled C4 sentences, decomposed by token type, compared against a GPT-2 parts-of-speech circuit from an
existing method. Units of analysis are circuits (causal graphs), representations (a chain of activation spaces
abstracting a circuit), and interpretations (abstract causal models).

## How they did it
Everything is built on a causal-abstraction formalism: a circuit is a deterministic causal model over the
model's computation graph with an input variable and an output equal to the model's own prediction; a
representation is a length-L chain abstracting that circuit, where every layer's hidden state can be linearly
decoded to the output and re-encoded from it up to error δ; an interpretation is an abstract causal model that
approximates the circuit's output within η. Implementations of an interpretation are defined as the class of
transition-function assignments admitting it under a chosen alignment class, and interpretive equivalence is the
Hausdorff distance between two implementation sets, with interpretive *compression* the diameter of one such
set. The estimator, Congruity, never interprets anything: for each of n rounds it draws an implementation of
each model by intervening on components judged unimportant (identified by patching-type scoring, deliberately
weaker than full circuit discovery), measures bidirectional linear representation distance between the two
models and each model's own variant, and checks that each model is closer to its own variant than to the other
model. Scores are averaged into a congruity number, with 95% bootstrap hypothesis tests. Three theorems relate
the estimator to the equivalence quantity: a representation-similarity term plus compression terms upper-bounds
the equivalence distance, a lower bound in the other direction, and a bound linking the estimator's expected
value to both.

## What counts as an explanation here
A *causal abstraction* in the formal sense — a high-level deterministic causal model aligned to the circuit,
with representations as its linear read-out chain — plus a relational judgement (equivalent or not) between two
such objects. It is explicitly partial: the paper states that equivalence alone does not identify an
interpretation, and offers it as a way to reduce the set of models one must interpret rather than as a
description of any single model. The nontrivial predictive content is the claim that representation geometry,
not circuit identity, is the right observable for comparing interpretations.

## How they judged success
On the toy task, ground truth is known, so calibration is checked directly: congruity should be high on the
diagonal (same interpretation, different implementations) and low off it, with bootstrap CIs and Wald intervals
over 20 resamples. On IOI, the target is the published disagreement in the literature — GPT-2 small and medium
share a circuit, Pythia shares a different but scale-consistent circuit — so within-family congruity should
exceed cross-family congruity. On next-token prediction, the criterion is a graded one against a control: the
all-token group is the baseline, and token groups expected to be syntactic (terminal punctuation, closing
brackets/quotes) should show a larger effect size (Cohen's d) than groups expected to need semantics (articles,
prepositions, general punctuation, opening brackets). Theorems are proved under four stated assumptions —
Lipschitz-representation alignment, bounded behavioural difference between models, a fixed distillation function
defining implementation distance, and a 1-Lipschitz map recovering implementations from representations.

## What they found
On the toy task the diagonal is significantly congruent and off-diagonal entries are low, with a visible block
structure: interpretations 1–4 (the sorting family) have average within-group congruity 0.43, against 0.01 for
the cross-group average involving the two counting interpretations — i.e. the score degrades gracefully with
algorithmic relatedness rather than being binary. On IOI, within-family congruity is high across scale
(Pythia-2.8B is congruent with Pythia-160M, so in principle only the small model needs interpreting) while
cross-family GPT-2/Pythia congruity is 0.13 against within-family values of 0.73 and 0.92; the authors flag two
smaller-scale exceptions against GPT-2-small/medium as unexplained and speculate about name-mover heads. On
next-token prediction, the all-token control scores 0.48 congruity with the POS circuit; articles and
prepositions are statistically indistinguishable from that control, terminal punctuation and closing
brackets/quotations are significantly higher with medium effect size, and general punctuation and opening
brackets are significant but with d < 0.3. Theory side: the upper bound decomposes into a compression term
(how much the interpretation collapses the circuit) plus representation quality and distance terms; the lower
bound is driven by the same d_repr numerator scaled by the Lipschitz constant of the alignment plus one, which
the authors use to argue that when representation identifiability is low, large changes in representation space
need not correspond to interpretive change — a caution for steering.

## Limits and open problems they name
The estimator needs unimportant components at the granularity the interventions define, so the level at which
interpretations are compared is set by the intervention set, not by the theory; the paper says openly that
little is known about the structure of those interventions. Equivalence does not yield a description — turning
it into actual interpretations is left open. The representation similarity is linear for computational
convenience, and the authors concede representations on complex manifolds may not be well captured by it, calling
for both representations and the similarity criterion to be broadened. High compression makes equivalence
trivial (every circuit implements the trivial interpretation), so the bounds degrade exactly where compression
is largest. Open hypotheses: sharp restrictions on the alignment class would tighten the bounds; sample
complexity of compression estimation is only bounded, not measured.

## What we take from it
A framing and a measurable object, plus one hard caution. The framing is the folder's cleanest statement that
"same mechanism" is a *set-valued* property: compare families of implementations, not two hand-picked subgraphs,
which is the formal counterpart of the empirical finding in the circuit-extraction entry that level of
comparison decides the claim. The measurable object is the compression–equivalence duality: shrinking the
implementation set (a sharper interpretation) enlarges the difficulty of showing equivalence, so any statement
of the form "our predictor identifies the mechanism" should come with both a diameter (how much variation the
description tolerates) and a distance (how far the comparison group is), because one without the other is
trivially satisfiable. The caution is the one that matters most for us: representation similarity is only
informative when the circuits are compared at a level where compression is low, so a frozen-representation
predictor that agrees with a reference explanation may be measuring shared encoding rather than shared
computation — and only a control at the level of interventions can tell the two apart. Data point worth reusing:
the graded 0.43-versus-0.01 block structure is a cheap way to show a comparison metric is sensitive to degree of
algorithmic difference rather than just to identity.

## Tags
unit: circuit | layer
form: causal-abstraction
method: decomposition | probing | ablation
eval: faithfulness | prediction/simulation | benchmark
goal: evaluation-of-methods | understanding
