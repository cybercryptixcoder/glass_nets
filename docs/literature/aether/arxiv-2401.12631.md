# A Reply to Makelov et al. (2023)'s "Interpretability Illusion" Arguments

- **Key:** arxiv-2401.12631
- **Authors / organisation:** Zhengxuan Wu, Atticus Geiger, Jing Huang, Aryaman Arora, Thomas Icard, Christopher Potts, Noah D. Goodman — Stanford University, Pr(Ai)R Group
- **Date:** 2024-01-23 (v1)
- **Link:** https://arxiv.org/abs/2401.12631
- **Kind:** paper (a technical reply)
- **Read from:** full text (v1 HTML, body read end to end: abstract, §§1–6, references, Appendices A–D)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The authors are the group that built distributed alignment search (DAS) and the causal-abstraction line:
train a learned change-of-basis so that a linear subspace of a network's activations can be swapped between two
inputs, then ask whether the swap produces the counterfactual behaviour a small symbolic model predicts. Makelov
and colleagues had published a critique saying that these subspace interventions manufacture causal pathways that
are dormant in normal forward passes, and called the resulting explanations illusions. This reply argues that the
label is wrong, that the critique's own technical definition of an illusion fails a case its authors consider
clean, and that the concrete examples in the critique come from a training-and-evaluation design that cannot
support the claim. It is a book-length comment on measurement rather than a new method, and its value to us is
the argument over what an intervention-based claim has to control for.

## What they studied

The unit of analysis is a *subspace direction* at an intervention site — for the worked argument, the MLP
down-projection in GPT-2; for the experiments, seven named activation streams above the final token of GPT-2
Small (block input, attention output, value output, MLP input, MLP output, block output, and per-head
representations). Two tasks carry the empirical half: indirect object identification, and factual recall over
subject–relation–object triples. No new models were trained; the object of study is the method by which an
alignment between a high-level causal model and a network is judged.

## How they did it

Three moves, in order. First a linear-algebra argument: Makelov's "illusion" is defined as the gap in effect
between intervening on the full direction and on its projection onto the row space of the downstream weight
matrix, and expanding the intervention equation shows that gap is the product of the input difference with the
direction's null-space part and its row-space part — so an illusion appears exactly when the direction is *not*
orthogonal to the null space. They then take the direction Makelov calls obviously non-illusory in his own toy
copy network and compute its projection onto the null space by hand; the projection is non-zero, so that
direction is classified as illusory by the critique's own definition. The reason, they argue, is that activations
live on a *data-induced submanifold* — the set of states inputs actually reach — which has no reason to be
orthogonal to a downstream component's null space; they agree that such cases are common and deny they are
anything but geometry. Second, they re-run the critique's IOI experiment in GPT-2 Small with the same dataset
generator, two training templates and a third unseen template for evaluation, 200 base/source pairs, ten epochs
of Adam at learning rate 0.01, batch 20, one DAS direction per site, and read the result as interchange
intervention accuracy (the share of counterfactual predictions matched, bounded at one). Third, they walk every
head at the relevant layers with two-sweep designs: leave-one-out (align with all heads but one) and cumulative
(start from the head whose removal hurt most, add heads one at a time), plus Boundless DAS, a variant that also
learns the subspace's dimensionality instead of fixing it at one.

## What counts as an explanation here

A *causal abstraction*: a high-level causal model, plus an alignment map from its variables to subspaces of the
network, validated by interchange interventions. The explanation is partial by design — it claims fidelity on
the set of interventions drawn from the data distribution and is explicit that behavioural equivalence under
those interventions cannot distinguish structurally different mechanisms. Nothing here is a description or a
feature dictionary; the deliverable is a number (interchange intervention accuracy) attached to a named alignment.

## How they judged success

Interchange intervention accuracy against the counterfactual label, with the model's own task accuracy as the
ceiling of interest; contrast against vanilla (non-learned) interchange intervention at the same site; the
leave-one-out and cumulative head sweeps as the localisation test; the number of non-zero entries in the learned
DAS weight matrix as a second, cheaper read of head importance; Boundless DAS as an ablation of the fixed-
dimensionality assumption; and a held-out template as the generalisation arm. For the reply's first half the
criterion is analytic — a definition is refuted by exhibiting a case it misclassifies.

## What they found

The obviously non-illusory direction in the toy network decomposes into a null-space part and a row-space part
of comparable size, and the paper works the arithmetic through to show the row-space-only intervention keeps a
small fraction of the effect — for a base value of one and a source value of five, the output under the full
direction is five and under the row-space-only direction 1.8. On IOI in GPT-2 Small: name-position information
aligns best at the eighth layer with the main residual stream (70%) and hardly at all with MLP activations (4%),
reproducing the critique's table; within the eighth layer the block input stream reaches about 11%, the attention
output about 48% and the block output about 69%, so the attention block is what writes the information in;
after that layer the attention streams carry none. Aligning with a single head gives 0% at every head, while
leaving out heads six and ten costs the most — but cumulative alignment shows head six alone at 0%, and head six
plus ten below 20%; the correct-object name needs on the order of eleven heads to approach the ceiling. On the
critique's own numbers the reply presses two points: an interchange accuracy of a few percent against a task
the model solves at over ninety percent is a failure to find structure rather than evidence about it, and the
fractional-logit-difference metric can be driven to roughly half its range by a probability change that leaves
the model's answer untouched. The factual-recall critique is answered by design rather than by numbers: a
single learned direction fitted to one pair of facts aligns the network to a degenerate high-level model that
merely returns the seen source answer, and would be expected to generalise to nothing else.

## Limits and open problems they name

They concede the phenomenon the critique describes is common and do not dispute its mechanism; they dispute only
its significance. They accept that causal abstraction cannot certify structural equivalence, that multiple
abstractions of the same network exist, and that for complex models several fully correct abstractions are
likely to survive. Boundless DAS edges DAS only on the object-name alignment, and the cumulative head sweep is
order-dependent in practice. The comparison to their own higher interchange accuracies rests on a different
model and task (Alpaca), so it is a plausibility argument, not a matched baseline.

## What we take from it

A *measure* and two *warnings*. The measure: interchange intervention accuracy with a held-out template and a
vanilla-intervention contrast at the same site, which is the cheapest way to tell an alignment that works from
one that fits. The first warning is a screening rule for any read-out we evaluate by substituting activations —
compute whether the substituted direction has support in the directions the downstream component actually reads,
and report it, because a null-space component makes the substitution inject information the normal forward pass
never uses, which inflates any effect we then attribute. The second warning is about our own holdout discipline:
the critique's factual-recall design is exactly our "fit one direction on one example pair and report the
counterfactual flip" failure, and both objections here — degenerate target model, no train/test separation —
apply word for word to a probe fitted per item and scored on the same item. The reply's own weakest point is
also instructive: its counter-evidence is mostly reproduction of the critique's numbers under a better metric,
and it never runs the matched-strength comparison that would settle the argument.

## Tags

unit: direction | layer | head
form: causal-abstraction
method: decomposition | patching | other
eval: patching | faithfulness | benchmark
goal: understanding | evaluation-of-methods
