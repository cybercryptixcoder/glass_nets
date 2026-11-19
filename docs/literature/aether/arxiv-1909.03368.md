# Designing and Interpreting Probes with Control Tasks

- **Key:** arxiv-1909.03368
- **Authors / organisation:** John Hewitt, Percy Liang; Stanford University
- **Date:** 2019-09-08 (v1; EMNLP 2019)
- **Link:** https://arxiv.org/abs/1909.03368
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Probing — training a small supervised model to predict a property such as part of speech from a frozen
representation — had become the standard way to argue that a network "encodes" a property, and the field was
reading high probe accuracy as evidence about the representation. The authors ask the obvious counter-question:
if a probe is expressive enough, it can learn the task from a lossless encoding regardless of what the encoding
contains, so what part of a high accuracy is the representation and what part is the probe? Their answer is not
a better probe but a matched instrument for pricing the probe's own capacity — a task that is, by construction,
learnable only by memorisation, run beside the linguistic task so that an accuracy can be read against the
amount of work the probe did that the representation did not do for it. It is a step toward making a
representation claim falsifiable rather than suggestive, and it is the paper that supplies the folder with a
first-class random-floor instrument for evaluating any read-out.

## What they studied

ELMo (the 5.5-billion-word pretrained contextual model), specifically the output of its first BiLSTM layer
("ELMo1") as the main representation, plus the second contextual layer ("ELMo2") and an untrained BiLSTM run on
ELMo's non-contextual character-CNN embeddings ("Proj0") as representations to compare. Two linguistic tasks:
English part-of-speech tagging over a 45-tag set, and dependency edge prediction (head index per token) after
converting Penn Treebank constituency trees to Stanford Dependencies. The unit of analysis is a *probe* — a
linear, bilinear, one-hidden-layer or two-hidden-layer classification head — and what is measured is the pair
(linguistic task accuracy, control task accuracy) for each probe and each representation.

## How they did it

Control tasks are built per word *type*, not per example: every vocabulary item is independently assigned a
random behaviour, and every token of that type receives the behaviour's output regardless of context. For
part-of-speech the behaviour is a tag sampled from the empirical tag distribution, so a control task has exactly
the linguistic task's output space and marginal label distribution but can only be learned by looking up word
identity and memorising a random map. For dependency edges the length-dependent output space forces a different
construction: three length-independent behaviours (attach to self, attach to the first token, attach to the last)
sampled uniformly per type, with each predicted edge scored independently and the root omitted. Because the map
is type-level, the ceiling on control accuracy is the fraction of evaluation tokens whose type appeared in
training, which also makes the control task a probe of how easily word identity is recovered from a
representation. The authors then sweep probe families (linear / bilinear, MLP-1, MLP-2) and six complexity
controls — weight-matrix rank or hidden width, dropout probability, training-set size, weight decay, and early
stopping at a fixed gradient-step budget — all optimised with Adam and a halving learning-rate schedule, and
report linguistic accuracy against control accuracy for each setting. Finally, the same probes are run identically
on ELMo1, ELMo2 and the untrained-encoder baseline, so that layer comparisons are read on both axes rather than
accuracy alone. "Selectivity" is the difference between the two accuracies, and the paper's prescription is to
pick the highest-selectivity hyperparameters that do not cost linguistic accuracy, which they then do by hand for
each probe family and re-report.

## What counts as an explanation here

A probe, and the explanation it licenses is a *relative* claim of the form "information of type X is recoverable
from representation R by a function of class F", priced by how much of the achieved accuracy could have been
obtained without the representation. It is deliberately partial and deliberately not mechanistic: nothing about
how the network computed the feature is offered. Its falsifiable content is a comparison — the control accuracy
is the number that makes the linguistic accuracy mean something, and the paper's contribution is the argument
that the pair should be reported together. There is also a lightweight causal reading they lean on for
interpretation: if selectivity is low, the probe's accuracy can be obtained from word identity alone, so the
probe is not a witness for anything the representation encodes about context.

## How they judged success

Two accuracies and their difference (selectivity), on the Penn Treebank development set, with default
hyperparameters (hidden width 1000, no dropout) as the reference setting; every complexity control is swept over a
stated grid and reported as a curve of accuracy against selectivity rather than as a single score. Baselines are
the same probe family on a random or untrained representation (Proj0) and the linguistic-task accuracy of the
same probe under heavy capacity constraints. Controls: the control task shares the linguistic task's input and
output space and label marginal, so the only difference is the type-level random map; reductions in training size
follow the earlier memorisation work and are tested on both tasks so that a selectivity gain bought by hurting the
real task is visible. A qualitative arm inspects aggregate confusion matrices of ten linear and ten MLP probes to
check whether selective and non-selective probes err differently.

## What they found

With default settings, MLP probes are barely selective: part-of-speech MLP-1 reaches 97.3 accuracy against 92.8 on
the control task, a selectivity of 4.5, and MLP-2 is the same (97.3 / 93.2 / 4.2); the linear probe gets 97.2 with
only 71.2 control accuracy, selectivity 26.0. On dependency edges the pattern is stronger and goes negative —
bilinear 89.0 / 82.4 / 6.6, MLP-1 92.3 / 93.0 / −0.7, MLP-2 93.9 / 92.0 / 1.9 — so the two most accurate probes
here are learning a task the representation cannot help with at all. Dropout, the field's default regulariser, does
not fix this and in places makes it worse (MLP-2 part-of-speech selectivity 4.2 → 3.4 at p = 0.4; linear 26.0 →
29.8 and bilinear 6.6 → 16.7 at the same setting), and early stopping moves little on part-of-speech but helps MLP
dependency probes. Small hidden width works: with rank-10 probes chosen by control-task selectivity the part-of-speech
MLP-1 goes from 4.5 to 16.6 selectivity at 97.2 accuracy, and dependency MLP-1 from −0.7 to 6.2 at 90.5 accuracy,
with linear probes best at 33.0. The layer comparison then inverts the usual reading: on ELMo2 the linear probe is
slightly less accurate than on ELMo1 (96.6 vs 97.2, a loss of 0.6) but much more selective (31.4 vs 26.0, a gain
of 5.4), the same direction across all four probes and both tasks; and Proj0 matches ELMo2's part-of-speech
accuracy (96.3 vs 96.6 linear) while being far less selective (20.6 vs 31.4), which the authors read as ELMo2
achieving its accuracy from contextual structure rather than word-identity lookup. The error analysis supports
the mechanist reading of selectivity: the MLP wins mostly on adjective-versus-noun distinctions inside compounds
while mistaking plural-suffix proper nouns for the plural proper-noun tag, a pattern the linear probe does not
show — a less selective probe combining spurious surface markers with real ones.

## Limits and open problems they name

Selectivity is defined against one specific control-task design, and the authors note the sampling distribution of
control behaviour is not principled (they sample from the empirical tag marginal "for intuition"). The control
task requires a type-level structure — it is not defined for tasks whose inputs are not tokens with reusable
identities, which is a real constraint for our own read-outs on activations rather than words. No selective probe
was found across a broad hyperparameter range except by hand-search, so finding one is admitted to be non-trivial
and the paper offers no procedure beyond sweeping and looking. Whether ELMo2's higher selectivity means it
represents part of speech better is left open — the paper explicitly refuses to settle which layer "better
represents" the property, offering only the alternative hypothesis. And the comparison is between probes, not
between explanations: nothing here addresses whether a *selective* probe's errors are the model's errors.

## What we take from it

- **Measure.** Selectivity — the gap between the real-task score and the matched random-task score of the *same
  read-out* — is the cheapest control in the folder that prices the reader rather than the representation, and it
  needs no human labels beyond the task's own output space. Our own account tables should carry a column of this
  kind beside every read-out score.
- **Design rule for the control.** The matched random task must share the output space and the label marginal of
  the real task and differ only in the part that cannot be learned from the representation. That is the concrete
  recipe for the "matched no-account arm with a stated reader" our claims have never run: same read-out class,
  same output space, target random.
- **Warning.** Capacity is the confound in every read-out comparison, including layer-to-layer comparisons; an
  accuracy difference of 0.6 can sit next to a selectivity difference of 5.4, and a random-representation baseline
  can match a trained layer's accuracy while being much less selective. Reporting accuracy alone is not a
  comparison between representations.
- **Warning.** The most popular complexity control (dropout here) bought nothing, while an "unintuitively small"
  hidden width bought a lot at no accuracy cost — so a read-out's capacity should be chosen by the control score,
  not by convention.
- Coreference `arxiv-2205.14140` (the naive-baseline-and-random-floor pair, the same demand on the causal-effect
  side), `arxiv-2403.19647` (the skyline/neuron arm that is the matched-but-illegible reader) and
  `arxiv-1705.05598` (a per-unit score with its own random-direction floor).

## Tags

unit: whole-model | feature
form: probe | description
method: probing | other
eval: benchmark | faithfulness | human-rating
goal: understanding | evaluation-of-methods
