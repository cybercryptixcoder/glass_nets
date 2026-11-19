# Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals

- **Key:** arxiv-2006.00995
- **Authors / organisation:** Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg — Computer Science Department, Bar Ilan University, and the Allen Institute for Artificial Intelligence
- **Date:** 2021-02-19 (v3; first posted 2020-06-01)
- **Link:** https://arxiv.org/abs/2006.00995
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why
Probing had become the field's default way of saying what a language model "knows": train a small classifier on a
layer's activations, and read a high score as evidence that the property is represented. The authors think that
inference is unsupported in one specific direction — a probe can extract information the model never consults, so
extraction and use are separate claims, and the probing literature routinely collapses them. Their goal is a
read-out whose output is not a decoding score but a behavioural one: what happens to the model's own predictions if
a property is taken away from the representation. The larger motivation is a stepping stone toward causal
attribution, using the closest thing available without retraining: a counterfactual on the representation the model
is actually handed.

## What they studied
BERT-base-uncased, analysed as an encoder (its transformer blocks) plus a "model" (the word-prediction / embedding
matrix at the output). The unit is a *linear direction* in a hidden representation — a property's subspace, removed
by projection — and, separately, a layer index when the removal is done inside the stack. Six properties are treated
as the things to erase: coarse and fine-grained part-of-speech (12 and 45 classes), dependency-edge identity
(41 classes), named-entity labels (19), and phrase-start and phrase-end markers (2 each), trained from the English UD
Treebank and OntoNotes with 100,000 random tokens per task. Two input regimes are studied throughout: the ordinary
unmasked encoding, and the masked encoding in which one token is replaced by [MASK] and only its contextualised
representation is read — the regime BERT was actually pretrained in.

## How they did it
The representation to be edited is projected off the property's directions using Iterative Nullspace Projection: a
linear SVM probe is fitted to predict the property, its nullspace is projected out, and this is repeated until the
last probe is within one point of majority accuracy on dev, so the surviving representation is one from which no
linear read-out can recover the property. That "amnesic" representation is then pushed through the model's own
output layer and the word-prediction behaviour is measured, with no retraining of the encoder. Two controls make the
number interpretable. **Rand** removes the same quantity of random directions from the representation, so the
comparison isolates *which* directions were removed from the bare cost of losing rank. **Selectivity** concatenates
a 32-dimensional learned embedding of the true property value back onto the amnesic representation and fine-tunes
only the output layer with the rest frozen; if the damage can be repaired from the property alone, the erasure was
about that property rather than a smeared removal of correlated ones. Measurements are word-prediction accuracy,
Kullback-Leibler divergence between the pre- and post-intervention output distributions, and — in the layer sweep —
the recovery of the property by probes on later layers. The layer experiment extracts the first k layers, projects
at layer k with a projection matrix learned there, and lets encoding continue, so "which block uses this" becomes a
measured accuracy differential rather than a place where a probe happens to be accurate.

## What counts as an explanation here
A *behavioural counterfactual attribution*: for a named property, a signed change in the task's own performance
attributable to removing that property, priced against a random-direction removal of equal rank. It is explicitly
partial and explicitly non-causal in the strong sense — the authors call the erasure an approximation to the abstract
property, note it only removes linearly-present information, and describe what is removed as whatever the probe's
directions happen to capture, spurious correlates included. It is not a description, not a dictionary, not a circuit.
It is designed to underwrite one narrowly-scoped prediction: if this property is erased from this representation,
the next computation degrades by about this much. The fine-grained and layer-wise versions give the same kind of
quantity indexed by label and by layer.

## How they judged success
The primary comparison is amnesic versus Rand on the same property and the same setting, in both accuracy points and
KL. Selectivity is scored as restoration of vanilla accuracy after re-supplying the property. Probing accuracy is
reported beside the intervention effect so the two can be related across properties, and that relation is
summarised as a Spearman correlation with its p-value. Controls are run per property, with the reported caveat that
the number of removed directions varies across properties and settings and is not controlled for. No ablation of the
encoder itself, no fine-tuned downstream task, no human rating.

## What they found
Erasing dependency information costs 87.0 accuracy points and fine-grained POS 81.8, coarse POS 32.2, NER 10.8,
while both phrase markers *improve* accuracy slightly (0.21 and 0.32 points up) — so a property the model can decode
is not necessarily one it uses, and one of them is mildly counterproductive where it is removed. Rand loses the same
rank but far less behaviour, so the effect is about the directions, not the dimensionality. Selectivity restores
performance for most properties; dependency does not fully recover, which the authors attribute to the large number
of removed directions carrying coupled information with it. The decisive number for the probing debate is the
correlation between probe accuracy across properties and their measured importance: the paper reports a Spearman
correlation of 8.5 with a p-value of 0.871, i.e. no relationship at all — and a reader should note that a
correlation coefficient above 1 means the printed figure is a scaling artefact rather than a coefficient, so what
this entry takes from it is the null result and the p-value, not the coefficient. In the masked setting the ordering
by property is similar but the numbers are far smaller in absolute terms, and the eroded-rank caveat bites harder
because masked prediction has less accuracy to lose. Breaking the property down by label changes the picture
completely: within coarse POS, erasing conjunction and particle distinctions costs 73.73 and 77.66 points while nouns
and numerals cost 8.65 and 6.91, which the authors read as closed-class vocabulary depending on POS and open-class
vocabulary not. Across layers, POS removal hurts most at layer 11 in the unmasked regime and at layer 3 in the masked
one, and NER at layer 11 (a 31.09-point drop) against layer 4 (masked) — a direct disagreement with the "pipeline"
story that assigns POS to early processing, obtained with an interventional measure rather than an extraction one.
Recovery results are regime-ordered too: in the unmasked setting, POS erased from the first seven layers is largely
re-learned by later layers (5.72 to 12.69 points of loss) while erasure from the upper part is not (16.57 to 46.39),
whereas in the masked setting erasure from the *second* layer leaves later layers unable to recover it (32.7 to 42.1
accuracy).

## Limits and open problems they name
The erasure is linear-only, so it is an approximation to the abstract property and a causal reading must be hedged;
the authors call for non-linear information-removal methods to swap in. The number of directions removed differs
across properties and settings and is uncontrolled, so no claim is made about which property is more important than
another, and figures from the two regimes are not comparable (the y-scales differ for exactly this reason). The
paper's own control for Hewitt and Liang's control task is argued to be unsuitable here rather than run as an
primary control — the appendix shows it, explains that erasing its shuffled-label directions deletes lexical
information central to language modelling, and prefers Rand plus selectivity. Fine-grained label results are run with
INLP capped at 60 iterations. Nothing is claimed about why the layer pattern differs between regimes.

## What we take from it
A **measure** and a **design discipline**. The measure is the amnesic effect: for any property or direction we claim
a model uses, erase it by projection and report the change in the task's own output, beside a same-rank random
erasure — the pair is what turns "the read-out recovers X" into "the model needs X here", and the random arm is what
stops a rank loss being sold as an attribution. The discipline is the selectivity repair test (give the removed
information back in a different form and see whether performance returns), which we can run on any projection-based
account and which cleanly separates "we removed the property" from "we removed a lot of correlated information". The
warning is the correlation-null: a score and a use can be entirely unrelated, so a probe's accuracy should never be
printed as a row in a use table without an intervention on the same axis. Two smaller borrowings: the masked/unmasked
split is the right precedent for our own "training-time regime versus deployment-time regime" comparison, and the
per-label breakdown is the cheapest way to find that an average over a property is hiding opposite signs.

## Tags
unit: direction | feature | layer
form: causal-abstraction | probe
method: probing | ablation | decomposition | other
eval: ablation | prediction/simulation | downstream-task
goal: understanding | evaluation-of-methods
