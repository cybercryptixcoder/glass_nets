# Probing Classifiers: Promises, Shortcomings, and Advances

- **Key:** arxiv-2102.12452
- **Authors / organisation:** Yonatan Belinkov; Technion – Israel Institute of Technology
- **Date:** 2021-09-22 (v4; accepted to *Computational Linguistics* as a squib, 2021-09-08)
- **Link:** https://arxiv.org/abs/2102.12452
- **Kind:** survey/agenda (a critical review — the paper calls itself a squib)
- **Read from:** full text (v4 HTML, body read end to end; references not read line by line)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Probing classifiers — train a small supervised model to predict a linguistic property from a frozen
representation, and read its accuracy as evidence about what the representation contains — had become one of the
two or three standard instruments in NLP interpretability, applied to embeddings, recurrent encoders and then to
every layer of BERT. The author's complaint is that the instrument was being read far beyond what it can support:
the same accuracy number had been glossed as representation *quality*, as *readability* of the information, as
*extractability*, and sometimes as an account of how the original model reaches its answers, and those four
readings are not equivalent. The article is a stock-taking that decomposes the framework into its named
components, then walks the component list to say which of the field's favourite conclusions each component
choice licenses and which it quietly forecloses. It is a step toward making representation claims relative
(comparisons with stated bounds) rather than absolute (a bare accuracy), and it is the map of the probe-side
critique literature for a project that intends to publish probe results and needs to know which criticisms are
already settled.

## What they studied

No new experiments: the unit of analysis is the *probing experiment* itself, taken as a nine-object structure.
The base components are the original task, the original dataset and the original model (with its performance);
the intermediate representations that model produces for a given input at a given layer; the probing task, the
probing dataset, the probing classifier, and the probing performance. The article adds a second ring of
components that the field's better work has accumulated: a baseline model and a skyline or upper-bound model, a
control task with its own control-task dataset and the derived selectivity score, a control function with its
information gain, the probe's minimum description length, a control dataset, and — in the causal line —
representations taken from the model *after an intervention*. The literature surveyed runs from early classifiers
fitted on static word embeddings through morpho-syntactic tagging of machine-translation hidden states to
sentence-embedding probes and, at the end, to probe-guided representation editing.

## How they did it

The method is formal, not empirical: define a model that maps input to output, trained on a dataset and scored by
a measure; denote by the representation of an input at some layer; define a probing classifier mapping
representations to a property; and observe that the reported score is a function of the classifier *and* its
dataset *and* the original model *and* the original dataset — four dependences that a bare accuracy hides. From
that definition the article reads probing as estimating the mutual information between representations and
property. It then reviews each critique against the works that made it, in five groups, and for each group states
what the proposed advance does and does not fix. The comparison group covers majority-class baselines, simpler-
representation baselines (static embeddings), randomized representations, skylines (human performance, literature
state of the art, a dedicated model trained to map input to property without the frozen-representation
restriction), control tasks, control functions, and control datasets. The classifier-choice group covers the
linear-versus-expressivity argument, the report-the-trade-off argument, minimum description length as a joint
measure of accuracy and complexity, pruning-based probes, and parameter-free probing that infers structure from
attention weights, representation distances or representational similarity instead of learning weights. The
causal group covers probe-gradient editing, counterfactual representations, iterative projection of a property
out of the representation, and adversarial removal with a positively trained guard probe. The last two groups cover
datasets-as-proxies-for-tasks and the requirement that the probed property be named in advance.

## What counts as an explanation here

A probe, and the explanation it licenses is explicitly *relative and graded*: the article's closing position is
that absolute claims about representation quality are hard to sustain, while relative claims — how extractable a
property is from this representation compared with that one, or from these representations compared with a random
or untrained baseline — are supportable when the controls and bounds are printed. It is not a mechanistic form at
all: nothing about circuits, directions or weights. The one mechanism-adjacent reading it admits is causal, and
only when the representation is actually intervened on and the effect on the original task is measured;
correlation between representation and property is described as necessary but not sufficient. Its own predictive
content is a component checklist — a probing claim is interpretable only if the original task, dataset and model,
the probing task, dataset and classifier, and the bounds and controls are all stated.

## How they judged success

No benchmark of its own; success is the coherence of the taxonomy and its coverage of the critique literature.
Every advance it endorses is stated with the measure it introduces — selectivity as the gap between probing
performance on the linguistic task and on a label-randomized control task; information gain as the change in
mutual information under a control function; minimum description length of the code that transmits the property
given the representations; probe performance and original-task performance measured *together* after a
representation edit; human, state-of-the-art and dedicated-model skylines as upper bounds; majority, simpler-
representation and randomized-representation probes as lower bounds. It also adjudicates between competing
advances: it notes that the control-task and control-function criteria were subsequently shown to be close to
equivalent, both in theory and in experiment, which is the one place the article compares two proposed fixes head
to head rather than listing them.

## What they found

The synthesis, in the article's own ordering. On bounds: randomized-representation probes show that even random
features yield decodable information, so any probing number read without that floor is uninterpretable — the
same conclusion later work made structural. On classifier choice: linear probes had been preferred on the
argument that an expressive probe risks inventing features the network does not use, but expressing the same
comparison as selectivity rather than accuracy can *reverse* the ranking of two representations, so a
"representation A beats B" claim is a claim about the probe class as much as about A and B; the proposed repair
is to report the accuracy–complexity trade-off across a range of probes rather than one point, with MDL as the
joint score. On causation: the surveyed causal studies disagree with each other in a way the article leaves
standing — probe-gradient editing was read as evidence that probes find features the model uses, while iterative
projection of probed properties out of the representation found that high probing accuracy does not entail a large
drop in the original task after removal, i.e. the opposite conclusion — and the article's summary is only that
causal approaches *may* reveal which properties are used. On datasets: because models are usually trained on
different corpora, statements about architectures are confounded with statements about data, and the article
names this as an open gap rather than a solved one. On properties: requiring the property in advance restricts
probing to annotated properties, biases the investigation toward properties the researcher already suspects, and
means the framework is mostly useful when one has prior expectations; one cited attempt learns latent clusters
instead and recovers both known and unannotated categories. The summary's practical advice is conditional on the
question being asked: control tasks are suited to word-level properties, control datasets to sentence-level;
complexity matters when ease of extractability is the question; intervention when use is the question;
parameter-free probes when the probe's own learning is the worry. It also concedes that probing has produced
surprisingly few examples of *action* — it names a handful where probing findings motivated an architectural
decoupling, a transfer-learning procedure, multi-task supervision on intermediate layers, and translation-model
selection, and no more.

## Limits and open problems they name

The article is a review of a field, so its limits are the field's. The labelled-property requirement is stated as
structural and unfixed. The dataset confound is stated as an ideal that nobody has run: comparing architectures
trained on one dataset, or one model trained on several. The effect of the *probing* dataset — its size and
composition — is called under-studied. Control tasks are noted to be well suited to word-level properties and of
unclear applicability to sentence-level ones. The causal line is left as a disagreement among the cited studies
rather than resolved. Two smaller limits are honest and worth carrying: the article does not claim every study
must run every control, only that the components be acknowledged; and it observes that probing results have
rarely been converted into improvements, listing the exceptions individually rather than as a body of evidence.

## What we take from it

- **Framing (the main yield).** The nine-component decomposition is the checklist our own probe-based claims
  should be reported against: original task/dataset/model, where the representation is taken from, probing
  task/dataset/classifier, the score, and the lower bound (majority, simpler representation, randomized
  representation) and upper bound (human, literature SOTA, dedicated model) it sits between. A bare accuracy from
  our runs should not be published as a finding.
- **Measure.** Parameter-free probing — inferring the property from attention weights, representation distances
  or representational similarity without fitting weights — is the cheapest way to dodge the capacity confound
  entirely, and the article places it at the low end of the complexity axis. For our read-outs of an activation
  space this is the arm we have never run: a property score computed from geometry, with no fitted reader to
  absorb the signal.
- **Design rule.** Report the accuracy–complexity trade-off as a curve, not a point, and use a joint score
  (MDL) rather than accuracy when comparing probes of unequal size.
- **Warning.** Probe accuracy and probe-guided removal can disagree in direction — high extractability with a
  small effect of removal on the original task — so an account that names a property must state which of the two
  it has shown. This is the same lesson the folder's deletion-test entries reach from the attribution side, and
  it means "the probe reads X from these activations" and "the model uses X" are separate rows.
- **Warning.** Control tasks are type-level and therefore not directly portable to activation-level read-outs;
  the sentence-level substitute the article endorses is a control *dataset* in which the property carries no
  information about the original task's answer. That is the construction our own probes should adopt.
- Coreference `arxiv-1909.03368` (the control-task paper, whose selectivity this article adopts as the standard
  probe-side instrument and whose word-type restriction it states), `arxiv-2104.07143` (the probing dataset as a
  first-class variable — the article's "datasets vs. tasks" section, made into an experiment) and
  `arxiv-1911.03429` (comprehensiveness/sufficiency — the attribution-side version of the
  extractability-versus-use distinction).

## Tags

unit: layer | whole-model | direction
form: probe | description
method: probing | ablation | other
eval: benchmark | faithfulness | prediction/simulation
goal: evaluation-of-methods | understanding
