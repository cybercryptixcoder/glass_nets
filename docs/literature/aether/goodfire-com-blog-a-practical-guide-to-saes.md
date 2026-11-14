# A practical guide to sparse autoencoders (SAEs)

- **Key:** goodfire-com-blog-a-practical-guide-to-saes
- **Authors / organisation:** Goodfire (blog post; research-engineering voice, individual author not named on the page)
- **Date:** 2026-09-28
- **Link:** https://www.goodfire.com/blog/a-practical-guide-to-saes
- **Kind:** blog/research post (practitioner guide)
- **Read from:** full text (~18.4k chars, one fetch, whole post including footnotes and the Silico walkthrough; no figures or tables inspected beyond the SAE/probe/black-box comparison table, which is text)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The post exists to puncture a specific misconception: that SAEs and interpretability are the same project. The authors report that SAEs are used less often, and for narrower purposes, than the publicity around feature demos suggests, and they want readers to choose a tool deliberately rather than defaulting to the most famous one. The larger goal is a division of labour inside the interpretability toolbox — unsupervised dictionary methods for exploration, supervised single-concept methods for precision, black-box judges where you have neither internals nor labels — with a stated decision rule for each. Nothing here is a new result; the value is the taxonomy and the failure modes the authors are willing to name about their own product's method.

## What they studied

No model was trained for the post. The unit of analysis is the *method*: the SAE (encoder into a wide sparse feature space, decoder back to activations, trained on reconstruction under a sparsity penalty), compared against probes (small classifiers on labelled activations) and black-box monitors (LLM judges reading inputs/outputs). Worked examples are drawn from published work — the auditing game, an implicit-planning discovery, vision conceptual blindspots, a fine-tuning "toxic persona" feature, an epigenetic foundation model's DNA-fragment-length features, hallucination/knowledge probes, and decomposition of a persona vector into finer SAE features.

## How they did it

The post is organised as three questions. First, what SAEs are and what they assume: activations described as a sparse combination of linear directions, with the explicit caveat that curved or multi-dimensional structures get split into fragments or missed. Second, which problems they suit — exploration where you do not know the target and precision is not essential: auditing for hidden objectives, model diffing before/after fine-tuning, dataset-level analysis, what a model knows, and validating or decomposing a probe/steering direction into finer features. Third, when to use something else: whenever the concept is known and labels can be obtained, a probe reads the full activations rather than a lossy reconstruction and generally matches or beats probes built on SAE features, and SAE features have tended to underperform steering vectors and prompting for steering. The training half walks the pipeline: pick a layer and hook site (middle layers for abstraction), collect hundreds of millions to billions of tokens of activations with *coverage matched to the deployment domain*, choose width/expansion factor, sparsity level and architecture (TopK / BatchTopK / JumpReLU variants), handle dead features, then evaluate on two axes — reconstruction quality including whether the model still behaves normally with the SAE swapped in, and feature interpretability via autointerp with held-out prediction of when a feature fires. It closes with an agent-driven walkthrough (their Silico tool plans layers, width, sparsity, data and compute from a natural-language goal).

## What counts as an explanation here

A feature dictionary: a labelled list of directions per layer, with each label meant to describe where the feature fires. The post is explicit that this is a *coarse map* and not a mechanism — features come without labels, autointerp labels are descriptions of maximal activations rather than statements about causal role, and the dictionary cannot see what its reconstruction error omits. It is a hypothesis generator pointed at a human or another tool for follow-up, not a predictive model.

## How they judged success

No experiments, so no metrics of their own. The evidence they cite is comparative: head-to-head results where probes on raw activations match or beat probes on SAE features; negative results for SAEs on downstream tasks (Kantamneni et al. and Google DeepMind's published negative results); the claim that SAE steering tends to lose to steering vectors and prompting, with the qualifier that careful feature selection and labelling can bring it close to fine-tuning. Their comparison table scores three methods on labelled-data need, unknown-concept discovery, precision on a known concept, internals access, upfront cost and per-use cost. Benchmark and dataset names are not given for the probe-vs-SAE comparisons — the post points at the sources rather than reporting numbers.

## What they found

The load-bearing claims are negative and hedged in a way that is unusually consistent with the rest of this folder: SAEs are unsupervised, scalable and unopinionated, but specific to one model and one layer, assumption-laden, unlabelled, and lossy; they are best for exploration and worst exactly where interpretability is used as a detector. Three reasons are given for the probe advantage — reconstruction error is invisible to anything built on features, the target concept may be split, lumped or absent in the dictionary, and feature labels can be subtly wrong. The complementary recipe is stated plainly: use the SAE to find something worth looking for, use an LLM judge to label more examples, then train a probe for production. On training, two practical points are sharp: coverage of the deployment domain (auditing a coding agent requires coding-agent transcripts, not generic web text) and the claim that the selection-and-labelling pipeline, not the architecture, decided whether steering came close to fine-tuning in one recent study. The footnoted alternative methods — transcoders, Matryoshka SAEs, crosscoders, temporal feature analyzers, block-sparse featurizers, J-lens, natural-language autoencoders — are listed but not evaluated.

## Limits and open problems they name

No numbers, no baselines of their own, and the comparative claims are attributed rather than measured here — the reader cannot check effect sizes from the post. Two failure modes are named without a fix: concepts represented as curved or multi-dimensional structures, which the linear-sparse assumption cannot hold; and autointerp descriptions derived from strongest activations, which miss behaviour in the tail of the firing distribution. Labelling cost at millions of features is called a significant compute cost with no estimate given. The post also does not say how to decide, for a *new* model, whether coverage is adequate before spending the training budget — the domain-coverage advice is qualitative.

## What we take from it

A decision rule and a warning, both cheap to adopt. The rule: before reaching for a dictionary in our own pipeline, ask whether the target behaviour is known and labelable — if it is, a probe on full activations is the better instrument, and our survey's dictionary-based read-outs should be paired with a probe baseline on raw residuals for exactly this reason. The warning is the one our project is most exposed to: an SAE is lossy and *specific to a layer and model*, so a dictionary miss is not evidence of absence, and any "we did not find a feature for X" claim needs the reconstruction-error and coverage denominators attached. Two habits worth importing into our write-ups: report whether the model still behaves normally when the reconstruction is swapped in (a behavioural check on the dictionary itself, not just an MSE), and score autointerp labels by their ability to predict firing on held-out inputs rather than by how well they read.

## Tags

unit: feature | direction | layer
form: dictionary | probe
method: SAE | probing | steering
eval: human-rating | downstream-task | faithfulness
goal: understanding | evaluation-of-methods | safety/auditing
