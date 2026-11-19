# CEBaB: Estimating the Causal Effects of Real-World Concepts on NLP Model Behavior

- **Key:** arxiv-2205.14140
- **Authors / organisation:** Eldar David Abraham, Amir Feder, Yair Gat, Roi Reichart (Technion – Israel Institute of Technology); Karel D'Oosterlinck (Stanford University / Ghent University); Atticus Geiger, Christopher Potts, Zhengxuan Wu (Stanford University)
- **Date:** 2022-05 (v1), 2022-10-12 (v2); NeurIPS 2022
- **Link:** https://arxiv.org/abs/2205.14140
- **Kind:** benchmark (with a comparative method evaluation)
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Explanation methods for NLP classifiers are compared only rarely and, when compared, only on synthetic or simplified data, so there is no agreed ground truth for what an explanation *should* say. The authors argue that if the aim is "why did this model produce this output for this input", the target quantity is a causal effect: how much the model's output moves when one named, human-meaningful concept is changed and everything else about the input is held fixed. Human-written counterfactual edits are the only cheap way to get that "everything else held fixed" condition, because pairs of inputs differing in exactly one concept are almost never observed in natural data. So the paper's larger goal is not a new explainer but a **measuring stick**: a dataset of validated edit pairs plus a metric, so that any explainer that claims to estimate concept effects can be scored against the model's actual measured response to the edit. They then use it to rank six existing methods and a naive baseline under one common metric, and the result is that almost nothing beats the baseline.

## What they studied

Short restaurant reviews from OpenTable: 2,299 originals tied to 1,084 restaurants, expanded by crowdsourced editing to 15,089 texts over four aspect concepts (food, service, ambiance, noise), each labelled positive / negative / unknown-expressed, plus a five-star overall rating, every text labelled by five workers. The unit of analysis is the **concept**, not the token, neuron or feature: the objects of study are the model's output changes under concept interventions. Models explained: `bert-base-uncased` as the main five-way sentiment classifier, with `roberta-base`, `gpt2` and an LSTM with dot attention in the appendix, each fine-tuned on binary, ternary and five-way versions of the task and averaged over five seeds. Explainer methods studied: approximate counterfactuals by sampling, CONEXP, S-Learner, TCAV, ConceptSHAP, CausaLM, INLP, plus a random-effect explainer as a floor.

## How they did it

Three crowdsourcing phases: workers rewrite an original review so that a chosen aspect flips to positive, flips to negative, or is hidden entirely, under instruction to change nothing else; a separate pool validates aspect labels and overall ratings five times per text. Edits with no label majority are quarantined into train only, and the whole edit-group of an original stays inside one split so no test item is a near-duplicate of a train item. Two train sets are published, an inclusive one with all group members and an exclusive one with exactly one text per group, precisely so that a model cannot learn the causal effect by averaging learning signals over near-identical inputs. Ground-truth individual effects come from the star labels of the edit pair (a look-up, not a model); the quantity a model is scored on is its own output difference for the same pair, and an explainer is scored on the distance between its predicted effect and that measured difference. Metric `ICaCE-Error` is the mean distance under three choices: cosine (direction only), L2 (direction and magnitude) and normdiff (magnitude only). Each explainer is fitted per the paper's own re-derivation in a unified notation, with the adaptations documented (TCAV output as a raw directional derivative passed through tanh; ConceptSHAP scoring function swapped from accuracy to class probabilities; CausaLM's average effect swapped for its individual variant), and CAVs learned with SVMs in the same way for both concept-vector methods so the comparison is like-for-like.

## What counts as an explanation here

A **per-example estimate of a causal concept effect**: a vector or scalar saying how much the classifier's output would change for this input if concept C moved from value c to c'. It is deliberately partial — one number per named concept, for a concept vocabulary fixed in advance by the experimenter — and it is meant to be predictive rather than descriptive: the object is to be right about the model's response, not to be readable. The paper explicitly distinguishes three quantities and evaluates only the first: the effect for *this* input, the average effect over the population, and the average magnitude. A model's own average effects are reported as a sanity table and compared with the human-labelled average effect, which is the closest thing here to "does the model compute what the data says".

## How they judged success

`ICaCE-Error` under cosine, L2 and normdiff distances, averaged per aspect and across five seeds with standard deviations, reported in a full appendix grid (Table 10) over four models × three task granularities × three distances × eight methods. Baselines: a **random-effect explainer** (the floor, effect = difference of two random probability vectors) and the **approximate-counterfactual sampler** (the naive method being challenged). Controls on the data side: an author-identity prediction experiment to show worker style does not leak (RoBERTa slightly above chance on the top-5 workers, 0.27 versus 0.16, flattening toward chance as k grows), a double-edit variability study (176 pairs of two different edits made from the same original with the same goal; most differ by at most one star), and the exclusive-versus-inclusive train split as a control on trivial learning of the effect. No human-rater study: nothing here asks whether a person understands the explanation.

## What they found

The model itself does carry the effects: its measured average concept effects track the human-labelled ones in sign and rough ordering (BERT five-way, negative-to-positive: food 1.90, service 1.42, ambiance 1.27, noise 0.75 on the star scale). The explainers mostly do not. On the five-way BERT model, the sampling baseline (Approx) is best or tied-best on two of the three distances, and on normdiff it wins outright at 0.44 against 0.54 for S-Learner, 0.62 for CONEXP, 0.73 for INLP and 0.78 for TCAV, with random at 0.67 — i.e. TCAV is worse than random on magnitude. INLP ties for best on cosine at 0.59 against 0.61 for Approx and 1.00 for random, despite using neither the intervention direction nor test-time concept labels. Across the appendix grid the pattern holds: the methods that know the intervention direction (Approx, CONEXP, S-Learner) win on direction, S-Learner usually wins on L2 in the fine-grained setting, and TCAV and ConceptSHAP frequently sit at or worse than chance, with ConceptSHAP the only method that does not degrade monotonically from ternary to five-way. Classifier quality is high enough that the comparison is not measuring a broken model (test macro-F1 for RoBERTa: 0.98 binary, 0.83 ternary, 0.70 five-way exclusive).

## Limits and open problems they name

One domain only — restaurant sentiment — and the authors present the negative result as a call to action rather than a general law. The explainers never exploit the counterfactual *training* structure the dataset provides; each is fitted the conventional way, and the authors say plainly that better methods should use it. CausaLM and INLP can only estimate interventions to the unknown value, because their counterfactual representation is trained purely to remove a concept. TCAV and ConceptSHAP underperform for reasons the paper does not diagnose ("further analysis is needed"). The authors also note the metric choice matters — direction, magnitude or both — and decline to declare one canonical. Evaluation is individual-level only; the population average effect and absolute effect are defined and left unevaluated.

## What we take from it

Three things, and all three are warnings about our own claims. (1) **A measurement, not a method**: the reusable instrument is the edit-pair design plus the per-example distance between predicted and measured effect. This is the cheapest form of "does the account predict the model's response to an intervention" we could run on our own read-outs — take two inputs that differ in exactly one named concept, predict each output from the account, and score the *difference*. It converts an account from a description into a prediction, and it is the same quantity a surrogate model is supposed to supply. (2) **The floor is the finding**: with a random-effect explainer and a naive sampler both in the table, five of six named methods fail to beat the sampler, and one is worse than random. Any table of ours that lacks a naive/sampled baseline and a random floor is unfalsifiable by construction. (3) **Ground truth by human edit is not free of its own noise**, and the paper measures its noise instead of assuming it: 99% of edits hit the intended aspect label, 88% of texts get a star majority, 176 double-edit pairs show how much two workers diverge when given the same goal, and a worker-identity probe bounds style leakage. That is the reporting checklist for any counterfactual benchmark we build.

## Tags

unit: whole-model
form: probe (explanations as per-concept effect estimates) | causal-abstraction
method: ablation (counterfactual input edits) | probing | other
eval: benchmark | prediction/simulation | faithfulness
goal: evaluation-of-methods
