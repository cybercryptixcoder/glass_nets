# On the Limits of Sufficient Explanations

- **Key:** arxiv-2611.18163
- **Authors / organisation:** Klaus Weinbauer, Thomas Gärtner, Sagar Malhotra — Machine Learning Research Unit, TU Wien; Peter F. Stadler — Bioinformatics Group, Leipzig University, Max Planck Institute for Mathematics in the Sciences, Santa Fe Institute
- **Date:** 2026-11-13
- **Link:** https://arxiv.org/abs/2611.18163
- **Kind:** paper
- **Read from:** full text (HTML v1, §§1–6 and the theorem statements; supporting appendices skimmed—some proofs were not checked line by line)
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Sufficient explanations — a part of the input that on its own guarantees the predicted label — are widely sold as the strong, faithful form of post-hoc explanation for graphs, strings and text spans. Formal work on them has lived in fixed-width feature spaces, where the question "does a sufficient explanation exist" has a trivially positive answer, and the practical literature reports sufficiency scores without any statement of what the task could possibly allow. The authors want the missing negative half: a characterisation of when no sufficient explainer can exist at all, and dataset-level statistics that tell a practitioner, before any explainer is trained, how much sufficiency is available. The larger goal is to separate the explainer's performance from the task's ceiling, so that a low sufficiency score stops being read as a broken method.

## What they studied
Classification on structured instance spaces — graphs (molecular activity, mutagenicity, enzyme class, protein function) and strings (peptide and protein toxicity predictions, SST-2 sentence sentiment) — where the unit of explanation is a *sub-instance*: a subgraph or a substring, not a feature vector coordinate. The formalism is general over any partial order of sub-instances; the empirical section covers ten datasets, from 183 instances (MUTAG) to 66,973 (SST-2). No neural network is trained — the analysis is over labels, instances and the inclusion relation only.

## How they did it
They define an explanation as a sub-instance selected from the instance under a *select relation*, an acyclic order that encodes which sub-instances an explanation formalism is allowed to pick (node-induced subgraphs, edge-induced subgraphs, arbitrary subgraphs, substrings). They then define a *co-bound*: an instance from which some sub-instance can be selected that is also a sub-instance of a differently labelled instance. The central theorem is that a sufficient explainer exists exactly when there is no co-bound, which makes existence a joint property of the classifier and the select relation rather than of the explainer. They extend the characterisation to class-wise sufficiency (a sufficient explainer for one class exists if every other class is monotone), to probabilistic sufficiency in the sense of Dasgupta et al. (an upper bound for any explainer derived from the trivial "return the whole instance" explainer's coverage and its same-label rate), and to deletion sufficiency (always achievable, and vacuous for monotone classes — a single node suffices to "explain" acyclicity or planarity, and so does the empty explanation). Three dataset statistics follow: the fraction of instances that are co-bounds, the inter-class relation ratio (share of sub/super-instance pairs among distinct instances whose labels differ), and the minimum accuracy loss needed to relabel a dataset co-bound free, computed over connected components of the relation graph.

## What counts as an explanation here
A sub-instance of the input that is sufficient for the prediction, together with a hard statement about whether any such object can exist for a given task. It is explicitly partial and explicitly *not* predictive of behaviour: the paper's product is a ceiling and an obstruction, not a predictor, and the authors say outright that similar-instance explainers, prototypes and counterfactuals are outside the formalism. The obstruction is in-distribution — unlike the usual out-of-distribution complaint about perturbation tests, it needs no perturbation at all.

## How they judged success
Proofs for the characterisations (Theorem 4.1 existence iff co-bound-free; Corollary 4.3 all constant-size instance spaces admit a sufficient explainer; Proposition 4.1 unbounded selection orders make every explainer sufficient; Theorem 4.3 class-wise sufficiency under monotone complementary classes; Corollary 4.4 for purely existential positive first-order graph classifiers; Theorem 4.4 the probabilistic bound; Propositions 4.4–4.5 and Theorem 4.5 on deletion sufficiency and consistency). Empirically: the three statistics computed per dataset, with wall-clock runtime for the co-bound search, and their argument checked against the label structure rather than against any explainer's output.

## What they found
Nine of the ten datasets contain co-bounds, and in seven of them co-bounds occur in every class, so no classifier that fits those labels admits a sufficient sub-instance explainer. SST-2 is the worst: 21% of instances are co-bounds, 24% of relations are between differently labelled instances, and relabelling to co-bound-free would cost 44% accuracy. MUTAG has 14% co-bounds against a 23% relabelling cost, while Mutagenicity has 0.3% co-bounds and a 0.35% cost — nearly explainable by this criterion. The instructive reversal is ENZYMES: only 2.2% of instances are co-bounds yet relabelling costs 82% accuracy (its constant-classifier accuracy is 16.67% on six balanced classes), so the co-bound count alone is not the measure. The minimum local probabilistic sufficiency bound is far below one for several datasets — 18% on ENZYMES, 58% on PROTEINS, 73% on MUTAG — meaning some single instance cannot be explained even probabilistically above that level by any explainer. ToxinPred2 is the one co-bound-free dataset (100% bound).

## Limits and open problems they name
The formalism covers only sub-instance selectors; prototypes, similar-instance and counterfactual explainers are left out, and the authors say the notion does not transfer to them. Class-wise and probabilistic results are existence statements, not constructions — nothing here tells you which explanation to emit when sufficiency is unavailable. Contexts in the empirical section are the ground-truth labels, so the statistics describe the dataset's ceiling and not any particular trained model's behaviour. The relation-graph computation is expensive at scale (5 hours for PROTEINS, 3 for MCF-7).

## What we take from it
A **measure and a warning**. The measure: any white-box model of a network that is scored by "does this piece of the input/state alone determine the output" inherits a ceiling from the data and the granularity of the piece, and that ceiling is computable from labels and the inclusion relation alone, before any probe is fit — so our evaluation tables should carry it, or at minimum should stop reading a low sufficiency number as a failure of the readout. The warning is sharper for us than for them: the mechanism they expose is *sub-instance containment*, and our probe battery has the same structure whenever a candidate state or feature subset of one input also appears in an input with the opposite label — the identical-answer requirement in the folder's stratification entries is the same obstruction arriving from a different direction. And the deletion-sufficiency result is the one to quote at anyone (including us) who reports a perturbation test as evidence: for a monotone class it is satisfied by an empty explanation, so a deletion test on its own tests nothing.

## Tags
unit: layer | feature
form: causal-abstraction
method: attribution | ablation | other
eval: faithfulness | completeness | benchmark
goal: understanding | evaluation-of-methods
