# RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations

- **Key:** arxiv-2402.17700
- **Authors / organisation:** Jing Huang, Zhengxuan Wu, Christopher Potts (Stanford University); Mor Geva (Tel Aviv University); Atticus Geiger (Pr(Ai)R Group) — affiliations from the byline block and printed e-mail lines.
- **Date:** 2024-02-27 (v1); this entry reads v2, 2024-08-26
- **Link:** https://arxiv.org/abs/2402.17700
- **Kind:** benchmark (dataset + method)
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Individual neurons carry several high-level concepts at once, so a method that claims to have found "the continent feature" has to be judged on whether the thing it found can be moved without dragging other attributes along. The authors' complaint is that the field had no shared, quantitatively controlled way to ask that question: featurizer methods (dictionaries, rotations, probes, masks) were each evaluated with their own metric on their own data, and nobody had compared them apples to apples. RAVEL is the answer — a benchmark whose task is attribute disentanglement, plus a new training objective, multi-task DAS, that is fitted against several causal criteria at once. The larger goal is the one this whole survey is about: making a localization claim a scored, falsifiable object rather than a story attached to a heat map.

## What they studied

Llama2-7B as the single target model. Unit of analysis is a *feature* — a set of neurons, a featurizer over those neurons, and a subspace within the featurized space — at one intervention site: the residual stream above the last token of the entity, across layers. The objects being located are entity attributes: five entity types (city, Nobel laureate, verb, physical object, occupation) with three to six attributes each and between 563 and 3,552 entities per type (e.g. city carries country, language, latitude, longitude, timezone, continent), queried through fifty to a hundred and fifty prompt templates per type, in plain-prose, zero-shot, few-shot and JSON formats. The published Llama2-7B instance filters down to 2,800 entities and about 15k–33k test pairs per type at 94–97% model accuracy on the attribute task.

## How they did it

The dataset is generated so that base–source pairs differ in exactly one attribute and its answer token, then split two ways to separate two senses of generalisation: an Entity split (same templates, unseen entities) and a Context split (same entities, unseen templates), each with a train/dev/test division. Evaluation is by interchange intervention: run the base prompt, replace the value of a candidate feature with the value it takes on the source prompt, and look at the next token. Two quantities follow from that — the *Cause* score (did the answer move to the source's value when the source's attribute differed) and the *Iso* score (did the answer stay put when the source's attribute was the same), averaged into a *Disentangle* score. Seven methods are fitted on the training split and scored on test: no featurizer (replace the whole representation), PCA, a sparse autoencoder (4,096 → 16,384 latents trained on Wikipedia sentences mentioning training entities), a relaxed linear adversarial probe in the R-LACE style, differential binary masking, DAS, and multi-task variants of the last two in which the loss adds the isolation objective as a second task per attribute. Hyperparameters (feature width via L1 coefficient or rank) are chosen on dev; a single test run is reported at the dev-best setting.

## What counts as an explanation here

A located subspace with an intervention-backed behavioural signature: "this feature causes this attribute and leaves the others alone." It is explicitly partial and local — one attribute, one token position family, one layer at a time — and it is predictive only in the narrow sense that a feature with a high Cause-and-Iso profile is expected to redirect future interventions of the same shape. The paper is careful that a failure to disentangle is itself an admissible finding ("evidence that two attributes might not be separable is an insight into how knowledge is structured"), which makes the benchmark's negative results first-class rather than noise.

## How they judged success

The Disentangle score, a weighted average of Cause and Iso, averaged over attributes and entity types; the full-representation baseline; and two generalization splits (Entity, Context) that are reported side by side. Feature-width sweeps from about 1% to 50% of the residual dimension produce the Cause-against-Iso curve. Per-attribute Cause matrices for the city type serve as entanglement diagnostics, and every supervised method gets the same matrix so that an entanglement pattern recurring across methods can be told apart from an artefact of one training objective.

## What they found

- The full-representation baseline scores about 40 on both splits, and the unsupervised featurizers barely move past it: PCA is at or below the baseline (39.5/39.1), the sparse autoencoder reaches 48.6/46.8.
- Attribute-supervised fitting does better (RLAP 48.8/50.9), and counterfactual-supervised fitting better still: DBM 52.2/49.8, DAS 56.5/57.3, and multi-task DAS 60.1/65.6 at the top on both splits.
- Adding the isolation task to the training loss lifts DBM by 1.5 points (Entity) and 4.1 (Context), and DAS by 3.6 and 8.3 respectively.
- ISO and Cause trade off with feature width: widening the intervened subspace raises Cause and lowers Iso.
- Entanglement is pair-specific and consistent across all five supervised methods: country-with-language and latitude-with-longitude resist separation even under multi-task supervision, while language-with-continent separates cleanly — a property of the model's structure rather than of the method, since the five methods optimise different objectives.
- Disentanglement emerges with depth: on the country feature, Iso climbs from about 0.5 to 0.8 between layer 8 and layer 16, and the best Disentangle score arrives only at layer 16.
- Cost note the authors print: SAE training is cheapest, while a 128-rank DAS or multi-task DAS fit over a thousand intervention examples takes tens of seconds, and evaluation runs at about twenty seconds per thousand examples.

## Limits and open problems they name

One model (Llama2-7B) — they explicitly ask that RAVEL be re-instantiated on any model one wants to make claims about, because the ranking of methods may not transfer. The intervention site is restricted to the residual stream above the entity's last token, although attributes may live across several tokens and several layers. The SAE featurizer is only invertible up to its reconstruction loss, which makes it the one method in the table that is not exactly invertible, and the paper says so. Hyperparameter selection on dev with a single reported test run means the table is a dev-best picture, not an averaged one.

## What we take from it

- **Measure: a two-sided score, not a one-sided one.** Cause alone is the thing every patching paper reports; Iso is the number that says whether the effect was targeted. RAVEL's Disentangle score is a concrete, cheap form of this that our batteries can adopt unchanged — move the feature, check the answer moved for the right attribute, check it did not move for the others.
- **Design: entangle the attributes on purpose.** The dataset is built so that attributes are correlated to different degrees (country entails continent, timezone implies country, latitude and longitude are independent with identical output spaces). That is how the finding "country-with-language resists separation" becomes interpretable: the benchmark contains a graded series of entanglement difficulties, so a method's per-pair profile is a curve over that series rather than one number.
- **Warning: the full-representation baseline should be printed in every table.** At 40 points it sits above PCA and not far under the SAE, which means a table without it would have made two featurizers look like improvements while they are doing nothing a whole-vector patch did not already do.
- **Warning: splitting the *experiment*, not only the data.** Entity and Context splits answer two different questions, and the ranking's stability across them is what lets the authors say "supervised fitting wins" rather than "supervised fitting wins on these templates". A method that wins on Entity and loses on Context would have been a different paper.
- **Method to reuse: adding the isolation criterion as a second training task.** Their multi-task variant is a small change to the loss that moves the score by several points, and it is the clearest demonstration in this folder that what a featurizer finds is a function of the criteria it was optimised against, not of the model alone.

## Tags

unit: feature | direction | layer
form: direction | probe | dictionary | steering
method: DAS | probing | SAE | decomposition | other
eval: patching | downstream-task | faithfulness | benchmark
goal: evaluation-of-methods | understanding
