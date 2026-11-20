# Forgetting Has Neighbors: Localized Collateral Forgetting in Machine Unlearning

- **Key:** arxiv-2605.31317
- **Authors / organisation:** Polina Dolgova, Sebastian U. Stich — CISPA Helmholtz Center for Information Security; Dolgova also Universität des Saarlandes
- **Date:** 2026-05-29 (v1, single version)
- **Link:** https://arxiv.org/abs/2605.31317
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Deletion methods ("unlearning") are sold as a cheap substitute for retraining a model without some training examples, and the field scores them with a handful of scalar averages over the retained set, the deleted set and a test set. The authors' worry is that those averages are a *summary* whose adequacy has never been checked at the level where the damage actually happens — a single example. Their question is therefore not "does the unlearner forget" but "where, exactly, does the unlearnt model disagree with the model that was genuinely retrained", and whether the disagreement has structure. The larger goal is to make deletion auditable: a method that is right on average but wrong on a whole neighbourhood of inputs is not a substitute for retraining no matter what its average says.

## What they studied

Image classifiers: a ResNet-56 on CIFAR-100 (a single class, "couch", deleted at 50% and at 90% of its training examples, plus five other deleted classes) and a ViT-Tiny on SVHN (digit 9, 50% and 90%). The unit of analysis is the **example**, not the model: every claim is a per-input comparison between an unlearnt checkpoint and the checkpoint retrained from scratch on the retained data. Methods compared are gradient ascent, random labelling, plain fine-tuning, Influence Unlearning, SalUn and AMUN, plus their own method. Theory is done in linear regression (gradient ascent on one deleted point) and logistic regression (random labelling, with and without retain-set fine-tuning), where proximity is read as projection onto the subspace spanned by the deleted examples.

## How they did it

Everything is measured relative to a retrained reference θ_retrain, so the quantity of interest is the pointwise discrepancy of predictions or of correct-class confidence, never an absolute change. For each retained or test example they compute a similarity score — cosine similarity of the penultimate embedding to the average normalized embedding of the forget set — bin the examples by that score, and plot the *drop relative to retraining* per bin. Two things make the story causal rather than descriptive. First, two closed-form propositions: gradient ascent on one point inflates the loss of a retained point in proportion to the squared inner product with the deleted point, and random labelling transfers its perturbation to a query through the query's projection coefficients onto the forget span — with an asymptotic result that random labelling does not converge to retraining at all but to the zero-logit predictor. Second, a mitigation built directly from that analysis: train a small ResNet-8 *teacher* on the 1000 retained examples most similar to the forget set, use its predictions on the forget set in place of random labels as soft targets, and fine-tune the full model on retained hard labels plus those soft targets. Ablations vary the support set (nearest versus random versus *farthest* neighbours), the teacher architecture, teacher accuracy and support size, and one ablation removes the teacher entirely and builds targets straight from neighbour labels.

## What counts as an explanation here

A **mechanism stated as a geometry**: the unlearnt model differs from retraining on an example to the extent that the example has a component in the direction (or subspace) of the deleted data, and the reason is that the surrogate target written onto the forget set is inconsistent with the local prediction structure that retraining induces. The explanation is partial by construction — it is proved in two stylized convex settings and carried to the deep models as a hypothesis that predicts the shape of the error-versus-similarity curve — but it predicts two concrete things: that deviations concentrate at high similarity, and that fixing the *targets* rather than the retain updates should reduce them.

## How they judged success

Runs are averaged over 5 seeds and reported as mean ± std. Aggregate columns are retain accuracy, test accuracy, forget-set accuracy, an SVC membership-inference score, a composite "Avg. Gap" (mean absolute deviation from the retrained values across the accuracy columns and the attack score) and wall-clock runtime. Localized columns are the same accuracies restricted to the deleted class — on retained, deleted and test subsets — plus per-similarity-bin drop in accuracy and in correct-class confidence relative to retraining. Baselines are the six published methods under their own tuned settings, and hyperparameter selection is done without touching the retrained model (choose the configuration keeping retain accuracy high and forget accuracy low); one separate retrain-free diagnostic — a single gradient-ascent step plus per-bin accuracy drop — is used to pick the support size.

## What they found

The locality effect is real and large. At 50% deletion of one CIFAR-100 class, every method overshoots forget-set accuracy relative to retraining (retrain 47.4%; random labelling 73.2, fine-tuning 79.0, gradient ascent 68.6, Influence Unlearning 70.6), and the damage shows up on the *retained* portion of the deleted class: gradient ascent falls to 66.6% against retraining's 99.7%, Influence Unlearning 74.6%, SalUn 81.2%. Their method reaches 97.4% there with 53.4% on the forget set, giving the lowest Avg. Gap of the table (2.4 against Influence Unlearning's 6.6 and gradient ascent's 7.7) — bought with runtime: 226.5 s against gradient ascent's 1.8 s and a retrain at 1558.9 s. At 90% deletion the ordering holds (LTD 2.7 Avg. Gap, Influence Unlearning 2.9, gradient ascent 7.6), and on SVHN with a ViT-Tiny the aggregate gap is 2.6 against AMUN's 4.0. The honesty check is worth recording: across five additional deleted classes their method has the best *aggregate* gap on all five but the best *affected-class* gap on only three, ranks second on the other two, and AMUN wins one of them. The support-selection ablation is the sharpest single result — random or farthest-neighbour support collapses forget-set accuracy to 0.3% and 0.0%, and affected-class forget accuracy to 20.0 and 15.9 against 53.4 for nearest support — and removing the teacher (using neighbour labels directly) also clearly loses on the same columns.

## Limits and open problems they name

The mitigation depends on the quality of the local support set and on the representation used to select it, which works because their forget sets are homogeneous (a fraction of one class); multi-modal forget sets are left to per-example or clustered neighbour selection sketched in an appendix, untested here. The theory is linear and logistic regression; the deep-network results are correlational support for the predicted shape. Evidence is confined to image classification, two datasets and two architectures, and the authors call extension to other deletions and architectures future work. Their own local metric is a similarity bin, which is itself a modelling choice, and their headline aggregate gap is one of several equally defensible composites.

## What we take from it

Two instruments and one framing. **(a) The reference model is the measurement.** Every number in the paper is a difference from a model trained the "expensive correct" way, and that is exactly the accounting our probe and circuit batteries need: an explanation's score should be reported as a discrepancy from a known-good reference behaviour, not as an absolute quantity, and it is only interpretable binned by the *structural* quantity the theory names. **(b) Bin the error by the variable the mechanism predicts, and print the per-bin curve.** A single aggregate says nothing about where an instrument fails; here the whole finding is a monotone curve against similarity to the forget set, which in our register is the same construction as plotting probe error against distance from a feature direction. **(c) The "kept component can be fixed while the neighbourhood is not" warning.** Their retain-fine-tuning proposition says that restoring agreement on the retained *training points* does not restore agreement on general queries, because the residual lives in the subspace of what was removed — a precise version of the general caution that passing the check set does not license claims off it. **(d) Neighbourhood support as a design principle**: their ablation shows that the *nearest* neighbours are what carry the correct signal and that random or antipodal support is worse than useless, which is a cheap, reusable control for any local surrogate we train to stand in for an expensive reference.

## Tags

unit: whole-model
form: causal-abstraction
method: ablation | other
eval: ablation | downstream-task | benchmark
goal: control/editing | evaluation-of-methods
