# A Benchmark for Interpretability Methods in Deep Neural Networks

- **Key:** arxiv-1806.10758
- **Authors / organisation:** Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, Been Kim — **Google Brain** (affiliation line and google.com addresses in the HTML author block)
- **Date:** 2018-06 (v1 2018-06-28; v3 2019-11-05). NeurIPS 2019 per the comments field.
- **Link:** https://arxiv.org/abs/1806.10758
- **Kind:** paper (benchmark)
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Feature-importance estimators are offered as explanations of a classifier's decision, and the field had no way to say which of them is more accurate. The authors name the two obstructions plainly: there is no ground truth for importance (if we knew it we would not be estimating it), and there are dozens of candidate methods with theoretical credentials that do not settle the question empirically. Their complaint about the standard cheap test — delete the "important" features and re-score — is that the modified inputs are off-distribution, so a drop in accuracy cannot be attributed to removal of information rather than to the distribution shift. The larger goal is a *benchmark*: an ordering of estimators by how well their rankings capture the features a network actually uses, priced by an experiment rather than by argument. It is the evaluation-side paper for the folder's whole line of attribution work.

## What they studied

A ResNet-50 image classifier on three large open image datasets — ImageNet (1,281,167 train / 50,000 test images, 76.68% top-1 unmodified), Birdsnap (47,386 / 2,443, 66.65%) and Food 101 (75,750 / 25,250, 84.54%). Eighteen estimators in total: three base methods (raw gradient heatmap, Guided Backprop, Integrated Gradients) and three ensembling wrappers around each (classic SmoothGrad, SmoothGrad-Squared, VarGrad), plus a squaring-only arm in the appendix. Two controls that never see the model: a random importance assignment and a fixed Sobel edge filter. The unit of analysis is the **input pixel ranking** — a feature-importance estimate, not a neuron, circuit or direction — and the thing actually measured is a **retrained network's accuracy on data with that ranking's top fraction neutralised**.

## How they did it

For every image in both train and test, compute the estimator's ranking, sort pixels, and replace the top *t* percent (t = 10, 30, 50, 70, 90) with the per-channel mean. That produces a new train set and a new test set *drawn from the same modified distribution*, which is the whole point: train and test are corrupted identically, so no off-distribution gap is left to confuse the reading. Retrain ResNet-50 from random initialisation on each modified train set, five independent runs per cell, and report mean test accuracy. Better estimator ⇒ steeper accuracy drop as *t* grows. The full grid is 540 modified datasets (180 per original dataset). Two auxiliary experiments: a 16-dimensional linear toy problem with four known-informative features and a deliberately inverted ("worst possible") ranking, where the same comparison is run with and without retraining to show what the naive deletion test reports; and KAR, a mirror benchmark that *keeps* the top *t* pixels and replaces the rest, so that a good estimator is the one whose kept set preserves accuracy. Training hyperparameters (learning rate, steps) are tuned per dataset and held fixed across estimators.

## What counts as an explanation here

An **ordered ranking of input pixels by estimated importance**, i.e. a direction-free attribution over inputs — the paper's object is not the explanation's readability but its correspondence to the features the network can be shown to depend on. It is explicitly partial: the benchmark scores the *order* of a ranking, says nothing about magnitude calibration, and cannot see inside the network at all. Its predictive content is a prediction about an intervention the authors actually perform — neutralise these pixels, retrain, and accuracy should fall by this much — so it is the folder's clearest instance of "an account is a claim about a counterfactual, priced by running it".

## How they judged success

Primary read-out: mean test accuracy of five independently retrained ResNet-50s at each degradation level, plotted as a curve and fully tabulated per cell. Baselines and controls: a random Bernoulli ranking (a hard floor that uses no model information), a Sobel edge filter (a fixed input-side heuristic), and a "keep instead of remove" (KAR) mirror. Correctness anchors: the toy linear problem with a known ground-truth ranking and its exact inversion, plus the no-retrain comparison which serves as the negative control for the benchmark itself. Reported statistic is the mean over five seeds with per-cell variance, alongside the accuracy *range* between the best and worst base estimator at the harshest threshold.

## What they found

Retraining is what makes the measurement honest, and the size of the correction is large: under the naive no-retrain protocol, replacing 90% of ImageNet pixels with the mean drives accuracy to 0.5% from a 76.68% clean baseline, whereas after retraining on the same 90%-removed data a model still reaches 63.53% ± 0.13. Even 10% of pixels retained trains a net to 63.53% (the paper's own numbers for the retained-fraction and the removed-fraction settings coincide at that value), i.e. images are hugely redundant and a deletion curve mostly measures redundancy plus distribution shift.

Against that backdrop the headline is a near-null: on the authors' reading, raw gradients, Integrated Gradients and Guided Backprop are **at or below the random assignment at every dataset and every threshold**, falling further behind as *t* grows. The ranking is cleanest on Birdsnap, where at *t* = 90% removed random scores 38.12, Sobel 44.33, GRAD 56.08, IG 60.30 and the Guided Backprop row 59.76 — so IG and GB clear random there while still trailing the model-blind Sobel filter; on Food 101 at the same threshold random is 68.37, Sobel 69.58, GRAD 78.36, IG 80.87. This entry therefore reports the paper's verdict as a *tie-with-or-fall-short-of the controls* rather than the flat "worse than random" of the abstract, because the tables put the base estimators above the random floor on two of the three datasets while leaving them no decisive edge over an edge detector. Classic SmoothGrad does not fix this and is often worse than the single estimate it averages, despite the extra compute (Birdsnap *t* = 90% removed: SG-GRAD 65.02, SG-IG 64.54 against GRAD 56.08, IG 60.30 — high in absolute terms, but still inside the narrow band). Two ensembles do win, consistently and by a lot, where winning means *losing the most accuracy under removal*: ImageNet at *t* = 90% removed, SG-SQ-IG 29.41 and VAR-IG 25.25 against random 63.53; Birdsnap SG-SQ-GB 5.21 and VAR-GB 5.20 against random 38.12. The three base estimators sit in a strikingly narrow band — best-to-worst spread at 90% removed is 4.22 points on Birdsnap, 5.17 on Food 101, and 3.62 on ImageNet — so the method choice among them barely matters. Per-cell variance is small throughout (maximum on ImageNet 1.32% for SG-SQ-GRAD at 70% removed; 0.12% on Birdsnap; 1.52% on Food 101). Squaring alone (no ensembling) buys only a small gain over random (GB at *t* = 0.9 improved by 8.43 ± 0.97 points), which is why the authors conclude the gain comes from the *combination* of squaring and ensembling, not from either. KAR is a much weaker discriminator than ROAR: keeping the top-ranked pixels leaves all estimators in one narrow band (Birdsnap, 90% of pixels *kept*: random 59.92, Sobel 59.08, GRAD 62.49, IG 65.50, GB 62.44, SG-IG 66.35 — against the ROAR spread of 5.21 to 65.02 under the same threshold).

## Limits and open problems they name

The evaluator is not the model that was explained: ROAR retrains from scratch, so the estimator's ranking is applied to a network whose weights differ from the one that produced it, and the authors argue the logic still holds by the redundancy reasoning but do not prove it. Models that gate features (decision stumps, L1-regularised feature selection) break the metric unless importance is recomputed after each retrain, which at this scale they do not do. Redundancy is an intrinsic blind spot: a correct estimator may show no degradation until a whole redundant set is removed, so the curve must be read across thresholds rather than at one point. They can only afford an image-classification grid, cannot include arbitrary estimators, and say outright that they do not understand *why* VarGrad and SG-SQ work — the behaviour that both "remove whole objects" is described as emergent and unnamed. Only one architecture, one family of estimators.

## What we take from it

1. **A prediction-under-intervention measure, and a warning about the cheap version of it.** The reusable instrument is: rank by the account, destroy the top-ranked part *in both train and test*, retrain, re-measure. Any evaluation of ours that instead ablates at inference time on the original model is measuring distribution shift as much as information — this paper puts a 63.53-versus-0.5 gap on that error, and the toy inverted ranking shows the naive test awarding a *wrong* explanation credit.
2. **A random floor is mandatory, not optional.** Every attribution claim in our write-ups gets a random-ranking row and a fixed input-side heuristic row (here Sobel), because three textbook methods with axiomatic guarantees failed to beat a coin flip. This is now the second floor-and-baseline rule the folder holds, arriving from the attribution side rather than the concept side.
3. **Narrow ranges are a finding, not a detail.** When a dozen differently-derived accounts of the same object all score within ~4 points and all sit at the floor, the instrument is telling us the accounts share a cause — the same lesson `arxiv-1810.03292` reaches by randomisation, arrived at here by an intervention and a retrain.
4. **Ensembling over noise is a real lever and the wrong one to copy blindly.** Squaring before averaging, or taking the variance of a noisy set, converts a floor-level account into a clearly-informative one, and the authors cannot say why. For us: an account whose power depends on an unexplained aggregation step is an account we cannot yet predict from, and the paper's honest label for it ("emergent behaviour") is the correct one to carry.
5. **Redundancy is the measurement's floor and also a claim about the model.** 10% of ImageNet pixels suffices for 63.53% — a fact about how distributed the network's information is, and the reason an account that names only one feature at a time can never be shown to be complete.

## Tags

unit: direction
form: direction
method: attribution
eval: ablation, prediction/simulation
goal: evaluation-of-methods
