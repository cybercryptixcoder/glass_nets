# Learning how to explain neural networks: PatternNet and PatternAttribution

- **Key:** arxiv-1705.05598
- **Authors / organisation:** Pieter-Jan Kindermans (Google Brain; part at TU Berlin, part during the Google Brain Residency), Kristof T. Schütt, Maximilian Alber, Klaus-Robert Müller, Sven Dähne (TU Berlin; Müller also Korea University and Max Planck Institute for Informatics), Dumitru Erhan, Been Kim (Google Brain)
- **Date:** 2017-10 (v2 2017-10-24; v1 2017-05-16)
- **Link:** https://arxiv.org/abs/1705.05598
- **Kind:** paper
- **Read from:** full text — v2 HTML, the camera-ready-length version; v1 HTML was also fetched and is the shorter five-author early draft, not read as the source
- **Relevance to us:** 3 (central)

## What they wanted, and why

The authors' complaint is that a family of widely used explanation methods — DeConvNet, Guided BackProp, LRP and its Deep Taylor Decomposition framing — were introduced on visual grounds and never check out on the easiest possible model. They take the network everyone treats as trivially analysable, a linear regression, and show that these methods produce, by construction, an answer that is not the answer. The larger goal is to give explanation a *quality criterion* that can be evaluated, so that a method can be scored offline instead of being judged by how satisfying its pictures look. From the criterion they then derive two methods they claim are sound in the linear limit and better on a real network. The framing that carries the paper is a vocabulary split between the **filter** (the weights, which tell you how to extract the output from data) and the **pattern** (the direction in the data along which the output actually varies).

## What they studied

Unit of analysis: individual neurons, each treated as a linear filter followed optionally by its ReLU, with max-pooling and softmax allowed; the full network explanation is the superposition of per-neuron explanations. Models: a seven-dimensional toy linear regression (signal on one axis, a distractor along the (1,1) direction, Gaussian noise) and pretrained VGG-16 on ImageNet at 224×224, described as five blocks of two or three convolutions plus a max-pool each, followed by three dense layers. Data: the ImageNet training set split in half — the first half to fit the signal estimators, the second half to fit the scoring vector — with all reported results on the untouched 50,000-image validation set. No language models anywhere.

## How they did it

1. Write the toy data as signal plus distractor, then solve for the optimal linear filter. The result is the paper's lever: the weight vector is orthogonal to the distractor, so its *direction* is set by the distractor and is not the signal direction.
2. Observe that this makes signal estimation ill-posed: any vector scaled to give unit output is a valid "explanation", so an infinite number of back-projections and root points satisfy the bookkeeping constraint.
3. Fix a quality criterion for a signal estimator: fit the best linear predictor of the neuron's output from the *residual* the estimator leaves behind, take its correlation with the output, and score the estimator as one minus that correlation (higher is better). It measures how much of the output is still recoverable from what you said was distractor.
4. Derive two estimators as closed forms from setting that covariance to zero: a linear one where the pattern is the covariance of input with output over the output variance, and a two-component one that fits separate patterns for the firing and non-firing regimes of the ReLU, so the gate's sign is not retained in the distractor.
5. Reframe LRP/DTD as special cases: assuming the whole input is signal gives the z-rule, and assuming the signal lies along the weights gives the w²-rule. PatternAttribution is then described as a learned root point, i.e. a learned estimate of the distractor.
6. Build the explanations as ordinary backward passes in which the network's weights are replaced by the estimated patterns (PatternNet) or by the pattern-weighted weights (PatternAttribution).
7. Score every estimator with the criterion on all of VGG-16, against a random-direction baseline; then run an image-degradation test on the whole classifier (sort 9×9 patches by summed attribution, replace the top n with the per-channel mean, n up to 100, measure confidence drop), and inspect the pictures.

## What counts as an explanation here

Two artefacts, both single-example and partial: a back-projection into input space in the original colour channels (PatternNet: what the neuron's signal looks like), and a pixel-wise heat map of contributions to the score (PatternAttribution: how much each part of the estimated signal counts). Both are neuron-local and additive, with no claim about interactions between units, no claim about the circuit, and no claim that the set of named parts is complete. The one predictive claim is behavioural: an estimator of higher quality should be better at telling you which patches the classifier actually uses, and that is tested by corrupting pixels, not by inspecting the map.

## How they judged success

Three arms, and it is worth noting what is and is not a control. (i) The correlation criterion on held-out data, per neuron, with the random-direction estimator as its floor. (ii) Image degradation on 50,000 validation images — steeper confidence decay is better — with random patch order as the floor and the trivial "the whole input is the signal" estimator as the do-nothing reference; note this test rewards an attribution for pointing at anything the classifier uses, so it is a test of the heat map, not of the signal theory. (iii) Qualitative inspection of single images and six random validation images, normalized for contrast. There are no human ratings, no downstream-task arm, and no ablation of the criterion itself.

## What they found

- In the toy model the trained filter points at (1,−1) while the signal direction is (1,0): the weights are orthogonal to the distractor and orthogonal-ish to nothing else. The learned linear pattern recovers the true signal direction; the filter-based pattern cannot by construction.
- On VGG-16, the random-direction estimator removes almost no correlation. The filter-based estimator (the implicit assumption of DeConvNet and Guided BackProp) removes some in the first layer — filters there resemble patterns — and much less in higher layers, ending "not much better than the random estimator". Both learned estimators remove far more across the board; the linear one and the two-component one are comparable in convolutional layers, with one exception, and the two-component one is best in the dense layers.
- In image degradation the two-component estimator wins by a large margin: random patch order is worst, then the linear estimator, then the filter-based one, then the trivial whole-input estimator slightly above them, then the two-component estimator clearly steepest. The ordering of the linear and filter arms here is the reverse of what the correlation arm mostly showed, which the authors do not discuss.
- Qualitatively: the trivial estimator's heat map is noisy because distractor contributions are multiplied in and only cancel in the forward pass; the two-component estimator produces crisp maps that recover the image structure, and PatternNet recovers a signal close to the original input without needing extra rectifiers in the backward pass, unlike DeConvNet and Guided BackProp.
- Cost: fitting the estimators takes 3–4 hours on four GPUs (comparable to a fraction of training the network), scoring them takes about 24 hours per estimator on a Tesla K40, and once fitted a single explanation is one backward pass, so it is at least as fast as real-time saliency work.

## Limits and open problems they name

The paper's own scope statement is restrictive: only linear neurons optionally followed by ReLU, max-pooling or softmax, networks treated without the final softmax layer, and every neuron analysed independently so that the whole explanation is a superposition. The distractor regime argument is supported empirically on one architecture and one dataset. The authors concede that estimating the signal from the constraint alone is ill-posed and that the covariance criterion still admits many solutions — they present two of them as choices rather than deriving a unique one. There is no analysis of what happens when the pattern estimate is wrong in a structured way, and no statement about how a superposition of per-neuron explanations could ever be additive at the output level, which is the assumption the attribution rests on.

## What we take from it

- **Framing, and the strongest one in the folder: weights are a filter, not a pattern, and the gap between them is the distractor.** A weight direction can be tilted arbitrarily by correlated nuisance structure in the data while the function stays identical, so no amount of reading weights yields the signal direction — that has to be estimated from data with the output as target. This is the constructive complement to `arxiv-1711.00867`'s checkered-box attack, which is the same statement obtained by attacking the input instead of analysing the estimator; the same authors wrote both.
- **A measure we can reuse directly: the residual-recoverability score.** Fit the best linear predictor of the output from what your account called unexplained and report one minus its correlation. It needs only held-out data and a linear regression, it has a natural floor (random directions), and it turns "the rest is noise" into a number. It is the per-neuron version of the unexplained-effect share that `arxiv-2403.19647` prices in effect units.
- **A design rule that costs one forward pass: an estimator whose fit is linear in the output is only right when signal and distractor are orthogonal, and a ReLU forces the fit to be piecewise.** Splitting the fit by the gate's sign is a one-line change that moved the degradation curve further than any other choice here. For us, any read-out fitted globally over a gated unit should be fitted per regime, and the per-regime split is itself a testable prediction about where the account's error concentrates.
- **A warning about the second arm.** The image-degradation test rewards an attribution for pointing at anything the classifier's answer depends on, including the distractor; combined with the fact that the trivial "everything is signal" estimator beats two of the three learned ones there, this is the folder's clearest case of an evaluation whose floor and whose target are not the same thing.
- **A limit on prediction.** Nothing here predicts a behaviour; the account is a per-neuron picture plus a score. It buys a criterion, not a model.

## Tags

unit: neuron | direction | weights
form: direction | description | replacement-model
method: weight-analysis | decomposition | other (learned signal estimator, fitted closed-form from data)
eval: faithfulness | ablation | benchmark (image degradation) | human-rating (qualitative only)
goal: understanding | evaluation-of-methods
