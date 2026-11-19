# The (Un)reliability of saliency methods

- **Key:** arxiv-1711.00867
- **Authors / organisation:** Pieter-Jan Kindermans, Sara Hooker, Julius Adebayo, Dumitru Erhan, Been Kim — **Google Brain**; Maximilian Alber, Kristof T. Schütt, Sven Dähne — **TU Berlin**. Kindermans and Hooker share lead credit; the thanks note records the work as done in the Google Brain Residency.
- **Date:** 2017-11 (v1 2017-11-02)
- **Link:** https://arxiv.org/abs/1711.00867
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Saliency maps are sold as evidence about *why* a network decided, but the field has no ground truth for what a correct map is, so the authors look for a property that can be checked without ground truth. Their observation is that a map which changes when the input is changed in a way the model provably ignores is not reporting the model's sensitivity — it is reporting the input's position. They propose an axiom, *input invariance*: whatever transformation is applied to the input, the saliency method must mirror the model's own sensitivity to that transformation, so two models that compute the same function must receive the same attribution. The paper then argues this is a *precondition* for reliability rather than a nice-to-have, and demonstrates that a subclass of common methods fails it. The larger goal is a set of checkable axioms for explanations in the spirit of the completeness and implementation-invariance axioms of Integrated Gradients, but with the model's *behaviour* as the benchmark rather than the method's internal accounting.

## What they studied

A small MNIST classifier: a three-layer multilayer perceptron with 1024 ReLU units per layer, trained with mini-batch SGD for ten epochs to 98.3% test accuracy (the authors concede the gap to state of the art and argue it does not affect the findings). The unit of analysis is the **input-attribution vector** — one score per pixel for the predicted class — produced by raw gradients, Guided Backprop, PatternNet, gradient-times-input, Integrated Gradients with several reference points, Deep Taylor Decomposition with several root points, and SmoothGrad wrapped around each. A single visual sanity check on ImageNet with VGG16 is included for the reference-point argument. The object compared is not a model but a **(method, reference-point) pair**, evaluated across two networks that compute the same function.

## How they did it

The core construction is exact and cheap. Take network 1, trained on MNIST encoded into [0,1]. Build network 2 for a shifted encoding ([-1,0], i.e. a constant vector −1 added to every pixel) by subtracting the shift from the first-layer biases: the first-layer pre-activations and hence every later activation are algebraically unchanged, and so are all weights except those biases. The two networks therefore define the same function and produce the same prediction on corresponding inputs, and the **input gradient itself is identical** for both. For each saliency method, compute the map on x₁ for network 1 and on x₂ = x₁ + m for network 2, and compare the two maps. If the method is input invariant the maps are identical; any difference is attribution to the shift, which the network provably cannot see. They run this for every method in the list, then repeat with a more hostile shift — a rendered image of a checkered box added to every MNIST digit — and finally with a hand-drawn cat used as the shift. In the last experiment they also invert the construction: choose a target attribution, solve for the shift vector that would produce it under gradient-times-input, clip the shift so the digit stays visible, and check which methods display the planted attribution. For the reference-point methods they vary the baseline (zero vector, black image, LRP root, PatternAttribution root) and report the same comparison per baseline.

## What counts as an explanation here

A per-pixel attribution vector for the predicted class, meant to be read as *which inputs drove this decision*. The paper's contribution is not a new attribution but a **property that an attribution must satisfy** and a template for testing it: an explanation is admissible only if it is invariant to transformations the model is invariant to. It is deliberately partial — it says nothing about whether an invariant method is *correct*, only that a non-invariant one is demonstrably reporting something the model does not use. Predictively, the axiom buys one specific claim: given the transformation, you can state in advance whether a method's output will or will not move, and the paper does exactly that per method.

## How they judged success

The judgement is qualitative and per-comparison rather than statistical: are the two heatmaps identical or not, on the same digit, for the two algebraically equivalent networks. Controls are strong for so small a paper — the model itself, its weights and its prediction are the control, so there is no need for a random baseline; the null hypothesis (invariance holds) is checkable by construction rather than by sampling. The variants that matter are the *reference point* (four baselines) and the *transformation type* (mean shift, checkered box, cat), and results are reported separately for each cell. There is no dataset benchmark, no human rating and no ablation study; the authors state that they only show at least one transformation that breaks each failing method, not that these are exhaustive.

## What they found

Raw gradients, Guided Backprop and PatternNet are input invariant — all three are functions of the weights and the input through the weight product, so an identical-weight model gets an identical map. Gradient-times-input fails, because the shift is carried into the multiplication by x; the paper notes (citing SmoothGrad) that multiplying by the input also silently restricts attribution to inputs that are non-zero. Integrated Gradients is invariant with a black-image baseline taken as the per-dataset minimum pixel value (the baseline shifts with the data, so the difference x − x₀ is unchanged) but fails with a zero-vector baseline; Deep Taylor Decomposition is invariant with a PatternAttribution root but fails with an LRP root, since LRP is the zero-root case of DTD and the attribution then depends on x alone. The checkered-box shift defeats *all* Integrated Gradients baselines tried. SmoothGrad inherits whatever its underlying method does — it averts noise but cannot repair a non-invariant base. The cat attack is the paper's headline: by adding a hand-drawn cat as a constant vector, every method except PatternAttribution-rooted ones will draw the cat as the explanation for an MNIST digit. The remedy they offer is data normalisation against the transformation considered, with an explicit caveat that this is a case-by-case patch and not a systematic guarantee.

## Limits and open problems they name

The scope is one transformation family, one small MLP, one toy dataset, and a single VGG16 illustration — the authors say plainly that they show only that *at least one* transformation breaks the failing methods. Reference-point choice is left as an open research problem: the right baseline is domain- and task-dependent, and without a principled criterion nothing can be said about reliability in domains where a human cannot eyeball the map. That last gap is the one they press hardest: the cat would be caught by looking at an MNIST heatmap, and there is no analogous way to catch a planted or accidental misattribution in an audio or language model, which are exactly the settings where an explanation is most needed. No claim is made that an input-invariant method is a correct or complete explanation.

## What we take from it

- **A rejection test with no ground truth and no training cost.** Build the equivalent model by folding the transform into the first-layer bias, then compare the account on both models. This is the analytical sibling of `arxiv-1810.03292`'s randomisation test: there the model is *destroyed* and the account should move; here the model is *preserved* and the account must not. For a weight-derived account of ours (a decoder, a direction, a feature dictionary), the corresponding check is trivial to run and cheap to report: recompute on a re-parameterised but functionally identical model and require bit-level or near-bit-level agreement.
- **Separate the account from its baseline.** The paper's sharpest structural point is that IG and DTD are not one method each but a family indexed by a reference point, and the invariance verdict flips with the index. Any of our read-outs that involves a reference — a mean activation, a zero-ablation, a "neutral" input, a comparison class — is a family too, and the honest report names the reference and prints the verdict per reference choice rather than one aggregate.
- **A designed adversary is a report card, not an anecdote.** Inverting the map to construct an input that displays a chosen attribution turns "this method can be fooled" into "here is the input that fools it", which is falsifiable by a third party. The same construction applies to any explanation we publish: can someone construct an input on which our account is confidently wrong in a way a reader would accept?
- **The warning that outlives the paper.** Visual plausibility is the one check that could have caught the cat, so it is also the check that is unavailable in exactly the domains where attribution is needed. Our own entries that lean on rendered figures inherit this; see the SSIM-versus-signed-correlation finding in `arxiv-1810.03292`.

## Tags

unit: feature
form: direction
method: attribution
eval: ablation | faithfulness
goal: evaluation-of-methods
