# Certified Adversarial Robustness via Anisotropic Randomized Smoothing

- **Key:** arxiv-2207.05327
- **Authors / organisation:** Hanbin Hong, Yuan Hong; Department of Computer Science, Illinois Institute of Technology, Chicago
- **Date:** 2022-07-12
- **Link:** https://arxiv.org/abs/2207.05327
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Randomized smoothing certifies a classifier by adding noise to its input and measuring how often the prediction survives; the certificate is a radius in which the smoothed prediction provably cannot change. Every accepted theory of this in 2022 assumed one fixed noise distribution applied identically to every pixel of every image — isotropic Gaussian, isotropic Laplace, and so on. The authors argue this throws away a lever: pixels are not equally informative, images differ from one another, and a single global scale is therefore never the right scale for all of them. Their goal is a smoothing theory in which each dimension carries its own noise parameters, plus a small network that predicts those parameters per input so the choice is made automatically. This is a step toward certified robustness that is adaptive rather than a fixed hyperparameter, while keeping the "works for any base classifier" property that made smoothing popular.

## What they studied

Base classifiers are ResNet-110 on CIFAR-10 (3×32×32) and ResNet-50 on ImageNet (resized to 3×224×224); both are trained together with a small five-layer convolutional "Noise Generator". The unit of analysis is the noise distribution itself — a per-pixel mean vector and a per-pixel variance vector for each individual input — so the object being certified remains the input-to-output map, and no statement is made about the weights beyond what the base classifier was trained to do. A second noise family (anisotropic Laplace) is developed in an appendix-style section to show the construction is not Gaussian-specific.

## How they did it

They first restate the isotropic certificate (the top-class probability minus the runner-up, pushed through the inverse normal CDF, times the noise scale over two), then replace the single scalar scale with a diagonal covariance and a mean vector. The generalized theorem keeps the same probability condition but replaces the scalar scale with the smallest per-dimension standard deviation; the binary-case corollary replaces the two-probability difference with a single inverse-CDF term. A second theorem gives the Laplace/ℓ1 analogue, where the per-dimension scale enters the same way and the radius is the larger of two existing closed forms. Because the certificate only "sees" the minimum variance, the training loss is built around that minimum: a cross-entropy term on noisy inputs, a term that pins the minimum of the variance map to a chosen target, and a small-norm penalty on the mean map. The generator has a dense-block trunk with two output branches, each passed through a sigmoid so the predicted scales and means stay bounded; training is joint with the classifier over five noise samples per image. Certification reuses the standard Monte-Carlo two-stage sampling (a selection pass, then a large pass for the confidence bound) but draws each sample from the generator's per-input distribution. They then compare certified accuracy curves against three published methods at matched noise targets, and add an experiment in which the test image is first attacked and then certified.

## What counts as an explanation here

A theorem plus a learned component. The theorem is a closed-form radius that is a function of the minimum predicted per-dimension scale and two class probabilities — it is complete for the smoothed classifier and says nothing about the base network's internals. The learned part is a small convolutional map from image to noise parameters; it is not an explanation of the classifier at all, but a learned policy that makes the certificate's free parameters input-dependent. Nothing here predicts the base classifier's behaviour under intervention; the predictive content is entirely about how the smoothed prediction moves when the input is perturbed.

## How they judged success

Approximate certified test accuracy — the fraction of the test set whose smoothed prediction is correct and certified at a given radius — evaluated with α = 0.001, 100 selection samples and 100,000 certification samples, on the full CIFAR-10 test set and 500 randomly drawn ImageNet images. Baselines are the isotropic Gaussian smoothing of the original paper, an optimization-based smoothing method, and a data-dependent method that tunes variance per input. Hyperparameters are matched by setting their variance target equal to each baseline's fixed scale. A separate experiment applies a white-box PGD attack (ℓ∞ budget 16/255, ten iterations) to the image before certification and reports certified accuracy on attacked inputs alongside clean ones, plus the percentage loss between the two. They also report wall-clock per-certificate time.

## What they found

Certified accuracy is higher than the isotropic baseline at essentially every radius on both data sets, and the gap widens with radius: on CIFAR-10 the margin is a couple of points at radius 0.25 and grows to roughly thirty points at radius 1.5, and the curve stays non-zero out to radius 3.5 where the baseline reaches zero by 3.25. Compared with the two adaptive or optimization-based baselines the margin shrinks but persists — on CIFAR-10 at radius 0.75 the reported values are 61% against 44% and 37%, and on ImageNet at radius 2.5, 34% against 25%. At clean inputs all methods sit within a few points of each other, so the improvement is concentrated where the certificate is doing work. Under the pre-perturbation attack both methods degrade, and the degrade is similar in shape, but the adaptive method ends up certified on a larger fraction of attacked images (for example 14% against 5% at radius 3.5). Per-certificate runtime is essentially unchanged, because the generator is small relative to the classifier's forward pass.

## Limits and open problems they name

The Laplace extension is confined to an appendix-style discussion and is not evaluated; the paper states other noise families would follow the same recipe but demonstrates only two. The certificate's dependence on the minimum variance is stated as an observation, not priced — that is, no analysis of how much the minimum's position in the variance map constrains what the rest of the map can do. The pre-perturbation study uses one attack (PGD, one budget, ten iterations) and the paper does not report an adaptive attack against the generator itself, which is the obvious next step since the generator is a trained component that an adversary could in principle target. Nothing is said about the generator's behaviour on inputs far from the training distribution.

## What we take from it

A warning, chiefly. This is the third paper in this sub-head where the reported number is produced by a wrapper with free parameters, and it makes the deepest version of the point so far: the certificate does not depend on the *mean* of the noise at all, so everything the learned generator does with the mean map is invisible to the guarantee and visible only through the two measured probabilities. That is a clean worked example of a component that improves a headline number while the theorem is structurally blind to it — exactly the blind-spot shape that the earlier floating-point bias result exploited deliberately. Two reusable method notes: (1) when a guarantee depends on one order statistic of a per-dimension parameter vector (here the minimum), a learned parameter map is being trained against a loss that only constrains that statistic, so the rest of the vector is unconstrained by the proof and can be moved freely; (2) matching a learned method to a fixed-scale baseline by setting the learned quantity's target equal to the baseline's scalar is a good habit for our own comparisons, because it makes the axis shared. Also worth keeping: the pre-perturbation experiment is a cheap falsifier of the *framing* of a certificate — if certifying a perturbed input certifies the perturbed label, the guarantee's protective value under an adversary who acts before certification is a separate, measurable thing.

## Tags

unit: whole-model
form: surrogate-model | direction
method: other
eval: benchmark | prediction/simulation
goal: safety/auditing | evaluation-of-methods
