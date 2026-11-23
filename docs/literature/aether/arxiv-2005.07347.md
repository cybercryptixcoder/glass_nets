# Towards Assessment of Randomized Smoothing Mechanisms for Certifying Adversarial Robustness

- **Key:** arxiv-2005.07347
- **Authors / organisation:** Tianhang Zheng, Di Wang, Baochun Li, Jinhui Xu; State University of New York at Buffalo and University of Toronto. (The first two authors are marked as equal contributors; one author's address is printed as State University of New York at Buffalo.)
- **Date:** 2020-05-15
- **Link:** https://arxiv.org/abs/2005.07347
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

By 2020 randomized smoothing had become the scalable way to certify ℓ2 robustness, and the field had settled on adding isotropic Gaussian noise to the input without ever asking whether Gaussian noise was the right choice. The authors think that is a gap: every smoothing pipeline has a noise *magnitude* setting, that setting is a free parameter chosen by the practitioner, and choosing it too large destroys clean accuracy while too small a value certifies nothing useful. The paper therefore sets out to answer two questions — is the Gaussian mechanism an appropriate instrument for certifying ℓ2 robustness, or is some other distribution better, and what mechanism, if any, can certify ℓ∞ robustness in high dimension. The larger goal is a *comparative* theory of smoothing instruments rather than more instruments.

## What they studied

No network is analysed. The unit of analysis is the **noise distribution** `M(x) = x + z` added to the input, plus the resulting classification distribution of `f(M(x))` for an arbitrary deterministic `f`. Two `f` are used in experiments — a 110-layer residual network on CIFAR-10 and a ResNet-50 on ImageNet — but only as the thing being wrapped; the measurements are of the smoothing wrapper. Data dimensionality `d` is the only property of the problem that the theory depends on (3072 for CIFAR-10, 150528 for ImageNet as printed in the paper; one figure caption prints 150568, a typographical difference the entry does not rely on).

## How they did it

The paper builds a generic framework that subsumes the two existing formalisms (the differential-privacy route and the Rényi-divergence route). It defines a robustness notion using the maximum over the Rényi order of `D_α/α`, so a mechanism is called robust if that quantity stays below `ε` for every point inside an ℓp ball; a postprocessing corollary then lets the deterministic classifier be arbitrary, which is what makes the framework network-agnostic. It proves that robustness under this notion implies both of the earlier notions with the same noise, and that the ℓ2 radius reproduces the Rényi-divergence bound exactly — so the framework is a relaxed container for the prior results rather than a competitor to them.

The actual assessment machinery is a metric plus a criterion. The **metric** is the expected magnitude of the noise a mechanism must add, measured as the expected ℓ∞ norm of `z`, in order to certify a given radius at a given `ε`; smaller is better, because noise costs accuracy. The **criterion** is a lower bound on that quantity holding for *any* additive-noise mechanism whatsoever, derived by importing a one-way-marginal mean-estimation lower bound from the differential-privacy literature: the authors exhibit a synthetic grid of `2^d` points from `{0, r/√d}^d`, note that all pairs lie inside the ball so the mechanism must be private across the whole grid, and then apply the DP lower bound at sample size one. Two mechanisms are then scored against the criteria — the Gaussian one for both norms, and an "Exponential mechanism" (a product-Laplace-like density with the ℓ∞ norm in the exponent) for ℓ∞, which they sample by drawing a Gamma scalar and then a uniform point in a cube, a recipe they take from prior work rather than invent. Experiments compare the certified radii at 50% certified accuracy against the predicted scaling, using the printed certification algorithm with `n` Monte Carlo samples.

## What counts as an explanation here

Not an explanation of any model. The paper's product is a **criterion**: an asymptotic lower bound on the noise any mechanism needs, stated in terms of `r`, `ε` and `d`, against which a mechanism's own noise requirement can be compared. The explanatory claim is comparative and partial — "the Gaussian mechanism is within a factor of `√(log d)` of optimal, the Exponential mechanism is within a factor of `√d`" — and it is deliberately asymptotic, so it says nothing about constants or about any particular network. It cannot predict a certified radius for a real model; it can rank instruments.

## How they judged success

Success is (i) the framework's containment theorems — same noise certifies PixelDP and the Rényi bound, with the exact-radius match for ℓ2 — and (ii) agreement between the *scaling* of the derived bounds and the measured radii. Empirically, the test is the certified radius at which each mechanism holds roughly 50% certified accuracy on CIFAR-10 and ImageNet, compared against the predicted `O(1/√(log d))` (Gaussian, ℓ2), `O(1/√(d log d))` (Gaussian, ℓ∞) and `O(1/d)` (Exponential, ℓ∞) forms. No baseline mechanism other than these two is run, and the criterion itself is never evaluated numerically against a candidate.

## What they found

The Gaussian mechanism needs `O(r√(log d)/√ε)` noise while the universal lower bound for ℓ2 is `Ω(r/√ε)`, so the gap is at most `√(log d)` — about 2.83 for CIFAR-10's dimensionality and about 3.45 for ImageNet's, which the authors use as the whole justification for calling Gaussian "appropriate." The Exponential mechanism's expected noise is `dσ`, i.e. it grows linearly in dimension, while the ℓ∞ criterion is `Ω(r√d/√ε)`, leaving a gap of `√d`. The paper's surprising result is that the *Gaussian* mechanism does an adequate job of certifying **ℓ∞** robustness too: with variance scaled as `dr²/2ε` its expected ℓ∞ noise is `O(r√(d log d)/√ε)`, again within `√(log d)` of the criterion. Empirically the three regimes separate as predicted: at 50% certified accuracy the Gaussian mechanism reaches an ℓ2 radius of 0.34 on CIFAR-10 and 0.29 on ImageNet; an ℓ∞ radius of 9e-3 and 1.4e-3 respectively; and the Exponential mechanism reaches ℓ∞ radii of only 3e-4 and 7e-6.

## Limits and open problems they name

The paper states outright that the lower bounds are not experimentally verifiable, because it is not known whether any mechanism attains them — so the entire assessment rests on a comparison to a floor nobody can reach, and the "gap" of `√(log d)` is a worst-case statement about a hypothetical optimum. Everything is asymptotic; no finite-sample or constant-factor version is given. The framework is stated to require additive noise, which excludes the Neyman–Pearson-optimal isotropic proof, and hence excludes the sharpest known ℓ2 certificates; the authors acknowledge their container is more relaxed than PixelDP in the opposite direction too, holding only at δ = 0. Experiments verify only the two mechanisms' scaling, not the framework's own predictions at finite `n`. The 50%-accuracy point is chosen as the comparison anchor with the concession that other anchors in a 40-60% window would not change the (asymptotic) readings.

## What we take from it

Three things. **A framing**: when a measurement pipeline has a free instrument parameter, the right question is not "what number does the instrument give" but "how far is this instrument from the best possible one", and that requires a lower bound the paper is willing to derive from an outside field (here, DP mean estimation). **A transferable measure**: expected magnitude of the intervention, with the cost of the intervention printed as the quantity being minimised — this is the cleanest instance in the folder of pricing the instrument rather than the model. **A warning**: the paper's headline "Gaussian is appropriate" is the statement "Gaussian is within a constant of a bound no one knows is achievable", and the constant is roughly `√(log d)`, i.e. a factor of three, which in a field where improvements are routinely announced in the third decimal place is a very loose notion of "appropriate". For us: any probe family we propose should be accompanied by a lower bound on what *any* probe of that form must spend, so that our method's advantage is measured against a floor rather than against a prior method.

## Tags

unit: whole-model
form: description
method: other
eval: benchmark
goal: evaluation-of-methods
