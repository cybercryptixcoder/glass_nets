# Certified Adversarial Robustness via Randomized Smoothing

- **Key:** arxiv-1902.02918
- **Authors / organisation:** Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter; Carnegie Mellon University, with Kolter also at the Bosch Center for AI
- **Date:** 2019-06-15 (v2; first posted 2019-02-08)
- **Link:** https://arxiv.org/abs/1902.02918
- **Kind:** paper
- **Read from:** full text (arXiv HTML v2, all sections and Appendices A-K)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Certified defenses at the time were either exact verifiers — which could not scale past networks of modest activation count — or conservative bounds that were sound but loose on expressive models. Both routes needed the network itself to be analysable. The authors wanted a certificate that comes out of the *distribution of a network's outputs under noise* rather than out of its weights, so that an arbitrarily large and unconstrained network could still be certified. Their contribution is a tightness proof: earlier noise-based guarantees (from differential privacy and from information theory) were valid but loose, and this paper claims the exact worst-case radius implied by the observed class probabilities alone. The larger goal is stated plainly in the conclusion: reduce robust classification to ordinary supervised learning on noisy inputs.

## What they studied

The object is a *smoothed classifier* `g`, not a network: `g(x)` returns the class the base classifier `f` most probably assigns when the input is corrupted by isotropic Gaussian noise of scale σ, a fixed hyperparameter. The unit of analysis is therefore a probability distribution over class labels at a point, summarised by the top-class probability `p_A` and the runner-up probability `p_B`. The base classifiers are a 110-layer residual network (CIFAR-10), a ResNet-50 from torchvision (ImageNet), and a 20-layer residual network for the baseline comparisons on CIFAR-10 and SVHN. No assumption about `f` is used anywhere in the guarantee.

## How they did it

The certificate is proved in two directions. First, a lower bound: given that `f` classifies the Gaussian ball around `x` as class `c_A` with probability at least `p_A` and as any other class with probability at most `p_B`, the smoothed classifier's prediction is constant inside an ℓ₂ ball of radius `(σ/2)(Φ⁻¹(p_A) − Φ⁻¹(p_B))`, where `Φ⁻¹` is the inverse Gaussian CDF. The proof identifies the adversarial base classifier that minimises the mass on `c_A` at a shifted point, shows via a Neyman–Pearson-type argument that it is a linear classifier whose boundary is perpendicular to the perturbation, and solves for the shift at which its mass crosses one half. Second, a matching upper bound: for any perturbation beyond that radius there exists a base classifier consistent with the same class probabilities whose smoothed prediction flips, so no superset of the ball is certifiable in general. Because the probabilities cannot be computed exactly for a neural base classifier, two Monte Carlo procedures are given — one that predicts by majority vote with a binomial test deciding whether to abstain, and one that certifies by taking a one-sided Clopper–Pearson-style lower confidence bound on `p_A` (with `p_B` set to one minus it) and converting it to a radius. Base classifiers are trained with Gaussian data augmentation at the same σ, justified in an appendix by Jensen's inequality relating augmented cross-entropy minimisation to maximising the log probability of correct noisy classification.

## What counts as an explanation here

Not an account of the network. The "explanation" is a per-input certificate: a radius, together with the class probabilities that produced it, and two theorems showing that radius cannot be improved given that information. It is deliberately partial — it predicts nothing about the network's internals, only that the smoothed decision cannot change inside the ball — and it is exact in the sense that the bound is attained by some base classifier, while being necessarily conservative for any particular one.

## How they judged success

Three measures: certified test-set accuracy at radius `r` (fraction of the set both correctly classified and certified at radius ≥ r, under failure probability α = 0.001 and 100,000 Monte Carlo samples); standard accuracy of the smoothed classifier; and a direct empirical probe of tightness, running a PGD-style attack against the smoothed classifier at 1.5× and 2× the certified radius. Comparisons are against the released models of three certified ℓ₂ baselines (a duality-based convex method, a Lipschitz-based method, and a layer-wise bound propagation method), each evaluated with its authors' own code and preprocessing. Controls include varying the number of samples, varying α, and varying the training-noise level independently of the prediction-noise level.

## What they found

On ImageNet with a ResNet-50 base classifier, certified top-1 accuracy at ℓ₂ radius 0.5 is 49% (standard accuracy 67%); at radius 1.0 it is 37% with σ = 0.5; at radius 3.0 it is 12% with σ = 1.0. On CIFAR-10 the best row is σ = 0.25 at 77% standard and 61% certified at radius 0.25, dropping to zero by radius 1.0, while σ = 1.00 holds 14% out to radius 1.5. Certified accuracy falls gradually with radius and then hits a hard cliff, which the authors attribute to the finite sample budget: radius grows only like `σ Φ⁻¹(α^{1/n})` even in the best case where every sample agrees, so certifying 4σ at 99.9% confidence would take on the order of 10⁵ samples. Against the strongest baseline (the convex dual method on the same small ResNet), smoothing wins at all but the smallest radii; with a larger base classifier it wins everywhere, which the authors attribute to the baseline's networks becoming over-regularised so that filters go exactly zero. Prediction (as opposed to certification) is cheap — 0.15 s per image with 100 samples at 65% top-1, abstaining 12% of the time, falling to 1% abstention with 10,000 samples. On tightness: an attack found a class-changing point within 1.5× the certified radius for 17% of certified ImageNet examples and within 2× for 53%.

## Limits and open problems they name

The certificate is a high-probability statement about a smoothed object that cannot be evaluated exactly, so every number in the paper is an *approximate* certified accuracy; the authors compute the honest lower bound in an appendix and show the gap is visually negligible, but they do not report it as the headline. Abstention is real: the prediction procedure can be pushed into abstaining at a high rate by a carefully placed perturbation, which the authors note would need a variant theorem certifying a margin rather than a half-space. Their own attack is "PGD-style" and the paper says outright that effective attacks on smoothed classifiers are out of scope, so the 17%/53% tightness numbers are a floor on looseness, not a measurement of it. σ is a single global trade-off dial, tuned per radius; training was Gaussian augmentation at matching σ, and an appendix shows mismatched training noise hurts, with too little noise worse than too much — but the authors concede the training scheme is probably improvable, e.g. by optimising certified accuracy at a chosen radius directly.

## What we take from it

A framing and a warning. The framing: a guarantee can be bought by changing the *object* — smoothing moves the certificate off the weights and onto an output distribution, which is why it scales where weight-based analysis does not, and it also changes what the number means (a probabilistic statement about a noisy ensemble, not a deterministic fact about the network). The warning is the one our thread keeps finding: the reported quantity is "approximate certified accuracy", and the exact-but-lower quantity sits in an appendix with a figure showing the two lines nearly overlapping. Relatedly, the tightness probe reports only the fraction of certified examples an attack could break at 1.5× and 2× the radius — a two-point sample of the looseness distribution, not the distribution, and the authors say a true attack is out of scope. For a predictive white-box account this is the cleanest available demonstration that an output-side certificate and an input-side one are different kinds of object, and that "tight" is a claim about a worst case, not about the average case the numbers are computed on.

## Tags

unit: whole-model
form: causal-abstraction
method: other
eval: ablation
goal: safety/auditing
