# Adaptive Randomized Smoothing: Certifying Multi-Step Defences against Adversarial Examples

- **Key:** arxiv-2406.10427
- **Authors / organisation:** Saiyue Lyu, Shadab Shaikh, Frederick Shpilevskiy, Evan Shelhamer, Mathias Lécuyer; University of British Columbia and Google DeepMind (the arXiv listing titles the paper "Adaptive Randomized Smoothing: Certified Adversarial Robustness for Multi-Step Defences"; the PDF/HTML body prints "Certifying Multi-Step Defences against Adversarial Examples", and the bylines are printed with the e-mail block, which is why the first author's affiliation is not separated from the rest)
- **Date:** 2024-06-14
- **Link:** https://arxiv.org/abs/2406.10427
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Randomized smoothing gives a classifier a provable radius by averaging predictions over noisy copies of the input, but the noise is fixed ahead of time: one scale for every pixel of every image. Several groups have tried to make the noise depend on the test input, and the authors' framing is that these attempts fail in one of two ways — either the variance is adapted per input with an analysis that does not account for the adaptation, so the guarantee is unsound, or the adaptation is sound but low-dimensional (a single scale chosen from the distance to the training set), so it buys little. Their goal is a sound certificate for a *sequence* of randomized steps in which each step may look at the output of the previous one and choose its own noise, including a step that outputs a full per-pixel mask. The larger aim is to make certified defences allowed to compute adaptively at test time, which is what unproven heuristic defences do, while keeping the guarantee end-to-end.

## What they studied

Base classifiers are ResNet-110 on CIFAR-10 and ResNet-50 on CelebA and ImageNet; the adaptive component is a small U-Net that outputs one weight in [0,1] per pixel. The unit of analysis is the randomized mechanism and its composition — the input-to-label pipeline as a whole, including the learned mask — with no claim about any individual weight or neuron. The object certified is the smoothed classifier's prediction under an ℓ∞ threat model.

## How they did it

They first re-derive the standard Gaussian-smoothing radius from f-differential privacy, noting that the tight hypothesis-testing analysis of earlier work is expressible as a Gaussian mechanism's privacy profile; a proposition turns any f-DP mechanism's trade-off function into a robustness statement, and the standard radius falls out as a corollary. The extension is an adaptive-composition theorem: if each of k Gaussian steps is a privacy mechanism at its own noise scale, the composition has a single equivalent scale, and the certified radius is the standard expression evaluated at that equivalent scale — so, in their words, adaptivity over multi-step computation costs nothing in radius. They then build a concrete two-step instance for ℓ∞: the first step adds Gaussian noise and a U-Net maps the noisy image to a mask; the second step multiplies the *clean* input by that mask, re-noises, and the two noisy branches are combined by a per-pixel minimum-variance weighted average before a classifier predicts. The mask's norm enters the second step's noise scale, which is the mechanism by which masking reduces the effective dimension and hence the noise. One noise budget is split between the two steps by the composition rule, at a fixed split of equal scales inflated by root two. Everything except the mask (which is post-processing of the first noisy draw) is trained end-to-end. They evaluate standard and certified accuracy against the static smoothing baseline, a static learned-mask baseline (the same architecture with the adaptive step removed), a learned-noise-distribution baseline, and the one prior sound input-dependent method, using three seeds and reporting means and deviations.

## What counts as an explanation here

A theorem plus a learned mask. The theorem is a closed-form radius in the composed noise scale and two measured class probabilities; it is complete for the smoothed pipeline and says nothing about the internals of either network. The mask is a learned, input-dependent policy whose effect on the radius is mediated entirely by its ℓ2 norm, which sets the second step's noise — so the explanation of any particular certified number is "the mask was sparse on this input, therefore less noise on the second pass", a functional account, not a mechanistic one. Nothing predicts what the base classifier does under an intervention on its weights.

## How they judged success

Certified accuracy at radius r (fraction of test inputs that are correctly classified with a certificate at least that large), standard accuracy at radius zero, both averaged over three seeds with standard deviations printed in the tables; Monte-Carlo certification follows the standard sampling procedure. Comparisons are against the canonical static method, a static-mask ablation of their own architecture, a train-time noise-distribution method with its main hyperparameter grid-searched, and the prior per-input variance method run with the authors' code. Two bespoke benchmarks are constructed so that adaptivity should matter: CIFAR-10 images pasted onto larger distractor backgrounds at several sizes and noise levels, and unaligned, randomly cropped CelebA faces for a mouth-attribute task. On ImageNet they add a random-mask control (an untrained mask network) and a variant where only the mask is trained on a frozen baseline classifier. They report certification wall-clock per input, and print all training hyperparameters in an appendix.

## What they found

On the distractor-background benchmark the adaptive method is the best of the compared methods on most settings; at the largest size and a middle noise level it beats the best baseline by 4.9 points of standard accuracy, and the gap grows with input dimension while its own accuracy stays roughly flat — at the smallest size the static mask is ahead by 0.5 points, at a middle size the adaptive one is ahead by 1.6, at the largest by 4.9. At the highest noise level the adaptive method is no better or slightly worse, because the first noisy draw is too degraded for the mask to be useful. On unaligned CelebA the mask is reported to become sparse and input-specific where the static mask stays near the identity, and at a middle noise level certified accuracy at a radius of 0.005 is reported to rise from 40.7 to 71.3 percent, with a 3.6-point standard-accuracy gain. On ImageNet with a middle noise level, training only the mask on a frozen baseline gives a one-point standard-accuracy gain, end-to-end training gives three points, and the adaptive models certify a slightly larger maximum radius where the baselines certify none; the untrained random mask lands well below, at 49 percent. Certification of one input takes roughly twice as long as the static method.

## Limits and open problems they name

The authors name computation as the cost: a bigger model, a second noised pass, and Monte-Carlo estimation over the whole pipeline, for training and for each certification. They do not report the radius curves numerically in the text (the curves are figures) and every accuracy comparison is standard or certified accuracy at a shared radius, not an area under the curve. They flag that combining with adversarial training, consistency regularisation, higher-order certification, double sampling and diffusion denoising is untested, and they describe extending to fully adaptive variance schedules governed by privacy accounting as future work. A reader can also note what the tables show but the text does not stress: at high noise the adaptive method's advantage disappears or reverses, and no experiment isolates the contribution of adaptivity from the contribution of simply having a trained mask model, since the static-mask baseline shares the architecture but not the training signal.

## What we take from it

- **Framing (the main import):** composition of adaptivity is a *bookkeeping* problem, and the bookkeeping was already solved in another field. A multi-step randomized pipeline can take its per-step noise as a budget split, and the certificate is the one-step formula evaluated at the harmonically combined scale. For us: when our account wants a model that reads its own intermediate state and then does something else, the sound way to bound what the whole chain can hide is to bound each step and compose — and the composed budget, not the step, is what a per-example diagnostic should be reported against.
- **Measure:** the mask's ℓ2 norm as the bridge between a learned, high-dimensional object and the scalar the theorem consumes. The entire benefit of the learned component flows through one norm, which is why the mask can be sparsity-seeking yet fully sound — and it is also why no ablation of the mask's *direction* exists. Warning for us: when a learned component's effect on a guarantee is a single scalar function of its output, report the component's effect at matched values of that scalar, or a reader cannot tell whether the direction is doing anything.
- **Warning:** "no cost to adaptivity" is a comparison between two *instances of the same formula* — the adaptive pipeline against a hypothetical single-step pipeline with the harmonically combined noise — not against the measured baseline in the tables. The real gains in the tables come from the mask's sparsity, i.e. from the second step getting less noise on informative pixels, which is a statement about dimension, not about adaptivity. Report the quantity that actually moved.
- **Method:** the distractor-background benchmark is a reusable construction — take a standard task, pad it with task-irrelevant pixels, and sweep the padding so that input dimension becomes an experimental knob while the task is held fixed. That is a cheap way to test whether a mechanism is paying the dimension penalty its theory predicts, and it is the strongest exhibit in the paper.

## Tags

unit: whole-model
form: causal-abstraction
method: decomposition
eval: prediction/simulation
goal: control/editing
