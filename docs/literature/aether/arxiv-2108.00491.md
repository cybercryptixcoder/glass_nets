# Certified Defense via Latent Space Randomized Smoothing with Orthogonal Encoders

- **Key:** arxiv-2108.00491
- **Authors / organisation:** Huimin Zeng (Technical University of Munich); Jiahao Su, Furong Huang (University of Maryland, College Park)
- **Date:** 2021-08-01
- **Link:** https://arxiv.org/abs/2108.00491
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Randomized smoothing is one of the few certified defences that scales to large images, but the price is paid at test time: to certify a single image you have to push many thousand noisy copies through the whole network. The authors ask whether the same guarantee can be obtained by injecting noise somewhere inside the network instead of at the input, so that the noisy copies only travel through the last part of the model. If that works, certification becomes cheaper in proportion to how much of the network the noise skips. The larger goal is a certifiable region for the *input* space that is as tight as the ordinary input-space certificate but computed in a much smaller, cheaper place.

## What they studied

Image classifiers on CIFAR-10 and ImageNet: a ResNet18 (modified) and a WideResNet34 with widen factor two. The unit of analysis is neither a neuron nor a feature but a *split point* — a network cut into an encoder `fe` and a classifier `fc`, with the smoothing applied to the latent tensor `z = fe(x)` that crosses the cut. The dimensionality of that latent tensor (and hence how deep the cut is) is the object under study, controlled by how many of the network's skip connections are replaced with orthogonal ones.

## How they did it

They take the standard smoothed classifier — Gaussian noise at the input, majority vote over noisy copies, radius from the gap between the inverse Gaussian CDFs of the top-two class probabilities — and move the noise to the latent space, giving a "partially smoothed" classifier `argmax P(fc(z + ξ) = y)`. That construction only certifies `fc` against latent perturbations, which is useless on its own because attackers perturb pixels. To map the latent radius back to the input they bound `fe`'s Lipschitz constant, and rather than estimating it (expensive, and loose when singular values are unevenly spread) they build it in: the encoder is composed of orthogonal convolutions, orthogonal fully connected layers, and GroupSort activations, each with Lipschitz constant exactly 1. Orthogonality is enforced by parameterisation — the Cayley transform of skew-symmetric convolutions, evaluated in the Fourier domain — not by a penalty, so it holds exactly rather than approximately. Residual blocks are made norm-preserving with Trockman–Kolter's convex combination of main branch and skip: `α f_main(X) + (1−α) X`. Theorem 1 then says an `L`-Lipschitz encoder yields a certified input-space radius of `Rz/L`; with the orthogonal design `L = 1` and the input radius is the latent radius itself. They argue (Section 3.2) that rescaling the Lipschitz constant buys nothing, since any `L`-Lipschitz encoder factors into a scale-`L` map and an orthogonal one, so the orthogonal choice is representative rather than restrictive. Experiments compare against input-space smoothing at the same three noise levels (σ = 0.25, 0.50, 1.0) plus an undefended column, reporting average certified radius, clean accuracy, and seconds per certified example, with an ablation over how much of the network is orthogonalised.

## What counts as an explanation here

Not a mechanistic explanation of the network — a *proof certificate* about it. The form is a theorem plus a construction: a stated radius inside which every perturbation is guaranteed to leave the label unchanged, derived by pushing a latent-space certificate through a Lipschitz bound that is a fact about the architecture rather than about any learned quantity. It is partial by nature: it constrains the model's behaviour only within a ball, says nothing about the rest of input space, and the ball is certified for the noise-augmented composition, not the underlying deterministic network.

## How they judged success

Three numbers per configuration: average certified radius (mean radius over test images), clean accuracy, and certification wall-clock time per example. The comparison is against input-space randomized smoothing on the same architecture family, with the fraction of orthogonalised skip connections as the ablation variable. Statistical confidence for the radius comes from Monte Carlo sampling of the class probabilities, but the paper does not report the sample counts, the confidence level, or an abstain rate in the text I read.

## What they found

On CIFAR-10 with ResNet18 the headline is cost: about 11.2 to 13.0 seconds per certified image at input-space smoothing versus about 7.8 to 8.0 seconds with half the network orthogonalised — the abstract and introduction quote roughly 12 to 8 seconds, a stated 33.3% saving. Average certified radius stays close (0.444 vs 0.472, 0.533 vs 0.564, 0.524 vs 0.532 at the three σ) while clean accuracy drops (86.9% vs 93.2% undefended; 77.6 vs 80.8, 66.7 vs 68.2, 50.7 vs 49.4 at the three σ — the last one favours their model). Deepening the latent space to 13 of 18 orthogonal skip connections cuts the time further (about 5.8 to 6.1 seconds) but costs more: clean accuracy 79.4%, radius 0.387/0.485/0.480. On ImageNet the time roughly halves (about 88 to 47 seconds per example) with clean accuracy 71.6% versus 74.7%, and radii of 0.532/1.063/1.854 against 0.654/1.166/2.123. They note the latent space is actually *larger* than the input at the shallow cut (32768 versus 3072 dimensions on CIFAR-10) and that the win comes anyway, which contradicts the simple view that sample concentration makes low-dimensional spaces cheaper.

## Limits and open problems they name

The accuracy price is attributed to the orthogonal encoder's limited expressiveness and to the absence of batch-normalisation-like stabilisers for orthogonal training; more norm-preserving modules means less capacity. The latent-space route also inherits randomized smoothing's noise-related accuracy loss on top. There is no discussion of how to choose the split point other than by sweep, and no comparison against architectures whose Lipschitz constant is *estimated* rather than imposed — the very baseline the argument in Section 3.2 is aimed at.

## What we take from it

- **A construction, and a warning about it.** The certificate is true because the wrapper was built so that it cannot stretch distances. That is exactly the failure mode of the previous entry in this sub-head: the guarantee is a property of the composition, and everything the model actually does is outside it. Here the wrapper costs 6 to 14 points of clean accuracy depending on dataset and depth — the wrapper's price is large and measurable, and it is reported, but it is reported in the accuracy column rather than next to the certificate.
- **A measure to borrow:** "seconds per certified example" as the axis on which a guarantee is defended. It is a real quantity, unlike the order-of-magnitude volume ratios of the previous entry, and it moves monotonically with a knob (how deep the cut is) that the reader can see.
- **A framing for our own account:** a certificate across a *cut* is a statement conditional on everything upstream of the cut. If we ever write a predictive claim about a network it should be explicit about which sub-computation it is claiming over, because the moment a map in front of the certified part can stretch inputs, the input-space claim is only as good as that map's Lipschitz bound — and if that bound is imposed by architecture, the model has been constrained to make the theorem true.
- **A gap to note:** no sample counts or confidence levels are given for the radii, so the radii are not auditable from the paper alone.

## Tags

unit: whole-model
form: causal-abstraction
method: other
eval: none-of-the-listed (certified radius under a proven Lipschitz bound; report average certified radius + clean accuracy + seconds per example)
goal: evaluation-of-methods
