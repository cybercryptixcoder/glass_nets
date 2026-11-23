# Estimating Neural Network Robustness via Lipschitz Constant and Architecture Sensitivity

- **Key:** arxiv-2410.23382
- **Authors / organisation:** Abulikemu Abuduweili, Changliu Liu; Carnegie Mellon University
- **Date:** 2024-10-30
- **Link:** https://arxiv.org/abs/2410.23382
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1; 51,218 chars clean, sections 1-5, Tables 1-3, Appendices A and B)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The authors work on robots, not on interpretability, and their problem is a practical one: a perception network that can be fooled by a small crafted input change is unsafe to deploy, and neither of the usual answers is satisfying. Attack-based evaluation only produces a spot-check, verification is sound but expensive and always approximate, so both give a verdict per input rather than a property of the network. They want a quantity that belongs to the *architecture* rather than to a single test case, so that a designer choosing depth, width or initialisation can predict in advance how sensitive the finished network will be. The Lipschitz constant is the obvious candidate because classical results make certified robustness a function of it, but it is normally computed *from trained weights*, one network at a time. The goal of the paper is therefore to replace a numerical measurement with a closed-form expression in the architectural parameters, and to see whether that expression explains the robustness differences between designs.

## What they studied

Plain multilayer perceptrons under $\ell_p$ perturbations, in a classification setting; convolutional layers, residual blocks and any norm other than the ones implied by the derivation are explicitly out of scope. The unit of analysis is unusual for this folder: not a neuron, feature, direction or circuit, but the *architecture as a whole*, summarised by three numbers - depth $M$, hidden width $d$ (one shared hidden width, input dimension $n$, output dimension $m$), and the scale $\alpha$ of Xavier initialisation. Weights are random variables with zero mean and a variance fixed by the fan-in/fan-out rule, activations enter only through $q$, the standard deviation of the derivative under a standard normal pre-activation ($q=\tfrac12$ for ReLU). Experiments are toy MLPs for the theory check and small MNIST classifiers (2-4 layers, width 128-512) for the robustness check.

## How they did it

The derivation is a chain of three imported results and one new variance calculation. First, the exact Lipschitz constant of a differentiable function is the supremum over inputs of the Jacobian's operator norm, and the operator norm is the largest singular value. Second, for a random matrix with zero mean, variance $\alpha^2$ and a finite fourth moment, the expected largest singular value is asymptotically $\alpha(\sqrt{N}+\sqrt{n})$ in the tall aspect-ratio regime. Third, the missing ingredient is the Jacobian's *variance*, which they compute by writing the Jacobian element as a sum over a hidden unit of three independent factors (upper weight, activation derivative, lower weight), so the variance of the product is the product of the variances and the sum over $d$ hidden units contributes a factor $d$. Substituting the Xavier variance $2\alpha^2/(\dim_{\text{in}}+\dim_{\text{out}})$ into each layer, multiplying through the depth, and applying the singular-value formula yields a single closed form in $(M, d, n, m, \alpha, q)$. They then check the form empirically on randomly initialised networks of varying depth, width and initial variance, using bound propagation as the numerical reference, with ten trials per configuration and smoothing on the curves, before training small MNIST classifiers with SGD to convergence and computing certified accuracy under an $\ell_2$ ball of radius one with interval bound propagation. Further experiments train under weight decay and track the Lipschitz constant across training itself.

## What counts as an explanation here

A closed-form scaling law, not an account of any particular network. The object it describes is the *expected* Lipschitz constant of a network drawn from an initialisation distribution, so the "explanation" is a statement about what a design choice does on average before training - more layers multiply the constant by a factor, more width divides it, larger initial weight scale multiplies it. It is partial by construction: it says nothing about the constant of a specific trained model, and the authors are explicit that they treat the Jacobian's distribution after training as unchanged in shape, differing only in its variance. The payoff form is monotone statements tied to a robustness threshold via the classical margin criterion, i.e. the expression predicts the direction and the shape of a trade, and predicts that a network is certifiably robust inside a radius proportional to its margin divided by the constant.

## How they judged success

Two comparisons, both narrow. Against numerical truth: the analytical curve is plotted against a bound-propagation estimate of the same quantity as depth (up to nine layers), width and initial scale vary, judged by visual agreement and nothing else - no relative-error table, no coefficient of determination, no test of the ratio at any point. Against task behaviour: standard test accuracy, certified accuracy at a fixed $\ell_2$ radius of one under IBP, and the measured Lipschitz constant from bound propagation, tabulated for nine architectures in a depth-by-width grid and for three initial scales. No attack baseline, no adversarial accuracy, no comparison against another constant-estimation method, no error bars on the trained-network results.

## What they found

The closed form tracks the numerical estimate closely on random networks, and it says the constant grows exponentially with depth and falls as one over the square root of width; both trends survive training on MNIST. The trained-network grid is the informative part: standard accuracy is almost flat across all nine architectures (from 0.942 for the narrowest shallow net to 0.971 for the widest deep one - deeper and wider is *more* accurate), while certified accuracy at the same radius collapses with depth, from about 0.87 at two layers to about 0.58 at four, and the measured constant rises from roughly 76 to roughly 103 across the same span. Width helps in every depth row but only by a few hundredths of a point in certified accuracy. The initial-scale table is the sharpest result in the paper: at a scale of 3.0 the network still reaches 0.974 standard accuracy while certified accuracy at the same radius falls to exactly zero, with the measured constant at 819.6 against 102.5 at scale 1.0. Weight decay moves the same dial in the helpful direction, with certified accuracy rising from 0.583 to 0.841 as the penalty takes the constant from about 103 down to about 40 - the largest certified gain anywhere in the paper, and it costs two and a half points of standard accuracy.

## Limits and open problems they name

They name three: MLPs only, MNIST only, and the omission of the margin, which their own robustness criterion makes half of the story - a network can be robust either by having a small constant or by having confidently separated logits, and the paper controls only the first. Two further limits go unnamed and are the important ones for us. The expression is an expectation over an initialisation distribution, so it cannot be applied to any single network; the post-training claim rests on an assumption - that the Jacobian's entries keep the moment conditions the random-matrix theorem needs - which the authors state plainly and defer to future work. And nowhere is the analytical value compared numerically to the bound-propagation value on a trained network, so the reader cannot tell whether the expression predicts the constant of a trained model or merely correlates with it. The grid points and the training curves make the omission conspicuous: the table already contains both the trend and the measured value, and the ratio is one division away.

## What we take from it

A framing and a warning. The framing: a quantity that is a property of a design rather than of an instance can be predicted from structural parameters alone, and the prediction is a scaling law - exponent in the depth, exponent in the width - which is the shape of claim our own account should be able to make about a probing budget or an explanation's fidelity, and which is more useful than a single measured number because it says what happens at sizes not tested. The warning is the one this folder keeps rediscovering in a new costume: this paper replaces a *measurement* with an *expectation* and then reports the expectation in a table next to the measurement without ever dividing them, so an estimator whose error is a distribution rather than a bound is presented as if it were the quantity it estimates. The measured constants here are also the honest denominator for run 651's pricing question, and the initial-scale row is the cleanest illustration in the thread of a quantity that can be driven to ruin while a headline accuracy metric stays flat - which is the same failure shape as run 650's accuracy-preserving bias trick, arrived at here through initialisation rather than through arithmetic.

## Tags

unit: weights
form: description
method: weight-analysis | decomposition
eval: benchmark | downstream-task
goal: understanding
