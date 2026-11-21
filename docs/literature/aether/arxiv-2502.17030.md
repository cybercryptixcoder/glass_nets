# Your Assumed DAG is Wrong and Here's How To Deal With It

- **Key:** arxiv-2502.17030
- **Authors / organisation:** Kirtan Padh, Zhufeng Li, Cecilia Casolo, Niki Kilbertus; Technical University of Munich, Helmholtz Munich, Munich Center for Machine Learning (MCML). Padh and Li are marked as equal contribution.
- **Date:** 2025-02-24 (v1); v2 2025-03-10
- **Link:** https://arxiv.org/abs/2502.17030
- **Kind:** paper
- **Read from:** full text (arXiv HTML v2; §1-§6, Appendices A, B.1-B.4, C, D.1-D.3 and E read; figures 1-6 not inspected)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Every effect estimate in the graphical tradition starts from a graph, and in practice that graph comes either from a domain expert or from a discovery algorithm — neither of which yields a single trustworthy DAG. Experts argue about individual edges rather than endorse whole structures, and discovery returns an equivalence class at best, is sensitive to thresholds and hyperparameters, and for constraint-based methods even depends on the input variable order. So the standard practice of committing to one graph and estimating the effect leaves the analyst unable to answer the obvious objection: what if the graph is wrong? Their answer is not to fix the graph but to stop reporting a point estimate: given whatever partial knowledge is available (edges held sure, edges ruled out), compute the smallest and largest value the query takes over every DAG compatible with that knowledge, and report those two numbers. The ambition is a cheap, general rebuttal to the graph-misspecification critique that works for non-linear mechanisms and for graph collections far too large to enumerate.

## What they studied

Fully observed structural causal models with finite-variance densities, no latent confounders and no assumption on the noise family or the structural functions; both linear additive and two sigmoidal mechanisms are evaluated. The unit of analysis is not a neuron or a feature but the *graph itself* — specifically the set of uncertain edges, whose presence or direction is the free variable being optimised over. Data are Erdős–Rényi synthetic graphs of 3 to 9 nodes (plus 10-25 node runs for runtime only), 5,000 samples each, plus the real-world Infant Health and Development Program (IHDP) semi-synthetic benchmark with 25 covariates and 747 samples.

## How they did it

The search space is a set of allowed DAGs defined by two lists: sure edges that must appear and forbidden edges that must not, with the remaining K possible edges free. The query of interest is written as a functional of the graph — for average effects, the adjustment formula says the interventional mean under a valid adjustment set is an average of a regression prediction, so for a fixed graph the query costs one regression (ordinary least squares in the linear case, a small MLP in the non-linear case). Bounding then becomes a discrete min/max over the 2^K edge indicator vectors subject to acyclicity. Three moves turn this into continuous optimisation. First, the acyclicity constraint is replaced by a smooth scalar penalty (they use the DAGMA constraint, and report that DP-DAG is the alternative sampler), making "is a DAG" a differentiable equality. Second, the binary edge vector is relaxed to real logits feeding a straight-through Gumbel-Softmax: the forward pass samples a hard binary matrix which is used to determine the valid adjustment set and hence the regression, while gradients flow back through the soft matrix. Third, the constrained problem is solved by an augmented Lagrangian whose multipliers and penalty weight are updated between rounds, the query entering with a plus or minus sign depending on which bound is being computed. They run the whole optimisation once for the minimum and once for the maximum, bootstrap the two selected matrices 50 times to attach an estimation standard error to each bound, and widen the reported bounds by one of those standard errors. As a second solver they also apply differentiable DAG sampling with a topological-order permutation, over which forbidden edges can be masked (removing edges cannot create a cycle) but sure edges cannot be enforced without risking a cycle.

## What counts as an explanation here

Not an explanation of the model in the interpretability sense: the deliverable is a pair of numbers and a claim about them — a set-valued answer to a causal question that is valid across every account consistent with the stated prior knowledge. Completeness is deliberately abandoned. They state explicitly that the object being produced is a bound *set* rather than an interval, and that only its minimum and maximum are reported. It predicts nothing about the data-generating mechanism; it predicts the range of readouts any admissible account can produce, which is precisely the quantity a reader needs when they cannot choose an account.

## How they judged success

Three measures, each computed per simulation: point coverage (a binary indicator of whether the estimated bounds contain the true effect), bound coverage (the overlap between the estimated interval and the true interval divided by the true interval's length, so 1 means the true bounds are fully contained), and bound narrowness (width of the estimated interval divided by the width of the overlap, so 1 means the estimate is at least as tight as the truth). Brute force over all compatible DAGs is the reference and the ground truth for small graphs, and runtime against brute force is the scalability axis. Uncertainty is generated two ways: a random selection of a preset fraction of true edges and true non-edges to declare known, or the intersection of several PC runs over permuted variable orders (edges present in every run are sure, edges absent from every run are forbidden, everything else uncertain).

## What they found

On the random-uncertainty grid (5,670 simulations) the estimated bounds very nearly always contain the true effect and the true bound set: point coverage 0.97 for the Lagrangian and 1.00 for the permutation sampler on the linear mechanism, 0.99 for the Lagrangian on the non-linear one; bound coverage 0.93 against 0.99. The price is narrowness — estimated intervals run roughly 2 to 2.5 times the width of the truth, and in the PC-uncertainty grid the Lagrangian's linear arm drops to 0.64 point coverage against 0.96 for the sampling arm, which the authors attribute to the sampler covering the true set better while the Lagrangian gives tighter but sometimes non-covering bounds. Runtime is the result they most want read: while brute force grows super-exponentially with the number of uncertain edges, their optimiser is near constant across orders of magnitude of search-space size, and they hypothesise this reflects fixed per-update cost of a d² parameter matrix in tuned frameworks. On IHDP, with the sure/forbidden lists derived from ten permuted PC runs rather than any expert knowledge, they report bounds of about 2.4 to 7.2 with a ground-truth effect of 4.7 — covered, and roughly a factor of two wide.

## Limits and open problems they name

No hidden confounding — a fully observed setting, which they concede is almost never true in practice. The relaxation is non-convex, so there is no guarantee the returned bounds are the optimal (tightest valid) ones; the empirical gap between the two solvers is exactly this failure in action. The Lagrangian does not guarantee a DAG at convergence and needs a post-hoc filter. The permutation sampler cannot enforce sure edges, so using that solver alone weakens the bounds. They also flag that their PC-derived uncertainty can itself exclude the truth, since an edge the algorithm always deletes will be declared forbidden. Future directions they name: latent variables, graph classes other than DAGs, other solvers, and alternative representations of uncertainty.

## What we take from it

A framing plus a method. The framing is the load-bearing part: turn "which account is right?" into "over all admissible accounts, what is the range of the readout?", make the prior knowledge an explicit pair of lists (must-hold, must-not-hold) rather than a chosen object, and report the extremes. That maps directly onto our own situation, where the account of a network is always one of many and the honest deliverable to a reader is the span of behaviour across the accounts we cannot rule out. The method is a reusable trick for when that span is too big to enumerate: keep the discrete structure, replace its hard constraint with a smooth penalty, sample the hard structure in the forward pass with gradients flowing through a soft version of it, and use multipliers to push the penalty toward zero — all while the quantity being extremised stays exactly the thing we care about (they are careful to note that this is only possible because the *estimate* is differentiable given the structure, and they optimise the query itself rather than a likelihood surrogate). Two concrete warnings transfer: their measured coverage narrowness of 2-2.5x is the honest cost of a relaxed search, and the drop from 0.96 to 0.64 when prior knowledge comes from an algorithm rather than a random selection is a warning that derived priors can be confidently wrong. Also worth stealing: widening each bound by one bootstrap standard error before checking coverage — the finite-sample analogue of run 540's perturb-the-statistic move, applied to the endpoints of a bound rather than to a decision.

## Tags

unit: weights
form: replacement-model
method: weight-analysis, decomposition
eval: prediction/simulation, benchmark
goal: understanding, science
