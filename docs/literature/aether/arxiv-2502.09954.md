# On Space Folds of ReLU Neural Networks

- **Key:** arxiv-2502.09954
- **Authors / organisation:** Michal Lewandowski, Hamid Eghbalzadeh, Bernhard Heinzl, Raphael Pisoni, Bernhard A. Moser; Software Competence Center Hagenberg (SCCH), with Eghbalzadeh at AI at Meta and Moser also at Johannes Kepler University Linz
- **Date:** 2025-02-14 (v1)
- **Link:** https://arxiv.org/abs/2502.09954
- **Kind:** paper (accepted at TMLR 2025)
- **Read from:** full text (v1 HTML, all eight sections, Appendix A on CantorNet, Appendices B-D, all figure captions; figures read from captions and prose only, not inspected as images; the paper contains no tables)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Successive ReLU layers had long been described informally as folding operators on the input space — Montúfar et al. said it in passing, Keup and Helias likened it to paper folding — but nobody had attached a number to the deformation. The authors want the folding story moved from metaphor to measurement: a scalar, computed from a trained network's own activations, that says how much a straight line in input space has been bent by the time it is read in activation space. The larger goal is a geometric characterisation of what depth buys a network, in the same family as linear-region counts, but sensitive to the *arrangement* of regions rather than only to how many there are. They also want the quantity to be interpretable as a network property (comparable across models) rather than as a dataset property.

## What they studied

ReLU MLPs only. The unit of analysis is the **binarised activation pattern** of a whole network: every hidden neuron across every layer, sign-clipped to a bit, giving a point in a Hamming cube whose dimension equals the total number of hidden neurons. Two settings: CantorNet, a hand-built two-input recursive function whose decision surface is an arbitrarily ragged fractal, realised exactly as a small ReLU network; and MNIST MLPs at constant 60 hidden neurons in six depth/width splits (2×30, 3×20, 4×15, 5×12, 6×10, 10×6), plus larger 2×300 and 3×200 networks. The analysis object is a *walk*: a sampled straight line between two input samples, and the sequence of activation patterns it visits.

## How they did it

First they prove that their discrete notion of convexity is the right one to use: a set of activation regions is convex in Euclidean input space exactly when the corresponding set of bit patterns is convex in the Hamming cube, where convex means every shortest Hamming path between two members stays inside the set. That equivalence only holds for hyperplanes that cut the whole input space, i.e. for first-layer neurons, which is why folding is claimed to be invisible in a one-hidden-layer network. Then: interpolate between two inputs with n equally spaced points, forward-pass all of them once, binarise all hidden activations, and record two range statistics along the walk — the largest Hamming distance reached from the starting pattern, and the total Hamming distance accumulated step to step. Folding is one minus the ratio of the first to the second. The ratio is 1 for a walk that only ever gets further from where it started (flat); it collapses toward 0 when the walk wanders far and returns near the start (folded). A global version takes the supremum over all walks. They also propose, and test, replacing all sample pairs with k-means centroids per class to cut the quadratic pair count.

## What counts as an explanation here

A bounded scalar per path, plus one scalar per network in the global version — a *property*, not a mechanism and not a prediction. The authors are explicit that a value of the measure tells you how twisted the space is along that path, not which neurons did the twisting and not what will happen under an intervention. The proof structure is meant to supply the interpretation: the score is a precise deviation-from-convexity, so a high value can be read as "the straight input line was folded back on itself". There is no feature dictionary, no circuit, no per-neuron attribution, and no attempt to predict behaviour from the score in the paper itself — prediction appears only as a hypothesis about generalisation.

## How they judged success

Almost entirely by qualitative and correlational evidence. On CantorNet the check is that a recursion of depth 1 on a first-layer-only walk scores zero while the same walk with all layers included scores one half, and a recursion of depth 2 on a specified horizontal walk scores about 0.7 — an illustration that the instrument responds to known structure. On MNIST the checks are: the aggregated median of non-zero maximum folding rises with depth, reported as a Pearson correlation of 0.987 against depth for the single random seed (seed 4) where every network trained well; networks that failed to reach high validation accuracy showing noticeably lower folding; and the proportion of paths exhibiting any folding rising from roughly a third and 0.44 in the small nets to 0.97 and 0.99 in the 2×300 and 3×200 nets. Sensitivity to k-means grouping is checked on 16 digit-class pairs over 0, 3, 6, 9 with 1 to 100 clusters per class. There is no baseline network, no untrained control, no shuffled-label control, no second dataset, and no statistics beyond the one correlation — all of which the authors themselves list as future work.

## What they found

The measure behaves as a bounded folding score (proved to lie in [0,1] for any path that crosses more than one region), and its boundedness is offered as the reason it can serve as a cross-model reference point. Deeper MNIST networks show higher median peak folding, and the correlation with depth is very strong for the seed where training succeeded — but the same paragraph concedes that deeper nets are highly initialisation-sensitive and that only two of five seeds reached the target accuracy, so the depth correlation rests on the surviving architectures. Bigger networks barely change the peak values while almost every path starts showing some folding, which the authors read as folding becoming pervasive rather than more intense. On CantorNet, folding reaches roughly 0.7–0.8, well above MNIST, which they present as evidence that the measure tracks self-similarity in the decision surface. Their stated hypothesis is that peak folding is associated with generalisation capacity, framed as "depth is necessary but insufficient". The grouping study suggests that as few as five centroids per class reproduce the ungrouped folding values, making the quadratic pair sweep cheap in practice.

## Limits and open problems they name

The folding-is-invisible-in-shallow-nets result is a theorem about first-layer hyperplanes, so it does not license any claim about deeper layers where adjacent regions may differ by more than one bit — the equivalence that gives the measure its meaning is exactly the one that fails there. Uniformly spaced walk points are admitted to be suboptimal (they do not guarantee one point per region), and the fix is deferred to a later submission, leaving the score dependent on an n that is bounded only by the neuron count. Attention layers are said to be out of reach in the current form; residual connections are claimed to extend straightforwardly and normalisation is claimed to leave the measure applicable, both without experiment. The generalisation hypothesis is untested, and the paper's own proposed tests — random labels, adversarial perturbations, augmentation, learning-rate and optimiser variation, and computing the score over subsets of neurons — are all listed as future work rather than run.

## What we take from it

A **measure** and a **warning**. The measure: a Hamming-space walk statistic computed from one forward pass over an interpolated line, which counts region *transitions* through the geometry of the walk rather than through a region enumeration, and which is provably bounded — the cheapest sentinel we have seen for "did this path leave and come back", the discrete analogue of the locality tests the folder has been assembling. Combine it with the folder's random-probe control: a path between two random points is the null that separates "the fold is a feature of this class pair" from "this network folds everything". The warning is the seed situation, and it is the one to carry into our own geometry work: the headline correlation was computed on the single seed out of five where every depth trained, and the authors report it in the same breath as the failure. Also worth banking: their own concession that the measure may miss folds because adjacent regions can differ by more than one bit is the same defect the later companion paper resolves by halving the step until the bit gap is one — so a walk statistic in our white-box model must declare its step rule and its one-bit assumption on the same line as its number.

## Tags

unit: layer
form: description
method: other
eval: benchmark
goal: understanding
