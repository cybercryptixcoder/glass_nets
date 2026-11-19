# Spectral Signatures in Backdoor Attacks

- **Key:** arxiv-1811.00636
- **Authors / organisation:** Brandon Tran (MIT, EECS), Jerry Li (Simons Institute, Berkeley), Aleksander Mądry (MIT, EECS)
- **Date:** 2018-11-01
- **Link:** https://arxiv.org/abs/1811.00636
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Backdoor poisoning — a handful of mislabelled training examples that make a model predict an attacker's target class only when a small trigger mark is present — had been shown to work and to leave accuracy on ordinary data essentially intact, and no defence was known. The authors wanted a *statistical property* of the poisoned training set that a defender could compute and act on, rather than an attack-specific heuristic. The larger goal is stated as a first step toward securing training pipelines generally: if every attack of this family must satisfy the property, then either a defence exists or the attacker must argue the property away. The motivating observation is that a backdoor has to be a strong predictor of its target label, so any representation the network learns for classification should amplify it.

## What they studied
CIFAR-10 image classification with a ResNet (3 groups of residual layers, filter widths 16/16/32/64, five residual units each) and a VGG model as replication. The unit of analysis is a **per-class set of representation vectors** taken from an intermediate layer (they report the second-to-last residual unit, chosen because it removed the most poison on average; many later layers worked). The triggers are single-pixel or small-shape marks (pixel, X, L) at chosen positions and colours; attack/target label pairs are partly hand-chosen (airplane→bird, automobile→cat, cat→dog, horse→deer) and partly random.

## How they did it
Train the network on the possibly polluted training set. For each label, collect the representations of that label's examples, centre them by their own mean, and take the top right singular vector of the centred matrix. Score each example by the squared projection of its centred representation onto that vector, drop the examples with the largest scores — a budget of 1.5× the assumed poison count ε — and retrain from a fresh initialisation on what remains. The formal half of the paper argues that when the mean of the clean distribution and the mean of the poison distribution are far enough apart relative to their variances (roughly, squared separation at least six variances over ε), the top eigenvector of the mixture covariance is guaranteed to correlate strongly with the direction between the means, so removing the largest ε fraction along it removes the poison. Controls: the same procedure applied directly to input pixels (fails, even with augmentation off), weaker per-example statistics — ℓ2 norm, correlation with a random vector (both overlap heavily), and a class-merging arm ("pets" = cats + dogs) that deliberately inflates within-class variance.

## What counts as an explanation here
A **spectral signature**: a claim that a small sub-population of training examples shifts the covariance spectrum of a learned representation along one identifiable direction, plus a population-level sufficient condition for that claim to hold. It is explicitly partial and predictive at the level of *distributions* — it does not say which neurons or features encode the backdoor, only that the poison is linearly separable in representation space — and it is falsifiable by the separation condition failing. Nothing about the model's internal mechanism is claimed.

## How they judged success
Two accuracies per arm: natural test accuracy (all 10,000 CIFAR-10 images) and poisoned-set accuracy (1,000 attack-label images carrying the trigger), measured before the removal step and after retraining, with the number of poisoned examples remaining printed. The reference point is a network trained on a clean dataset, whose poisoned-set accuracy is reported beside each row; the explicit target stated in the introduction is to land within one point of that reference. Tables 1 and 2 in the body plus Tables 4 and 5 in the appendix; no confidence intervals, and success is partly defined by the run having reached roughly 90% poisoned accuracy in the first iteration.

## What they found
A few percent of one label's examples is enough: with 250 poisoned images (5% of a class of 5,000), natural accuracy stays at the clean level around 92-93% while poisoned-set accuracy climbs near 90%. At the learned representation the top singular value of the class covariance roughly doubles when poison is added (1,194 → 1,613 for one experiment, 1,017 → 1,884 for the other) for a mean shift of order 6-7 in a mean norm of about 79-90, which is the concrete instance of the separation condition. After one removal step and retraining, poisoned accuracy falls to within about a point of the clean-trained reference in most rows (e.g. 89.8% → 1.5% against a reference of 1.2%), with the residual number of surviving poison examples ranging from zero to 57 in the hardest row. The pixel-level control fails, and so do the norm and random-vector scores, which is the paper's evidence that the learned representation, not the input, is what makes the poison visible. In the merged-class arm the method still removes everything in most rows, though the authors note that at this 5% rate one hand-chosen backdoor did not install at all, whereas 10% did.

## Limits and open problems they name
The guarantee is stated at population level with deliberately loose constants, and the finite-sample version is deferred entirely to an appendix. The threat model gives the attacker control of the data but not of training, and the triggers are small and fixed. The paper itself notes that a defence against small-ℓ0 perturbations would rule out this class of backdoor, and lists that as future work rather than an achievement. Two things they do not test, and which matter for us: no adaptive attacker who knows the detector, and no variation in the number of poison points actually used versus the assumed budget.

## What we take from it
Three things. (1) **A method**: per-class centring, one leading singular direction, squared projections as an outlier score — the cheapest possible "is this sub-population structurally distinct in representation space" test, applicable to any set of probe or activation vectors we collect. (2) **A measure worth copying**: the paper reports the leading singular value of the covariance before and after the manipulation, which is a direct, unit-bearing quantity rather than a score — the same discipline our own spectral entries in this folder keep reaching for. (3) **A warning about where a signature lives**: the same statistics on raw inputs carry nothing, and even at representation level weaker scalars (norms, random projections) blur the two populations, so "the direction exists" is a claim about a *chosen* representation and a *chosen* statistic, and neither choice is neutral. This is the origin of the spectral hypothesis that `arxiv-2602.15195` and `arxiv-2601.06305` operationalise on weights; the transfer in the other direction is that a detector on weight spectra should print the same before/after leading-value columns this paper prints on activations.

## Tags
unit: layer | feature
form: direction | probe
method: weight-analysis | other
eval: downstream-task | ablation
goal: safety/auditing | understanding
