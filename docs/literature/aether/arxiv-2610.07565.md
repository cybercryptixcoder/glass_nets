# Complementary Feature Domains: Information Preservation Does Not Imply Predictive-Contribution Preservation

- **Key:** arxiv-2610.07565
- **Authors / organisation:** Timothy Oladunni, Farouk Ganiyu-Adewumi; Department of Computer Science, Morgan State University
- **Date:** 2026-10-06
- **Link:** https://arxiv.org/abs/2610.07565
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Multi-representation prediction systems feed a model several encodings of the same observation (time, frequency, time-frequency, learned). A common intuition is that an invertible recoding of one representation must be harmless, because a bijection leaves the mutual information with the target untouched. The authors argue this is a statement about *representations* while prediction happens through a *restricted family of decision rules*; if that family cannot express the compensating inverse, two information-equivalent encodings admit different attainable decisions and therefore different realised utility. Their larger goal is a formal account of when the contextual role of one representation survives a recoding and when it does not, and a quantitative law for how far it can move. The work is framed as a transition theory rather than a new information measure.

## What they studied

No neural network is trained for the theory. The objects are: a representation system of m component maps applied to a source Z; a declared family of decision rules (the "realisation family") available from each subset of representations; and a bounded utility. The unit of analysis is the *contextual marginal contribution* - the change in best attainable utility when one representation is added to an existing coalition - collected over the whole lattice of subsets into what they call a contribution system. Empirically they study two synthetic finite cases (a four-state XOR, and three independent bits with four deterministic targets), a capacity stress test of four realisation families, and one controlled experiment on frozen ECG embeddings.

## How they did it

They define coalition value as the supremum of expected bounded utility over the declared decision family, then define the marginal of representation i given a baseline subset S as the difference between the value of S plus i and of S. Recoding is modelled as a componentwise bijection g applied to one representation, after which the available family may change from A to B. To bound what happens, they define a behavioural compatibility error per coalition: the symmetric L1(P) Hausdorff distance between the sets of random actions realisable before and after recoding. Assuming the utility is L-Lipschitz in the action, they prove that the change in a coalition's value is bounded by L times its compatibility error, and that the defect in a contextual marginal is bounded by the sum of the compatibility errors of the two coalitions straddling the marginal. Synthetic work: enumerate the exact affinely-realisable dichotomies on the XOR support and on its recoded support, then repeat under quadratic-logistic, RBF-SVM and depth-2 tree families. ECG work: freeze time, frequency and time-frequency embeddings for 928 observations, group duplicate-linked records into 491 integrity groups so no group crosses splits, take a fixed 648/143/137 train/validation/test split, standardise using training statistics only, apply a signed-cube recoding to the time-frequency block, and compare raw, recoded and exactly-inverse-restored inputs under one-versus-rest logistic regression with identical settings. Uncertainty comes from 5,000 paired bootstrap resamples over test integrity groups. The finite-sample defect is then read off the coalition accuracies.

## What counts as an explanation here

A *causal-ish invariance theorem plus a defect statistic*. The explanation is not a description of any network's internals; it is a claim of the form "this quantity, indexed by (representation, context), cannot move more than this function of behavioural incompatibility", together with a named statistic measuring how far it did move. It is deliberately partial: the bound is sufficient not necessary, so a defect below the bound is consistent with the theory while a defect above it falsifies the mechanism. It predicts nothing about a specific model's weights - only about how coalition-indexed contributions must behave under a recoding whose behavioural compatibility is measured.

## How they judged success

Exact population accuracies computed by enumeration on the synthetic supports; presence or vanishing of the affine-versus-recoded gap as realisation capacity increases; on ECG, the raw/recoded/restored accuracy triple per coalition, with the recoding gap stated as a point difference and a paired 95% integrity-group bootstrap interval, and restoration error with its own interval. The stated success criterion for the mechanism is the joint *failure-and-restoration* pattern: a nonzero, interval-excluding-zero gap under recoding that returns to exactly zero under the analytical inverse. There is no held-out benchmark, no comparison against another method's accuracy, and no significance test - the theory's claim is the *shape* of the pattern, not a margin over a baseline.

## What they found

Under an affine family the XOR support is not linearly separable and reaches a population optimum of .75, while the recoded support puts the two classes on the sign of the first coordinate and reaches 1.00 - a .25 gap at identical target information of one bit. The gap disappears exactly (1.00 versus 1.00, gap 0) for quadratic logistic, RBF-SVM and depth-2 tree families, so the phenomenon is a mismatch between the recoding and the family, not a property of the data. A stronger synthetic result: with two independent nuisance bits added alongside, *every* subset retains identical Shannon information, yet the coalition values under one fixed affine family still differ, so equality of the entire subset-information profile is not enough for equality of the contribution system. On the three-bit lattice, the parity target gives the third bit a full-context marginal of .500 under Bayes realisation but only .250 under an affine one, with contributions of zero in the empty and one-companion contexts - while majority, OR and AND show no Bayes-affine discrepancy there. On ECG, recoding the time-frequency block drops coalition accuracy from .7372 to .6277 (TF), .6569 to .5839 (T+TF), .7883 to .5912 (F+TF) and .7372 to .6058 (T+F+TF); the TF and F+TF intervals exclude zero, T+TF and T+F+TF straddle it. Applying the exact inverse restores every coalition to its raw value, with a restoration error of 0 and a [0, 0] interval in every context. The resulting defects differ by context - .1095, .0730, .1971 and .1314 for a baseline of nothing, T, F and T+F - so the same information-preserving intervention changes the time-frequency block's realised role by different amounts depending on what else is available.

## Limits and open problems they name

The bound is sufficient, not necessary; finite-sample error sits outside the population theorem, so no confidence statement accompanies the bound itself. The ECG arm is one dataset, one learner family, one recoding function, and the authors explicitly say the cubic map is chosen as a controlled perturbation rather than as a better representation. They say the experiments test the mechanism and not any claim about universal complementarity. Broader tasks, other realisation families, and uncertainty guarantees are named as future work. The synthetic witnesses are hand-built finite supports, so the demonstration is existential; nothing in the paper estimates how often such mismatches occur in practice.

## What we take from it

A **measure and a warning**. The measure: index a component's contribution by the context it is added to and report the vector, not its average - the parity signature (zero, zero, zero, .5) is invisible to any single context-free importance score, and the ECG defects that matter (.1971 with F already present) are the ones a global accuracy difference of .1095 would hide. The warning: an exactly invertible re-encoding of our own features or activations - normalising, whitening, rotating into a probe basis, changing a dictionary's coordinates and re-expressing the same activations - is *not* guaranteed to leave the downstream prediction unchanged even though it leaves the information unchanged. The check is cheap and we should copy it: apply the map, then apply its exact analytical inverse, and require the coalition accuracies to return to their originals; a failure to return means the map itself was the confound, and a return means the difference was real geometry rather than numerical damage. Reusable framing for our white-box reports: state the realisation family for every contribution number, because a component that looks redundant under a linear probe may be decisive under a richer one (and vice versa), and the field's habit of quoting a single Shapley average is exactly the averaging this paper shows can bury a representation-learner mismatch.

## Tags
unit: feature
form: causal-abstraction
method: other
eval: ablation
goal: understanding
