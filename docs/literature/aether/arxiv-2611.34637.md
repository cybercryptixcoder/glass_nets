# Carrying Feature Ablations Across Sparse Autoencoder Seeds

- **Key:** arxiv-2611.34637
- **Authors / organisation:** Karl Li; Columbia University
- **Date:** 2026-11-25
- **Link:** https://arxiv.org/abs/2611.34637
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Sparse autoencoders trained from different random seeds on the same activations learn different features, and the standard way to reconcile them is a one-to-one matching that maximizes the cosine between decoder directions. That matching licenses a claim the authors think is unearned: that a feature "recurs" across seeds, so that ablating it in one SAE edits the encoder the way ablating its partner edits it in another. Two directions can agree closely while the features respond to different tokens, so the edits can differ. The goal is a behavioural criterion for what "the same feature" means across training runs, and then cheap proxies for that criterion. This is the seed-stability question that any interpretability result quoted from one SAE has to answer.

## What they studied

Two frozen vision encoders, DINOv2 base and the vision tower of SigLIP base, in three settings that each change one thing from a development setting: DINOv2 layer 6 at 6,144 SAE features, SigLIP layer 11 at 6,144, and DINOv2 layer 11 at 3,072. Five SAEs trained from five seeds per setting, plus two control families (5 SAEs with decoder directions frozen at their random start, and 5 surrogate SAEs trained on Gaussian tokens matched to the layer's covariance). The unit of analysis is the matched feature pair: 128 sampled features per SAE, each paired with its Hungarian match in each of the other four SAEs, giving 2,560 pairs per setting.

## How they did it

Each SAE reconstructs scaled layer tokens as a sum of unit decoder directions with non-negative ReLU codes; a feature's contribution to a token is its code times its direction. Ablating feature *i* subtracts exactly its contribution from every token of the edited layer and runs the encoder's later layers and final normalization on the remainder. The same is done for *i*'s matched feature in a second seed's SAE. The two effect vectors are compared by an **output error**: squared distance between the two output changes divided by the squared size of the first change, summed over all tokens of a set of images. The pair passes at a pre-registered bound of 0.25, which is equivalent to the difference of the two effects having norm at most half the first effect's norm. Images are COCO 2017: the first 4,000 validation images train the SAEs, the next 500 are selection images (where predictions are formed and any fit happens), and the last 500 are scoring images, split into two halves so the test can be checked against itself. A separate 500-image fresh set from the COCO training split scores every setting again with all rules frozen. Two sanity-check copies of each SAE are constructed whose right answer is known: a rescaled copy (whose matching must recover the permutation and whose pairs must all pass) and a shuffled copy (whose decoder directions are unchanged, so cosines stay at one, but whose encoder rows move, so its pairs should fail). Two cheap rules predict the output test without running the later layers: a **cosine rule** (pass where decoder cosine ≥ 0.96 and encoder cosine ≥ 0.71, thresholds fitted on a separate development setting and then frozen) and a **layer rule** (pass where the same comparison of Eq. 2, taken at the edited layer on the selection images, is ≤ 0.25). The layer error reduces to 1 + r² − 2rcρ for unit decoder directions, with c the decoder cosine, ρ the cosine between the two features' codes, and r the ratio of code norms — so it is small only where all three are near one. Post-hoc additions are marked with a dagger throughout, including the permutation control that shuffles matches among trained features.

## What counts as an explanation here

A behavioural equivalence test between two learned dictionary elements, not a description of either. The explanation is the pair-level verdict "this ablation carries across seeds" plus the diagnostic decomposition that says *why* it does or does not: the layer error separates the three ways a match can fail — wrong direction (c), right direction on wrong tokens (ρ), or right tokens at the wrong strength (r). The authors are explicit that this is a screen, not a certificate: the output test itself is the ground truth, and the layer rule is an approximation to it whose errors they chase.

## How they judged success

Four pre-registered commitments, fixed before the three settings were measured: the 0.25 bound, the first two controls, the cosine rule, and three comparisons. Each comparison demands that one share beat a baseline by more than a required difference that shrinks as the baseline approaches one (min{0.1, (1−q)/2}). (1) Matched pairs must pass more often than control pairs. (2) The output test's split-half agreement — its agreement with its own outcome on the other half of the scoring images — must exceed the cosine rule's agreement with those outcomes by that margin. (3) A feature that passes must keep its category name. Intervals come from 2,000 resamples of the 640 sampled features, kept paired by feature, which measures spread over sampled features only (the trained SAEs stay fixed). Robustness is checked on a second feature sample, on the fresh images, on image-level pooling, and on a third independent image set.

## What they found

**The pass share depends heavily on the setting.** 304 of 2,560 pairs pass at DINOv2 layer 6 (0.119), 1,057 at SigLIP layer 11 (0.413), and 1,790 at DINOv2 layer 11 at half width (0.699); the development setting sits at 0.405. Fresh images reproduce these within a few pairs (316, 1,045, 1,783). Moving the bound to 0.1 and 0.5 moves each share but not the order of the settings.

**The controls are at zero.** None of the 3,200 pairs into random-decoder SAEs passes in any setting; of the surrogate-SAE pairs, none passes at SigLIP layer 11 and one passes at half width; and none of the 4,608 permuted pairs passes on the fresh images. Every matched share therefore clears the required 0.1 margin, though at DINOv2 layer 6 the interval's lower end (0.096) sits just under it.

**Only strong matches pass, and being strong is not sufficient.** No weak match (either cosine at or below 0.7) passes in any of the three settings, against 3 of 1,259 in development. But 208 of 512 strong matches at DINOv2 layer 6 still fail, so a high cosine is necessary and far from sufficient.

**The layer rule beats the cosine rule, and the gain survives three different checks.** On scoring images the cosine rule mispredicts 57, 66 and 86 pairs; the layer rule mispredicts 34, 23 and 34, a gap of 23 to 52 pairs with every interval above zero. The gap repeats on fresh images (42/27/28 versus 65/56/81), on a second independent sample of 128 features per SAE (32-38 versus 58-63), and at the image-level pooled output (44-93 versus 73-126). The layer error also ranks pairs almost as the output error does, with rank correlations of 0.863 to 0.984.

**Which component carries the advantage is left open.** The layer error with the decoder cosine set to one (i.e. c collapsed) mispredicts 33 pairs at DINOv2 layer 6 — fewer than the full layer rule's 42 — but 54 at SigLIP layer 11 against 27. Single-component rules (code norm ratio alone, code cosine alone) are far worse. The authors say the data do not settle which part of the layer error is doing the work.

**The cosine rule's misses are mostly conservative.** 53, 65 and 71 of its 57, 66 and 86 errors are passing pairs that it calls failing; the reverse error (a failing pair it accepts) is 4, 1 and 15.

**One layer-rule error is real and persists.** Of the 8 pairs where the layer rule and the output test disagreed in one direction on all four halves of the scoring and fresh images, one survived on a third, never-used set of 250 COCO training images: at DINOv2 layer 11 half width, feature 387 of seed 2 matched into seed 4, passed by the layer rule at 0.178 and failed by the output test at 0.451. Both stay on their side of 0.25 in at least 0.99375 of 2,000 paired image resamples. The final normalization does not create it — the ratio is already 0.408 before normalization.

**Screening saves nearly all the cost of the output test.** Combining the rules (run the output test only where the cosine rule and the layer rule disagree) sends 37 to 68 of each setting's 2,560 pairs to the expensive path, but 13 to 27 of the accepted predictions are still wrong — so mutual agreement can miss a failing pair. Timing: the layer rule adds 0.010 s per pair at DINOv2 layer 6 against 1.081 s for the output test, and 0.008 against 0.136 at SigLIP layer 11 (one later layer).

**Features that pass are almost the ones the cosines rank highest.** 50, 211 and 368 of 640 sampled features pass into all four other seeds across the three settings. Ranking by the worst cosine across seeds reproduces that set to within 3 features on every SAE. Those features usually reconstruct more of the layer's variance than equally many random features — on all five DINOv2 layer-6 SAEs and four of five SigLIP — but the advantage over cosine-ranked features is mixed, and on two half-width SAEs they lose.

**Meaning could not be checked.** COCO's annotations name only 3 to 65 of the 640 sampled features at the author's coverage rule, and after the held-out-image name check more than nine in ten features take no name. The third pre-registered comparison therefore could not be made.

## Limits and open problems they name

Whether the pass share keeps rising from layer 6 toward layer 11 and the output is untested (only two DINOv2 layers are measured). Nothing is tested on another encoder objective beyond DINOv2/SigLIP. Whether a passing feature keeps its *meaning* in the other seed is unanswerable with available annotations. The layer rule's one persistent error is reported but not explained. Several analyses were added after the outcomes were known (daggered), including the permutation control, the layer rule's threshold fit, the second feature sample and the image-level check; the third pre-registered comparison could not be made at all. The intervals reflect sampled features only, not the five training seeds — seed-to-seed variation is reported as a range over first SAEs, not as an interval. This is a vision-encoder study; language-model SAEs are not tested here.

## What we take from it

**A method.** The matched-pair output test is the right template for any claim of the form "unit *u* means the same thing in run A and run B": ablate in both, compare the *effects* on downstream output with the effect's own norm as the denominator, and require a same-run reproducibility check (split-half agreement) before the cross-run number is believed. For our white-box model the directly reusable part is the decomposition 1 + r² − 2rcρ — three separately measurable failure modes (direction, support, magnitude) of a matched unit.

**A warning.** Cosine similarity between two dictionaries' units has a large false-positive rate: 12% to 70% of high-cosine matches do not behave alike, and no weak match in the three settings behaves alike at all. A paper that reports "N% of features recur across seeds" from cosine matching has reported a direction agreement, not a functional one.

**A measure.** The comparison of the output test against a *cheap* proxy on the same pairs, with the proxy's errors counted in both directions, is the pattern we should copy for any pre-intervention predictor: report the conservative misses and the dangerous accepts separately, because they cost the researcher different things.

**A shortcut.** The combined screen (agreement between two independent cheap rules, expensive test only on disagreement) is a general cost-reduction pattern for our own evaluation loop, with the honest caveat printed in the same sentence: 13 to 27 accepted-but-wrong predictions per setting.

## Tags

unit: feature
form: probe
method: SAE | ablation | probing
eval: ablation | prediction/simulation | faithfulness
goal: understanding | evaluation-of-methods
