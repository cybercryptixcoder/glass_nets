# Causal Concept Explanations for Deep Neural Models

- **Key:** arxiv-2611.14407
- **Authors / organisation:** Josh Rountree, Pulkit Verma, Oswin So, Chuchu Fan, Julie A. Shah; MIT CSAIL + MIT LIDS + IIT Madras
- **Date:** 2026-11-10
- **Link:** https://arxiv.org/abs/2611.14407
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Concept-based explanations are usually correlational: a concept can be active whenever a particular output occurs without being in the causal chain that produced it, so the explanation looks right and misleads. The authors want an explanation whose causal standing is measured rather than assumed, because certification and auditing of learned controllers needs to know how much of the decision each named concept actually carries. Their larger goal is to move causal faithfulness from a property claimed by inspection into one that is designed for in training and then measured in evaluation.

## What they studied
Deep networks under concept supervision in three domains: an aircraft extended-trail-formation controller (four binary pilot concepts, two active per sample), CarRacing-v1 (three mutually exclusive concepts), and CUB-200 image classification (112 binary bird attributes). In every case a frozen pretrained backbone (an RL policy, a CNN policy, or Inception-v3) is wrapped by a small concept encoder plus a new head. The unit of analysis is a **concept slot**: a vector mediator whose length encodes presence and whose direction encodes expression. The causal object is the slot's contribution to the wrapper's output; no claim is made about the backbone's own internals or about input-to-output effects.

## How they did it
The wrapper defines a chain input → backbone latent → concept slots → output with no edge that bypasses the slots, so the direct effect of the input is zero by construction and the slots are the only mediators. Concepts are trained so that each active slot is *necessary* (replacing an inactive slot with a counterfactual draw should not move the output), *sufficient* (the active slots' contributions alone should reproduce the expert label), and *independent* (the per-slot changes from a zero reference should sum to the total change), plus an optional presence term forcing each active concept to carry at least an equal share. Counterfactual slot values come from a per-concept conditional normalizing flow fitted to the encoder's own representations, which is refitted periodically during alignment training. Because the value projection is bias-free, a zeroed slot contributes exactly zero value while keeping a positive attention weight, so zero is a legitimate reference point. Evaluation is deliberately separated from training: the reported effect fractions come from a Shapley-style estimator that uses **no** flow, no zero reference, and none of the training losses — it matches test inputs across two sets that differ in one concept label, builds hybrids with slots drawn independently per group from the matched counterparts, and attributes the output change to each group. Everything is averaged over ten seeds with two-sided 95% Student-t intervals.

## What counts as an explanation here
A decomposition of the model's output into per-concept fractions of an interventional indirect effect, reported per concept-label combination and split into the share carried by active versus inactive concepts. It is partial and deliberately local: an account of how much the model's *own* slots drive its *own* output, with the residual interaction term absorbing whatever cannot be attributed to a single concept. It predicts nothing about the frozen backbone or about the data-generating process.

## How they judged success
The primary metric is the fraction of the output effect that the active concepts carry, with the inactive fraction as its complement; a faithful model has the inactive share near zero. Supporting metrics are task error, concept accuracy, and the DCI disentanglement/completeness/informativeness triple. Baselines are the same architecture trained without alignment losses, a prior scalar-concept wrapper (CW-Net), a concept bottleneck model on CUB-200, and two disentanglement objectives (β-TCVAE, iVAE). Five ablation families (loss subsets, concept dimension, data quantity, label fraction, label noise, ontology removal/merge/placebo) are run, and DCI in particular is tested for whether it tracks the causal measure.

## What they found
On aircraft control, the aligned model routes 94.5 ± 2.2% of the effect through active concepts against 50.0 ± 6.2% for the prior scalar wrapper, with comparable task error and accuracy; on CarRacing the comparison is 89.8% against 33.1%, and on CUB-200 an unsupervised-by-design architectural version routes 91.3% against 41.6% for the concept bottleneck, whose inactive concepts absorb 58.4%. Two dissociation results carry the argument: iVAE scores DCI disentanglement 1.000 yet its active concepts carry only 47.6% of the effect, and adding 10% label noise drops the active share from 97.3% to 71.6% while concept accuracy barely moves. The placebo and dimension ablations are the sharpest limits: a placebo concept added with alignment still absorbs 36.2 ± 11.4% of the effect at width 256, a scalar (1-d) concept representation recovers only 69.9-70.8%, and the merge/remove rows on CarRacing fall to 56.9% and 65.8% against 75.3% for the full ontology — the routing survives only when the ontology is close to right.

## Limits and open problems they name
The causal claims are confined to concept-to-output effects inside the wrapper; effects in the data-generation process are left to future work. We add: the counterfactual draws rest on a learned flow and on positivity, so the necessity loss is only as good as the mediator model; the effect decomposition is relative (fractions summing to one), so an inflated passive share and a missing concept are not distinguished; the CUB-200 backbone was fine-tuned on a set overlapping roughly 80% of the authors' test images, which they state explicitly; and the alignment training buys no task improvement, only routing.

## What we take from it
A method and a measure. (1) The training-side idea that causal criteria can be *optimised* rather than audited — necessity, sufficiency, independence as losses over slot edits — is a template for making a white-box model's components load-bearing by construction. (2) The two-estimator discipline (train with a mediator model, evaluate with a flow-free estimator on held-out data) is the cleanest separation of fit from measurement the folder has. (3) Two warnings worth keeping: a near-zero residual interaction term in an additive decomposition can be produced by a single concept absorbing the effect, so additivity is not per-concept necessity; and an architecture can be causally faithful by design yet degraded by ontology misspecification or label noise that concept accuracy cannot see.

## Tags
unit: feature
form: causal-abstraction
method: training-interpretable; ablation
eval: ablation; prediction/simulation; downstream-task
goal: understanding; safety/auditing
