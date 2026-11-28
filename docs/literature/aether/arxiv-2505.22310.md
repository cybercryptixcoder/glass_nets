# From Dormant to Deleted: Tamper-Resistant Unlearning Through Weight-Space Regularization

- **Key:** arxiv-2505.22310
- **Authors / organisation:** Shoaib Ahmed Siddiqui, Adrian Weller, David Krueger, Gintare Karolina Dziugaite, Michael C. Mozer, Eleni Triantafillou; University of Cambridge, Alan Turing Institute, Mila, Google DeepMind
- **Date:** 2025-05 (v1 2025-05-28; v2 2026-01-14)
- **Link:** https://arxiv.org/abs/2505.22310
- **Kind:** paper
- **Read from:** full text (abstract, sections 1-8, all 47 references, Appendices A-K, Tables 1-3)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Unlearning work in language models had already shown that a "removed" capability can come back after a little fine-tuning, even on data that has nothing to do with the forgotten thing. Those studies are hard to interpret, though: "knowledge" and "capability" are vague targets, the correct answer after removal is undefined, and question phrasing itself moves the numbers. The authors want a setting where the right answer is unambiguous — deleting specific training examples from a small image classifier — so that a retrained-from-scratch model can serve as the exact reference point. On top of that they ask a second question: is there anything in the weights of an unlearned model that says in advance whether its "forgetting" will hold up?

## What they studied

ResNet-18 and ResNet-34 image classifiers trained on CIFAR-10 and CIFAR-100, with a higher-resolution check on Imagenette. The unit of analysis is the **whole weight vector of the model**, compared with the pretrained model and with the retrained-from-scratch model by L2 distance between parameters and by linear mode connectivity (accuracy along a straight interpolation between two parameter sets, batch-norm statistics interpolated too). Eleven approximate unlearning baselines are compared: SCRUB, Circuit Breakers, NegGrad+, catastrophic forgetting (weight decay), L1-sparse, selective synaptic dampening, random relabeling, weight attenuation, weight dropout, and TAR.

## How they did it

Pretrain for 300 epochs (Adam, lr 1e-4, cosine decay factor 0.1, batch 128, weight decay 1e-4). Choose a forget set of 500 examples — either 10% of one class ("sub-class") or 1% of the dataset regardless of label ("class-agnostic"). Run each unlearning method for a fixed 100 epochs at lr 1e-5 with no weight decay, so methods are compared at equal budget. Then run the relearning attack: fine-tune the unlearned model for 10 epochs on the retain set plus 0, 10 or 100 examples drawn from the forget set, and score accuracy on the **held-out remainder** of the forget set, which the attacker never saw. The retrained-from-scratch model is put through the identical attack as the reference. Forget sets are split by typicality using precomputed consistency scores from Jiang et al., with atypical examples as the main condition. From the weight-space reading they build three new methods: Weight Distortion (add zero-mean Gaussian noise, sd 0.02, to every weight, then fine-tune on retain), Weight Dist Regulation (maximise the Euclidean distance to the pretrained weights while minimising retain loss), and Connectivity-Based Fine-Tuning (maximise loss at the midpoint between pretrained and current weights, under a small 0.001 weight and a loss-magnitude cutoff of 50). Each is also tried as a second phase on top of an existing safeguard.

## What counts as an explanation here

A **weight-space diagnostic plus a causal story about it**. The explanation of why a method fails is a geometric fact about the unlearned model: it sits close to the pretrained model, and in many cases the two are linearly connected with no loss barrier between them — so a short fine-tuning trajectory can walk back to behaviour the output-level method suppressed. It is an explanation of *fragility*, not of where in the network the forgotten examples are encoded; no neuron, feature or circuit is named anywhere in the paper, and the authors say which layers matter is future work. It is predictive in one direction only: distance predicts resistance with the right sign, and they report that it is the more reliable of their two diagnostics, but no fitted relationship or threshold is given.

## How they judged success

Two quantities per cell of a scatter plot: test-set accuracy on the x-axis (utility) and accuracy on the held-out part of the forget set after the attack on the y-axis (failure of forgetting). A model counts as tamper-resistant if its post-attack forget accuracy stays at the level reached by the retrained-from-scratch model under the same attack — i.e. relearning gains nothing beyond learning the relearn examples anew. The trajectory plots sample accuracy every 10 optimisation steps; scatter points average test accuracy and forget accuracy over the last 50 steps. Controls: the retrain-from-scratch arm in every cell; typical and random forget sets as degenerate conditions where all methods should behave alike; three seeds' worth of settings across 21 unlearned models per configuration; a membership-inference check; and a quantization sweep to separate this recovery channel from another known one.

## What they found

The central result is that **fine-tuning on the retain set alone — zero forget examples — takes forget accuracy back from just above chance after unlearning to nearly perfect**, across a wide set of methods, whereas the retrained model stays around 50% on the same held-out items. On atypical CIFAR-10 forget sets the methods that suppress behaviour most cleanly (Circuit Breakers, SCRUB, random relabeling) are the ones that come back most completely; on typical or randomly drawn forget sets everything looks fine, because the retrained model already scores near 100% on those items. Class-agnostic forget sets of atypical examples give the widest spread between methods. TAR, built specifically for tamper resistance, adds nothing over the first-phase safeguard in their setting. On the weight side, methods with larger L2 distance to the pretrained weights resist better, and a loss barrier between pretrained and unlearned weights along the interpolation path tracks resistance; retrained-from-scratch has a large barrier and no connectivity. Weight Distortion and Weight Dist Regulation lead the frontier; CBFT makes a barrier but moves little in weight space and performs in between, which the authors read as evidence that L2 distance is the better predictor. Two honest qualifications carried in the abstract and body: the frontier trades against test accuracy — the most tamper-resistant arms have the lowest test accuracy — and on ResNet-34 with hyperparameters transferred unchanged from ResNet-18 (about 11M to 21M parameters), Weight Dist Regulation loses its resistance. Weight Distortion resists relearning even after a single unlearning epoch (Imagenette: forget accuracy 63% for retrain against 74% and 84% for the two new methods after 0 relearn; SCRUB at 93%). Membership-inference accuracy sits near chance for nearly every method except random relabeling.

## Limits and open problems they name

The tamper-resistance / utility Pareto frontier is their own stated open problem — they hope later work improves it. They flag that the finding is established on small vision classifiers with example-level forgetting, that quantization-based recovery in their models barely occurs (so the mechanism is not obviously shared with the LLM results they cite, which they attribute speculatively to the very large parameter counts of those models), and that layer-wise analysis of which weights matter for resistance is untouched. Observed but not resolved: CBFT is poor on CIFAR-10 and markedly resistant on CIFAR-100; the relative ranking shifts with model size when hyperparameters are not retuned; relearning is stronger when the "reminder" data were themselves in the training set than when a held-out set of the same distribution is used; single-method attention to the atypical forget set means all headline separation lives in one condition of one dataset; and there are no confidence intervals printed anywhere in the text — every cell is a trajectory or an average with no variance reported.

## What we take from it

- **A method: fine-tune on benign in-distribution data as the standard recovery probe for any claim that an internal state was removed.** It needs no gradient projection, no inverse, no optimisation against the model — one short fine-tuning run on data the model already saw. Their result that training-set data recovers more than held-out data of the same distribution is the detail to copy: the probe data should be drawn from what the model was actually trained on.
- **A warning that lands directly on the folder's readout rules:** their membership-inference table is nearly chance for exactly the methods that turn out to be fully recoverable. A leakage metric with the right null can still be blind to a state that is present and merely dormant. Every verdict we issue from one metric needs to name which channel that metric observes.
- **A framing for our own model: "dormant" and "deleted" need a behavioural definition, and the cheapest one is an attack with a reference.** The retrained model is not only a target; it is the null for the attack — the amount of recovery that a model which genuinely never saw the data would show. Failure is defined as beating that reference after the same fine-tuning.
- **A measure we can port: L2 distance in weight space, plus the presence of a loss barrier along the interpolation, as two cheap white-box features that predict a downstream behaviour** — with their own caveat intact: the prediction is ordinal, not calibrated, and it collapses on a bigger model with transferred hyperparameters.
- **A dataset-design lesson:** typicality of the forgotten items determines whether the whole experiment says anything. Their typical and random conditions are uniform no-op conditions; only the atypical forget set separates methods. Any probe of ours that uses random examples is measuring generalisation, not memory.

## Tags

unit: weights
form: description
method: weight-analysis | training-interpretable
eval: ablation | downstream-task
goal: understanding | safety/auditing
