# The Confidence Manifold: Geometric Structure of Correctness Representations in Language Models

- **Key:** arxiv-2602.08159
- **Authors / organisation:** Seonglae Cho, Zekun Wu, Kleyton Da Costa, Adriano Koshiyama; Holistic AI and University College London
- **Date:** 2026-07-29 (arXiv v2)
- **Link:** https://arxiv.org/abs/2602.08159
- **Kind:** paper
- **Read from:** full text (HTML v2, main body plus Appendices A-P)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The paper asks what shape the internal correctness signal takes, rather than whether it exists. Prior work says true and false statements are linearly separable in the residual stream; the open questions are how many dimensions the separating structure actually needs, how simple the decision rule can be before accuracy collapses, whether that direction survives a change of data distribution, and whether causal interventions confirm the picture or merely correlate with it. The authors frame their contribution as geometry, not a new detector, and insist the useful claim is about structure across many models at once.

Their larger target is an honest account of what "the model knows it is wrong" can mean: if the signal is a low-dimensional mean shift, then a detector is a very cheap object, and if it is not transferable without joint training, then the cheapness is bought with a distributional assumption that has to be named.

## What they studied

Eleven autoregressive decoders across five families, from 124M to 14B parameters: three GPT-2 sizes, Qwen2-1.5B and Qwen2-7B, Qwen2.5-14B, Mistral-7B, Llama-3.2-1B and 3B, Llama-3.1-8B and Gemma-2-2B. The unit of analysis is a supervised subspace of the last-token residual stream at one chosen layer per model, together with the per-layer intrinsic dimension of the representation manifold. Five QA datasets are used (TruthfulQA, SciQ, CommonsenseQA, FEVER, HaluEval), with TruthfulQA as the main stage because it ships matched correct/incorrect answer pairs per question.

## How they did it

Contrastive pairs from TruthfulQA give, for each question, one correct and one incorrect answer sharing the prompt, so topic and difficulty are held fixed. Activations are taken at the last token for every layer. An L2 logistic regression is fit to predict the binary correctness label; its weight vector is called the correctness direction. Dimensionality is measured two ways: an unsupervised Levina-Bickel nearest-neighbour estimate on pooled activations, and a supervised partial-least-squares projection whose number of components is swept from one to thirty-two, with a probe fit inside each subspace. A simple geometric rule (nearest of two class centroids) is compared against the trained probe and against richer classifiers (Mahalanobis, k-nearest neighbours, kernel SVM, nearest convex hull). Three interventions test causality: adding a scaled direction vector at every token of the optimal layer during generation; erasing the direction by projecting activations onto its nullspace; and a distributed-alignment search that learns a rank-k rotation and swaps the projected representation between a correct and an incorrect example, sweeping k explicitly. Transfer is measured by training on one dataset and testing on the others, with a leave-one-out joint-training variant. Controls include a random direction, its orthogonal component, length-only and length-residualised probes, surface-statistic correlations, a paraphrase clustering check, and nested cross-validation to check that layer and dimension selection did not inflate the held-out scores.

## What counts as an explanation here

A geometric description of a decision boundary: a 2-8 dimensional subspace, two class means inside it, and a decision rule that is the perpendicular bisector between those means. It is explicitly partial. The authors say they locate the subspace but cannot say what its individual directions mean, because a covariance-maximising basis is determined only up to a rotation within the subspace, so no single component has a readable semantics. The explanation predicts classification accuracy for any contrastive-style pair at the probed layer, and predicts the direction of a behavioural change under a push along the direction, but it does not predict which tokens or later computations are involved.

## How they judged success

Grouped k-fold cross-validation (five folds grouped by question so paraphrases cannot leak) with AUC as the headline metric; nested cross-validation with an inner three-fold loop for layer and dimension selection; per-fold standard deviations for the dimension sweep; paired t-tests across seeds with Bonferroni correction for the main comparisons. Interventions are scored by the percentage-point change in stated error rate over a coefficient sweep, against random and orthogonal direction controls, with p-values from a two-sample test at the endpoints. Necessity is scored as the AUC after erasure, with chance (about 0.50) as the reference. Causal rank is read from the DAS dimension sweep. Transfer is AUC on held-out datasets, with native refits reported beside the transferred values so that a chance-level transfer can be distinguished from an absent signal. Confound checks are a length-only probe, correlations with norm, mean activation and sparsity, and a paraphrase test.

## What they found

Correctness is decodable almost everywhere: the probe reaches 0.80 to 0.97 AUC across the eleven models, with optimal layers at the final layer for the GPT-2 family and at 46 to 75 percent depth for the larger instruction-tuned models. The dimension sweep peaks between two and eight dimensions for every model and then falls off, in one case from 0.76 at three dimensions to 0.62 at thirty-two, which the authors read as extra directions spending label budget on variance irrelevant to correctness. Centroid distance inside that subspace lands within 0.02 AUC of the trained probe across nine models, and the richer geometric classifiers do not beat it, so the boundary is a mean shift rather than a covariance difference; unsupervised alternatives are far weaker (pooled local-intrinsic-dimension scores 0.51 to 0.58, unsupervised features 0.51 to 0.62, individual pretrained SAE features under 6 percent of variance). Centroid distance is also the more label-efficient rule at realistic budgets. Steering moves the error rate monotonically, 9.1 points on the reference model and 2.8 to 10.0 points across six models, with random and orthogonal directions near baseline. Erasure drops AUC to chance on all five models tested. Only the alignment search bounds rank, peaking at one to five dimensions, and the authors are explicit that steering and erasure are rank-agnostic. Cross-dataset transfer of a single-dataset probe is near random (0.39 to 0.66), and in one case below chance with a near-orthogonal direction to the natively fitted one; joint training restores 0.73 to 0.91 for instruction-tuned models but barely helps the base GPT-2 family. Free-form generation costs a large amount of the signal in the one judge-free comparison available: 0.957 teacher-forced against 0.688 on the model's own generated answers.

## Limits and open problems they name

The authors list several. The subspace is located but its component meanings are not identified. Causal validation is in-distribution only. The transfer recipe's boundary is confounded, because every model below 1B in their grid is a base model and every model at or above 1B is instruction-tuned, so scale and tuning cannot be separated. Everything is teacher-forced, and their attempt at free-form evaluation was blocked by low agreement between their two correctness judges. Correctness is treated as binary, with negation and compositionality unaddressed. Encoders and non-transformer architectures are out of scope. The steering direction is dual-use.

## What we take from it

A framing and a cheap measure. First, the paper separates two numbers the folder has been sloppy about: the decision rule's dimension and the mechanism's causal rank, and it shows that only the intervention that explicitly sweeps rank licenses a rank claim, while steering shows efficacy and erasure shows necessity and neither bounds rank. Any rank statement our white-box model makes needs its own rank-sweeping intervention, not an inference from the other two. Second, the paper is a worked example of measuring the marginal value of a supervised projection over a projection-free baseline: a difference of means in the raw residual stream already reaches 0.72 to 0.91, and the projection adds about 0.04 to 0.08. Third, it provides a concrete instrument for the dimensionality question we keep asking (sweep the subspace size, fit the simplest rule inside it, report degradation beyond the peak and per-fold spread), and it names the reason the extra dimensions hurt rather than being neutral. Warning to carry: a low-rank, easily decodable direction is not thereby transferable, and a below-chance transferred score can mean a present-but-misaligned signal rather than an absent one.

## Tags

unit: direction | layer | feature
form: direction | probe
method: probing | steering | ablation | decomposition | other
eval: ablation | prediction/simulation | benchmark | downstream-task
goal: understanding | evaluation-of-methods
