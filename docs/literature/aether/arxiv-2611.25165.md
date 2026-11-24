# Separating Interpretability and Reconstruction in Sparse Autoencoders

- **Key:** arxiv-2611.25165
- **Authors / organisation:** Anh Tien Nguyen, Dimitris Samaras (Stony Brook University); Mahdi S. Hosseini (Concordia University, Mila)
- **Date:** 2026-11-21 (v2)
- **Link:** https://arxiv.org/abs/2611.25165
- **Kind:** paper
- **Read from:** full text (arXiv HTML v2, ~146k chars after whitespace collapse; abstract, §1-§6, and the supplement's metric definitions — MonoScore, binary purity, KNN/LP, ΔCE/KL — plus Tables 1, 2 and the supplementary tables referenced by number; figures 1-19 not inspected as images, only their captions)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

A standard sparse autoencoder asks one sparse code to do two jobs at once: reconstruct the activation faithfully and expose features that a human can read. The authors argue these objectives are not aligned, because reconstruction rewards any information that helps rebuild the vector, including information that carries no concept — a nuisance factor, a magnitude artefact, a dataset fingerprint. Their diagnosable consequence is that the dictionary spends capacity on directions that exist only to reduce squared error, which then mix distinct inputs into the same latent. The proposal is to stop asking one representation to do both jobs: keep a sparse branch for the readout and add a second, deliberately uninterpretable branch that absorbs whatever is left over. This is a decomposition-design paper aimed at the dictionary arm of the field, and it matters to us as a caution about what a reconstruction metric is actually measuring.

## What they studied

Frozen representations from three vision encoders (CLIP, SigLIP2, ViT-L/16) over ImageNet-1K and OpenImagesV7, and Pythia-12-layer activations over OpenWebText and a GitHub code corpus. Six SAE variants are covered — TopK, BatchTopK, JumpReLU, GatedSAE, MatryoshkaSAE and CB-SAE — each paired with its "Dual" counterpart, so the comparison is a paired AB test across 18 vision settings and language settings rather than a single architecture. Unit of analysis is the SAE latent (the dictionary atom), plus the branch as a second unit: the claim is about which branch carries which kind of information.

## How they did it

Motivating measurement first. They take a trained BatchTopK SAE on CLIP/ImageNet embeddings, split each reconstruction into the sparse part and the residual, and run nearest-neighbour retrieval with each part as the query, scoring the fraction of the 50 retrieved neighbours that share the query's class. The sparse reconstruction retrieves class-consistent neighbours and the residual does not, which is the evidence that semantic content concentrates in the sparse part.

Architecture. The sparse branch is any standard SAE, producing the code and a sparse reconstruction. The residual branch takes the leftover vector (input minus sparse reconstruction) through a projector and a nonlinearity and predicts it back. The final reconstruction is the sum of the two branches, so the L2 reconstruction loss is applied to the *sum*, not to the sparse branch alone.

The load-bearing design detail is that this sum creates a shortcut: the residual branch can reach zero loss by learning the identity on the residual, in which case the sparse branch is never pressured to carry anything meaningful. Their fix is a decorrelation penalty on the norm of the residual prediction projected onto the decoder directions *active for that sample*. They give the gradient argument explicitly: at the identity solution the reconstruction gradient vanishes but the decorrelation gradient is nonzero whenever the residual has a component along those active directions, so the identity is not a stationary point. Training is two-stage — warm up the sparse branch as an ordinary SAE, then add the residual branch and train jointly with the decorrelation term — and they sweep warm-up length.

Controls. Three matter for how much the result is worth. (1) A parameter-matched baseline: each standard SAE is widened until its parameter count equals the DualSAE's, so a gain cannot be explained by added capacity. (2) A sequential baseline: freeze a trained SAE and train only the residual branch, which restores reconstruction without touching the code. (3) Ablations on the decorrelation loss, on the expansion ratio (4, 8, 16), and on warm-up length. Everything is averaged over three seeds.

Metrics. Reconstruction via R², KNN accuracy and linear-probe accuracy computed on reconstructed embeddings; interpretability via MonoScore and a label-only binary purity; language-side preservation via ΔCE and KL after splicing the reconstruction back into block 12 of Pythia and running the rest of the model. They also run three downstream applications (topic modelling with an LLM-judge intruder test, VLM hypothesis generation, and positive-sentiment steering) so that "more interpretable" is cashed out in something other than a score.

## What counts as an explanation here

A better-shaped dictionary, not an explanation of a network. The deliverable is a decomposition in which the sparse latents are claimed to be nearer to one-concept-each and the residual branch is explicitly *not* offered as an explanation. Completeness is traded away on purpose: the sparse reconstruction is allowed to be incomplete, and the paper's own framing is that the residual carries "non-semantic information". So the form is a partition of the activation into an interpreted part and a discarded part, with the discarded part given a name and a job rather than a meaning. The predictive content is narrow but real: the sparse code alone should predict the class-relevant part of the input, and the paper tests that by classifying from each branch separately.

## How they judged success

Headline numbers are balanced improvements across paired settings: roughly 29% mean relative gain in reconstruction fidelity and 34% in monosemanticity across the encoder-SAE configurations. On ImageNet-1K specifically they report average gains of 27.1% in R², 1.8% in KNN accuracy and 1.2% in linear-probe accuracy, with +31.2% MonoScore and +113.8% class purity. The parameter-matched table is 18 paired rows with per-metric differences, all favouring DualSAE (most small, a few large: JumpReLU rows show R² differences from -0.04 to -0.25). Against the sequential frozen-SAE baseline, the residual branch alone lifts R² by 15.4-24.8% while leaving interpretability untouched, and DualSAE adds 10.5-43.3% MonoScore on top of comparable or better R² — that contrast is the paper's central argument that joint training, not extra capacity, is doing the work. In the language setting every variant improves on all four metrics on both corpora, with the contrast starkest for BatchTopK on code: ΔCE falls from 2.41 to 1.76 and KL from 2.41 to 0.77. Downstream, topic diversity improves in 89 of 90 configurations and intruder accuracy in 74 of 90; hypothesis generation yields more significant hypotheses in 17 of 18 settings; the steering example is qualitative.

## Limits and open problems they name

Only frozen representations of pretrained encoders and one language model — no end-to-end trained or multimodal reasoning models. Three new hyperparameters (decorrelation strength, residual-branch design, warm-up length) that the user must now choose. They recommend more modalities and adaptive schedules. What they do *not* name, but the text shows: the residual branch is a learned predictor, so the "non-semantic" label is an assumption about what it absorbs rather than a measurement of it, and the branch-decomposition evidence is qualitative t-SNE plus per-branch KNN. MonoScore also improves on OpenImagesV7 only unevenly, by their own account, and their absolute MonoScore values are not comparable to the original paper's because they compute the similarity with the same encoder being decomposed rather than an independent one — a change that removes a second model but also removes the check that the latent's coherence is not an artefact of that encoder's own geometry.

## What we take from it

A method, a control, and a warning. The method: when a dictionary is asked to reconstruct, the residual is a place to put things you are choosing not to explain, and naming that place makes the incompleteness of the explanation explicit instead of hiding it inside mixed latents. For us the analogue is direct — a white-box model that predicts part of a network's behaviour should say which variance it has declared out of scope, and test that the declared-out-of-scope part is genuinely not predictable from the interpreted part.

The control worth copying is the *sequential versus joint* baseline. Freezing the interpreted component, letting the second component absorb the rest, and checking whether interpretability moved is a cheap way to separate "this component got better" from "this component got more capacity". Every decomposition claim we make should carry that pair.

The warning is about MonoScore-style metrics. Their construct is activation-weighted pairwise cosine similarity between strongly co-activating inputs, computed *in the space the SAE was trained on*; the authors acknowledge that doing so can reward latents whose members are close in that embedding space for reasons unrelated to semantics, which is why they add an encoder-free purity measure. That is the right instinct, and it generalises: any coherence score computed in the same space the model was fit to compress will be inflated by the model's own metric, so we should pair every such score with a label-based or otherwise independent measure — which is the same lesson this folder keeps reaching from a different direction.

## Tags

unit: feature, layer
form: dictionary, decomposition
method: SAE, other
eval: downstream-task, ablation, benchmark
goal: understanding, evaluation-of-methods
