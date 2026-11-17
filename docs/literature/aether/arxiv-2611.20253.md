# Integrating Latent Block Models into Variational Autoencoders for Disentangled Representation Learning

- **Key:** arxiv-2611.20253
- **Authors / organisation:** Anasse Essalih (VEDECOM Institute; DAVID Lab, UVSQ / Paris-Saclay), Reda Khoufache (TSM Unit, UVSQ / Paris-Saclay), Laurent Durville, Mohamed-Cherif Rahal, Walid Kheriji, Fouad Hadj Selem (VEDECOM Institute), Mustapha Lebbah (DAVID Lab), Hanene Azzag (LIPN, CNRS / Sorbonne Paris Nord)
- **Date:** 2026-11-15
- **Link:** https://arxiv.org/abs/2611.20253
- **Kind:** paper
- **Read from:** full text (HTML build, ~22K clean chars: abstract, §§1–5, Table 1, Figures 1–2 via captions only, references)
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Disentanglement work in VAEs usually assumes one latent dimension per factor of variation, and subspace methods that break the assumption require the number and size of the subspaces as a user input. The authors want the axis structure itself to come out of the data: given only an upper bound on the number of groups, the model should decide which latent coordinates belong together. They get there by replacing the standard Gaussian prior on the latent code with a latent block model, a co-clustering prior that partitions observations into rows and latent dimensions into columns with one Gaussian per cell. The larger goal is a representation whose grouping is a learned parameter rather than a design choice, so that "which coordinates carry this factor" is read off the fitted model.

## What they studied
Two small image datasets and the latent code matrix of a VAE. MNIST is used for the row partition, judged against digit labels. Shapes3D is used for disentanglement, judged with the MIG, SAP and DCI (disentanglement / completeness / informativeness) scores. Unit of analysis is the **latent coordinate grouping** — a column cluster of the code matrix, treated as the carrier of one factor. No transformer or language model is involved; the encoder/decoder are convolutional and small.

## How they did it
A per-observation diagonal Gaussian encoder supplies means and variances for a code matrix Z of n rows and p coordinates. The prior over Z is an LBM: each row gets a categorical cluster label, each coordinate gets a categorical block label, and each entry is Gaussian around a block-specific mean with a block-specific variance. The approximate posterior is mean-field over codes, row labels and column labels. The objective is the ELBO with the usual reconstruction term, a per-entry cross-entropy between the encoder's code distribution and the block Gaussian it is assigned to, and two KL terms against the marginal cluster proportions. Row responsibilities are local to one observation; column responsibilities depend on the whole dataset, so they are updated from memoised per-batch sufficient statistics (counts and weighted first and second moments, indexed by coordinate and row cluster) that are swapped in as each batch is re-encoded, with the block parameters refreshed once per epoch. Notably the column partition is not part of the gradient path — it is a variational parameter updated in closed form. Training runs 200 epochs with the first 30 as an ordinary VAE before the block prior switches on, Adam at 5e-4, batch 256, ten seeds, one H100.

## What counts as an explanation here
A **partition**: a learned grouping of latent coordinates into blocks, presented as a claim about which coordinates jointly encode which factor. It is partial by construction — the block assignment names the carrier set and the block size, and says nothing about the mechanism by which the decoder reads those coordinates. The supporting read-out is a traversal figure: perturb one coordinate at a time and show which visual attribute moves, grouped by learned block. There is no dictionary, no circuit, and no causal intervention beyond traversal.

## How they judged success
Clustering: adjusted Rand index, normalised mutual information and accuracy of the row partition against MNIST digit labels, mean ± std over ten seeds at the final epoch. Disentanglement: MIG, SAP and the three DCI components on Shapes3D, compared against β-VAE, β-TCVAE, and each with an added HFS regulariser, all sharing the same convolutional encoder and decoder so the difference is attributable to prior and objective. Baselines use a ten-coordinate latent with β = 6 (γ = 10 for HFS); the proposed model uses fifteen coordinates so that the column partition is not forced trivial by p being close to L.

## What they found
On Shapes3D the model takes the best mean SAP (0.129 ± 0.027) and best DCI-D (0.899 ± 0.107), reaches 0.990 ± 0.018 informativeness (just behind the HFS variant at 0.991 ± 0.007) and 0.780 ± 0.095 completeness (third, behind 0.788 and 0.779 just above it), but MIG is 0.452 ± 0.124 against 0.616 ± 0.181 for β-VAE with HFS and 0.534 ± 0.204 for β-TCVAE with HFS. The authors read the split as the signature of block encoding rather than partial failure: DCI-D only asks that a coordinate inform one factor, which a block satisfies, while MIG rewards concentration in a single coordinate, which a block structurally cannot. On MNIST the row partition reaches ARI 67.4 ± 10.3, NMI 78.7 ± 6.2 and accuracy 75.3 ± 8.3, which they state plainly is below dedicated deep-clustering models such as VaDE. Block inspection is qualitative: one block drives object orientation only, a two-coordinate block drives wall hue, one block carries object scale, one collects coordinates the decoder does not use, and at least one block still mixes object hue, shape and floor hue.

## Limits and open problems they name
All blocks share a single observation partition, though different factors plausibly induce different row clusters; the named next step is a per-block observation partition, i.e. multiview clustering. The clustering result is below dedicated clustering methods. Disentanglement is competitive rather than ahead, and relies on no extra regulariser. The block diagnostics are figures, not scores, and no quantitative per-block purity or utilisation is reported. Two settings are chosen by the authors as sensitivities rather than derived: the number of coordinates, which must exceed the block ceiling for the column partition to be non-trivial, and the clustering ceiling L, which is set to the number of ground-truth factors.

## What we take from it
- **Method, transferable directly to our read-out stage:** fit the grouping of coordinates as a variational parameter in closed form beside a gradient-trained model, using memoised sufficient statistics so the global partition update does not require re-encoding the dataset each step. This is a cheaper route to "which directions belong together" than clustering a post-hoc covariance of activations, and it makes the grouping a parameter of the model rather than a second analysis with its own hyperparameters.
- **Measure:** MIG and DCI-D disagree in *sign* here for structural reasons — a metric that rewards one-coordinate concentration will always penalise a block encoder, and a metric that only asks each coordinate to inform one factor will always flatter it. Any of our claims that a subspace read-out "works" must name which of these it is the same family as, because the choice of metric is a choice of what shape of representation counts as explained.
- **Warning, and the sharpest thing in the paper:** an unused block. One learned block collects coordinates the decoder ignores, and the paper reports it as an observation rather than a failure. A block-structured partition can spend capacity on nothing, and a grouping learned as a parameter has no incentive to avoid that, because a block with no downstream effect costs nothing in the ELBO. Any decomposition we report should print per-component utilisation, not only per-component interpretability.
- **Warning on pollution of the fit:** the row partition is shared across all blocks, so a factor that induces its own observation clusters will be pressed into a grouping chosen by other factors — the authors identify this and defer it. In our terms, a joint partition over components and inputs is a constraint that forces every block to explain the same partition of the same data.
- **Shortcut worth noting, not taking:** the thirty-epoch plain-VAE warm-up before the structured prior activates, with the t-SNE figure drawn from a different schedule entirely (five pretraining epochs). The qualitative block behaviour is shown under a setting the headline numbers do not use.

## Tags
unit: direction, layer
form: dictionary, causal-abstraction
method: decomposition, other
eval: downstream-task, benchmark, ablation
goal: understanding
