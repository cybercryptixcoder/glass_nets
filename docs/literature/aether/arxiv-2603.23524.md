# Navigating the Concept Space of Language Models (Concept Explorer)

- **Key:** arxiv-2603.23524
- **Authors / organisation:** Wilson E. Marcílio-Jr (Adaption Labs, Curitiba, Brazil); Danilo Medeiros Eler (São Paulo State University / UNESP, Presidente Prudente, Brazil)
- **Date:** 2026-03
- **Link:** https://arxiv.org/abs/2603.23524
- **Kind:** paper (systems / tool paper with a single case study)
- **Read from:** full text
- **Relevance to us:** 1 (background)

## What they wanted, and why
Sparse autoencoders now emit tens of thousands of latents per layer, and every established way of looking at them is local: read one feature's top activations, ask a model to name it, search for features near a known one. That works for checking a hypothesis you already have and fails at the thing the authors say is now the real bottleneck — feature *discovery*, and specifically the ability to reason about the whole dictionary at once: which concept families dominate, which features are rare or isolated, how concept regions overlap. Their goal is an organisational layer over the outputs of existing SAEs rather than a better SAE. The paper says this outright: no new training objective, no new architecture, only the downstream problem of arranging what the SAEs already produced.

## What they studied
One model, one layer, one dictionary: a SmolLM2 MLP SAE from EleutherAI (the repository named in the references is the 135M model, 64x expansion), at layer 27, over 36,864 features, with top-16 activation contexts per feature drawn from a corpus of 1.7M sentences. The unit of analysis is the *explanation text* of a feature, not the feature's direction or activations — each feature gets a short description from GPT-5-mini, the descriptions are embedded with nomic-embed-text-v1.5, and the map is built over those embeddings. The authors state that the interface needs only four metadata fields per feature: the explanation text, its embedding, the top-k contexts, and the feature id.

## How they did it
The system, Concept Explorer, is a client over a hierarchy of two-dimensional projections produced by HUMAP (the first author's earlier hierarchical UMAP). HUMAP repeatedly (a) embeds the current subset with UMAP, (b) turns the kernel similarities into a Markov transition matrix, (c) picks landmarks by random-walk visit frequency to form the next coarser subset, and (d) keeps, for each landmark, the set of points in its region of influence so that a region can be reprojected one level down. The result is a nested sequence of levels from coarse to fine. The interface shows a level as a canvas with point size encoding influence-region size, a panel with the selected feature's explanation and top-k contexts, and controls for annotating, merging and reprojecting. Three design questions drive the layout: what the main concepts are, which concepts are rarest, and how concepts influence each other. The case study is purely qualitative — analysts explore the map, annotate coarse categories by hand, and drill into level-1 regions to see substructure; there is no measurement experiment anywhere in the paper.

## What counts as an explanation here
A hierarchy of neighbourhoods whose members are label texts. An explanation, for this paper, is the combination of a short natural-language description of a feature plus a position in a multi-resolution map; understanding a dictionary means being able to navigate to a region and read off what lives there. It is explicitly partial and organisational — the map says which concepts sit near each other and which are isolated, and makes no claim about causal role, mechanism or prediction.

## How they judged success
By demonstration on one model and by inspection. The evidence in the paper is three qualitative exhibits: a punctuation region at the second level with a visible spread of distinct sentence-boundary features; a rare feature inside that region specialising on em-dashes and list markers; and a level-1 landmark with twenty features in its influence region which, reprojected to level 0, splits into programming tokens and glyphs, placeholder markers including one for a Java method annotation, and enumeration separators. A table of twenty-odd feature ids with their verbatim GPT-generated explanations is printed as the appendix for the reader to judge. There is no baseline (no comparison against Neuronpedia, against k-means on the same embeddings, or against a flat UMAP), no user study, no task accuracy, no stability or coverage measure for the labels, and no quantitative criterion for "coherent". The code and application are promised after double-blind review and are not available in the read version.

## What they found
Nothing measured. The claim is that the multi-resolution map surfaces structure the authors' existing workflows did not: coherent concept families, meaningful subclusters within them, and rare concepts identifiable by small influence-region size or by distance from dense groups. Their own examples are the substance of the finding, and the strongest of them is a local one — several dozen features that plainly should share a region and a mechanism (sentence-final punctuation) but whose descriptions differ enough to be worth annotating apart. The appendix actually documents the thing the paper does not analyse: the "Punctuations" class contains around twenty features whose descriptions are near-paraphrases of one another, and blocks of features whose description is "token wrapped in double angle brackets" with the lexical content explicitly declared irrelevant — i.e. the map's resolution is bounded by how much the labeller varies its prose on identical behaviour.

## Limits and open problems they name
The paper is candid about being a system demonstration: one model, one layer, one dictionary, no evaluation beyond a use case, and no comparison to prior tooling. It also names a limitation of the interaction model — a single-level embedding of the whole dictionary produces visual clutter at this scale, which is the problem the hierarchy is meant to solve, but the paper offers no clutter metric. Not named by the authors but visible in the read text: the grouping is over description embeddings, so it inherits the labeller's biases and any two features with similar descriptions are placed together regardless of what they do; and the whole chain is a single labeller (GPT-5-mini) with a single prompt template, so nothing separates "these features are related" from "these descriptions are worded alike".

## What we take from it
Mostly a warning and a lookup structure, not a method. For our goal of a predictive white-box model, the transferable part is the *indexing* idea: a dictionary of thousands of entries is unusable unless there is a nested structure over it, and building that structure over item embeddings with progressive reprojection of regions is cheap and costs no extra forward passes. The part to copy with care is the fact that the hierarchy's edges are label-similarity edges, not behaviour edges. Our own version should be indexed on something that touches the model — decoder-direction cosine, activation co-occurrence on held-out text, or a shared-behaviour signature — and the label embedding should be one view among several, because the appendix here shows descriptions collapsing onto identical prose for features that are plainly distinct in what triggers them. And any "concept family" we report should carry a number the paper never provides: whether the cluster is more coherent than the same number of items grouped at random under the same embedding.

## Tags
unit: feature
form: description / dictionary
method: SAE / other
eval: benchmark (none: qualitative case study; read as no quantitative evaluation)
goal: understanding
