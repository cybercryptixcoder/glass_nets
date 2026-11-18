# Learning Cross-Domain Media Similarity from Amazon Reviews

- **Key:** arxiv-2611.22756
- **Authors / organisation:** Gabrielly Xavier dos Santos, Lucas Dias Cler, Luciana Fujii Pontello, Olga Goussevskaia (Computer Science Department, Universidade Federal de Minas Gerais, Brazil)
- **Date:** 2026-11-17
- **Link:** https://arxiv.org/abs/2611.22756
- **Kind:** paper
- **Read from:** full text (HTML v1, abstract through §7 and references; figures via captions only)
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Rated items are the input to almost every recommender, and the healthy ratings histories that make those systems work are held privately by platforms. Public review data exists, but the relational signal that a recommender really wants — which items were consumed by the *same* people — is not published. The authors reframe that asymmetry as a supervised-learning problem rather than a data-access problem: take public review text and review timestamps, predict the co-occurrence the platform would have measured, and check how far the prediction travels across two media categories that share nothing but their users. The larger goal is a similarity function an academic or independent developer can build from an open dataset and use where the proprietary log is absent.

## What they studied
Amazon Reviews 2023 (Hou et al.), restricted to two categories — books and movies/TV. Raw: 4.45M book items / 29.1M book reviews and 0.75M movie items / 17.2M movie reviews. After a strict 10-core filter (an item must keep at least ten reviews): 502,048 book items, 20.0M book reviews, 186,296 movie items, 15.7M movie reviews; 231,346 distinct items carry at least one retained pair. Unit of analysis: an **item pair**, labelled with a cosine over the two items' shared-user rating vectors.

## How they did it
Each item is described by four feature blocks: a 768-dimensional SBERT embedding of the ten most relevant TF-IDF keywords drawn from the item's concatenated review titles and texts (the paper's §5.2.1 says "top 50" keywords, remaining ambiguous about which count the reported 775-dimension vector actually uses); the Jaccard overlap of the two items' sets of active review years; the absolute difference in review counts; and the absolute difference in mean star rating. The blocks are concatenated and fed to XGBoost regressors with R²/RMSE/MAE scored against the cosine target. Pair sets are split by three regimes — SAME (intra-domain), CROSS (book–movie) and GLOBAL (both) — and four train→test configurations are run: CROSS→CROSS, GLOBAL→GLOBAL, GLOBAL→CROSS, and the zero-shot SAME→CROSS arms, with distinct co-occurrence thresholds for same- and cross-domain pairs to keep the cross-domain pair set from vanishing. A dummy mean predictor is the stated floor baseline. Attribution is done twice over: SHAP values on the fitted models, and a block-level ablation that removes one feature group at a time.

## What counts as an explanation here
Two things, neither of them mechanistic. First, a **predictive surrogate** — a gradient-boosted regressor over public features that stands in for the private co-occurrence signal. Second, an **attribution summary** of that surrogate: which of the four feature blocks the fitted model draws on, expressed as a percentage of total SHAP mass. Both are explanations of the *estimated* relationship, not of any network's computation; the paper never inspects a model's weights or activations beyond a tree ensemble's split structure through SHAP.

## How they judged success
R², RMSE and MAE per configuration; block-level ablation deltas; relative SHAP importance shares per feature group, compared across training regimes and across the two intra-domain transfer directions. There is no held-out-per-domain cross-validation beyond an 80/20 split, and the only baseline reported is the dummy mean predictor.

## What they found
Cross-domain R² falls monotonically as the train–test gap widens: CROSS→CROSS 0.96, GLOBAL→GLOBAL 0.85, GLOBAL→CROSS 0.32, and SAME→CROSS **−4.00** (RMSE 0.038). The headline is the attribution flip. Trained on same-domain pairs, the model puts **68%** of its SHAP mass on text semantics and 28% on the review-count (popularity) feature; trained on cross-domain pairs alone, it moves to **69% popularity / 29% semantics**. Removing the popularity block from the global model costs 0.85 → 0.65 on the general test and 0.32 → −0.99 on cross; removing the SBERT block costs 0.85 → 0.77 and 0.32 → −1.77. Removing Jaccard or ratings moves nothing. On intra-domain transfer, books→movies reaches R² 0.38 against movies→books 0.76 (RMSE 0.82 movie→movie, 0.85 book→book).

## Limits and open problems they name
The 10-core filter sacrifices statistical reach, and the cross-domain pairs are thin by construction — the authors admit this is what stops the model learning subtle semantic bridges across very different categories. Only two categories are studied, all interactions are positive-only, and the sole baseline is an uninformative mean predictor. Future work names music, a hierarchy/brand feature block, and a Pearson-based target.

## What we take from it
Three things, all guards rather than tools. (1) **A negative R² is the honest number for a real distribution shift**; this folder rarely prints one, and −4.00 for a zero-shot transfer with the same features and the same learner is the cleanest available demonstration that a model trained on the easy stratum can be worse than the mean on the hard one, and that no single summary across strata is meaningful. (2) **The attribution flip is a result about the *stratum*, not about the feature set**: the same 775 dimensions explain a same-domain task mostly through semantics and a cross-domain task mostly through popularity, so any claim of the form "X is the feature that drives this model" owes a separate attribution per stratum. That is directly relevant to our read-outs, where a probe fitted on easy pairs and a probe fitted on hard pairs will silently rank directions differently. (3) **A cautionary example on metric plumbing**: the R² column of the ablation table moves in the direction of the popularity block while the global-test R² and RMSE are consistent, the keyword count is stated as 50 in one section and implicit as 10 in another, and the MSSE/RMSE column labels do not match their contents — reported numbers in this paper are worth citing only alongside their table and section.

## Tags
unit: whole-model
form: replacement-model | graph
method: probing | other
eval: downstream-task | prediction/simulation | ablation | benchmark
goal: understanding | evaluation-of-methods | science
