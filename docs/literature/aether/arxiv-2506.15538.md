# Capturing Polysemanticity with PRISM: A Multi-Concept Feature Description Framework

- **Key:** arxiv-2506.15538
- **Authors / organisation:** Laura Kopf, Nils Feldhus, Kirill Bykov, Philine Lou Bommer, Anna Hedström, Marina M.-C. Höhne, Oliver Eberle; Technische Universität Berlin and BIFOLD, with the Munich Center for Machine Learning / TUM, ATB Potsdam, ETH AI Center and Universität Potsdam
- **Date:** 2025-06 (v1 2025-06-18; read v4, dated 2025-11-12)
- **Link:** https://arxiv.org/abs/2506.15538
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Automated feature description in language models has settled on one label per feature, which encodes a
prior that each neuron or dictionary entry means one thing. The authors take the superposition literature
at its word — features commonly respond to several unrelated patterns — and argue that the description
method, not the model, is what throws that information away. Their goal is a description pipeline that
returns a *set* of labels per feature, plus a numeric way to say how many distinct things a feature tracks.
The larger purpose is a vocabulary rich enough to compare features across models rather than a pile of
single-sentence captions.

## What they studied

Four models across two feature types: GPT-2 XL and Llama 3.1 8B Instruct at MLP neurons, and GPT-2 Small
and Gemma Scope at residual-stream SAE features (GPT-2 Small SAE at width 32k; Gemma Scope at width 16).
Three layers per model — early, middle, late (GPT-2 XL layers 0/20/40, Llama 0/20/30, GPT-2 Small 0/5/10,
Gemma 0/10/20) — and 60 randomly chosen features per model (59 for GPT-2 Small, where only that many have
reference annotations), 20 per layer. The unit of analysis is the single neuron or single SAE feature, and
the object being explained is a set of natural-language concepts rather than one.

## How they did it

Three steps per feature, run over the English training split of C4 with excerpts fixed at 512 tokens.
First, percentile sampling: rather than taking the top-k activating excerpts, they keep every excerpt
whose mean activation falls in the 99th–100th percentile and sample across that band in very fine bins,
producing 1,000 high-activation excerpts and so a spread of *kinds* of strong activation instead of 1,000
copies of the peak pattern. Second, concept clustering: those excerpts are embedded with a sentence
transformer (gte-Qwen2-1.5B-instruct) and partitioned by k-means into k = 5 clusters. Third, cluster
labelling: for each cluster the twenty highest-activating excerpts are shown to an LLM (Gemini 1.5 Pro in
the original run, later Gemini 2.0 Flash Lite) with the peak token spans bracketed, and the model writes
one short concept label per cluster. The output is five labels per feature. They then benchmark against
MaxAct, GPT-Explain, Transluce-Explain, Neuronpedia and the Output-Centric ensemble, and run ablations on
cluster count and on the language model used at both the generation and the evaluation stage.

## What counts as an explanation here

A set of short natural-language concept labels, one per activation cluster, meant to be jointly exhaustive
over the feature's observed behaviour — explicitly partial in each element and capped at five elements. It
is a description, not a surrogate: nothing here predicts an activation from text. Two derived quantities
are offered as the explanation's *quality*, not as additional explanation: a description score saying how
well each label separates concept text from control text on that feature, and a polysemanticity score
saying how mutually dissimilar the five labels are.

## How they judged success

Two activation-side metrics per label. AUROC: the feature's mean-pooled activations on 10 LLM-generated
concept-specific samples must separate from those on 1,000 random Cosmopedia entries, with 95 per cent
confidence intervals reported. MAD: the mean activation difference between concept and control samples,
normalised by the control standard deviation, reported as the percentage of features with a positive
value. Both are computed per description, and the paper reports a *max* variant (best-matching label per
feature) beside a *mean* variant (average over the five), which is the honest choice, since max rewards
having many labels. Controls are two randomisations on GPT-2 XL: randomising the sentences inside clusters,
and reassigning real descriptions to the wrong features. Polysemanticity is scored as the mean pairwise
cosine similarity of the sentence embeddings of the five labels — low similarity means polysemantic — and is
validated against seven human annotators who wrote their own labels for 8 features and rated their mutual
similarity on a 0–1 scale. One output-side check re-runs the FADE faithfulness metric on GPT-2 XL.

## What they found

On GPT-2 XL neurons PRISM (max) reaches AUROC 0.85 and 91.67 per cent positive MAD against GPT-Explain's
0.64 and 65 per cent; on Llama 3.1 8B it reaches 0.71 and 81.67 against Transluce-Explain's 0.59 and 63.33.
The advantage is uneven and the paper says so: on GPT-2 Small SAE features PRISM (max) is 0.57 against
Output-Centric's 0.57 with a lower positive-MAD share, and on Gemma Scope Neuronpedia beats it outright on
both metrics (0.62 and 63.33 against 0.54 and 38.33). Neuron features are more reliably describable than
SAE features on their numbers, which they read as further evidence against SAE-based interpretability
claims. The mean variants are consistently below the baselines' single labels on the SAE models, so the
multi-label gain is a best-case statistic. Ablating the cluster count shows the trade cleanly: raising k
from 5 to 10 lifts max AUROC to 0.88 and positive MAD to 93.33 per cent while mean AUROC falls to 0.61 —
more clusters sharpen the best label and fragment the rest. Both randomisations drop the score hard (random
sentences 0.68 max-AUROC and 65 per cent MAD; random descriptions 0.65 and 66.67), and swapping the
generator model leaves the ranking intact. Polysemanticity scores align with the human ratings on their
worked examples (a polysemantic GPT-2 XL neuron at 0.40 human against 0.38 model; two monosemantic Gemma
features at 0.90 and 0.80), and the authors' own reading of Figure 1 is that a neuron previously published
as possibly monosemantic splits into several unrelated concepts under this pipeline.

## Limits and open problems they name

Descriptions are bounded by what natural language can say, so graph-like or algorithmic mechanisms are out
of reach. A fixed cluster count guarantees the label set is a subset, not a cover — there is no claim that
the most salient concept is among the five. MAD is outlier-sensitive on heavy-tailed activations, which is
why AUROC is paired with it. Sampling only maximally activating corpus excerpts restricts interpretation to
concepts the corpus actually exhibits. They also flag, without running it, that simulation-style evaluation
rests on one model's ability to predict another's activations, and call for standardised human-evaluation
protocols.

## What we take from it

A method and a warning. The method: percentile-band sampling instead of top-k is a cheap way to get
*heterogeneity* into the evidence set before any model is asked for a label, and it is the step that makes
a multi-label answer possible at all. The warning is the max/mean split — the headline number is a
best-of-five statistic, and the same table shows the average label losing to cheap baselines on the SAE
side. Also useful: their polysemanticity score is a property of the *label set*, not of the feature, so it
inherits every weakness of the labeller, and the human study is eight features with seven raters, i.e. a
demonstration rather than a measurement. And the description benchmark they build is activation-side only,
with the one output-side check (FADE faithfulness on GPT-2 XL) relegated to an appendix figure.

## Tags

unit: neuron | feature
form: description | dictionary
method: other
eval: benchmark | human-rating | ablation
goal: understanding | evaluation-of-methods
