# Automated Interpretability and Feature Discovery in Language Models with Agents

- **Key:** arxiv-2605.01555
- **Authors / organisation:** Arnau Marin-Llobet, Javier Ferrando; Harvard University
- **Date:** 2026-05-02
- **Link:** https://arxiv.org/abs/2605.01555
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

The complaint is that automated interpretability has been built as a labelling service: an
LLM is shown a feature's top-activating examples and asked for one sentence, and nobody checks
whether that sentence predicts anything. The authors want the same tool to behave instead like an
experimenter — propose several rival hypotheses, design prompts that would separate them, look at
what the model actually does, and revise. Their second complaint is about where the work starts:
almost every method assumes the feature id is already known, which stops being reasonable when a
layer holds thousands of latents and the user's real question is "which features have anything to do
with Spanish". So the target is a single loop that both *finds* candidate features from a plain-language
objective and *sharpens* the story about them, with every step logged. This is the folder's own thesis —
a description should predict — arriving from the natural-language side rather than from the circuit side.

## What they studied

Gemma-2 family models at SAE feature layers 13 and 17 (the Ambrosia/Neuronpedia dictionaries), on
3,260 features split unevenly across three hypothesis-generating backbones (GPT-4o-mini 1,600,
Gemini 2.5 Pro 300, Claude 4 Sonnet 60). Feature discovery is benchmarked at layer 0 on 20 languages
(500 prompts each, 10,000 prompts), then on three non-linguistic categories — Python code, symbolic
math, harmful content — swept across layers 0, 8 and 11 of gemma-2-2b. Generality beyond dictionaries
is shown on a 12-layer GPT-2-style **weight-sparse** transformer from Gao et al. with 1,024 hidden
dimensions and 4,096 MLP neurons per layer, treating raw post-GELU neurons as the candidate features
over 16 Python-syntax categories. The unit of analysis is the SAE latent or the MLP neuron; the
unit of *prediction* is a token span inside a prompt.

## How they did it

Two sub-agents under a supervisor, both with a shared Python sandbox, persistent memory, and
transformer-lens / sae-lens available.

*FeatureExplainer* runs propose–test–revise. Starting from a seed hypothesis (the one-shot autointerp
label, a user hypothesis, or one generated from top activations) it writes 12 predicted-positive and
12 matched-negative prompts, runs them through the target model, records the feature's activation, and
scores the hypothesis on seven metrics: Detection F1 (an LLM classifier, labels hidden, is asked to
sort a shuffled positive/negative mixture — 5 per class — and this is scored as ordinary F1), Fuzzing F1
(randomly highlighted tokens are shown and the judge says whether they carry the activation), Surprisal
AUROC (judge coherence ratings 0–10 for positives minus negatives, rescaled to [0,1] with 0.5 = no
difference), embedding similarity (all-MiniLM-L6-v2 cosine between hypothesis text and each prompt),
Welch t-test p-value on the activation gap, Cohen's d, and a holistic 0–10 LLM-judge coherence score.
Hypotheses are ranked by *average ordinal rank* across the seven, Pareto-dominated candidates are
dropped, and the agent then reads its own weakest metrics and writes a revision that must be
semantically and structurally different from prior attempts. It stops when the top hypothesis is
unchanged, the average rank improves by less than 0.5 rank units, and new negatives stop producing
counterexamples. Budget four iterations. If two semantically distinct hypotheses clear both the rank
and coherence thresholds and neither dominates, the feature is flagged **polysemantic** and both are
reported. A steering test exists for causal validation but is deliberately excluded from the ranking.

*FeatureFinder* runs discovery. Given prompts with weak group labels, it builds an
N×D activation matrix with a fixed token reduction, log-transforms and per-prompt normalises
(sparsity and heavy tails), takes PCA (≤200 components), builds a k-NN graph on the top
min(n_pcs−5, 50) components with k=15, and runs Leiden community detection at resolution 0.5 to get
groups. For each (group, feature) it tests enrichment by Wilcoxon rank-sum against everything outside
the group, corrects with Benjamini–Hochberg, then filters hard: adjusted p < 0.001, composite score > 2.0,
log-fold change > 0.5, Cohen's d > 0.3. Retrieval is intentionally permissive and a separate LLM-judge
screening pass throws most of it away.

## What counts as an explanation here

A natural-language hypothesis about what a feature *responds to*, but scored as a classifier over
prompts rather than appreciated as prose: the primary metric family asks whether reading the sentence
would let you sort prompts into "fires" and "does not fire". It is deliberately partial and deliberately
plural — the system's own success condition includes reporting two incompatible sentences for the same
feature instead of forcing one. Nothing in it predicts an *intervention* outcome; the steering arm is
validation only, ranking is syntactic/semantic/statistical, and no hypothesis is ever used to predict
what happens to the model's output.

## How they judged success

Hypothesis quality: win rate under the 7-metric battery — for each feature, was the top-ranked hypothesis
the one-shot baseline or one of the four refined candidates (a within-feature, same-LLM comparison, so
the generator and the judge can be the same model). Discovery quality: clustering agreement (ARI, NMI,
V-measure, homogeneity, completeness, silhouette) against known language labels, versus K-Means,
agglomerative/Ward, and random assignment, plus an 18-configuration hyperparameter sweep over PCA
dimensions {50, 200}, k {10, 15, 25} and Leiden resolution {0.3, 0.5, 1.0}. Marker validation: t-test of
mean normalised activation on category prompts against random controls, plus a 75%-accuracy check on
LLM-generated test prompts for discovered MLP neurons. Ground truth for the weight-sparse model is
four layer-0 neurons that Gao et al. found by hand. Cost is measured with instrumented API accounting.

## What they found

*Refinement.* A refined hypothesis wins for 63.5 ± 2.6% of features with GPT-4o-mini, 79.2 ± 0.2% with
Gemini 2.5 Pro, and 83.3% with Claude 4 Sonnet (60 features, one layer). Most of the gain arrives in
the first one or two iterations. The distribution across iterations is flat and wide, not a ramp — with
Claude, iteration 1 and iteration 2 each win 16.7% and 21.7% of features, so the loop mostly trades
one constraint for another rather than monotonically climbing. When the baseline survives, the authors'
reading is that the feature is too poorly defined or too polysemantic for any hypothesis to dominate, and
the agent flags that instead of guessing.

*The two case studies are the honest part.* Feature L0_F9004: the Neuronpedia label is "expressions
related to knowledge and understanding, particularly about technology and troubleshooting"; the real
driver is informal French. The refined hypothesis gets Detection F1 1.00, Surprisal AUROC 0.71,
embedding 0.59, **judge coherence 0.40**, p < 0.001, d = 2.53 — and the plain baseline gets Detection
F1 0.89, Surprisal 0.77, embedding 0.52, **judge coherence 0.00**, p = 1.00, d = 0.00. Read the
activation statistics and it is night and day; read the LLM judge alone and the refined answer is
mid, and read the embedding similarity alone and the two are near-twins. Feature L4_F12773: the
polysemantic flag fires because a FeatureFinder-generated hypothesis ("coding snippets and equations",
d = 13.07, embedding 1.00) does not lose to a competing autointerp hypothesis ("medical research terms",
Surprisal AUROC 0.96 — the *highest* of the six, judge 0.80, but p = 0.064, d = 0.99). So the metric
battery itself is a contested field: two hypotheses can both look strong and disagree about what the
feature is, which is exactly the condition the paper calls polysemantic.

*Discovery.* k-NN+Leiden recovers language structure better than global partitioning: ARI 0.315 vs 0.130
(agglomerative) and 0.118 (K-Means), NMI 0.536 vs 0.304/0.278, random at ~0. **ARI 0.315 is the ceiling
here** — after cluster-label alignment, most of the variance is unexplained, and silhouette for the k-NN
method is 0.005, worse than K-Means at 0.112. In the 18-configuration ablation ARI collapses to ~0.03;
the authors attribute that to 200 vs 500 prompts per language and 17 vs 20 languages and explicitly say
the ablation shows relative robustness, not absolute performance. Hyperparameters barely matter for NMI
(0.151–0.175, under 13% spread), and PCA dimensionality is nearly inert (≤3% ARI change from 50 to 200
components, so the top 50 carry the relevant variance). Screening is brutal: across five language families
8,353 candidates are found and 69 survive — 4.0% for Germanic, 0.7% Romance, 0.2% Semitic. On
non-linguistic categories the retained marker counts are single digits (math 3/1/0, coding 3/2/0,
safety 0/1/3 across layers 0/8/11), with harmful-content markers appearing only at layers 8 and 11.

*The weight-sparse transfer is the strongest result.* With identical hyperparameters on raw post-GELU
neurons, the pipeline finds marker neurons for all 16 Python categories with effect sizes an order of
magnitude larger than the SAE features (semicolon-ending 152.3, double-quote strings 73.9, comments 72.3),
mean validation accuracy 75% over the 14 categories it ran, 12/14 above chance. It rediscovers 2 of the
4 neurons Gao et al. identified by hand, and the authors give a precise account of the two misses:
one neuron fires on *both* quote types so it cannot discriminate between their categories, and the other
acts on the residual stream delta while they measure post-GELU. They claim ~one researcher-day reduced
to under two minutes.

*Cost.* 13 LLM calls and ~16,867 tokens per feature against 727 for one-shot, i.e. 13× calls, 23× tokens,
24× wall-clock, 34× dollars — $0.0061 per feature, ~$6 per 1,000 on GPT-4o-mini, and by their pricing
projection under $125 per 1,000 even on the frontier tier.

## Limits and open problems they name

No monotone convergence guarantee. Hypothesis quality tracks the backbone's ability to write informative
test cases and is weakest on abstract or context-dependent features. FeatureFinder requires group labels,
so discovery is not fully unsupervised. Weaker generators refine poorly (GPT-4o-mini 63.5% vs Claude 83.3%),
which makes the method a moving target as models change. The safety case study deliberately withholds the
feature id and the intervention recipe.

What I would add, marked as mine rather than theirs: **there is no seed or run-to-run variance anywhere**,
so every win rate is a single draw over a stochastic agent; **the judge and the generator can be the same
model**, which is the reflexive confound the win rate cannot see, and neither the Detection classifier
nor the coherence judge is validated against a human; and **the FeatureFinder half is never ablated out
of the winning loop** — the tables report refinement against a one-shot label, never agent-with-discovery
against agent-without-discovery, so the discovery contribution and the refinement contribution are
confounded in exactly the way the paper's own thesis warns about. Net: a 34× cost multiplier
is justified by a battery that includes judges, not by an external task.

## What we take from it

**A framing.** Treat a verbal description as a *classifier over prompts* and score it that way. Detection
F1 on a shuffled, label-hidden positive/negative mixture is cheap, needs no ground-truth description, and
is the single most transferable number in this paper — a description that cannot sort its own activating
cases from matched controls has failed before aesthetics enter the discussion.

**A diagnostic reading rule.** When the two case studies disagree, they disagree in a useful pattern:
the activation-statistics metrics (p, d, Detection F1) separate the hypotheses cleanly, the *language-side*
metrics (embedding similarity, LLM judge) do not, and on L4_F12773 the judge ranks the weakest-discriminating
hypothesis second-highest. For our read-outs that means the judge is a preference measure, not evidence —
it can say a sentence is well-formed and on-topic, and that is all.

**A construction, not a judgement, for polysemanticity.** "Two semantically distinct hypotheses, both
above threshold, neither Pareto-dominating" is a definition of polysemanticity that needs no SAE anatomy
and no dictionary. It is exactly the criterion a predictive white-box model needs to refuse to give a
single answer, and it is cheap: it falls out of a battery you were already computing.

**A finding to reuse as a null model.** Clustering activation space finds language structure far better
than K-Means (ARI 0.315 vs 0.118) but reaches silhouette 0.005 — i.e., the neighbourhood graph is
*locally* informative while the global geometry is not well separated. Any claim of ours that "activation
space organises by concept X" should report both the graph-based agreement and the global separation,
because this paper gets to say the former and quietly has a bad number for the latter.

**A test-case discipline.** 12 matched positives and 12 matched negatives per hypothesis, with negatives
that are *matched* rather than random, is the cheapest possible falsification harness and it is what
converted a one-shot label into a falsifiable claim. Our own intervention predictions should ship with
the same shape of adjudication set.

**A warning about cost-shaped rigour.** 34× the price buys an audit trail, alternatives and a
polysemanticity flag — but the paper never shows that audit trail improving an external outcome. Given
the folder's comparative-utility argument, the untested question is whether 13 calls per feature beats
one call plus a better-designed prompt, and this paper cannot answer it because it never runs that arm.

## Tags

unit: feature | neuron | direction
form: description | dictionary | probe
method: agent | SAE | probing | steering | other
eval: benchmark | human-rating | downstream-task | prediction/simulation | ablation
goal: understanding | safety/auditing | evaluation-of-methods
