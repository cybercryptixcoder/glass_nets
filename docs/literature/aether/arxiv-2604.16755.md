# Machine individuality: Separating genuine idiosyncrasy from response bias in large language models

- **Key:** arxiv-2604.16755
- **Authors / organisation:** Valentin Kriegmair, Dirk U. Wulff; Center for Adaptive Rationality, Max Planck Institute for Human Development (Berlin), and Center for Cognitive and Decision Sciences, University of Basel
- **Date:** 2026-04-18 (v2; HTML page dated August 24, 2026)
- **Link:** https://arxiv.org/abs/2604.16755
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Every attempt to say what one language model is *like* compared with another runs into the same objection: a
difference between two models can be a disposition that depends on the specific stimulus, or a tendency to push all
answers the same way regardless of stimulus, or simple sampling noise — and a single score conflates all three. The
authors point at the sharpest published case: a classifier that names the source model from its free text at high
accuracy after paraphrasing, whose own authors found the tell consists largely of phrasing and formatting habits, i.e.
exactly the machine analogue of a response style. Their goal is a measurement in which the *stimulus-specific* part is
identified separately from the *global offset*, and then a demonstration that this part is not per-item noise but a
stable, model-unique signature. They also want a replacement for human questionnaires, whose construct validity they
explicitly decline to assume for a non-human respondent.

## What they studied

Ten instruction-tuned open-weight models spanning 20B–235B total parameters and four architecture families: Qwen3-32B
and Qwen3-235B-A22B, Mistral-Small-24B, gemma-3-27b-it, gpt-oss-20b and gpt-oss-120b, OLMo-3.1-32B-Instruct,
Falcon-H1-34B-Instruct, granite-4.0-h-small, and phi-4. The stimuli are **psycholinguistic word-rating norms**, not
personality items: 107,083 cue words drawn from the psychNorms metabase across 14 norms — valence, arousal,
concreteness, morality, gender association, humor, socialness, five sensory modalities from the Lancaster norms, and two
age-of-acquisition scales. Each model is asked, zero-shot, to rate each word on each scale with a single number.
Observations run to roughly 5.4 million per norm (~74.9 million total) because each word is rated five times at
temperature 1.0. The unit of analysis is not the model and not the word but the **word × model interaction** — the
per-word deviation of one model from the consensus — which is what the paper renames machine individuality.

## How they did it

A fully crossed random-effects model is fitted independently for each norm: a rating decomposes into a grand mean, a
random effect per word (shared consensus across models, "Trait"), a random effect per model (a global offset, "Bias"),
a random word × model interaction ("Idiosyncrasy"), and residual noise. The design is crossed in the strict sense —
every model rates every word in a norm five times — which is what makes the interaction separately estimable at all;
the authors note that this is precisely why human personality inventories, administered one respondent at a time,
cannot supply it. Fitting is lme4, with variance shares read as each component over the total. To show that the
interaction is not a fitting artefact, each norm is refitted on data simulated from the model *without* an interaction
term (parametric bootstrap, 100 iterations, fixed effects and both random-effect variances taken from the empirical
fit), and the observed idiosyncrasy variance is compared with the resulting null distribution. To show the interaction
is *coherent* rather than per-item noise, each model's estimated per-word deviations on a held-out norm are predicted
by ridge regression from its own deviations on the other thirteen norms, and the resulting five-fold cross-validated
within-model R² is divided by the mean R² obtained when another model's deviations do the predicting — a specificity
ratio. The same ratio is recomputed on raw ratings, with consensus and offset left in, to check that the fingerprint
belongs to the interaction and not to the ratings. Finally, each model's mean stochastic rating is correlated with the
published human norm for each dimension (Fisher-z averaged), and the same correlation is computed for a deterministic
single-sample rating so the two decoding regimes can be compared on the identical words.

## What counts as an explanation here

A variance decomposition, and deliberately nothing more mechanistic. The explanation of "model A and model B behave
differently on this item" is a statement about which term of the model the difference lives in: consensus (they agree
because the words mean something), offset (they agree on everything but shift), interaction (they differ word by word),
or noise. Completeness is claimed only for the partition, not for the sources — the paper says nothing about weights,
activations or modules, and it treats the interaction as an *object to be measured*, not as a mechanism to be opened.
What it does predict is structural: if the interaction is a real signature, it should transfer across dimensions for
the same model, and that transfer is the paper's main evidential claim.

## How they judged success

Four criteria, in increasing strength. (1) The interaction variance share exceeds a parametric null in which no
interaction exists — reported as a p-value equal to the fraction of simulated null variances meeting or exceeding the
observed one, with the null's upper bound quoted. (2) The share is large relative to the bias term, which is the
paper's stated substantive bar: individuality is present only if it outweighs directional response style. (3) The
specificity ratio exceeds 1 for every model, and exceeds the ratio computed on raw ratings. (4) Human alignment is
compared across decoding regimes, with the stochastic-minus-deterministic difference aggregated over models and norms.
There is no held-out behavioural benchmark, no intervention, and no preregistration; the null simulation, the
cross-validation and the 10-model × 14-norm coverage are the whole of the robustness apparatus.

## What they found

Shared consensus dominates: 32.1% of variance is word-level Trait. Idiosyncrasy averages 16.9%, ranging from 4.8% on
valence to 9.7% for gustatory ratings (the five sensory norms average 14.3%, individually 9.7–18.0%) up to 34.0% on
gender association, with arousal second at 22.6%. The parametric null's upper bound is below 0.033% on all fourteen
norms at p < 0.01, so the interaction is not a structural artefact of the crossed fit. Every model's idiosyncrasy
fingerprint predicts its own held-out dimension better than any other model's does: all specificity ratios exceed 1,
in the range 1.74–3.43, while the same ratio computed on raw ratings sits at 1.16–1.49 — so the model-specific
structure lives specifically in the part with consensus and offset removed. The norms differ in kind, not just in
amount: morality carries an unusually large model offset (bias rather than idiosyncrasy), while valence, socialness,
concreteness and age of acquisition are consensus-dominated. Human alignment is moderate (Fisher-averaged r ≈ 0.48–0.67
across models), which the authors read as the ratings being partly *sui generis* rather than noisy copies of human
judgments. And stochastic aggregation beats deterministic decoding on human alignment in every cell — Δr̄ = 0.032,
10 of 10 models and 14 of 14 norms — which they frame as a trade: greedy decoding buys reproducibility at the cost of
both the variance structure they are measuring and agreement with people.

## Limits and open problems they name

They state two open questions rather than a limitations list: whether these idiosyncrasies survive when models are
elicited in context rather than one word at a time, and whether they predict downstream behavioural differences — with
the explicit concession that only the second would establish individuality as consequential rather than merely
measurable.

Limits this reader adds: **(a)** the decomposition is variance-share arithmetic from one fitted model per norm, and
nothing checks the assumption that word and model effects are Gaussian and additive on the rating scale — a
non-additive but non-idiosyncratic response pattern (e.g. an interaction driven by scale-endpoint use on particular
word classes) would land in the same term. **(b)** The specificity ratio's within-model R² is cross-validated over words
but the comparison arm is a mean over other models, not a distribution, so "every model exceeds 1.74" is not accompanied
by any interval on the ratio itself. **(c)** The human-alignment correlations are computed on raw values across
different scale conventions (one dimension, arousal, is explicitly reversed at analysis time; valence is an explicitly
constructed two-unipolar-to-bipolar conversion), so small differences in r between models are not attributable to the
models alone. **(d)** The `Trait`-vs-`Bias` reading is asserted, not tested: a model that is consistently harsher is
called bias by construction, but the same pattern concentrated on one semantic region is what the paper elsewhere calls
individuality, and the boundary between the two is a modelling choice no arm varies. **(e)** The 5-repetition noise term
is the only partition of sampling variation, so any within-word instability that is systematic per model (e.g.
formatting-dependent parse errors surviving the retry ladder) is absorbed into idiosyncrasy. **(f)** No figure was
inspected here — Figure 1's two panels are reported as their captions and the surrounding prose, and every number in
this entry is a sentence of the text or a Methods/Appendix parameter, never a value read off a plot.

## What we take from it

A **method** and a **measure**, both cheap and both directly transferable to our own read-outs. The method is the
crossed random-effects partition: give every probe *stimulus* to every model, several times, and the interaction term
is the only one that carries disposition — the offset term is the thing that a good white-box model must reproduce and a
bad one invents. In our language this is the third rung the folder keeps asking for, done with arithmetic rather than an
extra arm: consensus = the input's own content, offset = a model-level constant (the analogue of an acquiescence or
temperature direction), interaction = the representation of *this input in this model*. The measure is the specificity
ratio: a per-item deviation profile should predict the same model's deviations on *other* items better than another
model's do, and the ratio should be computed on the residualised quantity while the raw-rating version is printed
beside it — that second column is what shows the residualisation is doing the work rather than the vocabulary. Two
warnings carry over unchanged: the parametric null must be run through the *same* fitted model without the term in
question (otherwise a structural artefact is indistinguishable from the finding), and the reconciliation with the
psychometric entries is that everything here is measured on stimuli with no self-report content, which is likely why a
16.9% interaction term appears here and not in borrowed human inventories.

## Tags

unit: whole-model
form: causal-abstraction
method: decomposition
eval: benchmark
goal: science
