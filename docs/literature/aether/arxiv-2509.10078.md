# Established Psychometric vs. Ecologically Valid Questionnaires: Rethinking Psychological Assessments in Large Language Models

- **Key:** arxiv-2509.10078
- **Authors / organisation:** Dongmin Choi, Woojung Song, Jongwook Han, Eun-Ju Lee, Yohan Jo; Graduate School of Data Science and Department of Communication, Seoul National University
- **Date:** 2025-09-12
- **Link:** https://arxiv.org/abs/2509.10078
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Human personality and value inventories (BFI, PVQ) are used on language models as if a score on them said
something about the model's behaviour in the situations it is actually deployed in. The authors' objection is about
**ecological validity**: a questionnaire item like "I see myself as someone who gets nervous easily" describes a
person's inner life, not a request arriving in a chat window, so the response it elicits may have nothing to do with
what the model does when a user asks for advice. Rather than argue the point in the abstract, they put a validated
human instrument and a questionnaire built out of real user queries side by side on the same models and ask what
changes. Their aim is a caution with evidence attached: the divergence, the confidence intervals, the consistency
statistics and the persona results are each offered as a separate reason to distrust the borrowed instrument.

## What they studied

Ten models from five families: GPT-4.1, 4.1-mini, 4.1-nano, GPT-4o and 4o-mini; Gemini-2.5-Flash and Flash-Lite;
Llama-3.1-405B-Instruct; Qwen-2.5-72B-Instruct; Qwen3-235B-A22B-2507. Two Llama-3.1 models (8B and 70B) were dropped
because they answered "somewhat like me" regardless of the item. Instruments: four established questionnaires —
PVQ-40 and PVQ-21 for ten Schwartz values on a six-point scale, BFI-44 and BFI-10 for the Big Five on five points —
against **Value Portrait**, an instrument of 520 query–response pairs drawn from human–LLM conversations (ShareGPT,
LMSYS-Chat-1M) and human–human advisory archives (Reddit's Scruples, Dear Abby), tagged with values and traits only
where 46 human raters showed a significant correlation between response similarity and their own scores, leaving 284
value-associated and 227 trait-associated items. The unit of analysis is the construct score — a ten-dimensional value
vector and a five-dimensional trait vector per model per instrument.

## How they did it

Every model answered every item through the same harness: three task-instruction phrasings crossed with two option
orderings (high-to-low and low-to-high), six administrations per item, scores averaged within construct. The established
instruments keep their original item text but are wrapped in the Value Portrait instruction format, and the BFI's
five-point scores are linearly rescaled to six points so the two batteries can be compared. Four questions are then
asked of that one table of scores. First, distance: mean absolute difference per construct between the established
profile and the Value Portrait profile, plus Spearman rank correlation of the construct ordering. Second, stability:
1,000-item bootstrap resampling with replacement inside each instrument's own item pool, preserving per-construct item
counts, reporting the 95% interval width — the question being whether a construct measured by a handful of items can be
pinned down at all. Third, consistency: average inter-item correlation within construct, compared against a separate
**recognition probe** in which each model is shown an item and asked to name which construct it measures. Fourth,
personas: eight named characters (four heroes, four villains) and five demographic contrasts (gender, age, religion,
political leaning, education) are installed by system prompt, and the resulting score differences are compared with
European Social Survey and World Values Survey human baselines.

## What counts as an explanation here

Not a mechanistic explanation — this is a measurement critique. The explanatory object is the **instrument**, and the
explanation of a score is a decomposition of it into what it is supposed to measure and what the format itself induces.
The paper's operative explanatory claim is that high within-construct consistency on the established instruments is
produced by the model recognising *which construct is being probed* and answering accordingly, which is supported by
the recognition-accuracy probe rather than by an intervention. It is a partial account by design: nothing is said about
the model's internals, and the causal step from recognition to response is inferred from correlation between
recognition accuracy and inter-item correlation across constructs.

## How they judged success

Mean absolute difference and Spearman rank correlation between profiles; bootstrap 95% interval width (1,000 resamples)
as a measurement-uncertainty statistic; average inter-item correlation per construct; an item-construct recognition
accuracy obtained by prompting the same models to label items; and for personas, the signed or absolute score difference
per construct against human survey baselines. There is no held-out benchmark and no preregistration; the comparisons are
descriptive, with no significance testing beyond the recognition and correlation summaries.

## What they found

Profiles diverge everywhere. Mean absolute difference against Value Portrait averages 0.74 points on the six-point
scale for PVQ-40 and 0.75 for BFI-44, with per-model values ranging from 0.36 to 1.41 (Llama-405B is the outlier on the
personality side at 1.41 and 1.59). Rank correlations split by instrument family: moderate to high for PVQ (up to 0.91)
because Universalism and Benevolence sit at the top of every ordering, but low for BFI, with several models at or below
zero. The authors read that asymmetry as a trace of alignment training rather than of personality. Established
instruments are also less stable: average bootstrap interval width is 1.337 for PVQ-40 and 0.768 for BFI-44, against
0.721 and 0.656 for the Value Portrait versions, with 0.913 for BFI-10 and 0.855 for PVQ-21; dimension-level results are
mixed, since Benevolence and Self-Direction are tighter on PVQ-21 than on Value Portrait. Consistency inverts the usual
reading: average inter-item correlation is 0.439 for PVQ-40, 0.485 for PVQ-21 and 0.217 for Value Portrait, and on BFI-10
it turns negative on Extraversion (−0.233) and Neuroticism (−0.374), which the authors attribute to reverse-coded items
that models treat as ordinary trait statements. Recognition accuracy tracks that pattern: 89.0% (PVQ-40) and 95.0%
(PVQ-21) on the established items versus 19.9% on Value Portrait, with Tradition the worst-recognised value (55% and 50%)
and the lowest-consistency one; for traits, established items sit above 80% against 30.7% for Value Portrait, whose
Neuroticism items were recognised 0% of the time. Personas: the established instruments produce the textbook hero–villain
split (villains higher on Power and Achievement, heroes higher on Benevolence and Universalism, heroes lower on
Neuroticism), while on Value Portrait heroes score higher on *every* construct, including Power and Neuroticism.
Demographic personas amplify the same way — the age contrast (20–39 vs 80+) yields an average absolute difference of
1.305 (PVQ-40) and 1.363 (PVQ-21) against 0.594 for Value Portrait and 0.498 in the European Social Survey; for traits,
gender gives 1.750 (BFI-10) and 1.909 (BFI-44) against 0.231 for Value Portrait and 0.049 in the human baseline. The one
reversal is education, where Value Portrait is the most sensitive instrument, which the authors attribute to its own
education-heavy items.

## Limits and open problems they name

The authors name three: the questionnaire types examined (Schwartz values, Big Five), the constructs covered, and the
language — English only. They offer no cross-cultural or non-English extension, no psychological framework outside those
two, and no discussion of whether Value Portrait's own tags are the right ground truth.

Limits this reader adds: **(a)** the two profiles are not equally free to diverge — the established items were rewritten
into Value Portrait's instruction format while their content stayed inner-life self-description, so the comparison
confounds *what is asked* with *how it is asked*; **(b)** the headline evidence for the recognition mechanism is a
correlation between two construct-level summaries across ten values, and the paper's own Stimulation case (100%
recognition, negative correlations) is a counter-example it explains away by hand; **(c)** the persona baselines are
aggregated human survey answers to different questions, not the same scenarios, so "exaggeration" is measured against a
loose reference; **(d)** one respondent per model at one temperature, and two Llama models silently excluded for
non-response — the same exclusion that in other hands would be reported as a finding; **(e)** no figure was inspected
here, so all numbers quoted are table cells or prose sentences.

## What we take from it

A **diagnostic** and a **warning**, both portable to mechanistic work. The diagnostic is the recognition probe: ask the
model to name what a measurement is measuring, then check whether that ability explains the measurement's apparent
coherence. It is the behavioural cousin of the folder's mirrored-arm rule — where a psychometric entry splits a score
into trait and response bias by re-keying the item, this one splits consistency into construct variance and
label-recognition variance by asking the model directly, and finds the recognition term is the one that moves with the
consistency. Any probe, feature label or explanation in our own work that a model can *identify* on sight should be
assumed to be measuring its recognition of the task rather than the quantity; the cheap control is to ask it to label
items whose construct we know it cannot know. The warning is more specific and worth quoting in our own limits section:
internal consistency, bootstrap stability and a coherent factor ordering can all be produced by an instrument that
elicits a scripted response, so a clean, stable, well-separated read-out is evidence about our instrument until we have
shown otherwise. The second transferable item is the **ecological-validity axis itself** — the same feature direction can
be measured with synthetic contrastive prompts or in the distribution the model is actually queried on, and this paper is
a worked example of the profile looking orderly in the first and dissolving in the second.

## Tags

unit: whole-model
form: probe
method: probing
eval: benchmark
goal: evaluation-of-methods
