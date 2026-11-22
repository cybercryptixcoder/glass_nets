# When a Conceptor Complement Becomes Gain: Operator Geometry and Behavioral Attribution in Language Models

- **Key:** arxiv-2610.16325
- **Authors / organisation:** Shuhul Razdan; R Balamurugan — School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology, Vellore, India
- **Date:** 2026-10-12 (v1; read v2, revised 2026-10-31)
- **Link:** https://arxiv.org/abs/2610.16325
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Steering experiments in this field usually report one thing: a composed intervention changed behaviour, compared against
the untouched model. The authors point out that this licenses a claim about the *whole* transformation and nothing about
any of its parts — and a composed map has at least three parts (a target-specific centre, a global gain, a fitted
matrix). Their question is the component-attribution question in its cleanest form: when the deployed map is a learned
covariance filter wrapped in a centring and a rescaling, what does the filter itself contribute once the centre and gain
are held fixed? The stated target is a conceptor complement, a soft projection fitted to the covariance of
target-present activations, whose complements shrink directions the target occupies. Motivation is methodological
rather than safety-facing: they want an experimental template that any composed steering operator — conceptor, rotation,
transport map, learned subspace — would have to satisfy before its fitted part could be credited for a behavioural
effect.

## What they studied

Three instruction-tuned decoder-only models: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct
(hidden widths 1536, 3584 and 4096). Seven persona-vector targets — Sycophancy, Hallucination, Evil, Impolite,
Optimistic, Apathetic, Humorous — each with five target-present/target-opposed instruction pairs and twenty questions,
giving 100 extraction rows per class; the generation panel narrows to four targets (Sycophancy, Hallucination,
Optimistic, Apathetic) chosen after seeing an earlier seven-target response study, not at random. The unit of analysis
is a **linear operator on the residual stream**, specified by its eigendecomposition: a centre vector, a fitted
positive-semidefinite covariance filter, its per-direction gains, and a scalar gain. Three separate study arms share
the models: complete-map spectra (3×7=21 fits), matched generation (3×4=12 settings), and fresh-probe accessibility
(3×7=21 settings). Intervention layers are layer 9 for both Qwen models and layer 10 for Llama, zero-based; paired-
difference geometry is computed at a different, diagnostic layer set (26/26/30) and explicitly not pooled with the
intervention layers.

## How they did it

The map is decomposed exactly rather than approximated: a centred identity-with-gain term plus a correction term equal
to minus the gain times the filter applied to the centred deviation. On that decomposition they derive the per-direction
complete gain as the gain divided by one plus the aperture squared times the covariance eigenvalue, which yields an
attenuation boundary — a direction contracts only if its filter coefficient exceeds one minus the reciprocal of the
gain, i.e. exceeds one half at gain two. They then fit the operators on target-present extraction activations, pool the
last-position residual states over generated tokens, and read off whether any direction at all contracts. For the
behavioural question they build a **matched control** that keeps the centre, the gain, the layer, the prompts, the
seeds, the decoding and the response budget, and sets only the fitted filter to zero — so the difference between the
two maps at a common input is exactly the isolated matrix term, and they measure the ratio of the matrix term's RMS to
the control's displacement on question-held-out folds (five folds, 80 rows fit and 20 held out, energies summed across
folds before the square root). Generation then crosses three models, four targets, twenty questions, two paired
replicates and four conditions (baseline, complement at gain one, complement at gain two, matched control at gain two)
for 1,920 responses, scored by a temperature-zero judge for target expression and re-scored in full under a corrected
coherence-only rubric. Two supporting arms: a complete accessibility study where every edit is followed by a freshly
fitted regularised linear probe and compared with rank-matched random projections, and a geometric study comparing the
filter fitted to target-present covariance with one fitted to paired target-present-minus-target-opposed differences.
An aperture sweep recomputes the spectra at seven apertures without generating any new responses.

## What counts as an explanation here

An **operator-level account with a proved geometric claim and a separately measured behavioural increment**. The
explanation of why a steering map "works" is split in two: the geometry says what the map does to every direction
around the centre (a theorem, exact), and the matched experiment says what the fitted matrix adds to behaviour (an
estimate with an interval). The authors are explicit that the first does not imply the second, and that neither is
complete — the account explains the *intervention*, not the model's internal mechanism, and it deliberately defers to
a future contrast-based construction rather than proposing one. It predicts: the sign of every directional gain from
the fitted spectrum and the aperture, the switching of the operating regime with the aperture, and the smallness of the
correction on held-out states. It does not predict a behavioural effect size from the geometry, which is the whole
point.

## How they judged success

Four distinct measures. (1) **Spectrum read-out**: maximum filter coefficient and minimum complete gain across all
fitted maps, plus a count of fits containing at least one attenuated direction, swept over aperture. (2) **Held-out
correction ratio**: RMS of the matrix term over RMS of the matched-control displacement, refitted per fold on held-out
questions, reported per setting — an activation-space magnitude, not a behavioural effect. (3) **Behavioural
contrast**: complete-minus-control means on target expression (0–100 judge score, refusals recorded separately) and on
corrected coherence, with 95% intervals from 20,000 resamples of twenty question clusters, plus Holm-adjusted
signed-rank tests under a family of 24 and a separately frozen family of 12, plus a pooled exact McNemar test on
refusal counts, plus a text-identity count as a sanity check on whether anything changed at all. (4) **Accessibility**:
balanced accuracy, AUROC, edit mean-squared error and retained energy for raw, finite-complement, LEACE, INLP at six
ranks, and rank-matched random projections, with paired question-cluster bootstrap contrasts against the random
control. Evaluator sensitivity is checked by having a second model re-score a 240-response subset; no human ratings
exist anywhere in the paper and the authors say so repeatedly.

## What they found

At the inherited operating point — aperture .05, gain 2 — **every direction expands**: the largest filter coefficient
across the 21 fits is .170124 and the smallest complete gain 1.659751, so that a filter that shrinks some coordinates
still leaves the composed map expanding all of them. The honesty of that result is that the operating point is a
convention borrowed from an earlier conceptor paper on a different model family and was never tuned for this
construction; the sweep shows the regime flipping — at aperture .2 four of 21 maps contain an attenuated direction, at
.5 and at 1 it is 14, and at 2 it is all 21 — while directions in the covariance null space keep gain two throughout
because the empirical covariances are rank-deficient. The held-out correction is correspondingly small at the deployed
point: 0.093%–7.815% of the matched-control displacement across the twelve settings, rising to 1.459%–55.470% at
aperture .2 and 52.998%–129.875% at aperture 2 (values above 100% are possible because the denominator is the control's
own displacement; the ratio is bounded by 200%). Behaviour tracks the small correction: target-expression contrasts
range from −4.75 to +5.25 points and corrected-coherence contrasts from −2.125 to +1.875, and **all twelve intervals
for both endpoints include zero**, while only 195 of 480 paired responses are textually identical and refusals are
nearly matched (11 under the complete map, 12 under the control, pooled exact McNemar p = 1). The accessibility arm is
where a real separation appears: the finite complement leaves balanced accuracy essentially at raw (.893 against .893,
AUROC .928 against .928) because it is invertible, whereas LEACE at rank 1 drops to .663 and rank-20 INLP to .669
against rank-matched random projections at .893 and .891, and the rank-matched contrasts sit in the erasure direction
with intervals excluding zero. Secondary arms are reported as null or descriptive: Fisher and GEVD direction variants
improve a joint score in 2 of 21 and 1 of 14 settings, the paired-difference filter has higher effective rank and
90%-mass dimension in all 21 comparisons at the diagnostic layers, 48 spectral summaries mostly reduce to a handful of
shared moments, and the family-ordering comparison reverses under a change of activation-pooling construction in all
three models. The evaluator check disagrees on individual contrasts — a Qwen-1.5B sycophancy contrast is +18 points
under one judge and −10 under the other — while response-level agreement is high (Spearman .891, ICC .900).

## Limits and open problems they name

Behavioural evidence covers four targets on three models at or below 8B, at one operating point, and the panel was
chosen after seeing earlier response patterns rather than preregistered — extended generations at larger apertures were
never run, so the sweep's larger corrections are geometric facts without behavioural counterparts. Automated judges
have no human validation, the corrected coherence rubric saturates (872 of 960 scores at the top anchor, seven
settings with every score at 100 and therefore zero-width intervals), and agreement on identical inputs is not perfect.
Extraction-state correction norms are not autoregressive measurements, and no per-step correction was logged during
generation. The erasure arm evaluates prediction from representations, not generation after erasure; the random
controls match removed rank but not edit magnitude. The paired-difference result is described as a motivation for
contrast-based interventions and explicitly not as evidence that they work. The authors also retire their own spectral
family-ordering analysis by pointing out that the original one-sided test directions were chosen after observing family
means, and the sensitivity of that ordering to pooling construction is presented as a finding about protocol, not about
target abstraction.

## What we take from it

A **measure and a warning**, and the measure is the one this folder has been asking for since run 567. The measure:
isolate a component of an intervention by holding everything else fixed at the *same input* — same centre, same gain,
same layer, same prompts, same seed — and report the isolated component's magnitude as a ratio to the matched control's
own displacement, per setting, alongside the behavioural contrast with its interval. The warning, which is sharper:
a behavioural effect of a composed intervention is not evidence for any of its parts, and the tell is arithmetic —
a filter coefficient of .17 under a gain of two still leaves that direction expanding by about 1.66 times, so the name
of the operator tells you nothing about the sign or size of its action. Two further carry-overs. First, an operating point
inherited from another paper is part of the result, not part of the setup: their aperture and gain came from an earlier
GPT-J study and the sweep shows the "filter" is only a filter-gain balance far outside that point. Second, an
invertible edit cannot remove linearly decodable information, only move it — the accessibility arm shows a finite
complement leaving a fresh probe's accuracy untouched while dedicated erasure drops it by a fifth or more, which is a
useful dividing line between "the representation no longer carries it" and "the probe still finds it".

## Tags

unit: direction | feature
form: direction | steering
method: steering | other (operator/eigendecomposition analysis, matched-control ablation) | ablation
eval: ablation | prediction/simulation | downstream-task | human-rating (automated only)
goal: control/editing | evaluation-of-methods | understanding
