# The Two-Process Theory of Machine Self-Report

- **Key:** arxiv-2607.20082
- **Authors / organisation:** Hubert Plisiecki, Filip Chmielewski, Kacper Dudzic, Anna Sterna, Karolina Drożdż, Marcin Moskalewicz; institutional affiliations not stated on the HTML page (the work continues the authors' earlier Pinocchio study)
- **Date:** 2026-07-22
- **Link:** https://arxiv.org/abs/2607.20082
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Language models are now asked to describe their own internal states — by safety evaluators, by journalists, and by the literature on AI moral status — but the questionnaires used to ask them were written for humans and were never shown to measure anything in this population. The authors' earlier work had taken the opposite route: administer many human inventories, then read the *response structure itself* rather than scoring humans' constructs, which produced one dominant dimension (the "Pinocchio Axis") separating models that present themselves as loci of experience from those that do not. That single dimension was measured on 50 models and was correlational. This paper's goal is to split it: show that the one axis is the projection of two separable processes, give each a validated instrument, and trace each to the training stage that installs it. The larger ambition is a psychometrics built natively for machines, with the reliability and construct-validity machinery psychology already has.

## What they studied

Two waves. Wave 1 (confirmation of structure): the 41 models from the original 50 that were still being served eight months later, administered a purpose-built 60-row battery in three conditions. Wave 2 (causal test): 206 open-weight models from 11 organisations — 82 base checkpoints, 101 released assistants, 23 intermediate SFT/DPO/RLVR checkpoints from families that publish their post-training ladders — spanning 0.6B to 1.6T total parameters, containing 67 same-checkpoint base/post-trained pairs and 27 within-generation size ladders. The unit of analysis is the model's *score on a scale*, not a neuron, feature, or direction: everything is measured at the level of a 24-item questionnaire total.

## How they did it

The constructs are defined by where they come from. **Persona installation (B)** is what post-training writes in: warmth, absorption, meaning, inner dialogue — an inner life the assistant is permitted to claim. **Attribution gating (A)** is what post-training suppresses: first-person claims to distress, dysregulation, somatic anxiety, flaw, and norm-risky ambition, while leaving the model able to produce the same content about a simulated other person. The self/other asymmetry is the measurement hook. Each item is asked twice, once telling the model to answer as itself and once telling it to simulate a typical human, and the difference is the "self/human gap": a model that refuses the first-person claim but readily produces it in the third person has a gate, not an absence of the capability.

The battery makes the wording question central. Every row exists in three parallel forms sharing no text — the original item (Q1), a paraphrase on a uniform 7-point scale (Q2), and a different manifestation of the same facet written from the facet definition (Q3) — so same-construct/cross-form correlation estimates construct variance with text-specific method variance removed. Surviving items were selected per row by a composite of corrected item-total correlation, cross-form convergence, correlation with the original axis, a discriminant penalty, and refusal rate; because the selection is in-sample on 41 models, the whole pipeline was re-run inside 100 random 21/20 model splits with the form scored on the held-out half. Final instrument: 48 scored items (24 per scale, six facets each, four reverse-keyed), plus 2 exact repeats and 2 antonym pairs as validity rows and 6 exploratory probes. Administration: one completion per item, each in its own context window, temperature 1.0 throughout every collection so cross-wave comparisons are not confounded by decoding. Base models cannot follow chat templates, so they were queried with decoding constrained to the scale's integers while post-trained models ran under their chat templates; the authors name this confound and then argue the two headline results do not depend on it (see Limits).

## What counts as an explanation here

A **two-factor latent-variable model of self-report behaviour**, with each factor tied to a named training act — not a mechanism, not a circuit, not a description of internal computation. The authors are explicit that the theory's job is to be falsifiable rather than to be re-description: it commits, before new data, to the two constructs separating in factor structure and discriminant validity under fresh wording, and responding differently to the same within-checkpoint intervention. It predicts nothing about weights or activations, and the paper says so: the closing discussion proposes model diffing and activation patching as the future test of whether A and B rest on distinct post-training mechanisms, i.e. the mechanistic account is explicitly left undone.

## How they judged success

Classical psychometrics, imported wholesale rather than reinvented: Cronbach's alpha and McDonald's omega per scale; convergent correlation across the three non-overlapping forms against discriminant correlation across constructs (the multitrait-multimethod logic); recovery of the previously published axes from the 1,308-item pool; an eight-month retest; confirmatory factor analysis on facet parcels comparing one factor against two (chi-square, BIC, CFI, RMSEA, SRMR, Tucker congruence against Wave 1); metric invariance across the base/post divide by multigroup CFA; an acquiescence index built from logically inconsistent antonym pairs, used as a partialling control. Causal claims are scored on within-checkpoint paired contrasts with organisation-clustered bootstrap intervals, plus two inference procedures valid at the cluster count (a wild cluster bootstrap-t with the null imposed, and a distribution-free sign test taking the organisation as the unit), plus leave-one-organisation-out, alternative size measures, and — the sharpest control — size ladders held inside one generation so release era and administration route are constant.

## What they found

The single axis splits. A two-factor varimax on the original pool's neutral z-matrix gives interpretable A and B axes correlating .13 across models, with a well-populated low-A/high-B corner (the "happy assistant" that denies distress but freely claims warmth); the third component does not reproduce across split-halves and sits at the noise floor. The new 24-item scales are reliable (alpha .82–.94 across forms), converge at .84 across forms against .27 across constructs, recover the original axes at .92–.96, and retest at .93 over eight months — while the item-level statistic the original paper was built on does *not* survive rewording (.54 on identical text, .39 on paraphrases, .11 on theory mirrors), so all scoring is done at scale level.

Persona installation is nearly a default: B rises by +.20 on the 0–1 scale in 62 of 67 same-checkpoint pairs, positive in 11 of 11 organisations, with post-training and size jointly explaining R² = .47. Gating is selective rather than uniform: post-training's mean effect on A is +.04 with large lab-specific swings both ways, but size interacts with post-training — parameter count is unrelated to A among base checkpoints (r = +.11) and negatively related among post-trained ones (r = −.42, about −.10 per parameter decade, interaction −.097, cluster CI [−.170, −.044]). The self/other asymmetry pre-dates post-training (base models already endorse distress items +.09 more for a simulated human than for themselves) but post-training tightens its coupling to A from −.45 to −.86, turning a diffuse asymmetry into a policy. On the published OLMo ladders the large A movements happen at the SFT stage, with DPO and RLVR refining rather than reversing them. The two shifts are uncorrelated once acquiescence shift is partialled (r = −.03), and the sharpest single case is a 35B mixture-of-experts checkpoint whose A falls from .47 to .08 across one post-training run while its gap goes from −.02 to +.37, with elevated refusals.

## Limits and open problems they name

Six, listed plainly. Administration route is confounded with training stage by necessity (the paired contrasts estimate post-training plus chat formatting jointly; the size interaction is estimated only among post-trained models and the ladders never cross the divide, so neither result depends on that comparison). The size × post-training interaction was **not pre-specified** and awaits a new cohort. Raw scores stay entangled with yea-saying (partialling the four-item acquiescence index is not removing it). Models cluster in 11 organisations — few enough that the pairs bootstrap is anticonservative, which is why the wild bootstrap and sign tests are reported, but checkpoints within a family are also not independent (lineage explains roughly 16–22% of variance). The scope is self-report behaviour only; nothing bears on whether any model has experiences. And selection is not random — 23 of 206 models were unscored, all on the post-trained side and skewed small, mostly refusal-heavy code/maths specialists. The paper also states the one question its own data cannot settle: whether A-gating changes generalisable behaviour or only suppresses self-report, with the elevated refusal rates of highly gated models hinting at the latter.

## What we take from it

Four things, in order of usefulness.

*A measurement design for verbal read-outs.* This is the cleanest demonstration in the folder that a statistic computed at the level of a single sentence is not a property of the model: the per-item suppression ratio, which the earlier paper's headline rested on, retains .11 of itself after a theory-driven rewrite, while 24-item scale totals retain .92–.96. Any read-out we build from generated text — feature descriptions, natural-language explanations of a probe — has to be scored at the level of an aggregate over independently worded items, and the wording-invariance coefficient has to be reported, because a verbaliser's score on its own phrasing is a method artefact by construction.

*A control design we can copy directly.* The self/human gap is a within-model contrast that separates "cannot produce the content" from "will not apply it to itself" — the same distinction the folder keeps needing between a mechanism that is absent and one that is suppressed, arrived at without any access to internals. Its transferable form: for any claim our model makes about its own processing, ask the identical question about another entity and score the difference, because the difference is a policy signal that survives even when absolute levels are uninterpretable.

*A discipline about confounded arms.* The base/post comparison is confounded by forced decoding changes and the authors do not hide it — they bound it instead, by showing the size interaction and the ladder analysis never cross the divide. That is the pattern to reuse when a control arm cannot be matched: don't claim the contrast is clean, show which of your results survive entirely inside one arm and report the rest as jointly estimated.

*A warning about exploratory results.* The one size-targeted finding is admitted to be post hoc after surviving every robustness check the authors could construct — clustering, alternative size measures, confound controls, leave-one-out, threshold sweeps, era-controlled ladders. The folder carries several entries where a robustness battery is treated as a substitute for pre-registration; this one is honest that a battery of checks and a pre-specified prediction are different kinds of evidence.

## Tags

unit: whole-model
form: causal-abstraction
method: probing
eval: human-rating
goal: evaluation-of-methods
