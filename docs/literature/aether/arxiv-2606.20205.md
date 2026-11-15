# Apparent Psychological Profiles of Large Language Models are Largely a Measurement Artifact

- **Key:** arxiv-2606.20205
- **Authors / organisation:** Jelena Meyer, David Garcia, Dirk U. Wulff; Max Planck Institute for Human Development / University of Konstanz / Barcelona Supercomputing Center / University of Basel
- **Date:** 2026-06-18
- **Link:** https://arxiv.org/abs/2606.20205
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Borrowed human psychometric instruments are now routinely used to hand LLMs a "personality" or a "risk
preference", and those profiles are then treated as stable descriptions with consequences for deployment, safety
assessment and the practice of using models as stand-ins for human survey respondents. The authors' worry is
measurement-theoretic rather than behavioural: a response can be perfectly systematic and still not come from the
trait the instrument claims to measure, because scales also pick up a directional response bias — a tendency to
answer toward one end of the scale, or one labelled option, regardless of item content. Human psychometrics has
known this since the acquiescence literature, and its standard defence is to include items where trait and bias
push in opposite directions. The question here is what happens when an instrument containing no such items is
administered to a language model. Their claim, stated ex ante and not preregistered, is that the profiles read
off such instruments are mostly artefacts of the instrument.

## What they studied

56 instruction-tuned LLMs treated as individual respondents: 46 open-weight models spanning roughly 1B to 70B
enumerated near-exhaustively on Hugging Face, plus 10 proprietary models from five providers (three Claude, three
GPT-5.4, one Gemini, two Qwen, one Grok), queried through native APIs in April and May 2026. Two batteries, both
previously administered to large human samples so that a reference distribution exists: the 300-item IPIP-NEO-300
Big Five inventory (contributing five instruments, human reference N = 20,993) and a 24-instrument risk-preference
battery mixing self-report scales and behavioural tasks such as lotteries, decisions-from-description and
decisions-from-experience (human reference N = 1,507 from a single in-lab session), for 29 instruments in the main
analysis and up to 30 in one elicitation condition. The unit of analysis is unusual and is the point of the paper:
not a neuron, feature or circuit, but **the instrument and the respondent-level score it produces**.

## How they did it

They write down a two-component measurement model: a response equals the scale midpoint plus a keying sign times
the respondent's latent trait deviation, plus a respondent-level response bias, plus item noise. Keying is the
lever — a forward item and a reverse-keyed item carry the trait with opposite signs and the bias with the same sign.
Three consequences are then engineered into experiments. First, averaging the model within keying gives two means
whose cross-respondent covariance is bias variance minus trait variance, so the *sign* of the forward-versus-reverse
correlation says which dominates and its magnitude says by how much; they compute it per Big Five trait across the
20,993 humans and the 56 models. Second, the half-difference of the two means estimates the trait and the half-sum
estimates the bias, giving per-respondent orthogonal coordinates, a variance-share statistic equal to (1 + correlation)
over 2, and a closed form that does not depend on how many reverse items an instrument has. Third, they correlate
each instrument's proportion of reverse-keyed items with its mean inter-item correlation across all 29 instruments,
for humans and models separately, and test that gradient under a 2×2×2 factorial of prompt perturbations —
each item in a fresh context versus sequential presentation, constrained text generation versus next-token logit
extraction, standard versus flipped option order — plus a teacher-forced condition in which each model is run over
the 1,507 recorded human answer histories. Every model is prompted with the original item wording verbatim, at
temperature zero, one respondent in its default state. Controls target the obvious alternatives: five models that
gave one constant answer to every item are removed and the signs are re-checked; the reversal is re-run on
negation-free reverse items and on the one risk facet whose items carry almost no negation; and the panel is split
at the median parameter count to defeat a small-model story.

## What counts as an explanation here

A **decomposition**, and specifically an orthogonal one: every respondent's score is split into a trait coordinate
and a bias coordinate that are uncorrelated by construction under the model, and the explanatory claim is about the
variance shares of those two coordinates rather than about any mechanism inside the network. It is partial and
deliberately parsimonious — the authors say the model could be extended with item difficulty or extreme-response
tendencies at the cost of identifiability. It predicts something narrow but real: an instrument's apparent
reliability, given only its keying balance. It says nothing about how the bias is produced; the paper argues the
bias's signature (two-sided across models, stable within a model, not reducible to sycophancy or social desirability)
is inconsistent with the post-training stories it considers, and leaves the cause open.

## How they judged success

No benchmark and no accuracy metric. The statistics are the argument: cross-respondent Pearson correlations with
Fisher-z intervals, a variance-share estimate derived from that correlation, cross-instrument correlations of
reverse-keying proportion against mean inter-item correlation with intervals, Cronbach's alpha reported alongside,
Mann–Whitney comparisons between open and proprietary models, and OLS slopes for bias against log parameter count.
The magnitudes are treated as lower bounds in the authors' own favour by noticing that finite-item noise attenuates
toward zero and so pulls both populations toward the midway point, which shrinks the human–model gap rather than
inflating it. The human sample serves as the reference against which every model-side number is read, and the
null is therefore a real one: humans answering the same items are the controlled comparator.

## What they found

The sign flips and it flips everywhere. Across all five Big Five traits humans show strongly negative forward-reverse
correlations, from −.69 to −.82, and the models show strongly positive ones, from +.61 to +.81; the illustrative
scatter is −.82 for humans against +.68 for models, and a downward slope is what a trait-driven responder looks like
while an upward slope is what a direction-insensitive one looks like. In the per-respondent decomposition the bias
accounts for 81–90% of between-model variance and only 9–16% of between-human variance. Capability attenuates it
without removing it: across the 46 open-weight models the correlation between absolute bias and log parameter count
lies between −.14 and −.25 and is significant on none of the traits, open and proprietary models do not differ
significantly on three of five traits (proprietary lower on agreeableness and neuroticism, p = .007 and .028), and
proprietary models are still far from human levels. The reliability result is the sharpest: across 29 instruments the
correlation between reverse-keying proportion and mean inter-item correlation is −.95 for models and only −.41 for
humans; instruments with essentially no reverse items average an inter-item correlation of .61 with alpha .85 to .96,
which is exactly what a coherent trait looks like, while balanced instruments fall to near-zero consistency and alpha
turns negative, reaching −.52 on one behavioural task. The gradient survives every prompting condition (r between
−.83 and −.95) and is unchanged between small and large models. Profiles themselves move: mean absolute bias runs
from .15 to .99 across models against .09 for the average human, proprietary models average .23 and open-weight .48,
risk instruments .51 against personality .30, and the forward-only and reverse-only versions of the same profile
differ by twice the bias.

## Limits and open problems they name

Each model is one respondent in its default state, though the authors note that within-model variation through
personas or steering is a live alternative and that the definition of a model "individual" is unresolved. Only
post-trained models are assessed. Only two content domains, personality and risk. Temperature is zero throughout,
removing sampling noise at the cost of representativeness. The model of responding is deliberately thin. The
proprietary subset is too small for inference on most comparisons. And they are explicit that the deeper question is
which problem this is — that human instruments may lack validity for systems without embodiment or volition, or that
models may have no stable dispositions to measure at all — and they decline to conclude the second from their data.

## What we take from it

For our goal, a **measure and a design rule**, both cheap. The measure is the split of a score into a half-difference
and a half-sum: whenever a read-out is computed by contrasting two conditions, running each condition against a
mirror-keyed version of itself turns one number into two, and tells you whether your contrast is measuring the
construct or the response tendency the format induces. The design rule is that a read-out validated only by internal
consistency has almost no evidential value, because internal consistency is exactly what a directional artefact
produces; what makes the check work here is a third column that *inverts the keying* rather than repeating the same
item. The warning is the one this folder keeps meeting from the neural side: a stable, reliable, reproducible score
across dozens of models can be a property of the instrument and not of the thing measured, and the diagnostic is not
to collect more models but to find the design feature — a reverse-keyed item, a matched-random arm, a
superfluous-but-plausible claim — that would let the artefact cancel while the construct double. Finally, the paper
practises a form of honesty worth copying: it names the confounds it cannot remove (no preregistration, single
respondent per model, borrowed instruments possibly invalid outright) and reports the resulting claim as bounded
rather than as a discovery.

## Tags

unit: whole-model
form: probe
method: probing
eval: benchmark
goal: evaluation-of-methods
