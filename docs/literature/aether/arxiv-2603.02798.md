# Guideline-Grounded Evidence Accumulation for High-Stakes Agent Verification

- **Key:** arxiv-2603.02798
- **Authors / organisation:** Yichi Zhang, Nabeel Seedat, Yinpeng Dong, Peng Cui, Jun Zhu & Mihaela van der Schaar; Tsinghua University / University of Cambridge / Thomson Reuters Foundational Research
- **Date:** 2026-03-03
- **Link:** https://arxiv.org/abs/2603.02798
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Agents are now making decisions in settings where a wrong answer is expensive — the motivating example is clinical diagnosis — and in those settings the cheap side of the problem is generating an answer, not checking it. They argue that checking is hard because it needs domain expertise, and that the usual fixes each fail in a known way: trained verifiers need expert labels nobody can afford, judges score against criteria they only implicitly hold, and sampling-based agreement can be confidently wrong when the model makes the same mistake every time. Their observation is that much of the needed expertise is already written down in professional protocols — guidelines, checklists, procedures — and that these documents specify both what counts as adequate evidence and what the acceptable process looks like. So the goal is to compile those documents into a probability that the agent's decision is correct, calibrated well enough that it can drive abstention or escalation.

## What they studied

Agents in an agentic clinical diagnosis workflow, on three diseases from MIMIC-IV (diverticulitis, cholecystitis, pancreatitis), with two backbones: Qwen2.5-7B-Instruct and Qwen3-30B-A3B-Instruct. Two thousand trajectories per backbone, deliberately balanced fifty-fifty between correct and incorrect final diagnoses, generated at temperature 0.9. The unit of analysis is the **trajectory prefix** — a sequence of observation-and-action pairs ending in a final diagnosis — and the quantity being estimated is a scalar posterior over a binary latent "this diagnosis is correct". An external clinical-guideline corpus supplies the knowledge.

## How they did it

They first write verification as sequential Bayesian evidence accumulation: take the logit of the posterior that the answer is correct given the prefix, and note that a Bayes update turns it into the previous logit plus one additive term per step, that term being a likelihood ratio of the step under correct versus incorrect outcomes. That step term is intractable in an open-ended setting, so they replace it with a surrogate. For every trajectory they retrieve guidelines matched to the final diagnosis (an LLM extracts the diagnosis from the raw prediction, then titles are keyword-matched and re-ranked by embedding similarity), and for every step an LLM judge is asked whether *this step*, given the history, is consistent with each retrieved guideline; the rating is the judge model's token probability over YES versus NO. Multi-guideline ratings are collapsed into one low-dimensional feature per step by a small statistic — their default is the pair (average, minimum) — so that the number of available guidelines does not change the feature's dimension. Step logits are summed with an exponential discount, giving larger weight to recent steps, into a single accumulated evidence score. Because only trajectory-level correctness labels exist, calibration is fitted on that score directly: Bayesian logistic regression with a Gaussian prior, sampled by MCMC, and the reported confidence is the posterior-mean sigmoid. The estimated entropy of that confidence is then used as a trigger: above a threshold, they collect two extra kinds of evidence — expanding the guideline set, and a "differential check" that retrieves guidelines for competing diagnoses and subtracts the strongest competing logit from each existing rating, scaled by a rectification factor, so that a step which is merely generically plausible loses support.

## What counts as an explanation here

The product is a **calibrated scalar** — "probability this diagnosis is correct" — reported as a trajectory over steps, plus an interpretable decomposition: which guideline a step violated, and how much each step's rating moved the accumulated logit. It is explicitly partial: the additive structure is asserted by Bayes but the per-step evidence is a surrogate, and the appendix states cleanly that discrimination is necessary but not sufficient for the sufficiency assumption the bound needs. It predicts the correctness of a completed trajectory (or a prefix), not the next token, and it can rank candidates for selection.

## How they judged success

Discrimination by AUROC; a risk-at-threshold measure (error rate on the top half of most-confident samples); calibration by expected calibration error and Brier score. Two properties are checked separately: separation of prefix scores between correct and incorrect trajectories, and linearity of the score against correctness in logit space (reported as an R-squared of 0.943 with guidelines and non-significant separation without them). Baselines are token likelihood of "true", generative judge, self-consistency and semantic entropy over 16 samples, self-verification, retrieval-augmented token likelihood, a medical process reward model, and an outcome reward model trained on exactly the same 100-trajectory calibration set. Ablations: component-wise addition (process context, guideline, accumulation, calibration), guideline quality (empty placeholder versus short abstract versus full content), aggregation function, discount factor, rectification factor, active-verification arms, cross-backbone transfer (verifier trained on one agent, applied to the other), Best-of-N selection with a fixed sampling budget, and a computational-cost comparison in LLM calls and tokens. A three-clinician study over 50 trajectories supplies interpretability and utility ratings plus first-error-step localisation.

## What they found

Multi-guideline aggregation dominates the baselines on both discrimination and calibration, and the margin over the best baseline grows on the harder disease. The component ablation shows where the gain comes from: process context alone is small, guideline grounding is the large step, accumulation adds more, and calibration barely moves AUROC while collapsing calibration error. Guideline quality matters monotonically — an empty guideline is worst, a short abstract is a clear improvement, full content is best. The active-verification arms each help and combining them helps most, with the run-everything variant reported as an upper bound, and most of the benefit arrives at low trigger rates. Cross-backbone verification keeps its discrimination but loses calibration when the strong model's verifier is applied to the weaker agent, which they read as evidence that the *scoring* generalises across agents while the *calibration* is partly agent-specific. Best-of-N selection raises accuracy well above the strongest sampling baseline at the same number of candidates. Cost is the strong point: at one guideline per step, verification uses a small fraction of the calls and tokens that self-consistency needs while scoring higher.

## Limits and open problems they name

The sufficiency assumption is assumption, not measurement — they say directly that strong discrimination does not establish it, and that the residual information the score drops is exactly what active verification is there to recover. The calibration for evidence is partially agent-specific. Competing guidelines for differential checks are picked by similarity within the disease's diagnosis list rather than generated per case. The impact statement adds that guidelines themselves can be flawed or stale, that the method should complement rather than replace qualified judgement, and that broader healthcare validation would be needed before deployment.

## What we take from it

- **Method / framing:** the additive-logit decomposition of a prefix into per-step evidence is the reusable piece, and it is a *measurement recipe* rather than a model: one judge call per step per guideline, a token-probability rating, a discounted sum, a low-capacity calibrator fitted on few labels. For us it is a template for turning a sequence of per-step readings from a network into a single quantity that has probability semantics, without training anything that needs step-level labels.
- **Warning:** this is the folder's cleanest case of a *judge read-out* whose calibration was fitted on labels the judge never saw — the judge produces YES/NO token probabilities, and the linear calibrator is the only thing standing between those and a probability. The component ablation is the receipt that the judge alone is not calibrated: adding calibration leaves AUROC unchanged and cuts calibration error by roughly an order of magnitude. Any read-out of ours that is going to be thresholded gets the same treatment — separation first, then a separately fitted calibrator, then the calibration number reported next to the discrimination number.
- **Method:** the differential check is a cheap, one-line correction for a specific failure — support that is high because the criterion is generic rather than specific to the candidate answer — and it works by subtracting the strongest score obtained against *competing* answers in logit space. Our white-box model has an exact analogue available: a unit or direction that responds to a concept should be scored net of its response to the model's own alternative completions.
- **Shortcut:** guideline grounding is doing almost all of the work of the "expertise", and the ablation on guideline quality makes the dose-response explicit. If we want to claim our read-out is knowledge-grounded rather than criterion-grounded, the same dose-response is the test: swap the grounding for an empty placeholder and report what survives.

## Tags

unit: whole-model
form: probe
method: probing, other
eval: prediction/simulation, benchmark, human-rating
goal: safety/auditing
