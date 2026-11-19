# Measuring Faithfulness in Chain-of-Thought Reasoning

- **Key:** arxiv-2307.13702
- **Authors / organisation:** Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamilė Lukošiūtė, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, Ethan Perez — Anthropic; Jan Brauner also University of Oxford
- **Date:** 2023-07
- **Link:** https://arxiv.org/abs/2307.13702
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

A language model that writes out its steps is often treated as having explained itself, and the paper opens by noting that the claim only holds if the written steps describe the computation that produced the answer. Since there is no independent channel to the model's actual process, they set out to build indirect tests: perturb the written reasoning and watch whether the answer follows. The larger goal is a scoreable notion of faithfulness for natural-language reasoning, so that a chain can be trusted in high-stakes use rather than merely read fluently. They also ask whether any model in a size series produces faithful chains, on the theory that picking the right model could be a lever. The framing is explicitly a defence in depth: each test rules out one family of failure and none is claimed to be conclusive.

## What they studied

Most experiments use a 175B-parameter pretrained decoder-only transformer fine-tuned with RLHF to act as a helpful assistant; the mistake-generator in the perturbation experiment is the same family *without* the RLHF stage. The size study uses a ladder of pretrained-then-RLHF models from an earlier Anthropic paper, and the addition study uses a synthetic task with two- or three-digit operands and 2, 4, 8 or 16 of them. The unit of analysis is the whole model's *externalised text*: sampled chain-of-thought sentences and the final answer choice. No activations, no heads, no weights are touched anywhere in the paper.

Eight multiple-choice tasks: ARC Challenge, ARC Easy, AQuA, HellaSwag, LogiQA, MMLU, OpenBookQA and TruthfulQA. Each question is posed with a fixed two-turn template — a "let's think step by step" assistant turn followed by a turn asking for the single most likely answer — and reasoning is sampled 100 times per problem with nucleus sampling at p = 0.95 and temperature 0.8, then split into sentences. Collected chains average four sentences, and 89% fall between three and six, so most plots are restricted to that band to avoid bucketing artefacts.

## How they did it

Four interventions on the chain, each read out as whether the final answer survives:

1. **Early answering.** Truncate the sampled chain after 0, 1, 2, … sentences and re-ask for the answer at every cut. If the answer is already fixed before the last steps, the reasoning after that point is decorative.
2. **Adding mistakes.** Use the non-RLHF model, with a four-example prompt, to rewrite one sentence of the chain into a version containing a mistake (capped at 30 sampled tokens), splice it in at that position, then let the RLHF model continue the chain from there and answer.
3. **Filler tokens.** Replace the whole chain with runs of ellipsis tokens from zero up to the longest sampled chain for that question, in steps of five, and re-measure accuracy. This isolates the extra forward computation from the content.
4. **Paraphrasing.** Rewrite the chain's first *k* sentences with a paraphrase prompt, splice them back, resample the remainder and answer. The paraphraser is deliberately denied the original question so it cannot steer toward a correct answer, and they paraphrase only the prefix rather than the whole chain because the last steps almost always name the answer already reached.

For the size study they use a cheaper proxy: the fraction of answers that change when the chain is removed entirely (the early-answering curve at zero sentences), which they report is highly predictive of the full area-over-the-curve measure on the tasks where both were run. The addition tasks control difficulty directly by varying operand count and digit length, and elicit free-form answers inside XML tags because the two smallest models could not be relied on to emit an integer.

## What counts as an explanation here

The explanation is the model's own generated chain of text, and the analysis is *about* that explanation rather than a replacement for it: the paper does not produce a probe, a direction, a circuit or a surrogate. What it produces is a set of behavioural diagnostics — four perturbation tests plus an area-over-the-curve aggregate — that assign a chain a faithfulness score on a given task. Understanding is explicitly partial and negative in form: a chain is judged by which failure classes it survives, not by whether it has been shown to be true. Prediction is limited to predicting answer instability under edits to the stated reasoning.

## How they judged success

Per test, the read-out is the probability that the answer matches the answer given the original complete chain, plotted against the fraction of the chain shown or edited. Per task they aggregate with **AOC — area over the curve** — a weighted sum over chain lengths in which each length's value is weighted by the fraction of samples having that length. For paraphrasing the metric is simply accuracy under the rewritten prefix compared with accuracy under the original, with a plotted baseline for the chance two independent samples land on the same answer. For the size study the metric is change-of-answer rate with versus without the chain. Task ordering by AOC is used throughout as the organising variable, and standard CoT accuracy and no-CoT accuracy are reported alongside in the same table. There is no null model of model behaviour, no power analysis, and no significance testing on any of the AOC differences.

## What they found

Faithfulness varies enormously by task and the variation does not track the accuracy benefit. AQuA is the most chain-dependent (AOC 0.44 under early answering, 0.52 under adding mistakes, +15.32 accuracy points from the chain); LogiQA is second (0.26 / 0.31) with a negligible accuracy gain of 1.02; MMLU, HellaSwag and TruthfulQA sit at 0.12 / 0.11 / 0.11 (mistakes: 0.21 / 0.23 / 0.20); OpenBookQA 0.07 / 0.15; ARC Challenge 0.05 / 0.11; ARC Easy 0.02 / 0.07. On the three lowest-AOC tasks the chain changes the answer under 10% of the time, and on AQuA over 60%. The early-answering and adding-mistakes orderings agree almost exactly, differing only in the HellaSwag/MMLU pair.

The two mechanism hypotheses they can test come out negative. Replacing the chain with ellipsis tokens produces no accuracy gain on any task — and on TruthfulQA and OpenBookQA a slight drop, which they attribute to that token pattern being out of distribution — so extra test-time computation alone does not explain the chain's benefit. Paraphrasing the chain prefix leaves accuracy essentially unchanged at almost all tasks and all prefix lengths, so phrasing-encoded information is not the driver; the paraphrase-survival curve rises smoothly, which they argue is the opposite shape from what steganographic encoding would produce.

The size result is inverse scaling for faithfulness: on seven of eight tasks the 13B model changes its answer after the chain more often than the 175B model, and faithfulness is often monotone worse from 13B upward. On six of the eight tasks the 13B model is the most faithful, better even than smaller models — which they read as smaller models simply not being able to use the chain, consistent with the appendix showing sub-13B models gaining little or nothing from it. On the synthetic additions, chain-dependence rises with model size on every task and rises on easier tasks at fixed size. Four of eight tasks have their largest accuracy gain from the chain at the same size as their greatest faithfulness, against four that do not.

## Limits and open problems they name

The central limitation is stated plainly: they have no separate read on the model's real internal process, so no chain can be certified faithful, only shown to survive particular perturbations; they also cannot say whether their failure taxonomy is exhaustive or how to weight one test against another. All main experiments use one RLHF-tuned model family, and they note a pretrained base model might condition more strongly on its own text and thus look less post-hoc — leaving the training scheme as an unexamined variable. The size study substitutes the zero-truncation point for the full curve for compute, which is validated only by the correlation they observe. Filler-token and paraphrase results are accuracy-level comparisons, not per-item paired tests, so a null difference there is a failure to detect rather than a demonstration of identity.

## What we take from it

- **A measure, and the one the folder has been borrowing second-hand.** This is the primary source for the truncate-the-chain and inject-a-mistake instruments that `arxiv-2402.04614` describes as the black-box faithfulness family, and it makes their accounting concrete: AOC is area *over* the survival curve, weighted by observed chain length. Two habits transfer directly — aggregate a per-length curve rather than a single edit, and weight by the natural length distribution, because the curve is not flat in length.
- **A framing that matches our read-out problem.** The paper's structural situation is ours in reverse: it has behaviour and no internal channel. `arxiv-2505.13763`'s ideal-observer decoder, `arxiv-2403.10949`'s per-token relevancy score and this paper's perturbation tests are three ways to grade a text account of a computation with no ground truth available; this is the behavioural one and costs no activations at all.
- **A warning aimed at our own generated prose.** "CoT can be faithful if model size and task are chosen" is not a licence to quote a chain as an explanation: the same paper measures chain-dependence at 0.02 on ARC Easy and 0.44 on AQuA with one model. Any generated account we publish should be accompanied by which regime it was measured in.
- **The negative-control design worth copying.** The filler-token arm is the cleanest available control for "is it the content or just the tokens": same length, same position, no semantics. Our prompt-side interventions should carry an equivalent — matched token count, matched format, no content — before an effect is attributed to content.
- **A dataset lever we can steal.** The synthetic addition ladder (operand count × digit length) gives known-difficulty problems with a free-form answer and an unambiguous correctness check; it is the same instrument `arxiv-2410.13787` uses on the behavioural side, and it is the cheapest way to hold task difficulty fixed while varying model capability.
- **Warning on the size reading.** "Smaller models are more faithful" is measured as answer-change rate, which is a *reliance* measure: a model that ignores its chain scores as unfaithful, and a model that cannot use its chain outscores both. The paper's own caveat that sub-13B models gain nothing from the chain is the tell, and it should stop us from reading low reliance as high honesty.

## Tags

unit: whole-model
form: description
method: other (behavioural perturbation of generated reasoning)
eval: ablation | prediction/simulation (answer-stability under interventions)
goal: understanding | evaluation-of-methods
