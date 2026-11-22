# Introspection Fine-Tuning (IFT): Training Small LLMs to Introspect

- **Key:** arxiv-2607.14111
- **Authors / organisation:** Ely Hahami, Ishaan Sinha, Lavik Jain — Harvard College (Departments of Mathematics and of Computer Science), Cambridge, MA
- **Date:** 2026-05-08
- **Link:** https://arxiv.org/abs/2607.14111
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why
The prior introspection work, on very large proprietary models, reported that a steered model can often say that an unusual thought was injected and name its topic. The authors ask whether that holds for small open models, whether it improves with scale, and whether it can be trained in where it is missing. The motivation is safety-adjacent and concrete: a model that reliably reports on its own anomalous internal states could act as its own monitor, whereas reports that are decoupled from internal state give false assurance. Along the way they discover that the standard measurement used to show this ability is itself broken at small scale, and most of the paper's value for us is that diagnosis.

## What they studied
Six instruct-tuned models from two families: Llama-3.2 at 1B, 3B and 8B, and Gemma-4 at roughly 2B (E2B), 4B (E4B) and a 26B mixture-of-experts with about 4B active. Concepts come from two banks of 500 vectors each — concrete nouns whose vector is one word's last-token state minus the mean of 50 baseline words, and abstract concepts whose vector is a positive-minus-foil mean over twenty sentences a side, both partly LLM-generated. The unit of analysis is a scalar hidden state at one layer, turned into a steering vector and added back at the same layer; the quantity measured is the model's own next-token choice among labelled positions.

## How they did it
They first reproduce the field's standard instrument: inject a concept vector, ask whether the model detected an injected thought, and read the Yes-minus-No logit gap against an unsteered baseline. To separate detection from mere agreeableness they build a **factual-no control** — questions whose correct answer is obviously no, such as whether humans breathe underwater — and compare the injected-minus-baseline shift on the introspection question with the same shift on the control, over a grid of 40 (layer, strength) cells. Finding the two shifts nearly identical, they propose two tasks that are immune by construction because they ask for a *relative* judgement among labelled positions: **sentence localization** (ten or five sentences, inject into one, name which) and **strength comparison** (two sentences at different strengths, name the stronger, with the strengths swapped in a second sub-trial so position preference cancels). They then fine-tune with LoRA on localization examples the model generates for itself — sample ten background sentences, pick an injection position and strength, inject, and train cross-entropy on the correct index token — under six conditions crossing a fixed injection layer with a randomly sampled one, and Gaussian noise vectors with semantic vectors and a semantic-plus-concept-name-target variant. Strength comparison is never trained on.

## What counts as an explanation here
Not a mechanistic account. What the paper offers is a **behavioural capability measurement** — a chance-referenced accuracy for the model's report about its own perturbation — plus a training recipe that installs the capability, and one diagnostic finding about a measurement instrument. It predicts nothing about internal structure; it is a claim about what the model can be made to say about its state, and the paper is careful enough to give the two designs that make chance the right null.

## How they judged success
Peak and grid-average accuracy, each against an explicit chance level (10 per cent for ten-way localization, 50 per cent for strength comparison), plus MMLU and Winogrande before and after fine-tuning as a capability cost. The confound diagnosis uses Pearson correlation between the introspection shift and the control shift across the sweep, a mean net shift with a standard deviation, and the claim that all points lie on the identity line.

## What they found
On Llama-3.1-8B the two shifts correlate at r = 0.999 and the mean net introspection signal is −0.006 ± 0.033 logits — that is, nothing survives once a factual-no control is subtracted. Under the confound-free tasks, localization accuracy rises with scale (Llama-1B at or below 20 per cent chance, 3B about 65, 8B about 88) and Gemma strength comparison reaches about 85, 90 and 95 per cent at 2B, 4B and 26B. Fine-tuning lifts Llama-1B average localization from 9.6 to 60.6 per cent and transfers zero-shot to strength comparison (30.2 to 52.2 for the reasoning variant; 47.8 for the plain semantic one), with smaller gains at 3B (14.4 to 34.7) and 8B (16.8 to 28.3). Random-layer training beats fixed-layer nearly everywhere, and Gaussian noise vectors train far worse than semantic ones (14.9 against 60.6 at 1B), the authors' reading being that noise teaches detection of a perturbation rather than the concept. MMLU stays within about six points of baseline and Winogrande within about two and a half, with most conditions within a point.

## Limits and open problems they name
Up to 8B dense and one 26B mixture; whether the gains extend further is untested. The setup is a controlled single-vector injection, and transfer to naturalistic anomalies is an open question. The reasoning-target variant helps one metric and hurts another, so supervision design is unresolved. Beyond their list: every fine-tuned arm is a **LoRA adaptation of the model that is then evaluated on its own reports**, with no adapter applied to an untrained-output control and no per-condition seed spread; the "100 per cent best-cell" numbers appear for baseline and fine-tuned arms alike, so the headline contrast rests on grid-averaged accuracy, and the 1B baseline strength comparison (30.2) sits *below* chance, which means the beginning of that comparison is itself a broken instrument. The two reported 26B settings also give 100 per cent for the baseline 3B and 8B best cells, i.e. the sweep is broad enough that "peak" is not informative at those scales.

## What we take from it
A **measure, a warning, and a method**, in that order of value. The warning is the reusable one: the binary read-out with a factual-no control is the cheapest confound audit in the folder, and this is the folder's first case where the audit is run on the *question format* rather than on the direction — and after the audit the claimed effect is gone. The measure is relative judgement among labelled positions with an explicit 100/N chance level and counterbalancing for position. The method is self-generated supervision: build training items from the model's own perturbed forward passes, and randomise the nuisance (here the injection layer) so the learned strategy is not specialised to one site — which is exactly the discipline our own account wants when a read-out must generalise across layers.

## Tags
unit: layer | whole-model
form: probe | steering
method: probing | steering | training-interpretable | other
eval: downstream-task | benchmark | ablation
goal: safety/auditing | evaluation-of-methods
