# Idiosyncrasies in Large Language Models

- **Key:** arxiv-2502.12150
- **Authors / organisation:** Mingjie Sun, Yida Yin (equal contribution), Zhiqiu Xu, J. Zico Kolter, Zhuang Liu; Carnegie Mellon University / UC Berkeley / University of Pennsylvania / Princeton University
- **Date:** 2025-02-17 (v1; this copy under the arXiv.org perpetual non-exclusive licence, i.e. not the later revision)
- **Link:** https://arxiv.org/abs/2502.12150
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Two questions sit behind this paper. The practical one is provenance: when a piece of text has no watermark, can an
outside observer say *which* model wrote it, including which variant of a family? The methodological one is the one that
matters for us: if the output distributions of two models trained on largely the same corpora with the same objective
are distinguishable at near-ceiling accuracy, then something systematic and model-specific is being written into
behaviour by weights and post-training, and a scientist ought to be able to name what that something is. The authors
borrow an old computer-vision idea — train a classifier to recognise which dataset an image came from, and read the
accuracy as a measure of how much unexplained systematic bias the data carries — and transplant it from datasets to
models: a classifier that tells models apart is a measurement of how much of each model is not explained by the shared
recipe. Their ultimate interest is not attribution but diagnosis: they want to know whether the distinguishing signal
survives semantic destruction, where in the text it lives, and what it implies for training on synthetic data.

## What they studied

Fourteen generators in three groups. **Chat APIs:** GPT-4o, Claude-3.5-Sonnet, Grok-2, Gemini-1.5-Pro, DeepSeek-V3.
**Instruct models** of similar size across families: Llama-3.1-8B, Gemma-2-9B, Qwen-2.5-7B, Mistral-v3-7B. **Base
models:** the same four without post-training. Also a within-family size ladder (Qwen-2.5 at 7B/14B/32B/72B), base vs
instruct pairs, and Phi-4 as an extra open-weight model in the similarity experiment. Prompts come from UltraChat for
chat and instruct models (11k responses each, 10k train / 1k validation) and from FineWeb for base models. The unit of
analysis is the whole **response text**, embedded and classified; the second half of the paper reduces that unit to word
frequencies, markdown element counts, and a judge's prose description.

## How they did it

For each group of models, the same prompt set is answered by every model, and a sequence classifier — a text embedding
model fine-tuned with a classification head over the first 512 tokens — is trained to map a response back to its
source, using all pairwise combinations as binary tasks and the full set as an N-way task. Then the classifier is
re-run under a ladder of transformations that each destroy one kind of information and preserve another: stripping
special characters; shuffling words (order destroyed, word multiset kept); shuffling letters (word identity destroyed
too); keeping *only* markdown structure and replacing all words with a placeholder token; and having a second model
paraphrase, translate into Chinese, or summarise the text (word distribution destroyed, meaning progressively
compressed). Two prompt-level controls add instructions to force equal length and plain-text formatting. Further
controls vary the embedding backbone, truncate the classifier's input to a fixed number of tokens from one token
upward, vary the number of training samples per model, and ask the classifier to separate responses from the same model
under different decoding strategies. Finally, a TF-IDF logistic regression on unigrams and bigrams is fitted so its
coefficients can be read out as each model's characteristic phrases, and an LLM judge (identity anonymised, four
different judges as a robustness arm) writes open-ended comparisons that are then collapsed into bullets.

## What counts as an explanation here

Two things, neither of them mechanistic. First, a **diagnostic score**: classification accuracy is treated as the
quantity of idiosyncrasy, with chance as the null and the transformation ladder used to attribute the score to a level
of the text (letters, words, ordering, formatting, semantics). Second — the paper's actual explanation — a list of
**concrete surface habits**: characteristic phrases ranked by logistic-regression coefficient, markdown element count
distributions, first-word distributions, common-word frequency profiles, plus natural-language characterisations from
an LLM judge. It is a partial, descriptive account of *what* differs, explicitly not an account of *why*; there is no
probe, no circuit, no intervention on internals, and nothing that predicts behaviour on an unseen input.

## How they judged success

Held-out validation accuracy against chance level for the N-way task (20% for five chat models, 25% for four). The
controls are the transformation ladder itself plus prompt-level length and format instructions, decoder-strategy
classification as a *negative* target, five embedding backbones from ELMo to LLM2vec, an input-length sweep, a
training-size sweep, and the within-family and base-vs-instruct splits. Similarity metrics (ROUGE-1, ROUGE-L,
BERTScore with human-baseline rescaling) are compared across-model against within-model on the same prompts. The judge
arm is replicated across four different judge models. No statistical intervals are reported anywhere; every number is a
single accuracy or a single similarity score.

## What they found

Separability is close to total between proprietary chat models: 97.1% on the five-way task against 20% chance, with
every binary pair at 97.2–99.9%. Instruct models are 96.3% (87.3% for base models), and the classifiers transfer
out of distribution across four prompt corpora (89.9–99.8% off-diagonal). Within one family the task gets harder but
does not collapse: 85.5% on the best binary Qwen-2.5 pair and 59.8% on the four-way, and base vs instruct of the same
model is separated at 96.8%. On the transformation ladder the pattern is the paper's real result: letter shuffling
destroys the signal (97.1% → 39.1% on chat models, at chance for the four-way), while word shuffling barely dents it
(→ 88.9%) and stripping punctuation and markdown costs only about two points; markdown structure *alone*, with every
word replaced, still yields 73.1% on chat models and 77.7% on instruct models but 38.5% on base models; paraphrasing
(→ 91.4%) and translation into Chinese (→ 91.8%) leave it nearly intact, while summarising drops it to 58.1% yet stays
far above chance. A single token of a post-trained model's response already carries ~50% of the five-way information.
Separating decoding strategies of the *same* model is near chance (highest 59.6%, and 37.9% for five temperatures
against 20% chance), and ten training samples per model still give 40.3% on the instruct group. On the implication
side, fine-tuning Llama-3.1-8B and Gemma-2-9B on the same synthetic corpus cuts their separability from 96.5% to
59.8%, while two Qwen-2.5-7B runs fine-tuned on synthetic data generated by Llama and by Gemma respectively remain
98.9% separable — the teacher's fingerprint travels with the student. In the similarity experiment, an excluded model's
outputs are most often assigned to ChatGPT (82.8% of Grok's), and ChatGPT's and DeepSeek's own outputs are frequently
read as Phi-4 (55.9%, 76.0%).

## Limits and open problems they name

The authors are explicit that the mechanism is not located: the paper documents that the signal lives in word choice
and in semantic content rather than in letters or word order, and that formatting contributes, but cannot say what
causes a model to prefer one word or one phrasing over another. Their own follow-up framing treats the classifier as a
tool for measuring similarity and inheritance rather than as an explanation. Implicit limits the reader can see and the
paper does not state: every comparison is between *different prompt sets or models*, so the classifier may be reading
topic or prompt-answering convention rather than disposition, and the only control against that is the instruct/base
and within-family splits; no null arm exists in which the classifier is trained to separate the *same* model's two
sampling runs except the decoder-strategy task, which is a different object; accuracy is a pooled number with no
per-class breakdown beyond the state of the confusion matrices; and a 90%-plus score after paraphrase is compatible with
the classifier reading the question's topic rather than the model's answer to it, which the paraphrase arm does not
separate.

## What we take from it

Three things, in order of usefulness. **A decomposition-by-destruction ladder** is the cheapest attribution tool in the
paper and it costs nothing but text rewrites: destroy one level of representation at a time and read the score's decay
as which level carries the signal — letters, word multiset, word order, formatting, semantics. Applied to our own
read-outs this is a ready-made robustness battery, and it is the same shape as the paraphrase-invariance arms this
folder has collected, generalised to a whole ladder rather than one rewrite. **A sharp negative control**: the
classifier that separates one model's decoding strategies sits near chance (59.6% at best), which is what makes the
97% between models meaningful rather than an artefact of the classifier being good at text. That is the matched-format,
mismatched-object rung we keep asking for, supplied by the task design itself. **A warning with teeth**: idiosyncrasy
is inherited through synthetic data (98.9% separable students from two different teachers, and shared SFT collapsing
96.5% → 59.8%), so a probe or a description of a model's "style" measured on a distillation descendant may be a
measurement of the teacher — for us, this is a concrete reason to measure any behavioural signature on a lineage we
control, and to treat cross-checkpoint signatures as potentially inherited rather than acquired.

## Tags

unit: whole-model
form: description
method: other
eval: benchmark
goal: understanding
