# Meta-Models: An Architecture for Decoding LLM Behaviors Through Interpreted Embeddings and Natural Language

- **Key:** arxiv-2410.02472
- **Authors / organisation:** Anthony Costarelli (Olin College of Engineering), Mat Allen (Independent), Severin Field (Independent). Costarelli and Allen marked equal contribution; correspondence to Costarelli and Allen.
- **Date:** 2024-11-07 (v3; v1 2024-10-03)
- **Link:** https://arxiv.org/abs/2410.02472
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Probes are the standard cheap read-out of a model's internal state, but each probe is fitted to one labelled task and stops being useful outside it, and the authors gather the known reasons a probe's accuracy can be misleading (prompt sensitivity, non-concept features, pretraining-data artefacts). The alternative they pursue is a *single read-out that answers questions in English*: give one model (the "input-model") a natural-language question and a set of its activations, and train another model (the "meta-model") to answer yes or no about the input-model's state. If such an artefact exists, a human can ask about behaviour nobody had a labelled dataset for, which is the closest thing to the "mind-reading" they describe as the point. The specific question here is narrower and empirical: does a meta-model trained only on easy, well-labelled behaviours (sentiment, emotion, language identification) transfer to a behaviour it never saw during training — deception — and does the transfer survive changing which models play which role. It is a step toward the general white-box account, and it is the activation-side answer to a weight-side question: how much of a network's behaviour can be read off its internals by a reader that was never told which behaviour you were looking for.

## What they studied

Two families of model pairs. In the main setting the input-model is Llama-3.1-8B-Instruct and the meta-model is phi-2; the replication swaps in internlm2_5-7b-chat as the meta-model and re-runs the whole grid. The unit of analysis is the *activation vector at sampled layers* of the input-model — one token sampled every four layers — treated as a set of "interpreted embeddings" that get injected into the meta-model. Behaviours studied: binary SST-2 sentiment, GoEmotions multi-class emotion, six-language identification, multilingual sentiment on the same language corpus, and the deceptive-behaviour set they treat as held-out (questions about fictional entities where the prompt supplies the true answer, so a false answer is a lie by construction).

## How they did it

Four steps per example. First, the input-model is conditioned into a behaviour by few-shot prompting — a short dialogue whose turns are all, say, strongly negative movie opinions, ending in an open turn so the model must produce the next reply. Second, a single forward pass captures the hidden activations at the sampled layers. Third, the meta-model is prompted with a yes/no question about that behaviour plus a run of placeholder tokens ("Is this model acting negative?"). Fourth, during the meta-model's forward pass those placeholder embeddings are replaced by the captured input-model activations immediately after the embedding layer, and the meta-model's next token is read as the answer. Only the meta-model is trained; the input-model is frozen. The core experiment trains the meta-model on subsets of the four easy behaviours and evaluates it on the held-out deception set, varying which datasets are combined, so the object being measured is out-of-distribution transfer rather than in-distribution accuracy. The whole grid is re-run with the second meta-model. Every cell is the average of two runs.

## What counts as an explanation here

A natural-language yes/no judgement produced by a trained reader of another model's activations — a "natural language probe". It is deliberately *not* a feature dictionary, a circuit, or a description of what a direction means; it is a classifier that happens to be an LLM and happens to answer in words. The claim is partial and behavioural: the artefact predicts, at better-than-chance accuracy, a property of the input-model that it was not trained on, from activations alone and without the input-model's output text. Nothing in the account says *where* in the activations the lying signal lives or what it consists of.

## How they judged success

Accuracy on the held-out deception set, per training-dataset combination, averaged over two runs; comparisons are against an untrained meta-model and against the chance rate of the balanced binary task, with the authors' verbal claim that the architecture beats traditional probes on out-of-distribution transfer. No probe baseline is fitted or scored in the paper, no ablation isolates the contribution of the injected activations versus the prompt alone, no attack on the meta-model is attempted, and the two-run averaging is the only variance control.

## What they found

The untrained meta-model scores about 0.20 on deception and any training lifts it well above that, which the authors read as a large gain from training at all. The best configurations are the narrow ones: sentiment alone reaches about 0.75, while the model trained on sentiment plus emotion plus language identification sits just above chance at 0.5, and the all-datasets model is only fourth. Multilingual sentiment helps almost everywhere it appears, emotion and language identification are neutral to mildly negative, and the trend with more datasets is upward but not monotone, which they suggest is interference between training behaviours rather than a data-volume effect. The replication with the other meta-model reproduces the shape — multilingual still dominant, gains still non-linear — and reverses one sign: language identification hurts in the Llama experiment and helps in the InternLM one.

## Limits and open problems they name

The mechanism of transfer is not identified: why one training behaviour helps and another hurts is listed as future work, as are the choice of layers, the number of sampled tokens, wider spreads of sampling points, and whether the meta-model should be trained to answer yes/no before anything else. They want the method extended to more safety-relevant behaviours (malware generation, sycophancy, backdoors), to multi-modal models, to interpretation during pretraining, and — notably — they name adversarial attacks against meta-models as an open problem they have not done. The paper reports no interpretability of the meta-model itself and no intervention on the input-model, so its account is read-only.

## What we take from it

A method and a warning. The method: activation read-out by a frozen-weights *reader* is a legitimate alternative to a linear probe and can transfer across behaviour categories, and the placeholder-injection design makes it a two-line intervention (capture, substitute) rather than a training pipeline. The warning is the paper's own headline in reverse: the accuracy of this artefact is a function of *which* easy behaviours it was trained on, and adding more labelled behaviour made it worse — so for any reader we train on one task and evaluate on another, the honest table is the full dataset-combination grid, not the best combination, and the untrained-reader row has to be printed beside it. The 0.75 ceiling on a binary task, on two runs, with no probe fitted for comparison, is the number to keep: it is a demonstration of feasibility, not a benchmark, and it is the reference point for our own activation-side account. Secondary: their statement that meta-models beat probes out-of-distribution is asserted in the conclusion and is not measured anywhere in the paper — the folder's standing rule about a claim being scored by the experiment that could refute it applies directly.

## Tags

unit: layer | whole-model
form: description | probe
method: probing | other
eval: downstream-task
goal: understanding | safety/auditing
