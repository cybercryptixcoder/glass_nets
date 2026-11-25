# The Hard Decision Layer: Evidence for Committed Inference in Transformers

- **Key:** arxiv-2607.21613
- **Authors / organisation:** Ashwath Vaithinathan Aravindan; Mayank Kejriwal — University of Southern California / Information Sciences Institute
- **Date:** 2026-05-25
- **Link:** https://arxiv.org/abs/2607.21613
- **Kind:** paper
- **Read from:** full text (v1 HTML; all seven tables read cell by cell, figures read from their captions and from the prose that describes them, never inspected as images)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The paper asks where inside a transformer a multiple-choice answer is actually settled. Prior accounts of the same question come in two shapes: learned early-exit routers that decide per layer how much depth to spend, and post-hoc layer pruning that measures which blocks can be dropped. The authors argue neither describes how the model *itself* organises its computation, and that what is missing is a static architectural property marking the point at which the answer stops being revised. Their larger goal is efficiency — if the answer is fixed at depth d, everything after d is compressible — and secondarily an argument that transformers allocate depth in a task-shaped way rather than a dataset-shaped one.

## What they studied

Four instruction-tuned models spanning 2B-8B params and four architectures: Granite-3.3-2B-Instruct, Qwen3-4B-Instruct-2507, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct. Four multiple-choice benchmarks: CommonsenseQA, QASC, MMLU-Pro, SuperGPQA, sampled at 100 questions each and normalised to exactly four options. The unit of analysis is a **layer**, and specifically the layer at which the ranking of the four option tokens settles. No neuron, feature or head is examined anywhere in the paper.

## How they did it

A single forward pass per prompt with gradients off, capturing the residual stream at every layer for the final prompt position. Each intermediate hidden state is passed through the model's final layer norm and then the unembedding matrix (the standard logit lens), and only the four option-token logits are retained. For every question the four options are ranked at every layer, and the authors aggregate by *output rank*: they take the four options' final-layer ranks as a fixed labelling of the series, then average each labelled series across questions with its standard deviation. Layer accuracy is the fraction of questions whose correct option ranks first at that layer. The Hard-Decision Layer is then **defined as an argmax over layers of the drop in the average rank of the option that ends up first** — the argmax is taken over mean rank, not over accuracy, so no new measurement enters the definition. Robustness arms: (a) LoRA fine-tuning of Qwen and Llama on 8,000-10,000 samples per dataset, one epoch, rank 32, learning rate 2e-4, batch 64, prompt length 1024, via the Tinker library, followed by re-extraction; (b) option count varied to 3 and 5 on QASC; (c) option label format varied to Arabic numerals and Roman numerals on QASC; (d) a side experiment on GSM8K free-form generation capturing residuals at output-token positions 10, 50, 100 and 200 and tracking the top-10 logit-lens tokens along depth.

## What counts as an explanation here

A **single integer per model** — the index of one layer — plus a qualitative claim about what happens there ("the model commits"). The artefact is a *schedule* of where computation is spent, not a mechanism: there is no unit, edge, direction or dictionary behind the layer, and nothing in the paper says which computation inside the critical layer performs the collapse. It is explicitly meant to be partial but is presented as a general property of the architecture, and it does predict one thing — that truncating the model at that layer should preserve most of its multiple-choice behaviour, which the discussion recommends but, notably, never tests.

## How they judged success

Three quantities, all descriptive: the HDL's layer index and its stability across datasets, fine-tuning, option count and label format; the jump in logit-lens accuracy at the HDL versus the jump from there to the final layer; and the mean option-1 rank and its standard deviation before and after the HDL. Crucially, **random chance is the only baseline** (0.25 for four options, 0.33 for three, 0.20 for five) and it appears only as the level of the pre-HDL accuracy curve. There is no null model for the HDL itself — no permuted-label run, no shuffled-dataset run, no test of whether a steepest-drop point at that depth would arise from a random rank trajectory — and no confidence interval on the layer index. The one claimed ablation study is the option-count and label-format sweep, which are format variants rather than interventions on the computation.

## What they found

A single HDL per model, consistent across all four datasets: Qwen at layer 25 of 36, Llama at 18 of 32 (with one exception discussed below), Granite at 32 of 40, Mistral at 20 of 32. Expressed as proportional depth these are 69%, 56%, 80% and 63%, and the authors note a trend, on four points, that larger models commit earlier. Pre-HDL accuracy sits at or below chance in the large majority of model-dataset cells (Qwen 0.23-0.31, Llama 0.24-0.29, Granite 0.27-0.32, Mistral 0.27-0.38), the accuracy at the HDL jumps sharply, and the remaining layers add little: Qwen on CommonsenseQA goes 0.26 to 0.87 at the HDL and then *falls* one point by the final layer; its QASC cell goes 0.31 to 0.70 and gains two more points; Granite on QASC 0.27 to 0.51 then to 0.67. Llama is the model where the post-HDL layers still do real work and where the story is least clean: the jump at layer 18 is smaller than Qwen's and the remaining layers add more — 0.63 to 0.80 on CommonsenseQA and 0.59 to 0.74 on QASC — and on SuperGPQA Llama's HDL is detected at layer 23 rather than 18 in the base model, with post-HDL accuracy of 0.41 falling to 0.32 at the final layer. That SuperGPQA cell is the paper's one visible counter-example to invariance, and the fine-tuned Llama on the same dataset puts the HDL back at 18. Fine-tuning leaves the HDL where it was in eleven of the sixteen base/fine-tuned cells and moves it only in Llama on SuperGPQA; option count and label format leave the position unchanged in every cell, though Roman numerals produce a visibly gentler accuracy climb (Qwen 0.27 to 0.57 at the HDL and then a further twenty points to the final layer, against one to three points for alphabets and Arabic numerals). In the GSM8K arm the top-1 token stabilises around the same layer for every generation position measured, but the authors describe the effect as much less pronounced than in multiple choice and read it as suggestive only.

## Limits and open problems they name

Restriction to direct answer selection with no chain-of-thought; analysis of only the canonical option letter, ignoring alternative surface forms of the same answer ("b", "second", "B)") that intermediate layers might prefer; and free-form generation needing far more work before anything generalises. To these the paper adds a caveat in its own pruning recommendation — when prompt formatting differs, the last layers become essential, so aggressive truncation is unsafe — which is in direct tension with its claim that the HDL deserves to be treated as the effective end of the model.

## What we take from it

- **A warning, mostly.** Our model is supposed to predict what happens under intervention. This paper offers the most tempting short-cut form of that — "the answer is fixed after layer k, so intervene near k" — and its own Tables show why that is unsafe: on Llama/SuperGPQA accuracy **decreases** from the HDL to the final layer (0.41 to 0.32), and on the fine-tuned Llama/MMLU-Pro cell the post-HDL value falls below the pre-HDL value in a way that no single commitment layer explains. A collapse in the *ranking* of the eventual answer is compatible with the answer still changing, so rank stability is not commitment.
- **A measure worth stealing, in a fixed form.** The paper's `output rank` aggregation — label each option series by its final-layer rank, then track that labelled series backwards — is a clean way to draw a backward trajectory rather than a forward accuracy curve, and it is the one instrument here we would reuse.
- **A gap to exploit.** The HDL is defined by an argmax on the same curve it then describes, with chance as the only reference. The natural repair is to compare the observed steepest drop against the steepest drop of a matched null trajectory (same per-layer rank dispersion, ranks shuffled across layers), which would turn "there is a critical layer" from a description into a test. Any layer-depth claim in our own work should ship that null.
- **A prediction we would check before quoting.** The discussion's recommendation to prune everything after the HDL is never evaluated. That is a one-line experiment — truncate and measure — and its absence is the clearest sign of what the paper is: a descriptive schedule, not a causal account of where the computation is.

## Tags

unit: layer
form: probe
method: probing
eval: benchmark
goal: understanding
