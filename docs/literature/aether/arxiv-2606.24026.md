# Can Language Model Agents be Helpful Circuit Explainers in Mechanistic Interpretability?

- **Key:** arxiv-2606.24026
- **Authors / organisation:** Ayan Antik Khan, Harsh Kohli, Yuekun Yao, Huan Sun, Ziyu Yao (affiliations absent from the abs page; the acknowledgements name the Office of Research Computing at **George Mason University**)
- **Date:** 2026-06 (v1 2026-06-23, v2 2026-09-02)
- **Link:** https://arxiv.org/abs/2606.24026
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Circuit *localisation* has been automated — ACDC, EAP, attribution methods — but the step after it, saying what the localised components actually do, is still done by hand, one paper at a time, and there is no standard way to score it. The authors ask whether a language-model agent can do that explanation step once a circuit has been handed to it: assign each component a functional role, justify it with a causal experiment, and recover the task the circuit as a whole implements. The larger goal is scale — the human loop of "look at activations, guess a role, design a patch, check it" is the part of mechanistic work that does not grow with model size. They also want a *controlled* evaluation, because on famous circuits an agent can recite published roles instead of deriving them.

## What they studied

84 semi-synthetic decoder-only transformers taken from InterpBench, which retrains Tracr/RASP-compiled models under Strict Interchange Intervention Training so their weight distributions resemble naturally trained models while their circuit stays known by construction. 163 annotated components in total (120 MLP, 43 attention), one to ten components per circuit, average 1.94. Each component carries a role tag from a five-class taxonomy (Mapper 72, Combiner 33, Aggregator 32, Indicator 15, Router 11) plus a free-text note. The two IOI tasks are deliberately excluded as too well known. A separate case study uses a 10-component circuit for three-operand addition in Llama-3-8B, taken from the All-for-One subgraph. The unit of analysis is the transformer component — attention head or MLP sublayer — inside an already-given circuit.

## How they did it

The benchmark is built by tracing: for each localised component they read the RASP source and InterpBench's high-level/low-level correspondence map, find which RASP variable the trained component implements, and label it from that variable's role. Each item ships with the task description, the RASP program, five input-output examples and the per-component annotations.

Their agent, HyVE, runs one component at a time through a three-stage loop. *Observe*: it writes an observation plan (goal, steps, tensors needed, expected pattern), writes Python against a helper library for attention patterns and activations, executes it, and summarises. *Hypothesise*: it states a short natural-language claim grounded in that observation, with the role taxonomy withheld at this stage so the label does not constrain the reasoning. *Validate*: it writes a validation plan naming the intervention (ablation, activation patching, interchange intervention), writes and executes the code, and issues a binary verdict; a refuted hypothesis goes back into the next round as context. After every component is processed it assigns tags, writes notes, and synthesises a circuit-level summary from which a task description is derived. Code runs in a sandbox against a TransformerLens-loaded model, five execution attempts per stage, three hypothesis iterations per component; the whole thing is a LangGraph state machine.

Four backbones: GPT-5.4, Claude-Sonnet-4.6, Gemini-3.1-Pro, and Qwen-3-Coder-30B-A3B (temperature 0 for the closed models, 0.7 for Qwen).

## What counts as an explanation here

A **role dictionary plus a task description**: one taxonomy tag and one short natural-language note per component, plus a paragraph describing how information flows through the circuit and a derived description of the task. It is partial by construction — the circuit is given, so nothing here is found — and it is textual rather than a surrogate or a program. The notes are meant to describe *task-specific* behaviour ("computes a per-position flag for whether the token is x"), not a generic feature. The paper is explicit that the taxonomy exists only for scoring and is not offered as a general ontology of transformer components; the agent's real output is the prose.

## How they judged success

Three levels. Component level: tag accuracy against the ground-truth tag, and description quality on a 3-point LLM-judged scale (0 wrong, 1 partial with an incorrect mechanistic sub-claim, 2 correct) rescaled to [0,1]. Task level: binary LLM-judged correctness of the derived task description. Process level: the fraction of code executions that run without error. Two LLM judges (GPT-5.4 and Gemini-3.1) score independently and the **lower** score is kept, to avoid over-crediting and to blunt self-preference. They validate the judges against humans: two annotators, blinded and randomised, on 10 tasks / 17 components / 68 component instances; human-human agreement κ = 0.80 on description quality and 0.96 on task accuracy, human-judge agreement κ = 0.75 and 0.86. They also report per-judge scores, three repeated runs with standard deviations, and two ablations (no observation stage, no validation loop). Validation-plan soundness is scored by humans on a 0-2 scale but is *not* adopted as an official metric because its agreement is only κ = 0.46.

## What they found

Useful explanations, no uniformly best backbone. Best tag accuracy 0.79 (Claude), best description quality 0.59 and best task accuracy 0.83 (Gemini, eight points above second place), best code execution 0.93 (Claude). GPT-5.4's 0.52 execution rate visibly caps its final explanation quality. Qwen is far behind on the judged metrics (0.25 description quality, 0.25 task accuracy) despite a 0.67 tag accuracy.

The diagnosis splits the loop. All three proprietary backbones produce almost perfectly observation-grounded hypotheses (mean 1.94/2, 94.1% fully grounded), while Qwen scores 1.41 and over-specifies its hypotheses with unsupported per-neuron or positional mechanisms. Validation-plan soundness inverts the ordering: GPT-5.4 1.71, Gemini 1.41, Claude 1.24, Qwen 0.71 — so the backbone with the best plans executes them worst, and the one with the best execution writes weaker plans.

Both stages earn their place: removing observation or validation costs 1-18 points of task accuracy for every backbone, and the full pipeline wins 14 of 16 component-level comparisons. Against Claude Code given the same context but no imposed workflow, HyVE scores higher on tags (0.81 against 0.77) and description quality (0.64 against 0.60) and matches task accuracy; manual inspection finds Claude Code performed an actual causal intervention in only 15 of 31 cases, and in 4 it never loaded the model at all.

On the Llama-3-8B addition circuit, human raters (majority of three) scored Claude 8 correct / 2 partial / 0 wrong, GPT-5.4 6 / 3 / 1, Gemini 1 / 2 / 7. Gemini's failure mode is named: it treats positional or logit-lens signals as causal. The trace of one backup head is the clearest single artefact — the first hypothesis (a primary C-router) is refuted by an ablation showing no drop, and the second (a backup active only when two stronger C routes are removed) is supported by ablating the three together.

## Limits and open problems they name

Explanation only, post-localisation; end-to-end discovery is out of scope. The benchmark circuits are smaller, more structured and more algorithmic than real mechanisms. One agent design, one prompt set, one helper library — the authors expect constrained execution interfaces to cut the code errors. They explicitly flag data leakage as a threat to any evaluation built on published circuit explanations, and note that while RASP-derived ground truth dodges it inside the benchmark, the concern persists for the real-model case study. Validation quality has no reliable automatic metric (κ = 0.46), which is why it is reported as a diagnostic rather than scored.

## What we take from it

A **framing and a measure**. The framing: decompose "explain this" into three separately scored stages — is the hypothesis grounded in what was observed, is the experiment sound for that hypothesis, and did the experiment actually run — because here the two middle stages dissociate sharply across backbones and a single end-to-end accuracy number would have hidden it. Any agentic or semi-automated account we build should report that triple, and should keep a refuted hypothesis in the transcript rather than overwrite it: the refuted-then-revised trace of the backup head is the most legible piece of evidence in the paper.

A **warning about the target**, which is the part that matters most for us. Ground truth here is a role tag read off a RASP program and a human-written note, and the note itself is what the LLM judge compares against — so a high description score means *agreement with one annotator's phrasing*, not completeness or mechanism. The Llama-3-8B case study is honest about the alternative: the reference roles there were built by hand with targeted interventions, keeping causally redundant components in on purpose, precisely so that "logit-lens-positive but causally weak" could be scored as wrong. That construction — retaining the decoys — is the reusable idea for our own validation sets: build the gold set so that a plausible-but-incorrect answer is representable, or the benchmark measures fluency.

A **shortcut**: Tracr plus interchange-intervention retraining gives circuits whose components are known by construction, with scaffolding code already public; the price is model size and realism, and the authors' own case study is the only check that it transfers.

## Tags

unit: circuit
form: description
method: agent
eval: benchmark
goal: evaluation-of-methods
