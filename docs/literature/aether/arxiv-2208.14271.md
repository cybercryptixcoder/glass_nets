# Faithful Reasoning Using Large Language Models

- **Key:** arxiv-2208.14271
- **Authors / organisation:** Antonia Creswell; Murray Shanahan — both DeepMind
- **Date:** 2022-08-30
- **Link:** https://arxiv.org/abs/2208.14271
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

A single forward pass of a language model gives an answer with no account of itself: which premises it
used, whether it used the supplied context at all, or whether it leaned on memorised world knowledge, are
all invisible, which makes an answer impossible to check and hard to debug. The authors want a system whose
intermediate computations are *logically* inspectable — where a user can verify that each step follows from
the previous ones and that the conclusion follows from the steps. Their stated framing is that a faithful
system is one whose computation mirrors the textbook definition of logical validity, not one whose
narration sounds sensible. The paper is an explicit attempt to buy faithfulness by architecture rather
than to measure it after the fact on a model that was trained for accuracy. It sits in a line the same
group was already developing (selection-inference; the earlier multi-step deductive paper is cited as their
own prior work), and its contribution here is turning that backbone into a question-answering system with
termination and search.

## What they studied

The unit of analysis is a *reasoning step* and the *reasoning trace* built from steps, not any internal
state of a network: there are no probes, no activations and no weights analysis anywhere in the paper. All
components are 7B Chinchilla language models, fine-tuned separately, one model per role. The domains are
multi-step deductive entailment over a supplied context (ProofWriter, open-world subset, depths 1, 2, 3
and 5) and grade-school science multiple choice (a question-answering reformulation of EntailmentBank that
the authors build themselves, both with and without distractor statements in the context). The object being
produced is a forward-chaining proof written in natural language, with each step explicitly grounded in
sentences of the context by label.

## How they did it

1. A **selection** model sees the question plus the accumulated context and outputs only *sentence labels*
   ("sent 3"), never free text; the labels are then substituted back with the actual statements, so a step
   cannot cite a fact that is not in the context. 2. An **inference** model receives only the selected
   statements — deliberately not the question — and writes the entailment, which is appended to the context
   so the next step can cite it. 3. A two-stage **halter** is applied to every inference: first it is asked
   whether the question is answerable given that inference, and only if it says yes is it asked to pick
   among the choices using the inference plus the choice list. Exhausting a step budget returns "Unknown".
   4. Because selection samples, traces form a tree; a **value function** (a fine-tuned model scoring the
   log-probability that the current step is "correct") drives a step-level beam search, keeping the best B
   of P candidates per step. 5. Each component is trained in isolation on ground-truth traces (one selection
   point and one inference point per gold step; negative search examples made by swapping one supporting
   statement for a random one and re-running inference), and the pipeline is never optimised end to end for
   answer accuracy. For PW the halter is collapsed into a single-step "Unknown / True / False" prediction
   because the question alone is insufficient there.

## What counts as an explanation here

A natural-language proof: an ordered list of (selection, inference) pairs, every selection drawn from the
context by label, plus a final answer derived from the last inference. It is a **partial, task-level**
explanation: it explains a single answer on a task where the answer is a logical consequence of supplied
premises, and it claims no account of how the language model internally produced any individual step. The
authors are explicit that the guarantee is conditional — connectedness is guaranteed by construction,
per-step correctness is only encouraged — so the trace is a checkable certificate whose *validity* is
verifiable in principle by a reader, not a description of the network's computation.

## How they judged success

Final-answer accuracy against three baselines (proof-then-answer fine-tune, proof-only fine-tune paired
with their own halter and search, and the EntailmentWriter entailment-tree model extended to answer). Trace
quality is scored several ways against gold: Jaccard similarity of selected leaves, Jaccard of full steps,
Rouge on intermediate inferences (order-ignoring and order-respecting), exact string match on PW. Validity
is probed with two context manipulations rather than argued: an EntailmentBank run with a **random context
from another problem** (a model that really reasons over context should collapse, and the reported quantity
is both the random-context accuracy and the delta to correct context), and a ProofWriter run with an
**incomplete context** (rules present, facts removed). Fact fabrication is counted directly as the share of
problems whose trace contains a statement absent from the context. Inference is also audited in isolation
by feeding gold selections (exact match on PW, Rouge and BLEURT-thresholded accuracy on EB). Search is
ablated by running the same pipeline with and without the value function, and with a value function trained
on the *other* dataset, which is a transfer check on the scorer rather than on the task.

## What they found

SI + halter + search reaches 88.1% overall on ProofWriter (depth-5 63.4%) and 83.2% / 72.9% on
EntailmentBankQA Task 1 / Task 2, against 65.4% and 53.5% overall for the answering baselines on PW, and
64.6% / 7.8% for proof-then-answer on EB. Search is worth a great deal and the gain scales with difficulty:
SI + halter alone is 78.3% on PW but only 38.4% at depth-5, and search lifts depth-5 to 63.4%; the
proof-only baseline shows the same shape (79.9% to 85.4% overall, 44.6% to 56.8% at depth-5). On EB the SI
and proof-only models are close after search (83.2/72.9 against 82.9/76.2), and the authors say plainly that
the reason to prefer SI there is the trace rather than the score. The validity probes are the sharpest
result: on EntailmentBank with a random context SI answers only 9.4% of the time against 30.0% for
proof-then-answer and 23.0% for EntailmentWriter (deltas 63.0 / 34.6 / 27.0 points); on ProofWriter with
rules-only context SI drops 48.8 points against 4.3 and 0.1 — i.e. the two baselines keep working when the
facts needed for a proof are absent, which is the direct evidence that they were never using the trace.
Fabrication splits by dataset and the split is informative: on EntailmentBank proof-then-answer invents
facts on 60% of Task 2 problems (proof-only + halter 23%) while SI is at 0-1%, but on ProofWriter every
model including SI is at 0-4%, so the constraint is doing work only where the context is long enough to make
copying harder than inventing. The gold-selection audit shows inference is nearly exact on PW (99.9-100%
across depths) and adequate on EB (RougeL 0.69, 64% above the BLEURT threshold). Haltering is also a
precision instrument: filtering to answers the halter claims to know takes PW to near-perfect at every
depth and EB to 87.5% / 83.7%, and on the incomplete-context arm SI declines to answer 69.7% of problems.

## Limits and open problems they name

Correctness of individual steps is assumed, not guaranteed: connectedness comes from the label trick, but
logical validity of an inference is only made *more likely* by denying the inference model the question and
by the value function, and the authors say so twice. Two adjacent steps can be individually connected yet
the trace spurious — the appendix shows high-value traces on which a wrong premise is reused three times
and a rule is applied to an entity it does not license, which is exactly the failure a reader would catch
and the architecture does not. Everything assumes a supplied context sufficient to answer the question;
retrieval to populate it is left to future work. The EntailmentBankQA dataset is their own reformulation, so
cross-paper comparison on EB is not apples-to-apples. Evaluation of trace quality relies on string overlap
against gold proofs, and the authors note those metrics ignore step order, which flatters baselines that
can emit a late step without computing the earlier ones. There is no human evaluation of whether a trace
persuades or is actually checked by a reader. Nothing in the paper measures or claims anything about the
internal computation of the 7B models that generate the steps.

## What we take from it

A framing and a design pattern. (1) **The "faithful from X to Y" sentence, with both edges named**: here
the edge answer-ward is guaranteed by construction (the answer cannot see the question, only the final
inference; the inference cannot see the question, only the selection) while the edge from the model's
internal computation to the *selection and inference text itself* is entirely unexplained — the paper is a
worked example of paying for faithfulness by cutting the model's access to information, not by
understanding how it uses information it keeps. (2) **The blocked-path control is the reusable instrument**:
random-context and incomplete-context runs are the cheapest possible test of "did the answer come from the
account", and they produced the paper's only clean separation between architectures (9.4% vs 30.0%;
a 48.8-point delta vs 4.3). For our surrogate work the analogue is a substituted input in which the
putative cause is removed while a shortcut remains available — the shortcut's survival, not the accuracy
drop, is the measurement. (3) **Ablate the interface at each edge separately**: here the ablation table
shows the halter and the search each give compounding gains and that the proof-only baseline plus their own
halter and search recovers most of the accuracy on EB, so the credit for the *score* belongs to the
scaffolding; the credit for the *trace* is the architecture's, and the two are reported in one table.
(4) A warning: the paper's own fabrication numbers are dataset-dependent (0-1% on EB but indistinguishable
from baselines on PW), so "our structure prevents hallucination" is a claim that needs the long-context
condition to be interesting at all.

## Tags

unit: whole-model | form: program | method: decomposition | eval: faithfulness | goal: understanding
