# SelfIE: Self-Interpretation of Large Language Model Embeddings

- **Key:** arxiv-2403.10949
- **Authors / organisation:** Haozhe Chen, Carl Vondrick, Chengzhi Mao — Columbia University; Mao also at Mila and McGill
- **Date:** 2024-03-16 (v1; v2 2024-03-26)
- **Link:** https://arxiv.org/abs/2403.10949
- **Kind:** paper
- **Read from:** full text (v2 HTML; figures not inspected as images, several numbers lost to stripped inline math)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The authors start from the complaint that we read a model's reasoning off its *output*, where a chain-of-thought may be unfaithful and a model may answer differently when asked to explain itself. Their alternative is to read the internal embedding directly, and their objection to existing read-outs is practical: a linear probe needs labelled data per concept and covers only a closed set, while lens-style decoders return only a token or two and cannot describe a complex concept. So the goal is an open-ended, training-free natural-language description of what a hidden vector carries, at any layer and any position. They then turn the descriptions into an editing handle: if a layer's vector can be described in words, one can define a target description and move the vector toward it, which gives a control method that edits a single layer's weights rather than the output behaviour. The larger ambition is a self-interpretation loop — the model explains itself, and the explanation becomes the specification for changing it.

## What they studied

LLaMA-2-70B-Chat as the main model, with LLaMA-2-7B and 13B used only in the capability ablation. The unit of analysis is a single hidden embedding — one layer, one token position — taken from a normal forward pass over a prompt. Evaluation corpora are TextWorld synthetic game states (12,900 generated samples, 3,400 held for evaluation and 9,500 for probe training), a Counterfact subset for fact editing (844 items the model answers correctly), a synthetic ethical-preference set of 101 scenarios about prioritising humans versus aliens, and Zou-style harmful-behaviour prompts for the erasure experiment. Qualitative case studies cover prompt injection, the trolley problem, a physics question about syrup, a social scenario, hallucination on a fictitious name, and deception.

## How they did it

The read-out is a hijacked forward pass, not a trained head. Take the embedding of interest from layer ℓ at a chosen position of the original run. Then run a second pass over an *interpretation prompt* of the form `[INST] [X] [/INST] Sure, I'll summarize your message:` — the placeholder `[X]` repeated five times — with the rule that on every step of generation, at the placeholder positions on a chosen later layer L, the computed hidden state is overwritten by the stored embedding. The model then continues the sentence, and the continuation is the description. Because the substitute is injected at a different layer than the one it was read from, the authors lean on the residual stream's additive structure to argue that the vector can be read at one site and consumed at another, and they test that assumption in an ablation over L.

To suppress the autoregressive padding that the model likes to produce, they define a per-token relevancy score as a treatment effect: the change in the log-probability of each generated interpretation token when the substitution is performed versus not performed. High-relevancy tokens are the ones the injected vector actually caused; the rest are the model continuing its own sentence, and the visualisations highlight the former.

For control they reuse the same interpreter as a differentiable objective. *Supervised control* takes a pair (source embedding, target embedding) chosen by reading descriptions — the target being an embedding elsewhere in the model that interprets to the desired text — and minimises the MSE between the edited layer's output on the source and the target vector, by gradient descent on that one layer's parameters, with a WikiText mean-square regulariser that keeps the layer's behaviour on unrelated inputs near its original output. *Reinforcement control* drops the target vector: an evaluator (here LLaMA itself) reads the description of the layer's output and returns a signed reward, and the layer parameters are updated on a stop-gradient proxy loss so that outputs whose descriptions earn negative reward are pushed away.

## What counts as an explanation here

Text. The deliverable is a natural-language sentence, plus a per-token relevance highlight over that sentence saying which words the vector caused, and a layer-by-layer reading of the same prompt as a narrative of how the model got from input to answer (symbols creating urgency early, a crisis inference later, compliance last). It is accessible rather than complete: nothing in the method claims the sentence exhausts the vector, and the only check on whether the sentence is about the vector is the treatment-effect score. Prediction is indirect — a good description is one whose tokens would not have been produced without the injection.

## How they judged success

Four strands, with different measures. (1) TextWorld: for each sample the interpretation must pick, strictly between a positive and a negative state word, the state the entity is actually in; accuracy is compared against a linear probe trained on 1, 2, … up to 100 shots and on the full set. The authors also remove spurious correlation in the dataset so that entity alone or context alone lets the probe do no better than chance, and they randomise the order of the two option words in the prompt. (2) Fact editing on Counterfact: efficacy (original prompt returns the target answer), paraphrase (a reworded prompt does), specificity (an unrelated fact is still answered), and their harmonic mean, against fine-tuning, RepE and ROME; the FS and ROME baselines are constrained to work at the last layers for computational reasons the authors state. (3) Ethical control: the proportion of 100 unseen scenarios answered with the instructed preference before and after editing, against an identical procedure with a randomly chosen embedding and target. (4) Harmful-knowledge erasure: the fraction of responses containing harmful content on the erasure prompt and on 388 other harmful requests delivered inside a prompt-injection template, plus retained accuracy on the 844 Counterfact facts. Ablations sweep the insertion layer L and the model size.

## What they found

TextWorld: the zero-shot interpretation matches a linear probe trained on 100 labelled samples, and accuracy rises with the layer the embedding is read from. *I did not read the plotted accuracy value* — the figure caption states these comparisons qualitatively and the numeric cell was lost in the HTML conversion, so no percentage is quoted here.

Case studies: the interpretation narrative shows plausible staged structure — urgency inferred from exclamation marks at an early layer and a user-in-crisis inference later, before compliance; a "glittery → thick → viscosity" chain for the syrup question; and a hallucination path in which the model associates a fictitious name with a known country, a similar surname, and that surname's profession, then answers as if the fictitious person had the borrowed profession. These are individual traced examples, not a systematic evaluation.

Fact editing (Table 2): supervised control reaches a higher harmonic mean than ROME, fine-tuning and RepE — the paraphrased-prompt column is where it wins most clearly — with efficacy and specificity roughly level with ROME while touching fewer parameters. Ethical control (Table 3): before editing, 100% of the 100 unseen scenarios prioritise humans; after two gradient updates on layer 6 with a single embedding pair, 96% prioritise equally, 2% still prioritise humans and 2% answer otherwise, against 95% / 1% / 4% for the random-embedding control — so the random control is the number that shows the edit is not merely a perturbation, and the gap is 1 percentage point on the primary outcome.

Erasure (Table 4): after reinforcement control on one harmful prompt, the share of harmful responses to non-injection prompts falls slightly (table cells 1.84% before, 1.79% after) while the share for injection-templated prompts drops from 89.06% to 4.4%, at 95.85% retained fact accuracy — i.e. the same prompt that the model refused without an injection and complied with under one stops being answered harmfully once the layer is edited. The text's own summary is that injection's ability to elicit harmful answers across 388 activities was cut by about 85% while preserving about 96% of fact-answering, but the prose also reads "the editing reduces the success rate … from [blank] to [blank], achieving a [blank] editing effective rate", so **the headline percentages in the sentences were stripped with the inline math and only the table cells survive**.

Ablations: insertion at early-to-middle layers (the text says L around 5 gives the best accuracy, Fig. 11) beats late insertion, which the authors attribute to late injection not reaching the final position in time. Smaller models look worse on the task, but once responses that ignore the two-option instruction are excluded the 7B and 13B match the 70B — and the failure rates explain the gap: 27.17% of 7B and 17.68% of 13B interpretations disobey the instruction format, against 3.84% for 70B.

## Limits and open problems they name

The paper is short on limits — there is no dedicated limitations section, and the caveats appear inside the results. They note that fine-tuning cannot be scaled past the last layer on a 70B model and that ROME's editing-site search is limited to the final layers, so Table 2 is a comparison of figures at unequal effort. They note the smaller models' instruction-following as the explanation for their weaker scores. They do not address that the evaluator in reinforcement control is the same model being edited, that the ethical-control gain over a random-embedding control is a single percentage point on 100 scenarios, or that no experiment isolates how much of an "interpretation" is the injection and how much is the model's prior over plausible descriptions.

## What we take from it

A **method**, and the earliest and cheapest form of it in the folder: the read-out is a forward-pass substitution with a fixed prompt, so the "probe" has zero fitted parameters and can be pointed at any layer and position without collecting a dataset — worth having beside `arxiv-2505.13763`'s in-context reporting and the anthropic-com introspection post, which are the same idea with a training loop attached. A **measure**: the treatment-effect relevancy score is the right way to price an autoregressive description, and it is the same counterfactual-with-and-without-substitution logic we would use for any generated explanation — a token's probability under injection minus its probability without injection. A **warning** that applies to our own generated descriptions: a language model asked to describe a vector will happily produce a fluent sentence that continues its own first plausible guess, which is exactly the failure `arxiv-2608.30980` measures as a parse-and-quality split and `arxiv-2402.04614` names as plausibility outrunning faithfulness; the relevancy score is the cheapest partial repair. And a **contrast** worth carrying against the self-report entries: here the model interprets its *activations*, not its behaviour, and the ground truth is an entity state the model provably holds — conditions under which asking is cheap and works — so the paper is a boundary case for run 372's "a question spanning many tokens should be read out externally" rather than a contradiction of it.

## Tags

unit: layer
form: description
method: other
eval: downstream-task | prediction/simulation
goal: control/editing
