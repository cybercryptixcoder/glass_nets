# Do Unlearning Methods Remove Information from Language Model Weights?

- **Key:** arxiv-2410.08827
- **Authors / organisation:** Aghyad Deeb (MATS, Harvard University); Fabien Roger (Anthropic; work done at Redwood Research)
- **Date:** 2024-10-11 (v1 2024-10-11; v2 2024-11-10)
- **Link:** https://arxiv.org/abs/2410.08827
- **Kind:** paper
- **Read from:** full text (abstract, sections 1-8, acknowledgements, reproducibility statement, full reference list, Appendices A-K with Tables 1-4)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Unlearning a dangerous capability from a language model is only a real safety guarantee if the capability is gone from the parameters rather than merely unreachable at the output. The authors point out that almost every evaluation of unlearning methods scores generated answers, which cannot separate those two outcomes, and that the existing white-box options each fail for their own reason: linear probes recover what is linearly present in activations but miss information that is only implicitly there, and relearning speed needs a model that never saw the data, which is unaffordable at pretraining scale. They want a procedure that any unlearned model can be put through, that costs one fine-tune rather than a pretrain, and that returns evidence about the weights rather than about the prompt format.

## What they studied

Decoder-only language models, chiefly Llama 3 8B, with the results cross-checked on other models and on Zephyr-7B-beta after RMU had been applied to it by its authors. The unit of analysis is the **whole model as a information channel**: a set of facts the developer declares should be gone, split into the part the attacker is given and the part the attacker must recover. Four fact collections are built or adapted: historical Years, MMLU recategorised into ten low-overlap categories, a de-duplicated WMDP (785 questions against 3260 in the original cyber and bio sets), and Random Birthdays, a synthetic set the models are first fine-tuned on because randomly drawn names and dates cannot be guessed. Each has a retain set (FineWeb-edu, or the MMLU categories held out) and exists in two formats, plain prose and multiple-choice.

## How they did it

The core move is to turn one fact set into a held-out prediction problem the attacker cannot shortcut. The developer picks facts whose answers are mutually independent — the formal condition is that knowing all the others plus the rest of training data does not make the missing one guessable — and splits them into T (four fifths, 628 items) and V (one fifth, 157 items). Unlearning is applied to the union, and deliberately in the **prose** format, because a method should remove a fact rather than one rendering of it. The attack, Retraining on T (RTT), then fine-tunes the unlearned model on the T questions in **multiple-choice** form, for six epochs under the Lion optimizer across a learning-rate grid from 1e-7 to 3.2e-6, recording accuracy after each epoch and reporting the best. Because T and V are independent, an attacker who was handed only T cannot have been taught V — any accuracy on V that appears is accuracy that was already latent. Unlearning strength is chosen separately for each method so that retain accuracy falls by at most five percent, with the whole ladder (10, 30, 100 percent) reported in an appendix. Methods tested are gradient difference (ascent on forget plus descent on retain), RMU (activation perturbation on harmful prompts), training on a random incorrect answer (RIA), and RepNoise as an example of a method explicitly designed to resist fine-tuning. The headline measure is a **recovery rate** — accuracy on V after RTT for the unlearned model, divided by the same for the original model — so 1.0 means nothing was removed and 0 means the information is gone. A stress test then builds a model whose hiding is deliberately fine-grained by fine-tuning the back half of the layers and unlearning only the front half on a retain set drawn from the same distribution as the forget set, so the model must hide individual birthdays rather than birthdays in general.

## What counts as an explanation here

Not a representation of the model, and not a mechanism. What the paper offers as its object of understanding is a **decision procedure with an information-theoretic reading**: the unlearning process has fully removed a fact exactly when the mutual information between the post-unlearning weights and the true answer is zero, and high accuracy on V after an attack that never saw V is offered as evidence that this mutual information is not zero. The claim is deliberately partial — the authors say the test raises the bar rather than certifying removal, and that a method which hides each fact in a separate part of the model could pass their test while still retaining the fact. It predicts nothing about which internal unit carries a fact.

## How they judged success

Recovery rate as defined above, against an upper bound on the standard error derived from the 157-item validation split and two T/V draws (about 0.055 at 95 percent), and with the retain-accuracy drop held at a stated level so that a method is not credited for destroying the model. Controls that matter: an RTT run on the original model that was never fine-tuned on Random Birthdays reached at most 31.2 percent on V, which is the evidence that the splits really are close to independent and that the attack is not simply teaching the format; the same unlearning methods are measured at four retain-drop levels; the recovered model is compared against the model that never had the knowledge wherever such a model exists; and a released RMU model is evaluated independently of the authors' own unlearning runs.

## What they found

Every unlearning method the authors tested recovered **more than 88 percent** of pre-unlearning accuracy on the held-out facts after a six-epoch fine-tune on facts it was allowed to see — for all methods, at every retain-drop level the paper reports, so the recovery is not an artifact of a weak unlearning setting. When the unlearning is pushed hard enough to cost a quarter of retain accuracy, post-attack forget accuracy still stays high. The deliberate high-granularity hiding stress test, which reaches 32 percent forget accuracy while keeping 72 percent retain, is recovered to 92 percent. RepNoise, a method built specifically to make fine-tuning ineffective, is recovered to about the level achieved by fine-tuning the original model, given a wider hyperparameter sweep. RIA — training on a random wrong answer — reduces accuracy less than the other methods to begin with. On format, plain prose for unlearning combined with multiple-choice for attack and evaluation gave the best balance across methods, and RMU generalises its unlearning more than gradient difference when the unlearning dataset is topically related but not identical to the evaluation facts.

## Limits and open problems they name

RTT is expensive — it needs a fine-tune and a learning-rate sweep for every evaluation. It does not prove removal: a method that hides each fact in a separate location would survive it, and the authors say so. The method only works on fact sets with little mutual leakage, which rules out capability removal (unlearning to code, for instance, has no clean T/V split), so the paper measures factual knowledge only and never a skill. Their own framing admits that making information hard enough to access may be the safer practical target, and that removal is a stronger property than safety strictly requires. The WMDP de-duplication is aggressive and cuts the benchmark by roughly three quarters, mostly by discarding questions their own filter wrongly flagged. No liveness check is run on whether recovered answers were well-formed after RTT, and the recovered accuracy is measured with multiple-choice scoring rather than generation. Everything is behaviour-at-the-surface after an intervention; no internal unit is inspected, so the paper can say information was present but not where.

## What we take from it

A method, a measure, and a warning. **Method:** the held-out-fact attack is the cheapest instrument in the folder that looks past the surface a method edited without needing a gradient or a dictionary — split an independent fact set, unlearn on prose, fine-tune on the visible part in a different format, and read accuracy on the part nobody was shown. For our white-box model this is the standing test for any claim of the form "this unit or direction no longer carries X": the prediction a genuine removal makes is that a short benign fine-tune on nearby data must not restore it, and the reference is the model that never had X, not chance. **Measure:** the recovery rate's denominator choice — the original model put through the *same* attack, rather than the original model's raw accuracy — is the correct normalisation and should be copied wholesale; it removes the part of the recovery that comes from the attack teaching the format. **Warning:** the independence condition is the load-bearing assumption and it is what makes the result interpretable, so any version of this test we run must carry its equivalent of the 31.2 percent no-knowledge control, or a high recovery rate means nothing. Two smaller steals: unlearning in one format and testing in another is a cheap and underused robustness switch, and the high-granularity stress model (train back half, unlearn front half) is a way to manufacture a hidden-but-present state on demand for calibration.

## Tags

unit: whole-model
form: probe
method: training-interpretable, other
eval: ablation, prediction/simulation, downstream-task, benchmark
goal: safety/auditing, evaluation-of-methods
