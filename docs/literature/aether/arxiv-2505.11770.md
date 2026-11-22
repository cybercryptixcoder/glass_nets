# Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors

- **Key:** arxiv-2505.11770
- **Authors / organisation:** Jing Huang, Junyi Tao, Thomas Icard, Diyi Yang, Christopher Potts; Stanford University (the HTML byline lists Stanford for every author)
- **Date:** 2025-11-10 (v2; v1 posted 2025-05-17). Comments field: ICML 2025
- **Link:** https://arxiv.org/abs/2505.11770
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Causal-abstraction work normally runs one way: a model succeeds, so we search its weights for the abstract
mechanism that explains the success. This paper asks for the reverse direction, which it calls *prediction*: once
we have identified an abstract causal mechanism for a task, can we use it to say in advance whether a particular
output will be correct, especially when the input has been perturbed in ways that leave the task itself unchanged?
The motivation is practical rather than explanatory. Behavioural testing cannot enumerate a compositional input
space, confidence scores are miscalibrated away from the training distribution, and the internal-feature probes
that have been proposed for hallucination detection are chosen by location heuristics rather than by any account of
what the model is actually computing. If mechanisms are what make a model generalize, the features that implement
them should also be the features that tell you when the model has stopped implementing them.

## What they studied

One decoder-only language model family, chiefly Llama-3-8B-Instruct, with Llama-3-8B base, Llama-2-13B-chat and
Llama-3-70B-Instruct for a size comparison on one task. Five tasks spanning three kinds of structure: symbolic
manipulation (indirect object identification, and a numeric price-interval comparison), knowledge retrieval (a city
to country attribute task, and multiple-choice MMLU), and refuse-to-answer instruction following on a Harry Potter
unlearning prompt. The unit of analysis is a **variable-level subspace** of the residual stream: each task's
high-level causal model is taken from prior interpretability work, and each of its intermediate variables is located
to an orthonormal subspace at a chosen layer and token by distributed alignment search. There is also a comparison
arm in which the causal variables are located but the probe is trained the ordinary way, plus two feature families
that never use a causal model at all.

## How they did it

Every task is turned into a template with input variables in known positions, so all valid prompts can be
enumerated and the ground-truth answer is a pure function of those variables. They run the model, parse its output
by task-specific rules, and label each output correct or wrong, discarding undecided or off-instruction outputs
(under a tenth of responses). A balanced set of inputs is then used for three purposes: a *verified* set of labelled
in-distribution examples, a train split for anything that needs supervision, and a test split. Distribution shift is
defined against the **task mechanism**, not against the training corpus: the test inputs are perturbed only in
background conditions the mechanism does not use — a different currency symbol, added typos, names transliterated
into another language, the prompt template reworded, different in-context demonstrations, option markers renamed,
an extra distractor option, a longer passage — and the perturbations are selected so that task accuracy drops
noticeably. Localization trains the alignment on counterfactual input pairs drawn from the verified set, sweeping
intervention dimension over powers of two and intervention site over variable tokens and nearby positions, and
picks the configuration by interchange-intervention accuracy. Two prediction methods are then built on the located
variable. **Counterfactual simulation** takes a test input's value of the variable, transplants it into a verified
example whose other variables were set differently, and reads the resulting output: marginalising over background
values from many verified examples gives an estimate of the output distribution conditional on the variable alone,
and the score is either the likelihood of the expected tokens or, more sharply, whether the decoded variable value
matches the one the high-level model predicts. **Value probing** fits a small classifier on the variable's subspace
to predict which discrete value it holds, and scores correctness by the classifier's probability for its chosen
value, so a representation sitting near a boundary or outside the region all values occupy is flagged. Baselines
are: first-n-token and answer-token confidence, with and without temperature scaling; ordinary correctness probes on
the last prompt token and on the background variables; and an ordinary probe on the located causal variables. They
compare in-distribution and out-of-distribution test sets by AUC-ROC on the balanced task, and separately plot
interchange-intervention accuracy against the resulting AUC.

## What counts as an explanation here

A **causal abstraction used as an instrument, not as a description**. The high-level model is a small causal graph
over task variables; the account of the network is the pair (graph, located subspace per variable). The claim the
paper makes for it is predictive and partial: it does not say what every feature does, only that this subspace
carries this variable, and the payoff is a forecast about a single output's correctness. The authors are explicit
that the graph is usually imported from prior work rather than found, and they gesture at auto-interpretability as
a way to generate candidate graphs — a natural-language description of a subspace would become a three-node
chain — but nothing in the paper depends on that route being solved. Nothing here is a full white-box model: there
is no claim that the located variables exhaust the computation, and the background complement is treated as an
unnamed remainder to be marginalised away.

## How they judged success

AUC-ROC on a balanced correct/wrong test set, mean over three random folds with a standard deviation, at both an
in-distribution test split and several out-of-distribution splits per task; a reported ten OOD settings across the
five tasks. Controls and comparisons: causal features against output probabilities at two temperatures and against
probes placed by location heuristic; the located variable probed in the ordinary supervised way as a bridge arm
between the two families; a size comparison on one retrieval task; and a check of whether better simulation implies
better prediction (interchange-intervention accuracy against AUC) including the regimes where the relation breaks.
Hyperparameters for every method are chosen on a validation split drawn from the training distribution, and the
tables flag values within one standard deviation of the task's best.

## What they found

Causal features beat everything else out of distribution, and the gap is largest where accuracy fell most. The
strongest arm, counterfactual simulation with an output mapping, scores the highest AUC on eight of the ten OOD
settings; their headline is an average AUC gain of about 13.8 points over the prior baselines. In distribution
counterfactual simulation is the best method on the two symbolic tasks — at or near 0.999 on indirect object
identification where the mechanism is fully known, and clearly best on the city-attribute task — while ordinary
probes on the located variables are comparable or slightly ahead (by one or two points) on the two tasks whose
mechanisms are only partly understood. Value probing, the cheap no-forward-pass variant, never wins and is often
the worst causal arm, doing worse than plain probing on the city task. Confidence scores are competitive in
distribution wherever outputs are tightly constrained (multiple choice, city lookup, the identification task) and
collapse under shift, with the best temperature differing between the in-distribution and OOD settings; on a
retrieval task the answer-token confidence under a distractor perturbation falls to around chance. Probes placed on
the last prompt token can look strong in distribution purely because that location coincides with a causal variable,
and are erratic out of distribution (one setting drops them into the 0.4s). Scaling the model does not rescue
confidence scores. The relation between simulation and prediction is monotone over most of its range but flattens as
interchange accuracy approaches one, and there are points where an earlier layer localizes well yet predicts
correctness poorly — the authors read this as representations the model computes whether or not the answer ends up
right. In a worked case they note the model may skip a second variable once the first has decided the answer, a
shortcut visible in their diagram.

## Limits and open problems they name

Two, both inherited from the tools the method is built on. The predictor itself does not assume linearity, but the
localization they use locates an orthonormal subspace, so a non-linearly encoded concept would be badly served; they
say transformer representations are neither purely linear nor purely non-linear and leave the practical cost as an
empirical question. And the high-level model is supplied by human prior knowledge, so the framework's reach is set
by how much of a task's mechanism is already known — the places it underperforms the plain probe are exactly the
mixed-subject benchmarks where the mechanism is only a format. Not named but visible in their set-up: the verified
set is labelled in distribution and the localization is fitted there, so both the variable and the decision boundary
are calibrated on the regime the model is good at; the OOD splits are perturbations the authors chose from
stress-test recipes; and every task was filtered to prompts the model mostly gets right, with invalid outputs
silently removed.

## What we take from it

- **A framing, and the cleanest one in the folder**: the same abstraction that explains a success can be run
  forward to forecast a failure, and the two directions are not the same experiment. Their Figure 4 is the object we
  should copy — simulation fidelity on one axis, behavioural prediction on the other, so that "our account is
  accurate" and "our account is useful" are separate claims with a measurable relation between them.
- **A method**: score an output by transplanting the located variable's value into a verified example and reading
  what comes out, marginalised over background draws. It needs no supervised training on wrong examples, only a
  verified correct set and a located variable, and it converts our account into a per-input number rather than a
  corpus-level fidelity score.
- **A measure and a warning in one**: interchange accuracy saturates before prediction does, and a well-localized
  early layer can be a poor predictor. Fidelity of the abstraction is therefore not a proxy for predictive value,
  and any account we report should be scored twice.
- **A dataset recipe**: distribution shift defined against the mechanism rather than the corpus, built by perturbing
  only background conditions and *selected* for a real accuracy drop — this is how to make a shift test that
  isolates whether the model is still running the program.
- **A warning about the do-nothing-ish baselines**: the honest competitor here is not the confidence score but the
  location-heuristic probe, and it wins in distribution on two of five tasks. Any predictive claim we make must name
  the location-heuristic probe as well, not just the cheap baseline it beats out of distribution.
- **A leftover for us**: value probing is the cheap version of the same idea and loses nearly everywhere, which
  suggests the prediction signal lies in the *consequence* of the variable's value (what the downstream computation
  does with it) rather than in the value's decodability. If that holds, a surrogate that predicts forward from a
  located subspace should be our predictor, not a classifier on the subspace.

## Tags

unit: direction | layer
form: causal-abstraction | probe | replacement-model
method: patching | probing | decomposition | attribution
eval: prediction/simulation | patching | downstream-task | benchmark
goal: understanding | evaluation-of-methods | safety/auditing
