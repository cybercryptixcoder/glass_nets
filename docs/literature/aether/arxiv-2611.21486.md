# Differentiable Exact Learning

- **Key:** arxiv-2611.21486
- **Authors / organisation:** Hristo Papazov, Francesco D'Angelo, Nicolas Flammarion — Department of Computer Science, EPFL
- **Date:** 2026-11-16
- **Link:** https://arxiv.org/abs/2611.21486
- **Kind:** paper
- **Read from:** full text (HTML v1, §§1–5 + Appendices B, D, F, G, H, I in part; proofs skimmed for their statements, not checked line by line)
- **Relevance to us:** 3 (central)

## What they wanted, and why

The paper is after genuine algorithmic recovery rather than surface fit: given only traces of an expert working in an
environment, can gradient descent land on a controller that keeps behaving correctly far beyond every length it ever
saw, and can that controller then be *proved* equivalent to the algorithm that generated the traces? The motivation is
that length generalization in arithmetic has sat stubbornly short of the training distribution, and that passing a finite
suite of long tests does not distinguish an algorithm from something that merely looks like one for a few more digits.
The authors want a demonstrator where every step of that chain — supervise, learn, extract, certify — is made explicit,
so that "the network learned the rule" becomes a checkable statement rather than an extrapolation from accuracy curves.

## What they studied

The unit of analysis is a small discrete machine: the *grid agent*, a finite-state transducer that walks a two-dimensional
symbolic grid, reads one cell per step, writes one symbol and moves. Standard Turing machines are the single-row special
case, so the class is computationally universal. Four arithmetic agents (binary and decimal addition and multiplication,
11/19/32/108 states) are the training targets, plus a 15-state universal interpreter `U` that executes an encoded Turing
machine on an encoded input. The learners are Differentiable Finite-State Transducers (DFSTs) — three input-indexed
tensors and an initial state vector, no nonlinearity — with Softmax DFSTs, RNNs and GRUs as comparison families.

## How they did it

An expert agent's run is reduced to *policy-trajectory observations*: for each step only the read symbol, the written
symbol and the motion are recorded, never the expert's internal state. A controller of matching hidden dimension is
trained by next-action prediction with an MSE loss against those one-hot pairs, Adam, batch 32, deterministic
initialisation (transitions at identity, outputs at zero). Datasets are tiny and deliberately narrow — 25 traces for
binary addition, 250 for decimal addition, 100 for binary multiplication, 10,000 for decimal multiplication, with
training operands of at most 3 digits for addition and 5 for multiplication; the universal task trains on 75 traces from
5 programs on inputs of at most 5 symbols. Evaluation reports two length-generalization caps: uniform (10 random operand
pairs of that exact length must match the expert action-for-action) and robust (also every repeated-digit pair, which
catches the long-carry and symmetric-input failures that pure random sampling misses). Success is then turned into
certification by extraction: *metric extraction* clusters hidden states by Euclidean proximity, merges clusters into a
deterministic transition graph, and retries with a halved radius on inconsistency; *annealing* sharpens a Softmax DFST's
columns toward vertices and reads off argmax transitions. Each extracted discrete controller is then proved equivalent
to the expert by an invariant-plus-induction argument over algorithm phases, with the general-equivalence obligation that
cannot be discharged by testing reduced to a finite mechanical check.

## What counts as an explanation here

A recovered program: a discrete finite-state controller, small enough to read as a transition table, plus a proof that
it computes the target function on all inputs in its domain. This is the strongest form of explanation in our survey's
terms — not a description of a circuit and not a surrogate that approximates the network, but an exact artefact extracted
from the trained network by a fixed rule and then verified. Two levels are kept explicitly separate: the *certificate*
concerns the extracted discrete model; the *transfer* of that certificate to the neural controller needs an extra
assumption (empirically checked graph stability for metric extraction, a uniform leakage bound for the annealed softmax
networks). The paper is honest that the neural model is explained only under that bridge.

## How they judged success

Two length-generalization metrics at fixed caps (5,000 digits per operand for addition, 1,000 for multiplication;
500 symbols for the universal task), each requiring exact write-and-move agreement at every timestep; counts of total
consecutive correct actions on structured worst-case probes; per-task comparison against the Neural GPU as an external
reference; extracted-controller stabilisation and probability concentration as diagnostics of the extraction step;
constructed *spurious* controllers as an adversarial control on the supervision set; eight equivalence certificates, five
from metric extraction and three from annealing, with proof checkers in the public repository. A GPU-nondeterminism-free
set of six runs per family covers DFST, RNN and GRU.

## What they found

- **Extreme extrapolation from tiny data.** DFSTs trained on at most 3- and 5-digit operands reach both caps — 5,000-digit
  addition and 1,000-digit multiplication, in binary and decimal — with 25/250/100/10,000 traces and 891/8,227/5,280/162,108
  parameters respectively, against the Neural GPU's ~200k samples, 20 training digits, and robust ceiling of ≤20 for
  addition and ≤8 for multiplication.
- **The count is in actions, not digits.** Two 5,000-digit all-9 operands require 50,070,010 consecutive correct
  write-and-move decisions; two 1,000-digit all-9 operands require 15,012,991. Successful decimal DFSTs reproduce both.
- **The universal interpreter is learned, not hand-built.** A 4,335-parameter 15-dimensional DFST trained on 75 traces from
  5 programs (inputs ≤5 symbols) reaches the 500-symbol cap on all 10 tasks including 5 held-out programs; a 30-dimensional
  Softmax DFST (12,720 parameters, 150 traces) does too. The longest universal trajectory is 15,316,822 matched actions
  (the copy task); reverse is 14,911,144.
- **The supervision set provably cannot tell the right rule from a wrong one.** The authors build a 14-state spurious
  binary-addition controller (expert: 11 states) that reproduces the expert's trace on every operand pair of length ≤3 —
  all 196 of them — and then stops on the fourth left-operand digit, leaving the input untouched; `1111+1` is the
  smallest witness. A 38-state spurious multiplier (expert: 32) does the same from the sixth digit.
- **Capacity is not what selects the right rule.** The wider Softmax DFSTs (dimensions 22, 33 for addition; 64, 96 for
  multiplication) can express those spurious controllers and still reach the robust cap in five of six configurations in
  each sweep. The paper calls the cause a useful implicit bias of gradient descent under this parametrisation and
  initialisation, and says characterising it is open.
- **The recipe is not about the DFST family.** All six DFST/RNN/GRU runs reach both binary caps from the same 25
  addition or 100 multiplication traces, within one million updates.
- **Certification succeeds.** Eight proofs establish exact agreement between extracted controllers and the experts on
  arbitrary nonempty operands, unequal lengths and leading zeros included, and for the universal case on every correctly
  encoded machine and every nonempty binary input; the general-equivalence check is finite and machine-verified while the
  unbounded part is the induction.

## Limits and open problems they name

Softmax DFST optimisation is harder than the plain DFST: the annealed variants are the ones with interpretable transition
probabilities but they could not fit even the small decimal training sets, so the decimal certificates all come from
metric extraction. Metric extraction depends on a cluster radius and on the sampled trajectories stabilising (stability was
observed after 23 sampled trajectories, which is evidence, not a theorem), and the transfer of any certificate to the
neural controller rests on an assumption — graph stability or a leakage bound — that is empirically supported rather than
proved. The paper also proves that no finite test horizon can certify equivalence in general, so the certificates have to
be per-algorithm inductive arguments, and it flags hyperparameter configurations where nothing generalises and the
ground-truth procedure is provably not learned. The agent perceives one cell at a time; richer perception and larger motion
vocabularies are left to future work, as is any explanation of the implicit bias that does the real work here.

## What we take from it

- **A template for the strongest claim we can make.** If we ever want to say a model "has learned the rule", this is the
  shape: train on behaviour traces, extract a discrete mechanism by a fixed rule, prove equivalence on the domain, and
  state the bridge assumption separately. It is directly reusable as the ceiling we compare a *predictive* white-box model
  against — a predictor that reproduces the extracted controller's transitions is a weaker but far more general object,
  and this paper shows what the exact version costs.
- **A control we should copy: the exhaustive-lookalike controller.** Rather than asking whether our supervision is
  "enough", construct an alternative that matches on every training input and differs just outside it. Here that took 14
  states against the expert's 11, and it is the cleanest demonstration in the folder that test-set agreement at the
  trained length says nothing about the rule. For our battery: any probe or surrogate trained on a fixed length or a fixed
  set of contexts should be accompanied by a hand-built lookalike that diverges one step past the boundary.
- **Robust versus uniform metrics.** Their robust cap additionally demands perfect agreement on repeated-digit probes,
  which is precisely where the Neural GPU's random-input success hid its failures. Any length-generalisation number we
  report should come with a structured worst-case probe beside it, since random sampling will not find the long-carry case.
- **Two extraction routes with different costs.** Geometric clustering of hidden states needs no probability semantics and
  worked where the softmax route failed; annealing gives interpretable columns but a harder optimisation. If we ever mine a
  discrete mechanism out of a trained model, having both routes and reporting which one succeeded is a cheap robustness
  check on the extraction itself.
- **A warning about citing generated proofs.** The equivalence arguments here were drafted by a model and verified by the
  authors plus mechanical checkers for the finite obligations. That is a defensible division of labour, but it means the
  "proved" in the abstract should be read as "checked at the finite parts, argued inductively elsewhere".

## Tags

unit: whole-model | circuit
form: program | causal-abstraction
method: training-interpretable | other | decomposition
eval: downstream-task | prediction/simulation | benchmark
goal: understanding | evaluation-of-methods | science
