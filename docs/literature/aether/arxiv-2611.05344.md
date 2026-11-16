# A Circuit for In-Context Learning

- **Key:** arxiv-2611.05344
- **Authors / organisation:** Andrew Gao, William Gao (equal contribution; Stanford University)
- **Date:** 2026-11-03
- **Link:** https://arxiv.org/abs/2611.05344
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

In-context learning (ICL) is the clearest case of a model acquiring a function inside a single forward pass, with no weight
update; earlier work had located candidate parts (label tokens, induction heads, task vectors) but not how they compose
into a computation from demonstration to prediction. The authors set out to give that composition for few-shot sentence
classification with arbitrary labels, and to do it in a form that is causal at every link rather than suggestive. The
larger goal is the one this folder is chasing: an account of a behaviour specific enough to predict what happens under
intervention, to be re-found in other models, and to be built by hand in weights. A secondary goal is to explain a
developmental fact — that the algorithm assembles in a fixed order during pretraining, and that a small model never
develops it at all.

## What they studied

Three pretrained base transformers of 3–4B parameters — Qwen3-4B-Base (36 layers, primary), SmolLM3-3B and
Llama-3.2-3B — plus a size ladder inside one family (Qwen3 at 0.6B, 1.7B, 4B, 8B), the Qwen3-4B instruct model,
OLMo-2 1B and 7B stage-1 checkpoints, and two models they build themselves: a hand-wired three-layer attention-only
transformer and a 3.4M-parameter four-layer transformer trained from scratch. The unit of analysis is the attention head,
grouped into **populations** (sets of roughly twenty to ninety heads with a shared direction) and into per-position
key–value **slots** in the residual stream; the sub-unit is the key–value group inside a head. Data are synthetic
sentence-classification **batteries**: a concept battery (six demonstrations, 3+3, over axes like animal/food,
sentiment, English/Spanish, question/statement), concept12 (twelve demonstrations), concept12n (four natural concepts in
four label variants — arbitrary symbols, rare words, the concept's own words, and those words exchanged), and slotgate
(four demonstrations in token-identical pairs that differ only in whether every label repeats). Every prompt is a list of
`sentence -> label` lines, a query sentence and an arrow, scored on the next token, with **margin** = correct minus other
label log-probability in nats.

## How they did it

The design is a fixed subtraction: a **twin** prompt identical everywhere except inside one span, a **handoff** (patching
an activation from the twin into the intact run), and a **hold** (fixing a downstream component at its intact value while a
handoff runs). Swapping the two labels throughout moves the answer; patching only the keys and values at the label tokens
isolates where the label identity lives, and patching only a changed sentence isolates where the class key lives.
**Knockout** is mean-ablation of a head's output, and every named set of heads is selected on one draw of prompts and
tested on a fresh draw with the same number of heads from the same layers as the control. Steering adds the model's own
fitted directions (a task direction, a query code, a gate direction) to the residual stream at the dose the model itself
uses, with same-norm random directions as controls. On top of the causal work they run an exact **accounting**: because the
residual stream is a sum of component outputs, the answer logit decomposes identically into per-head and per-MLP terms,
which they first check reproduces the model's own margin. Faithfulness, completeness and minimality are then tested in the
manner of the circuit-evaluation literature: keep only the named heads and substitute a reference value for every other
head's output **at the query position only**, so the slots still build normally. Predictions are written down before the
concept transfer, size-ladder and instruct experiments. Intervals are paired bootstraps over prompts (4000 resamples).

## What counts as an explanation here

A **circuit of named component types with typed links**, not a single head list. Concretely: a key–value memory at the
label tokens (writers write the class of the preceding sentence as the slot's key in middle layers; the label identity
remains the slot's value, read through the readers' value matrix); about ninety **reader** heads in the upper layers that
match the query's class against slot keys and copy values out; two heads that write a **query code** consumed by the
readers; about twenty-four **brake** heads that write against the most recently seen label; a **copier** band in the last
layers that writes the logits; an attention **sink** at position zero that supplies a null action; and a single residual
direction **g**, computed from whether any label repeats, that opens the read by lowering each reader's sink logit. It is
explicitly **partial**: it covers two-class label retrieval at up to twelve demonstrations in 3–4B base models, and does
not cover instruction following, long-form tasks or many-class binding. Each link is labelled **forced** (follows from
attention and softmax and should exist in any transformer that does the task), **implementation** (set by training, e.g.
which heads, how many, how strong), or **population** (no single head is necessary). It predicts: the shape of the read as
a softmax law, the behaviour of the gate under steering, the fate of prompts whose class is not linearly encoded, and
which ability appears when with model size.

## How they judged success

Four measures per link. (1) **Paired margin change in nats with a 95% bootstrap interval**, against a twin whose effect on
the whole prompt is known, so a handoff's share of the total can be read directly. (2) **Random controls matched on count
and layers**, which is the baseline in every population result. (3) The **read law**: fit a three-constant softmax
expression per head (two identifiable, since the rest-mass and sink terms enter only through their sum) to matching-slot
attention mass on twelve-demonstration prompts, then apply the same constants to a held-out regime — one slot per class —
and to an extrapolation regime (eighteen demonstrations, three to nine slots per class), with a pre-declared pass mark of
median error below 0.07. (4) The **circuit-evaluation triad**: an exact-decomposition check (reconstruction error against
the model's own margin), faithfulness/completeness by keeping only the named heads against same-size random sets and
keeping no heads, and minimality by single-head knockout. Extra controls: a same-class, same-length sentence swap to show
only the class reaches the answer; a reference-value study showing that the choice of ablation reference decides whether
the faithfulness test can rank head sets at all; and predictions registered before the transfer and size experiments,
with misses reported.

## What they found

**The slot dissociates into value and key.** Exchanging labels and patching label-token keys/values in the reader layers
alone carries most of the exchange (in SmolLM3, −2.94 of −5.55 nats); the eight sharpest readers' **values** carry −2.16
while their keys carry +0.56, the wrong direction. Three of 72 key–value groups in layers 21–29 carry about half the effect
and one group in layer 24 a quarter. Changing a sentence with the label fixed instead routes the effect through the
**keys**: −2.35 of −2.87 sits in layers 21–29, the readers' keys carry −0.55 and values +0.10, and in Llama the effect sits
only in the two middle layer bands. A same-class same-length sentence swap moves the answer by +0.08 nats, so only class
information reaches the answer. Writers number about twenty heads with no head above 5% of the effect, and they write a
single task direction per task; the same dose spread over four layers beats a single-layer write (under a controlled
steering test, +0.56 against +0.37 at one dose, and +0.69 against +0.20 at double).

**The read follows a softmax law within a prompt length and fails on longer prompts.** Matching-slot mass is fit to within
0.1 for 127 of 140 Qwen heads and 124 of 125 Llama heads; kept constants predict the one-slot-per-class regime at median
error 0.07 / 0.04; extrapolating to eighteen demonstrations gives median error 0.15 / 0.13 with 92–94% of heads
over-predicted, because each head's sink mass and total matching mass are roughly fixed and added matching slots divide a
fixed class budget. The failure itself yields two facts the rest of the account uses: the gate acts on a per-head constant
null mass, and the read must be summed over many heads.

**A single direction gates the read.** Recurrence of labels is what opens retrieval (accuracy about 20% with one
demonstration per class, 60–80% with two, 96% with six, and a label appearing once is still retrieved if *others* repeat).
The direction g, fit at the query arrow in layer 23, separates recurring from token-identical unique-label prompts with
held-out sign agreement 1.00 and is orthogonal to the task direction; sixteen duplicate-token heads in layers 1–9 push it
(+18.3 to −0.3 when ablated, against ≤0.7 for sixteen random heads and 4.2 for induction-scored heads). Adding g at the
natural dose lowers readers' sink logits by 1.75–3.65 nats, moves the match-vs-other gap by only +0.34 to +0.48, and a
same-norm random direction moves either by ≤0.2; the sign in weight space is selective (g projects onto sink-versus-slot
functionals and not onto match functionals the same heads use, |cos| ≤ 0.02). Roughly a quarter to a third of g's effect on
the answer travels through the readers' queries; the rest takes unnamed routes.

**The query carries its own code, written in the readers' coordinates.** Two heads (L22h10, L23h26 in Qwen) carry 40% of
the query-flip effect, any random pair under 0.05; their write has cosine 0.07 with the task direction but 64% of its
energy lies in the span of the readers' match functionals (4% for a random subspace). Holding downstream attention queries
removes most of the effect, and holding late MLPs alone makes it *larger*, so those MLPs oppose the query code.

**Readers and brakes are populations, and the reason is gain, not ambiguity.** No single head moves the margin by more
than 0.14 nats (largest reader 0.09, brakes +0.13) against a 1.3-nat margin; yet twenty readers knocked out on a fresh
draw cost −0.74 nats (random +0.02) and twenty-four brakes raise it by +1.39 (random +0.22). The reason a single reader
suffices to *separate* the classes (its own class bias is 0.2–0.4 nats) but not to *write* the answer is a gain bound: no
head writes more than 0.2 nats directly, so at least seven must contribute, and in practice 14 heads carry half the flip
and 46 carry 80%. Amplifying three readers by 2 gives +0.40 and by 8 breaks the model; amplification cannot stand in for
the other seventeen. The brakes' job is anti-recency — readers prefer late slots by 0.5–0.85 nats, and with brakes
removed the model follows the last label.

**The circuit is complete within its layers and not minimal.** Keeping 79 heads at the query retains about 60% of the
margin, far above same-size random sets and above keeping no heads; with 189 heads (after a natural-label screen) the
circuit alone matches or exceeds the intact model, only the encoder band still cost anything, and the same protocol closes
in the other two models. Minimality is where the account departs from a sparse circuit, and the paper says so.

**Implementation is free, algorithm is conserved.** Llama's readers sit in layers 14–27 of 28, its gate is much weaker
(67% of unique-label prompts still get a label, against 10% in Qwen), and its brakes are invisible to a single-head census
(288 heads, max 0.12 nats) yet move +1.90 as a set. Across OLMo-2 checkpoints the parts appear in dependency order —
copying a repeated label, then the null action (1B between 315B and 839B tokens; 7B already at 42B), then class-keyed
retrieval — and 1B never develops the readers. A trained-from-scratch model grows the parts, but **no gate**: gating
appears only in a training distribution where retrieval on unique-label prompts is penalised, and steering along the g
direction then moves answers where it does nothing in the control.

**The word-label split explains the flipped-label curve.** With the concept's own words as labels the margin is
decomposable into a demonstration-following part and a word-meaning part of comparable size (≈2.2–2.9 nats each), and the
meaning part does not run through the circuit but through late MLPs and the brakes. Across 0.6B→8B the ratio of meaning to
demonstration shrinks (1.03 → 0.73), and the quantity that tracks the turned-positive flipped-label margin at every size is
the brakes' summed hold on word meaning (+1.73, +2.06, +3.61, +4.03 nats). Instruction tuning doubles both parts by about
2.5 and keeps their ratio: post-training did not rewire the readers, it added braking.

**The hand-built model closes the loop.** A three-layer attention-only transformer whose weights are written by hand
reproduces the read law to machine precision and the gate's behaviour (sink mass 0.67 → emptied by adding g), and capping
each reader's output makes the margin grow with the number of readers — sufficiency, though not that training would choose
these parts.

## Limits and open problems they name

Scope is narrow and they say so: two-class retrieval, short sentences, single-token labels, ≤12 demonstrations, ≤8B, three
base models. Minimality fails for a statable reason, and the paper substitutes populations for a minimal head set, noting
it did not run the conditional (necessary-once-others-removed) test. The read law is a fixed-prompt-length law; why each
head's null and matching budgets stay fixed is unexplained, and the constants are not separately identifiable. The
duplicate-head-to-gate link replicates at the level of the gate *feature* in all three models but at the level of behaviour
only in Qwen. Where the brakes come from in training is open — 400 steps of many-shot LoRA tripled the readers' necessity
and left the brakes unchanged. Several registered predictions failed and are reported as such (the up-transfer of the law,
a monotone brake count with size, a single gate dose clearing zero at 4B). The encoder correspondence ("a class not linear
at the arrow cannot become a key") is correlational. The paper also flags a generalisation it declines to defend as more
than a guess: that the gate is a candidate mechanism for whether a model consults its context at all.

## What we take from it

- **Method — the twin/handoff/hold triple as the unit of causal evidence.** Any intervention on a network we build should
  come with a twin that differs *only* inside the operated span, a handoff that carries a known total effect, and a hold
  that blocks the suspected consumer. This paper's own appendix shows why the naive version proves nothing: patching at
  every layer of a twin reproduces the twin by an induction argument that holds for a randomly initialised transformer, so
  the informative handoffs are the *restricted* ones. Our probe battery should therefore report, for every patching claim,
  the layer-restricted effect alongside the all-layer harness check.
- **Measure — an ablation reference must be distribution-matched or the ranking test cannot work.** Replacing heads with a
  value taken from natural text destroyed the named circuit and a random set equally, and keeping *no* heads scored the
  same; only a mean over matched prompts ranked head sets. A subtler trap they hit in their own first version: the
  reference must be matched on the label pair too, since mixing pairs moves results by up to a nat. This is the fold's
  "control must be matched on the variable the claim says is irrelevant" rule, arriving as a concrete protocol.
- **Framing — concentration versus summation is a property of the step, not of the network.** Where gain demand is low,
  the model uses a handful of heads; where it exceeds one head's output-value gain, no single head is necessary and the
  right object is a direction plus a matched random subspace. This gives us a way to state a mechanistic claim that is
  neither "head X does it" nor "it's distributed, sorry": say per step whether the model concentrated or summed, and give a
  gain bound that predicts which.
- **Warning — a survival or completeness number is meaningless without its layer scope.** Their faithfulness test holds
  the *query position* only, and therefore certifies the readout given intact slots, not the slots' construction; the
  slots have to be evinced separately. Circuit completeness claims in our own writing should name the position and the
  layers they cover.
- **Shortcut with a predictive payoff — probe the class at the query before trying to teach retrieval.** Writing the task
  direction into slots by hand raised accuracy on prompts whose class was not linearly encoded (50%→71%), while
  key-only, query-only and random vectors did nothing. If a class is not linearly represented where the read happens, more
  demonstrations cannot fix it: a cheap diagnostic that turns a negative ICL result into a representation diagnosis.
- **Caution about metrics that are monotone by construction.** The authors' own prediction — the seven largest direct
  movers should remove 0.3–0.5 nats — came out with the right sign and an interval containing zero, and they label it weak
  support rather than a confirmation. That disposition (report the miss, do not dress it) is the one to copy.

## Tags

unit: head | feature | direction | circuit | layer
form: graph | causal-abstraction | direction
method: patching | ablation | steering | probing | other
eval: ablation | patching | prediction/simulation | faithfulness | completeness
goal: understanding | science
