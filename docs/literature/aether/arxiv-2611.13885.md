# Are Causal Neuron Circuits Unique and Transferable?

- **Key:** arxiv-2611.13885
- **Authors / organisation:** Ömer Tekin Yavuz; İletişim Yazılım, Bursa, Turkey (single author, industry R&D)
- **Date:** 2026-10-30
- **Link:** https://arxiv.org/abs/2611.13885
- **Kind:** paper
- **Read from:** full text (12-page PDF, all of it, incl. Appendices A–B; no HTML version exists for this id, so the PDF was converted locally)
- **Relevance to us:** 3 (central)

## What they wanted, and why

A standard piece of interpretability evidence is a group intervention: pick some internal components, remove
or overwrite them, and watch the answer change. The paper opens on a concrete public example — an 18-neuron
group in Qwen3-32B whose removal turns a Pride Lands riddle answer from the uncle to the cub — and asks what
that kind of demonstration actually licenses. It establishes that the group matters, but says nothing about
how much of it is needed, whether a different set would do the same, whether the neurons act together or
independently, or whether the effect survives a rephrasing. The author's framing is that public write-ups
rarely report how group size was chosen or how subsets behave, and treats these as open measurement questions
rather than as faults in the original work. The target is therefore a *structure claim* about a group that has
already passed its causal check: what does "these 18 neurons" mean?

## What they studied

Four base (pre-trained, not instruction-tuned) checkpoints from three families: Qwen3-0.6B and Qwen3-1.7B,
Llama-3.2-1B, Gemma-3-1B. The unit is one coordinate of the gated-MLP activation entering the down-projection,
at one layer — i.e. a *neuron*, not a feature, direction or head. The causal object is a **group of 12 neurons**,
small enough that all 4,096 of its subsets can be enumerated exhaustively instead of sampled. Two studies: a
two-option capital-knowledge task in Qwen3-0.6B (10 discovery questions, 10 held out), and 20 hand-written
two-hop relation chains in the three larger models, each chain with a first-hop question, a second-hop question
and three phrasings of the composed question. All interventions act only at the last prompt token.

## How they did it

Neurons are ranked by a first-order estimate of their effect on a logit margin — activation times the gradient
for zero ablation, the activation difference times the gradient for patching — and real single-neuron
interventions are then measured, every neuron in two chosen layers for Study 1 and the estimated top 300 for
Study 2. The 12 candidates are the 6 strongest by *measured* effect plus the next 6 by estimate, so that
neurons which are weak alone but possibly synergistic are not excluded in advance; the group is chosen on one
prompt per question and then frozen, with rephrasings and new questions never used for selection. Four
interventions are available on a set: zero ablation, mean ablation over an independent reference set,
resampling (writing the value the neuron takes on a different prompt), and patching (writing values from a
designated source prompt). The metric is the drop in the margin between the correct answer token and an
alternative token, normalised by the clean margin, so 1.0 means the margin reaches zero and above 1.0 means the
alternative is now preferred; exact ties under bfloat16 are not counted as flips. For every frozen group all
4,096 subsets are measured and five things are derived from that table: additivity (fit of subset effect by the
sum of single effects, R² and slope through the origin), pairwise interaction terms, *minimal sets* (effective
subsets with no effective proper subset, at 30/50/70% of the full effect and at the flip threshold), *disjoint
alternative* pairs of minimal sets sharing no neuron, and *transfer* of the same frozen group to other prompts.
Controls: 12 random layer-matched neurons per question (one draw, fixed seed) plus, in Study 1, high-activation
neurons with low estimated effect; every batch carries an unmodified row and deltas are computed against it.
Intervals are 95% bootstrap over questions or chains. Study 1 runs in float32; Study 2 in bfloat16 with a
measured noise floor from 30 re-measured subsets per chain under a different batch size and uncached forward
passes, and interaction thresholds are set to the larger of 5% of the full effect and three times that floor.
Appendix B lists every definition changed after seeing data.

## What counts as an explanation here

A **structural description of a causal handle**, and deliberately nothing more. The paper does not build a
circuit, a graph, a replacement model or a program; it says plainly that it does not recover connectivity and
that its minimal sets are minimal *relative to the candidate group*, not "the model's circuit". What it offers
is a small set of measurable properties a neuron group can have — additivity, interaction, minimality,
multiplicity, transfer, content-specificity — plus the demonstration that these properties vary independently
of whether the group passes a causal test. It is a partial explanation by design, and the claim it can support
is predictive in a narrow sense: knowing the clean margin and the size distribution of sufficient subsets tells
you whether a group-level statement will replicate across paraphrases and across inputs.

## How they judged success

Comparisons are always against 12 random neurons under the identical intervention, and the central addition is
a **content-matched control**: replacing the selected neurons with their values from *other questions* (same
correct letter, different correct letter, or the mean over other questions) rather than with zeros or random
values. Effects are expressed as a fraction of the clean margin with bootstrap intervals; interactions are
thresholded against both a relative floor and a measured bfloat16 noise floor; uniqueness is counted as disjoint
pairs of minimal flipping subsets; transfer is a Spearman correlation across all 4,095 subset effects between
the original phrasing and its rephrasings, and a raw effect on a different chain. Held-out questions re-run the
entire selection procedure to check the patterns are not an artefact of one prompt set.

## What they found

**Study 1 — a group that passes every usual check and still has no evidence of content.** Zero-ablating the 12
neurons lowers the margin by 0.27 (interval 0.26–0.30) while 12 random neurons give −0.00; the subset effects are
almost perfectly additive (R² 0.99, slope 1.00); zero interacting pairs; 15–306 minimal sets at the 50%
threshold with disjoint pairs in 8 of 10 questions; and effects on rephrasings correlate with the original at
Spearman 0.96–1.00 across all 4,095 subsets. One neuron, (14, 692), is the strongest single neuron for all 10
questions and stays strongest when the option order is swapped. **But** replacing the same 12 neurons with their
values from other capital questions leaves the answer essentially unchanged (0.00, −0.01, −0.00 for the three
replacement schemes), and the activations are nearly constant across questions — median absolute deviation 0.12
against a median magnitude of 2.53. Zero ablation removes a near-constant signal the answer depends on; nothing
in the tested set shows it carries anything specific to the particular capital. The whole pattern replicates on
the 10 held-out questions, and the discovery-frozen group lowers the margin on them by 0.24.

**Study 2 — the group is tied to its chain, and rarely needed in full.** All three models know nearly every
single hop but compose only about half the chains, and their characteristic error is to answer with the bridge,
which is exactly the shift being induced: chains are included when both hops and the first phrasing are right
with a clean margin of at least 1, giving 11/11/10 chains. Patching the 12 selected neurons with first-hop
activations flips the answer in 10/11, 10/11 and 8/10 chains and makes greedy generation begin with the bridge in
9/11, 9/11 and 8/10; 12 random neurons never flip anything. Patching raises the bridge logit (median +4.1) *and*
lowers the answer logit (−3.7), while zero ablation mostly just lowers the answer (−5.4, bridge +2.7) — two
groups with the same provenance but different signatures. **The full group is not necessary:** the smallest
flipping subset is 4 / 8 / 7 neurons on the headline chain, and across chains it ranges from 1 to 8; effects
accumulate smoothly rather than in a step. **Uniqueness is model-dependent:** disjoint pairs of minimal flipping
subsets exist in 3/10, 7/10 and 7/8 flipped chains, and on the seven chains shared by all three models in 1/7,
4/7 and 5/7. The author flags that the four four-neuron Qwen3 subsets checked by generation all contain the same
two neurons, so disjointness there rests on the margin criterion and was not confirmed by generation.
**Interactions are small:** additive fits give R² 0.90 / 0.89 / 0.84 with slopes below one (0.78 / 0.80 / 0.83,
subsets do less than the sum of their parts), and only 4–11 of ~660–726 pairs exceed the interaction threshold.
**Transfer is asymmetric:** subset-effect rankings carry to rephrasings at Spearman 0.98 / 0.98 / 0.97, but the
same frozen neurons applied to a different chain's composed question move it by only 0.18 / 0.09 / 0.11 against
1.52 / 1.89 / 1.82 on their own chain. An exploratory, explicitly post-hoc split uses the clean margin as the
moderator: pooling the three models over the 28 flipped chains, disjoint alternatives appear in 13 of 15 chains
with margin below 4 and 4 of 13 above it, and the four included chains no subset ever flips all have margins
above 4.6.

## Limits and open problems they name

Pilot scale: models of 0.6–1.7B and at most 11 chains per model; the 32B model in the motivating example did not
fit in the available GPU memory. Every necessity and uniqueness statement is relative to 12 neurons chosen by
single-neuron and first-order criteria, which favour individually important neurons and could miss purely
synergistic ones. One random group per question rather than a distribution of random groups. Interventions at the
last prompt token only, so earlier-position computation is untested. bfloat16 in Study 2 with a noise floor up to
0.50 logits, so smaller interactions may be invisible. The margin cut-off of 4 was chosen after seeing data and
margins are confounded with chains. First-token margins stand in for multi-token answers, and generation checks
look at the first line only. The author's own framing of what this means for the motivating example is careful:
his groups are his own, he did not test the published 18 neurons, and he does not claim to refute them.

## What we take from it

Three things, in order of transferability. **A content-matched control, as a standing requirement on any
necessity claim.** The random-neuron control answers "is this group doing something the average group is not" —
it does not answer "is this group carrying the input-specific thing it is being credited with", and Study 1 is a
clean counterexample where the first question is answered yes and the second is unanswerable from the data: a
group that is causal, additive, transferable and *nearly constant across inputs*. The cheap version of the right
test is to write in the value the same neurons take on a different input and measure again; a group whose
behaviour does not change under that substitution is a gate or a scalar, not a representation of the content the
prompt names. This is the neuron-granularity twin of work this folder already holds at feature granularity, and
it is the single most reusable item here because it costs one extra intervention per claim. **Sizing claims
should be distributions, not counts.** "These 18 neurons" reads as a necessity claim, and the exhaustive table
shows why it isn't one: the smallest sufficient subset ranges from 1 to 8 across chains, the median and the
strongest subset diverge, and effects accumulate smoothly, so the headline number is close to arbitrary — it is
an existence proof for a sufficient intervention. Reporting the *size distribution of sufficient subsets* costs
2ⁿ measurements at n = 12 (cheap, and the paper's practical point) and converts a number into a curve. **The
dataset is worth stealing as a testbed, not as a benchmark.** Two-hop chains are a good choice not because of
accuracy but because the wrong answer is *content* — the bridge entity — so a patching effect can be given a
direction and a target (bridge up, answer down) rather than a scalar magnitude; that is what lets the paper
distinguish a group that pushes toward the bridge from one that merely disrupts the answer, and it is why the
random control is strong (it never flips) while the *other-chain* control is what limits the claim. Fourth, and
a warning about thresholds: uniqueness is not a property of a group but of a group *plus a threshold plus a
margin plus a selection rule*, and the paper demonstrates the dependence directly — the same frozen neurons look
unique or redundant depending on whether the criterion is half the effect or an actual flip, and the flip
criterion itself is margin-relative. Fifth, a piece of experimental hygiene worth copying wholesale: every
definition changed after seeing data is listed in an appendix, including the one that moved a headline count
from 4/10 to 3/10 and the pilot threshold that reached nothing and was replaced. That appendix is why the
exploratory result is legible as exploratory rather than folded into the findings. Finally, the honest negative:
this is what a *sizing* study can do, and it is not a circuit. The author says so in the abstract, the
introduction and the limitations, and the folder should file it that way — a set of measurements any group
demonstration ought to carry, not a method for finding the mechanism.

## Tags

unit: neuron
form: description | probe
method: patching | ablation | other
eval: ablation | patching | faithfulness
goal: evaluation-of-methods | understanding
