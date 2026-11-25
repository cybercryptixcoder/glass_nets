# Detection vs. Execution: Single-Bucket Probes Miss Half the Mamba-2 State Sink

- **Key:** arxiv-2606.00930
- **Authors / organisation:** Yuhang Jiang (independent researcher)
- **Date:** 2026-05-30 (v1; CC BY 4.0)
- **Link:** https://arxiv.org/abs/2606.00930
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

The standard move in mechanistic interpretability is to read a probe, take the units with the strongest
representational signature, and call that set the circuit. The author asks whether that move — signature to
locus — survives at intervention scale on a substrate that is not a transformer: selective state-space models,
where a token-dependent gate Δ decides how much a boundary token is written into the hidden state (a "state
sink" analogous to the attention sink, measured in gate space rather than attention-score space). The suspicion
is that the labelling step is granularity-conditional: fine enough units can be monosemantic enough for the
probe to point at the mechanism, while coarse units (heads) are forced by the architecture to multiplex several
functions, so a single-class probe will return only the function it happens to test for. The larger goal is the
folder's familiar one, stated as an audit: know which interpretability findings were localised under an
assumption that the substrate violates.

## What they studied

Seven checkpoints: Mamba-1 at 130M / 370M / 1.4B / 2.8B, Mamba-2 at 130M / 1.3B / 2.7B, plus four Pythia
attention checkpoints (160M, 410M, 1.4B, 2.8B) as a cross-architecture baseline. Data is wikitext-2 and the
Pile-10k subset, 30 documents per cell, plus RULER single-key NIAH at 1024 and 2048 context. The unit of
analysis is the **channel** in Mamba-1 and the **head** in Mamba-2 (Mamba-2's Δ projection is shared across the
channels inside one head, n_heads between 24 and 80; Mamba-1's is per-channel with D_inner from 1,536 to
5,120), with attention heads in Pythia classified by attention-to-BOS strength. The whole comparison turns on
that pair of unit classes.

## How they did it

Specialty is assigned by a deterministic probe: for each unit, average pre-softplus Δ activation on tokens of a
class (BOS, newline, punctuation, whitespace-word), subtract the unit's mean on a reference token set, threshold
the difference at 0.5 (pre-registered as the gate's "on" state), and classify the unit as a c-specialist if
exactly one class clears, **dual** if exactly two do, generalist if three or more, silent otherwise. The
pre-registered intervention grid is six factors: four models × six interventions on the gate / timestep / input
projection (gate_zero, gate_one, gate_mean, delta_zero, delta_median, u_zero) × seven unit scopes × six layer
scopes × four tasks × three seeds, 12,096 cells of which 6,060 were evaluated (formal-task loaders deferred per
pre-registration). The statistic everywhere is a **differential**: mean ΔNLL of the specialist set minus the mean
over a size-matched random complement, with the Mamba-2 headline cells using a 30-seed bank of stratified
non-adjacent complements (master seed 13, gap ≥ 61) and a 5,000-replicate bootstrap over documents for the CI.
The key design step is the decomposition: the same gate_zero ablation is scored against two different target
sets — tokens predicted in BOS context, and tokens whose target is a newline — for each head set, so a set can be
scored by *what it damages* rather than only by how much. Controls: a magnitude-matched non-specialist complement
per layer (top-K non-specialists by mean |Δ_pre|, K matched to the specialist count) on the retrieval task, a
random channel-bucketing sweep on Mamba-1 designed to reproduce Mamba-2's heads-per-class ratio without its
architecture, three alternative specialty thresholds, and four analytic / prior-work sink-mass scores
cross-validated against the probe's labels by top-10% Jaccard.

## What counts as an explanation here

A **partition of one layer type's units into two functional roles**, grounded in intervention, plus an
architectural account of why the partition exists: single-bucket BOS-specialists are the execution layer, dual
heads are a representational detection layer that the single-bucket probe excludes by construction, and the
head-shared Δ projection is offered as the mechanism that forces a head to multiplex boundary classes and
therefore generate dual heads rather than single-class ones. It is a causal localisation of a phenomenon, not a
model of a computation: nothing here predicts a next-token distribution from the partition, nothing is a circuit
graph, and the author is explicit that necessity was measured and sufficiency was not.

## How they judged success

ΔNLL differentials against a 30-seed size-matched random complement with bootstrap CIs; cross-corpus agreement
on the same cells (wikitext-2 vs Pile-10k); cross-scale monotonicity; a 4× context-length replication (512 → 2048);
RULER NIAH accuracy and per-token NLL on the answer span, with a magnitude-matched complement as the strong
control; four Pythia checkpoints as a cross-architecture negative; a bucket-width sweep for the substrate-capacity
alternative; threshold sensitivity of the headline ratio; and pre-registered prediction checks for the
phenomenology (§ 3.3), including a 3-of-4 and 4-of-4 conjunction across four bimodality tests (Hartigan dip,
Silverman peaks, GMM Δ BIC, KDE local modes).

## What they found

At **channel** granularity the probe's labels track causal mass cleanly: ablating BOS-specialist channels on
Mamba-1 130M gives +3.76 nats against the random complement, identical to two significant figures on wikitext-2
and Pile-10k, falling to +2.13 / +1.64 at 2.8B; the gate_one variant amplifies the effect by an order of
magnitude (+26.62 / +31.53 at 130M, +19.94 / +19.88 at 2.8B), and 46% of the all-layer effect sits in layer 0 at
130M but only 13% at 2.8B. At **head** granularity on Mamba-2 the correspondence breaks. The single-bucket set is
still positively causal but 2-3× smaller (all 8 headline cells, differentials +0.60 to +2.48, CIs excluding
zero), and it covers only 4.9-8.2% of heads against Mamba-1's 19.4-29.4% of channels. The dual set — 27-35% of
heads at the larger scales — carries 4.8-6.4× the single-bucket differential in aggregate, realises a BOS-newline
Δ_pre cosine of 0.89 against 0.54 for the specialists (stable under size-matched subsampling of 200 random
subsets), and yet **fails the functional test**: under gate_zero the dual set is null or negative on BOS-context
prediction at 1.3B (-0.40 / -0.14) and negative on newline targets at 2.7B (-5.79 on wikitext-2, CI
[-6.25, -5.33]), while the small specialist set is positive on newline targets in all six cells it was tested in
(+3.11 / +0.75 / +7.42 / +1.57). So the aggregate dominance is a set-size artefact: per unit the dual set adds no
execution mass (≤1.5× at every threshold), and its dominance ratio swings from 1.02× to 6.41× with the
classification threshold, which the author prints rather than defends. Behaviourally, ablating the specialist
heads destroys RULER retrieval — 1.00 to 0.00 at 1024 context on both Mamba-1 2.8B and Mamba-2 2.7B, preserving
1.00 under the size-matched complement, with the Mamba-2 magnitude-matched complement still at 1.00 (0.97 at 2048),
whereas on Mamba-1 the magnitude-matched complement collapses retrieval — while dual ablation lands in between
(0.40 at 1024, 0.27 at 2048), showing partial but non-identical retrieval substrate. Pythia at comparable
heads-per-class shows neither phenomenon: single-bucket selectivity is clean on all four checkpoints (+0.59 to
+2.22, all CIs excluding zero) but the dual set shrinks to 2-12% of heads with ablation effects ≤0.16 nats.
Randomly bucketing Mamba-1 channels to Mamba-2's ratio keeps single-bucket specialty at 0.292-0.302 against
Mamba-2's 0.050 and dual specialty at ~0.28 against 0.354, with sub-nat differentials in the bucketed ablation
grid — so substrate capacity alone does not produce the split; the head-shared Δ projection is implicated, with
head dimension, state width and training data explicitly left unisolated. Side results worth noting: the probe's
cross-method agreement with prior sink-mass scores is only 0.07-0.15 Jaccard against a pre-registered 0.4
(prediction failed, reported as such), and three findings that appeared on wikitext-2 but not Pile-10k are
demoted to corpus-specific observations in an appendix.

## Limits and open problems they name

Only four Mamba checkpoints are in the intervention grid (the others support phenomenology only); hybrid
SSM-attention models are out of scope; formal tasks (parity, Dyck-2, S3) were deferred under the pre-registration,
so the "causal substrate" being measured is next-token NLL on boundary tokens plus retrieval, not a task where a
circuit is independently known. Mamba-2 effects are small (+0.25 to +2.48 nats), one 1.3B cell goes non-significant
at 2048 context, and the dual set's CIs at 130M are wide. The bucketing control rules out substrate granularity
alone but not head dimension, state width or training data. Necessity is what was tested; the
phrasing throughout
is necessity. Two untested token classes (punctuation, whitespace-word) may form additional execution layers. At
4096+ context baseline retrieval has already collapsed, so the intervention has nothing to measure against. The
magnitude-matched control itself behaves differently across the two architectures (a genuine difference in how
much of the causal substrate is recoverable at channel granularity), which the paper reports but does not
analyse. Several pre-registered predictions failed (cross-method Jaccard) or demoted (three wikitext-specific
sign reversals), and the write-up treats those as informative rather than repairing them.

## What we take from it

Four mechanical things. **First, the counting rule that makes the dual-head number honest: a set's aggregate
differential is not evidence about its units — divide by set size and re-run before interpreting.** The 4.8-6.4×
dominance dissolves to ~1× per unit, and the paper states that the ratio's range across thresholds (1.02× to
6.41×) exceeds its value, so the headline was demoted in favour of the class-conditional test. Any white-box claim
of the form "component group X carries most of the effect" that we produce must carry a per-unit column and a
threshold-sensitivity row. **Second, the decomposition move is directly reusable: score a set by which target it
damages, not only by how much.** One ablation, two target classes, and a set that looks dominant becomes
sign-negative on one target and null on the other — that is a partition of function recovered from a single
intervention grid, and it is the cheapest form of the "which sub-behaviour does this unit serve" question our
model needs. **Third, the signature-versus-function dissociation is the general form of what run 753 proved for
score families and what run 751 found for signs: representation similarity (cos 0.89) is selection-constructed
here — dual heads are *defined* as units active on two classes — so a similarity computed over a set chosen by
that similarity cannot be treated as evidence about function.** The paper says this itself and demotes its own
cosine to a representational claim. Rule for us: when a unit set is defined by a criterion, any metric that is a
monotone function of that criterion is a restatement, not a measurement. **Fourth, the architectural-driver control
is the right shape for a negative claim about a mechanism: reproduce the coarse unit budget without the
architecture and show the phenomenon does not appear.** Random bucketing to the same heads-per-class ratio keeps
single-bucket specialty at ~0.29 rather than 0.05, which is what makes "the head-shared Δ projection" a serious
hypothesis instead of an association — even though the author honestly lists the three confounds the control does
not remove. For a non-transformer white-box model, the further lesson is that **the unit class is an architectural
choice that can manufacture or destroy the partition we are trying to read**, so Mamba-style gates should be read
at channel granularity where the projection is per-channel, and head-level readings on head-shared projections
inherit this paper's caveat.

## Tags

unit: feature (head / channel as the atomic unit) | form: causal-abstraction (functional partition of units, intervention-grounded) | method: probing, ablation, patching, other (analytic weight-derived sink scores) | eval: ablation, causal-differential vs size-matched complement, downstream-task (RULER NIAH), cross-corpus and cross-scale replication, benchmark | goal: understanding, evaluation-of-methods
