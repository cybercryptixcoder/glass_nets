# CIRCUS: Circuit Consensus under Uncertainty via Stability Ensembles

- **Key:** arxiv-2603.00523
- **Authors / organisation:** Swapnil Parekh; Intuit
- **Date:** 2026-02-28 (v1); 2026-03-20 (v2)
- **Link:** https://arxiv.org/abs/2603.00523
- **Kind:** paper
- **Read from:** full text (arXiv HTML v2; S1-S5, references, Appendices A-H; all numbers from printed tables and prose, none from figures)
- **Relevance to us:** 3 (central)

## What they wanted, and why

The paper's premise is that a mechanistic circuit is not a property of the model alone: it is a property of the
model *and* the analyst's pruning threshold. Two thresholds give two circuits, and practice nevertheless reports
one pruned subgraph as though it were the computation. The author wants to turn that analyst-choice variability
into a diagnostic rather than hiding it: if the same attribution graph is pruned under many configurations and an
edge survives most of them, that edge is robust structure; if it survives few, it is an artefact of a threshold.
The framing is explicitly imported from stability selection in variable selection — a per-edge selection
frequency standing in for a posterior inclusion probability — and the stated goal is that an analyst can
*abstain* from reporting unstable edges instead of asserting them. It sits squarely on our question of what a
white-box readout is allowed to claim.

## What they studied

Models: Gemma-2-2B (50 prompts) and Llama-3.2-1B (20 prompts), each with a publicly released cross-layer
transcoder, so the attribution graph has feature nodes produced by the CLT rather than by an SAE. Prompts are
short factoid completions (capitals, arithmetic, trivia) with one target token; the author names reasoning and
in-context learning as untested. The unit of analysis is an **edge** of the attribution graph — a directed
direct-causal-effect link between two feature nodes — and the secondary unit is the pruned subgraph, compared at
a matched edge budget. Uncertainty is varied over one axis only: the pair of cumulative-influence thresholds
(node threshold, edge threshold) that prunes the graph.

## How they did it

One attribution run per prompt produces the full weighted graph. A configuration is a (node threshold, edge
threshold) pair; applying it yields one pruned edge set, called a view. Each edge's stability score is the
fraction of the B views it appears in. The **strict consensus** keeps only edges present in every view; a looser
threshold tau keeps edges above tau. Edges are then tiered: core (in every view), contingent (in at least half but
not all), and noise (in fewer than half, flagged for rejection). The author's stated design constraint is that
configurations must be **non-nested**: if one pruned set always contains another, the all-views consensus
degenerates exactly to the tightest single view, which is verified empirically on nested threshold triples where
the consensus matched the loosest configuration on every prompt. Non-nesting is obtained by crossing the node and
edge thresholds in opposite directions, so one configuration keeps many nodes and few edges and another the
reverse. Evaluation uses B = 25 (a 5x5 grid) for the stability distribution and B = 9 non-nested configurations
for the per-prompt win rates, which is a change of design set between the two analyses and is reported as such.
Baselines at the same edge budget: union-pruned (top edges of the union ranked by absolute influence) and random
(ten seeds). A residual boosting round can add a second tier from the residual graph when the core's retained
influence is low.

## What counts as an explanation here

A **per-edge stability score in [0,1]**, plus a three-way partition of the union graph into core, contingent and
noise. The explanation is deliberately partial and advisory: it does not claim to be the computation, it claims to
say which structural assertions survive variation in analysis choices, and it licenses withholding the rest. There
is no dictionary, no natural-language description and no surrogate model — the subgraph is still the CLT's
attribution graph, and the paper is a statement about how to summarise a family of such graphs. Whether it can
*predict* anything is limited to predicting the edge's own inclusion frequency, which is by construction an
empirical frequency over the chosen family rather than a posterior over all possible pruning strategies.

## How they judged success

Four families of measure, all printed in tables or prose. (1) **Influence retained**: the share of absolute edge
weight in the circuit, summed over edges, over the same sum for the full graph — used as a structural coverage
proxy, and the paper says explicitly it is not a causal measure. (2) **Win rate against each baseline at matched
edge budget**, per prompt, counted out of 50 (Gemma) and 20 (Llama). (3) **A per-logit distributional fidelity
measure**: the KL divergence between the fraction of subgraph influence attributed to each output logit and the
same fraction for the full graph, compared against union-pruned. (4) **Activation patching**: patch the
consensus circuit's node activations from a source prompt with a different answer into a target prompt, three
conditions (consensus nodes, equal-count random nodes, equal-count nodes matched on influence), n = 20, Wilcoxon
signed-rank. Controls that matter: a nesting-matched triple to demonstrate the degenerate case, a
leave-one-configuration-out check, mean pairwise Jaccard across views as a diversity measure, bootstrap intervals
on consensus size and retained influence, and a comparison of three alternative aggregation rules.

## What they found

Consensus circuits come out roughly forty times smaller than the union at comparable retained influence. In the
main B = 9 non-nested setting the core retains .808 +/- .051 of attributed influence on Gemma and .892 +/- .052
on Llama (Table 1 prints them without a leading zero), winning 47/50 and 19/20 prompts against union-pruned and 37/50 and 20/20 against random; the match
count — prompts where consensus equals a single configuration — is 8/50 and 0/20, which is the paper's evidence
that the consensus is a new set. In the B = 25 analysis the strict consensus has a bootstrap 95% interval of
180 to 265 edges at retained influence 0.72 to 0.74, and mean per-edge influence rises monotonically with
stability, with all-config edges carrying about seventy times the mean influence of single-configuration edges
(figure only — the multiple is quoted from the caption and prose, not from a table). Distributional fidelity is
the largest reported gap: KL of 0.0010 +/- 0.0008 for consensus against 0.40 +/- 1.74 for union-pruned over 20
prompts, with the union's variance attributed to heavy-tailed outliers from greedy influence ranking — the abstract
calls this a 400x difference, which is a ratio of the point estimates against a baseline whose dispersion exceeds
its own mean on that scale. Causal validation: mean recovery 19.4 for consensus, 6.8 for random and 1.6 for
influence-matched, p = 0.0004, with consensus at 57 percent of an oracle ceiling of 34.2 (random at 20 percent).
The summary appendix adds an ablation result at n = 20 (16/20 wins over matched, difference 6.0 with interval 2.7
to 9.2, p = 0.012), a patching result (17/20, difference 17.8 with interval 10.8 to 23.1, p = 0.0026), a
prompt-level Spearman between retained influence and output probability of 0.69 with interval 0.28 to 0.89, mean
retained-influence interval width of 0.074 for consensus against 0.103 for a single configuration, and the tier
statistics: stability exactly 1 on 2.5 percent of edges (625 edges, in the boost appendix's B = 3 setting) with
73 percent below 0.5, and 6.2k contingent edges carrying 45 percent of total influence. Aggregation alternatives do
not beat strict consensus: ranking by stability times influence drops to 7/50 and 2/20 wins, and the adaptive
threshold rule selects the strict consensus on every prompt.

## Limits and open problems they name

The stability score is a conditional frequency over the chosen configuration family, not a posterior over pruning
strategies, so the stability-coverage curve moves when the family changes — the author says this twice, once in
the method and once in the limitations. No formal false-discovery control is claimed: the original stability
selection result rests on exchangeable subsampling, and a deterministic threshold grid is not exchangeable, so the
analogy to posterior inclusion probabilities is explicitly an analogy. Two models of 1-2B parameters, twenty to
fifty factoid prompts, one behaviour family; generalisation to larger models and to reasoning or in-context
learning is named as open. B differs between the stability analysis and the win-rate table. Most important for
anyone building on this: the piece of evidence the paper itself floats as a sanity check is thin — the
stability-versus-ablation-effect Spearman is 0.29, printed as a single number with no interval, and it is the only
place a stability rank is compared with an actual intervention effect.

## What we take from it

Three things. First, a **method we can reuse as-is**: config-bagging is the cheapest uncertainty estimate for any
ranking we publish, because it needs one expensive graph and B cheap re-prunings, and the non-nesting principle is
the whole trick — if our configurations are nested the consensus is guaranteed to collapse to one member, so the
design set must cross at least two thresholds in opposite directions. Second, a **measure**: the per-edge inclusion
frequency is a direct answer to the folder's question of which structural claims to withhold, and it is the
complement we have been missing beside point-estimate edge scores — a claim carries its own stability number
instead of an implicit one. Third, a **warning about the fidelity metric that most tempts us**: retained influence
is a sum of absolute weights and the KL here is over influence shares, so both are functions of the same quantity
the pruning optimises; the paper's honest sentence that retained influence is not a causal measure is the correct
label for the .808 and .892 numbers, and only the patching row is independent evidence. The 0.29 rank correlation
against ablation is the number to remember: choosing edges by stability is not the same as choosing edges with
effect, and the gap between those two is exactly the accounting a predictive white-box model has to close.

## Tags

unit: feature | circuit
form: graph | probe
method: attribution | ablation | patching
eval: patching | faithfulness | ablation
goal: understanding
