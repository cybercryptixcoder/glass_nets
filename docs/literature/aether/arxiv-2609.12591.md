# Where Decoder Cosine Similarity Fails for SAE Feature Flow Discovery

- **Key:** arxiv-2609.12591
- **Authors / organisation:** Hendrik Droste, Christian Medeiros Adriano, Kathrin Korte, Holger Giese; Hasso Plattner Institute (Potsdam) / IT University of Copenhagen
- **Date:** 2026-09-11
- **Link:** https://arxiv.org/abs/2609.12591
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1, 4 pages + appendices; 146,209 raw / 33,626 clean chars / 429 lines; `ltx_Math 142 / alttext 142`, exact equality)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The paper asks a narrow question about how one layer's computation feeds the next: if an SAE feature in the residual stream after layer `i` changes when layer `i`'s MLP runs, can you predict *which* upstream features caused that change using only the geometry of the decoder directions? The motivation is model adaptation — fine-tuning, editing and alignment procedures add capabilities while trying to keep old ones, and the authors want a way to read the feature-level mechanism by which one layer's update is produced rather than only parameter-level or representation-level deltas. They frame this as a step toward a "transition atlas" that could later be compared across stages of continual adaptation. The larger goal is a readable account of what a sublayer *computes*, expressed as feature-to-feature relations rather than a list of units.

## What they studied

Two models, one layer transition each. Pythia-160M is the primary study: the step from layer 7 to layer 8 (`L7 -> L8`), with three custom-trained TopK SAEs (dictionary width 33,024, top-k 100, input dim 768, 8B training tokens on `pile-uncopyrighted`, 128-token context, seed 43). Gemma-3-4B is the cross-model check: `L21 -> L22`, using pretrained Gemma Scope 2 width-16k `L0`-big residual and MLP SAEs. The unit of analysis is a **triple** — a residual-state feature `s_k` before the target layer, an MLP-output feature `u_j` in the target layer, and a residual feature `t_l` after it — so the object is coarser than a neuron but finer than a circuit, and explicitly cross-layer.

## How they did it

Three stages. **Extraction:** run the model over 20M tokens, collect the pre-layer residual, the target layer's MLP output and the post-layer residual, encode each with its SAE, and store only the top 64 active features per token. **Scoring:** define `Δz` for a target feature `t_l` as the difference between encoding the post-layer residual with and without the layer's MLP output added, in the same post-layer residual SAE basis — i.e. how much that MLP moved that feature. Then rank triples by an **interaction score**: the four-way conditional expectation `E[z_l | s_k,u_j] − E[z_l | s_k] − E[z_l | u_j] + E[z_l]`, divided by the target feature's standard deviation. This is a difference-in-differences, so a triple scores high only if the update feature predicts something *beyond* what the state feature and the update feature predict separately over their own marginals. Pairwise marginal baselines for `(s,t)` and `(u,t)` are accumulated first, precisely so that common features are not ranked for being common. **Candidate pruning:** per token only the strongest `n` local triples ranked by `|z_sk · z_uj · Δz_l|` are kept out of the Cartesian product of retained features, which takes 199M raw candidate triples down to 262,735 filtered ones on Pythia. **Validation:** for each selected triple, find token positions where both `s_k` and `u_j` are active, ablate *only* the decoded `u_j` contribution from the MLP output, reconstruct the residual, re-encode, and take the normalised effect `A_z = (E[z_l] − E[z̃_l]) / std(z_l)`. A triple is "strong" if `|A_z| >= 2` — magnitude only, because they are asking whether geometry recovers large effects, not whether it gets their sign right. A smaller complementary experiment ablates `s_k` and measures the effect on `u_j`.

## What counts as an explanation here

Two competing kinds are put in opposition. **Geometry** is the pairwise decoder cosine between two features' decoder directions — free, needs no activations and no interventions, and asserts that two features "write in similar directions". **Atlas** is a scored, causally validated relation over three features at once, with the interaction score as the ranking statistic and the ablation as the warrant. The paper's claim is that the first is not merely weaker but *systematically wrong* as a discovery instrument: a transition can be strong and causal while both of its pairwise cosines are near zero, because two low-cosine features can jointly push the residual into a third feature's activation region. The explanation is partial and local (one layer boundary, one model at a time) and it does not predict behaviour — it predicts feature-level *state changes* under an ablation.

## How they judged success

The measures are: fraction of filtered triples that pass `|A_z| >= 2`; and, inverted, the fraction of strong triples that a cosine threshold would have recovered. They report direct-write recall (strong triples with high update-target cosine), the joint condition `c_s,t < τ AND c_u,t < τ`, and a top-k overlap baseline between a cosine-ranked list and the causally validated set. Thresholds `τ = 0.4` and `τ = 0.7` are taken from prior work, and two main numbers are reported for each so the result cannot be a threshold artefact. The Gemma check adds per-baseline overlap counts.

## What they found

On the Pythia full atlas, 262,735 triples were filtered and 38,125 (14.5%) reached `|A_z| >= 2`. High update-target cosine *does* capture direct MLP writes — 2,254 of 3,211 triples (70.2%) with `c_u,t >= 0.7` are strong, and 3,876 of 5,855 (66.2%) at `>= 0.5` — which reproduces the prior direct-write finding. But of the 38,125 strong triples, 33,550 (**88.0%**) have *both* the state-target and update-target cosine below 0.7, and 23,677 (62.1%) still have both below 0.4. The stronger conditioning effect runs state → update: across the 4,096 highest-ranked `(s,u)` pairs and 500k tokens, 3,765 pairs with at least ten active tokens reach `A_z >= 2` (91.9%), with a median update-feature effect of `A_z = 40.6` — ablating the state feature strongly moves the MLP update feature. On Gemma-3-4B the direction is the same but weaker: of 12,139 strong candidates within the top-30,000 validated subset, 6,506 (53.6%) have both cosines below 0.7 and 2,291 (18.9%) below 0.4, but a top-30k update-target cosine baseline alone already overlaps 8,278 strong candidates (68.2%) while the state-target baseline overlaps only 1,676 (13.8%).

## Limits and open problems they name

The Gemma run validates only the top 30,000 ranked triples, so it is not a full-atlas recall measurement and its overlap numbers are inflated relative to a full run — the authors say so explicitly and decline to make a cross-model claim from it. Only one layer transition per model, one SAE architecture for the causal part (custom TopK on Pythia, Gemma Scope on Gemma), one training seed, and no SAE-family comparison. The atlas itself is "precision-oriented", not an exhaustive enumeration: the per-token pruning to the strongest `n` local triples is a choice that the paper says it can only now estimate the sensitivity of. The `|A_z| >= 2` criterion is magnitude-only by design, so sign prediction is untested. Everything is inference dynamics in a fixed model; the stated extension to learning dynamics across adaptation stages is unrun.

## What we take from it

Three things, all method. **(1) A control we should copy: the difference-in-differences interaction score.** When our white-box model claims edge `A -> B` matters, the natural confound is that both are simply frequent or both correlate with a third common feature; the four-term conditional expectation subtracts exactly the two marginals, and it is as cheap as counting co-occurrences. **(2) A warning with a number attached.** A geometry-only baseline on feature relations — pair decoder cosine, and its relatives — recovered only about one in eight of the causal transitions at `τ = 0.7` and still missed the majority at `τ = 0.4`, in the one setting where both were measured. Any claim of the form "features X and Y are related because their directions align" needs the ablation arm beside it. **(3) A dataset/measure pairing:** the normalised ablation effect in target-feature standard deviations (`A_z`) is the measure that lets effects be compared across features with different activation scales, and the paper's reason for normalising (massive-activation outliers dominating raw effects) is a concrete failure our own profiler will hit. What it does **not** give us: any evidence at scale (160M and a 4B partial check), any behavioural consequence of a validated triple, or any statement about whether the atlas triples survive an SAE retrain — which is the folder's standing question about dictionary instability.

## Tags

unit: feature
form: dictionary
method: SAE | ablation | decomposition
eval: ablation | faithfulness | benchmark
goal: understanding
