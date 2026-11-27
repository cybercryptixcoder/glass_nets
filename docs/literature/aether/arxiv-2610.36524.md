# Causal Localization of Evidence Effects in Transformers

- **Key:** arxiv-2610.36524
- **Authors / organisation:** Jipeng Li, Pramod Khargonekar, Yanning Shen (affiliation not printed in the extracted text; UC Irvine inferred from the author group, unconfirmed)
- **Date:** 2026-10-27
- **Link:** https://arxiv.org/abs/2610.36524
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Circuit-finding methods all need a behavioural target to localize: a prompt pair that differs in the thing whose mechanism you want, and a readout whose movement you score. Natural-language targets are hard to control because changing which side of an argument is better supported also changes the wording, the associations the model brings, and the surface statistics. The authors want a target that is fully controlled — same template, same symbol identities, same length, same local answer context — while still having a labelled language task downstream to check that whatever they find is not an artefact of the toy input. Their larger goal is a localization procedure whose output (a small set of attention heads) is scored through interventions, comes with a statistical statement about how reliable the selection is, and is shown to keep its effect outside the prompts it was selected on.

## What they studied

One frozen model throughout the main study: EleutherAI/pythia-410m-deduped, 24 layers × 16 heads = 384 heads, no training or fine-tuning. Unit of analysis is the individual attention head's write to the residual stream at the final prompt position (after the output projection), and the discovery unit is a *set* of heads evaluated jointly. The controlled target ("controlled evidence aggregation", CEA) varies only the count and the ordering of two symbols in a 200-symbol evidence region followed by a 50-symbol tail; the readout is the logit difference between two single-token answers (`various` / `successful`) at the answer position. A separate natural-language (NL) bridge task — 16 reports about two mutually exclusive states, with direct and negated forms balanced so that each state word occurs eight times regardless of which state is supported — provides a labelled behaviour the symbolic sets are tested against. A six-model sweep (Pythia-70M/160M/410M/1B, GPT-2, Qwen3-4B-Base) checks that the machinery does not depend on one checkpoint.

## How they did it

Each prompt is `Tokens:` + the symbolic sequence + `Choose one word: various or successful` + `Answer:`, with tokenizer audited so both answer strings are single tokens. Four ordering regimes are fixed in advance: Front, Back, Interleave, Random (Front/Back/Interleave are deterministic sequences; Random draws a permutation). The "count effect" is the difference in mean logit difference between a low-count and a high-count condition, at (50,150) for the main experiment.

Head attribution for head i is the projection of its write onto the difference of the two unembedding rows, `s_i = <v_i, d>`, and the contrast of that quantity across counts ranks heads. Selection then runs in three stages: rank all 384 heads by absolute attribution contrast and keep 192; score those with a single-head joint removal/recovery score and keep 48; run forward greedy search on the 48-head pool with a budget of 12 heads. The greedy objective is a weighted sum of two *normalized* scores, each computed on a held-out evaluation split: **shrink**, which is how much the count effect is removed by zeroing the selected writes (1 − |ablated contrast| / |clean contrast|), and **match**, which is how closely replacing those writes by means from the opposite count condition reproduces the count effect in both directions (high-to-low and low-to-high), averaged, with overshoot explicitly penalized. Selection and evaluation use disjoint prompt batches; selection uses 48 prompts per count, evaluation 96. Controls are layer-matched random heads (same head count, same number of heads per layer, 120 draws per seed).

Three audits follow. First, the selection objective's operator dependence: the same selected set is rescored under eight ablation/patch operator combinations (pooled vs directional donor banks, matched-example donors, joint vs independent-head resampling, a Gaussian operator matching mean and full empirical covariance). Second, an exhaustive local check of the greedy objective over registered 8-head pools (all 256 subsets), measuring negative marginal gains and the empirical submodularity ratio. Third, a selection-matched permutation null in which the high/low labels of matched pairs are swapped and the *entire* adaptive pipeline is rerun for 99 trials. For transfer, the selected sets are held fixed and tested across counts, prompt-cue wordings, orderings, readout pairs, and models; for the NL bridge, sets selected on symbolic prompts are ablated on labelled NL reports and the accuracy drop is compared against sets selected on NL prompts and against layer-matched random sets.

## What counts as an explanation here

A compact set of attention heads plus two operational scores: the set removes most of the behavioural contrast under zero-ablation and reproduces it under bidirectional mean patching. The authors are explicit that this is a *partial* and *operational* account: their own framing is that "necessity" and "sufficiency" refer only to these tests, not to an isolated circuit; recovery is "an effect of replacement within the full model, not proof that the selected heads form a sufficient subnetwork"; and the interventions identify writes at one position, not head–MLP paths. So the explanation predicts something (that editing these writes will move the contrast, in both directions, on unseen counts and on a labelled NL task) but does not claim completeness, uniqueness, or global minimality.

## How they judged success

Shrink and bidirectional match on held-out splits; a joint threshold of shrink ≥ 0.85 and mean match ≥ 0.70 to define a "successful" set; layer-matched random controls for every selection; budget-matched baselines (sensitivity-only ranking, single-head activation-patching ranking, same-layer top-norm heads, random heads); a "transfer ratio" that places a transferred set's normalized joint score between the random reference (0) and the target-selected set (1); restricted NL accuracy plus two paired metrics — PairAcc (both reversed queries correct; 0.5 is what an answer that ignores the evidence and only uses query orientation would get) and MinPairAcc (the weaker of the positive- and negative-margin halves, which is 0 for that orientation-only rule and 0.25 for independent fair guesses). Everything is reported with cluster-bootstrap intervals (1,000 replicates over matched prompt pairs, endpoints 2.5/97.5) where genuine sampling units exist, and the authors go out of their way to say which numbers get *no* interval: the deterministic regimes contain a single unique symbolic input, so repeating them is not prompt sampling.

## What they found

Main symbolic result at (50,150) on Pythia-410M: the selected sets remove 65.5–91.6% of the count effect with mean match 0.759–0.974, against layer-matched random shrink of 0.020–0.069 and random high-to-low match of 0.048–0.075. Compactness swings enormously with ordering: Front needs one head (shrink 0.906, match 0.974), Interleave three (0.916 / 0.914), Random 7 ± 1 (0.866 ± 0.021 / 0.823 ± 0.084), and Back exhausts the 12-head budget without crossing the shrink threshold (0.655 / 0.759).

The set keeps working off its selection pair: held fixed over a count sweep, ablation removes 0.646–0.896 of the clean slope, and on three unseen count pairs shrink stays 0.612–0.839 with match 0.689–0.874. Order-only contrasts — identical symbol counts, only the ordering changed — produce output effects of 0.321–1.846 that the selected sets remove (shrink 0.603–0.842) while random sets remove almost nothing (0.049–0.081).

The readout-pair audit is the sharpest generalisation test: reselecting for a new answer pair works (mean shrink 0.642–0.861, match 0.701–0.884 over seven pairs including a deliberately symbol-overlapping A/B pair), but *reusing* the set selected for `various/successful` degrades with how unrelated the new pair is — yes/no transfers almost fully (0.772 / 0.810, Jaccard 0.610), cat/dog barely half (0.382 / 0.431, Jaccard 0.240). A registered scan over 1,024 fixed-prompt fixed-set readout conditions finds only 0.060 pass both thresholds overall and 0.064 among the 960 evaluable ones.

The NL bridge: symbolic-selected sets ablate NL accuracy by 8.2 points at k = 8, NL-selected sets by 15.2, layer-matched random by 0.8, on a base accuracy of 0.772. Transfer ratio 0.555 in the symbolic-to-NL direction, 0.599 the other way; selected-set Jaccard overlap between the two is only 0.296 ± 0.041. In the expanded FP32 bridge across three models, ratios are 0.520 / 0.557 / 0.438 (Pythia-410M, Pythia-1B, GPT-2) with accuracy drops 7.4 / 8.6 / 4.9 points.

Scope and operator: the same heads have essentially no effect when the intervention is applied at symbolic-sequence positions instead of the answer position (seq shrink 0.001–0.059 versus 0.655–0.916 at the last position) — the heads act where the readout is built, not where the evidence sits. Splitting heads into their QK (routing) and OV (writing) parts shows the effect is mostly in the write: OV-only match 0.85–0.92 against QK-only 0.12–0.35 for Front/Interleave/Random, with Back mixed (0.72 QK / 0.61 OV).

Their own audits are where the paper earns its keep. Attribution correlates with intervention: over 500 sampled head sets per regime, cumulative attribution correlates 0.841–0.923 with ablation gain and 0.755–0.884 with the two patch gains — an association, which they stress is not an error bound. The greedy objective is *not* well behaved: negative marginal fractions run 31% (Front), 17% (Back), and similar for Interleave and Random, minimum empirical submodularity ratios are −5.8, −2.6, −4.9, −3.9 across the four regimes, and the theorem they prove is for a different (raw, un-normalized) objective than the one the algorithm optimizes, so they say plainly that greedy search under an identical objective was never isolated. Against the selection-matched null, only the Random regime has enough unique pairs to test; observed utility 2.69, adjustable p = 0.01 at the 1/100 resolution floor.

## Limits and open problems they name

Compactness is not uniqueness or minimality, and the head sets are not guaranteed to be stable — the three documented Random selections overlap at minimum 5/9. Interventions are at one position and on head writes only; the authors list earlier-token computation and head–MLP paths as untested. The NL bridge shows the fixed symbolic set matters for prediction but cannot separate shared evidence *processing* from a shared yes/no *readout*: both tasks end in yes/no, so a set that just writes the answer is indistinguishable from a set that aggregates evidence. Lexical balancing rules out total-word-frequency shortcuts but not phrase-level or partial-report strategies. Theoretically, attribution, shortlist recovery and greedy optimality each need their own assumptions, and success at one step does not establish the next. The declared decision criterion for the NL bridge requires prompt-pair bootstrap intervals that the original BF16 study does not contain, so they record the criterion as not fully established rather than claiming it. Both tasks the review would care about — order-invariance and readout-invariance — hold only partially: sets tolerate cue rewording (shrink 0.64–0.93, match 0.70–0.96) but the off-diagonal regime transfer at equal budget spans − 0.84 to + 0.89.

## What we take from it

Four things worth stealing, and one warning.

(1) **The shrink/match pair as the reporting unit for any localization claim.** Zeroing a component and replacing it with a donor mean are different questions — removal can be mere disruption, and recovery can be achieved by components that were never "necessary" — and the paper's whole audit programme exists because the two disagree. For our white-box model, every nominated component (head, direction, feature, module) should carry both numbers on the same held-out split, with the match averaged over *both* patch directions, because a one-directional match is trivially satisfiable by a site that only ever pushes the logit one way.

(2) **The readout-generation and readout-pair stress test.** Their reselect-versus-reuse split is the cheapest way we have seen to quantify how much of a "circuit" is the model and how much is the measurement coordinate: reselecting costs nothing (0.642–0.861), reusing the same heads across unrelated token pairs nearly halves the effect. Any direction or head set we nominate should be evaluated under at least three readout pairs of graded relatedness and the Jaccard overlap printed beside the score.

(3) **Intervention scope as a printed number, not a methods detail.** The seq-versus-last comparison (0.001–0.059 against 0.655–0.916 for the same heads) is the same scope-ladder lesson the folder recorded from run 926, arriving from the position axis of a symbolic task; it should be routine to print the effect at the content positions and at the decision position for every set we report.

(4) **The selection-matched null.** Swapping labels *within matched pairs* and rerunning the whole adaptive pipeline is a stricter null than permuting scores, because it re-does the greedy search that is itself adaptive; the honest tail is that only one of four regimes had enough unique pairs to test.

The warning: **the greedy objective's own pathologies are the finding.** Negative marginal gains on up to 31% of edges and empirical submodularity ratios down to − 5.8 mean the "compact set found by greedy search" is not a certified near-optimal set; our own selection procedures should print the fraction of negative marginals and the empirical ratio, and should never report a greedy set as minimal.

## Tags

unit: head | circuit | layer
form: graph | probe | steering
method: ablation | patching | attribution | probing | decomposition
eval: ablation | patching | faithfulness | completeness | downstream-task | benchmark
goal: understanding | evaluation-of-methods
