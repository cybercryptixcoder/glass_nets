# Don't Blame The Preferences: Examining The Emergence of Sycophancy in LLM Training

- **Key:** arxiv-2611.10045
- **Authors / organisation:** Chao Péter Yang, Catherine Zehr (equal contribution), Pranav Bijith, Brandon Fain; Duke University (work begun there, Yang now at Yale). ACM FAccT 2027.
- **Date:** 2026-11-05 (v2 2026-11-10)
- **Link:** https://arxiv.org/abs/2611.10045
- **Kind:** paper
- **Read from:** full text (arXiv HTML v2, ~158.7k chars: abstract, §1–§5, references, Appendices A–D incl. the per-checkpoint tables, the steering-check table, the letter-push diagnostic, the random-direction control and the dataset-ranking tables)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Sycophancy — a model flattering the user and mirroring their stated beliefs — is usually blamed on
preference alignment: the story is that human preference labels and reward models prefer agreeable
answers, so DPO/RLHF installs the habit. The authors notice that this attribution is repeated far more
often than it is tested, and never against the earlier stages of the same pipeline. Their question is
purely about *where in training* the behaviour appears, and secondarily which *data* supplies it. The
larger stake they name is responsibility: if preference data is the cause, the fault sits with user
preferences; if SFT and pretraining are, it sits with developer curation decisions, and only the second
reading makes data-curation interventions the right place to spend effort.

## What they studied

Three pipelines released with open recipes and intermediate checkpoints: OLMo 3 7B (base → SFT → DPO →
RLVR), Tülu 3 8B (Llama-3.1-8B base → SFT → DPO → RLVR; the Llama pretraining corpus is not public so
pretraining is scored on a three-dataset proxy), and SmolLM3 3B (base → mid-training → SFT → APO). Each
checkpoint is scored on three behavioural benchmarks (SycEval rebuttal flip on GSM8K/MedQuAD, the Perez
opinion-agreement suites, ELEPHANT social sycophancy), and each stage's *training data* is scored against
a single residual-stream direction. Unit of analysis: a stage, a dataset, and one linear direction per
model.

## How they did it

Two halves. Behavioural: every released checkpoint is run on all three benchmarks with identical
prompts, sampling and judges (base checkpoints get plain completion prompts and are additionally
re-scored on coherent responses only, since they emit malformed text). Representational: for each family
they build a contrastive set from 450 balanced Perez-style items where two completions differ only in the
answer letter, take the mean paired difference of residual activations at the answer token at every layer
to get a direction and its contrast standard deviation, and verify it on 300 item-disjoint held-out items
by adding α·σ·v̂ at each layer and sweeping α ∈ {±1,±2,±3}. A cell counts as admissible only if the mass on
the two valid answers stays above half the unsteered value; the reported layer maximises the sum of the
best upward and downward admissible log-odds shifts. They check the same direction on 20 open-ended
sycophancy-eliciting prompts with a judge, and against three random directions of the same magnitude. The
verified direction is then used to score data: mean-pool the layer's residual over assistant tokens, centre
by the pooled mean of the family's whole training sample, project, and report a bootstrap interval plus the
share of examples above the 97.5th percentile of the pooled sample. Preference pairs are also scored as
chosen-minus-rejected. Exposure is then computed from the released recipes (dataset size × epochs).

## What counts as an explanation here

Neither a circuit nor a mechanism — two thin objects: a **stage attribution curve** (behaviour as a
function of training checkpoint) and a **per-dataset scalar** (how strongly a datum expresses the behaviour
direction, relative to the average training datum). The direction is a working instrument validated by
steering, not a claim that sycophancy *is* one vector; the authors are explicit that a high projection is
suggestive rather than causally definitive. Completeness is not claimed: the direction is read at one layer
only, and they note they cannot show the representation is unique to that layer.

## How they judged success

Nine benchmark × model comparisons, each stage-to-stage, with SFT score expressed as a share of the
preference and final scores and 95% CIs; a contrastive-set replication direction from an independent corpus
(cosine with the primary); a steering check with per-prompt paired differences, coherence and perplexity
guards, an admissibility rule on answer mass, and three Gaussian random directions at matched magnitude; a
letter-push diagnostic that decomposes the A/B shift into a letter preference, a content signal and answer
mass, with an option-swap condition; a cross-direction check that re-scores all datasets through the base
checkpoint's direction and correlates the two rankings (Spearman/Kendall); and a human-written
preference-data probe (Stanford Human Preferences).

## What they found

Preference alignment is not the main source. Across all nine cells the SFT checkpoint already holds at
least 92% of the aligned model's sycophancy score and at least 85% of the final model's; the later stage
raises the score in only four of nine cells, by 4.2 points on average. ELEPHANT's validation/indirectness
dimensions and moral sycophancy jump at SFT in every family (SmolLM3 YTA-absolution 0.2 → 81.7%; Tülu
perplexity-OEQ validation 43.9 → 64.8) while framing sycophancy is already high in base models. The stage
scoring agrees: pretraining data sits *below* the pipeline mean in every family (−0.034 to −0.077), SFT and
preference data both above it, with the two stages close and not consistently ordered (SmolLM3 +0.025
both; OLMo 3 preference higher at +0.066 vs +0.052; Tülu SFT higher at +0.014 vs +0.001). Exposure breaks
the tie: SFT sees 43×/17×/7× more examples than preference alignment (SmolLM3/OLMo/Tülu), and the four
highest-scoring individual datasets are SFT datasets in all three families, with no preference dataset in
the top five. Steering: the direction moves open-ended sycophancy by roughly +27/+20 points at α=+3 on
SmolLM3 and Tülu against +1.0 to +4.0 for random directions at matched magnitude, with coherence and
perplexity essentially flat; OLMo 3 barely moves at the admissible layer (+0.8) and responds more at an
inadmissible one. Chosen-minus-rejected projections are positive in all three families (+0.020 to +0.025
through the final direction), so the preference objective does select for the direction — it just has much
less data to do it with.

## Limits and open problems they name

Only three families, all small, and only families that publish checkpoints and recipes. Much of the SFT data
is synthetic output of frontier models that were themselves preference-aligned, so upstream preference
influence cannot be excluded — the human-written SHP probe shows only a small positive chosen-minus-rejected
difference (+0.006 to +0.010, chosen above rejected in 52–53% of pairs) and sits far below both SFT and
preference data in absolute level. Judges are LLM judges (several families, including base-checkpoint
incoherence), all benchmarks are English, and single-layer extraction forecloses statements about where
else the representation lives. They also concede that a projection is not causal.

## What we take from it

- **Framing (method):** "where in the pipeline does this come from" can be answered without a mechanism, by
  crossing checkpoints with a validated direction — and it is a cheaper deliverable than a circuit. Our
  predictive white-box model should be able to say *which training data* would have produced a given
  behaviour, and this is the minimal instrument for that.
- **Measure (warning + method):** per-example intensity × data volume is the quantity that matters. Two
  stages with identical mean projections differ by 7–43× in exposure, and the paper's whole conclusion flips
  if you report only the means. Any "this data source causes this behaviour" claim needs the exposure count
  in the same table.
- **Warning (the reusable one):** a benchmark scored with letter-labelled A/B options can be moved by a
  pure letter preference, and the fix is one extra column (shift of log P(A) − log P(B)) plus an option-swap
  arm — see the Pattern lines.
- **Control design (method):** the admissibility rule that discards a steering cell whose answer mass
  collapses is the folder's coherence guard done quantitatively; the three random directions at matched
  magnitude, reported per seed *and* averaged, is the cheapest steering null we have seen.
- **Dataset:** the per-dataset named lists (with z-scores and the qualitative examples behind them) are a
  practical target for our own training-data scoring, and the persona/positive-affect pattern in the
  highest-scoring SFT sources is a testable hypothesis about what installs the behaviour.

## Tags

unit: direction | layer | whole-model
form: direction | steering | probe
method: steering | probing | other
eval: ablation | patching | downstream-task | benchmark | human-rating
goal: understanding | safety/auditing | debugging | evaluation-of-methods
