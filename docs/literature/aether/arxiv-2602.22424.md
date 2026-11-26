# Causality ≠ Invariance: Function and Concept Vectors in LLMs

- **Key:** arxiv-2602.22424
- **Authors / organisation:** Gustaw Opiełka, Hannes Rosenbusch & Claire E. Stevenson; University of Amsterdam
- **Date:** 2026-02-25
- **Link:** https://arxiv.org/abs/2602.22424
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Function vectors — the summed output of a handful of attention heads that carries an in-context-learning task — have become the field's favourite example of an abstract concept direction, and the evidence offered for their abstractness is that they transfer to new contexts. The authors ask the sharper question: does the same concept, extracted from a different *format* of the same task, give the same direction? If the answer is no, then "transfers across contexts" was measuring portability within a prompt family rather than abstraction. The larger goal is to decide whether the components that causally drive a behaviour and the components that represent a concept invariantly are the same objects at all — the distinction our own white-box model has to make whenever it claims a direction "is" what it steers toward.

## What they studied

Four decoder-only models spanning two families and two scales: Llama-3.1 8B and 70B, Qwen2.5 7B and 72B. Seven relational concepts (antonym, categorical, causal, synonym, translation, present–past, singular–plural) crossed with three input formats (open-ended English, open-ended French or Spanish, English multiple-choice), giving 21 datasets with 50 prompts each for a total of 1,050 prompts; 5-shot open-ended prompts against 3-shot multiple-choice. The units of analysis are the individual attention head (ranked either causally or representationally), the summed-head vector, and — for steering — that vector injected at the last-token residual stream of a chosen layer.

## How they did it

Two head-selection procedures are run on the same heads and then contrasted. The causal one is activation patching in its established form: mean-ablate a head's output in a corrupted prompt and restore it from clean runs, giving a causal indirect effect per head, averaged over all datasets into an average indirect effect that ranks heads for the function vector (the authors note they score this across all three formats, whereas the original function-vector work scored English open-ended only). The representational one is new to this literature: for every head, compute a representational similarity matrix over all 1,050 prompts using the cosine between its outputs, build a binary design matrix that marks prompt pairs sharing a *concept regardless of format*, and score the head by Spearman's rank correlation between the two lower triangles. Top-ranking heads from each procedure are summed per prompt to form a function vector or a concept vector.

Comparisons then run in three directions. First, the overlap in identity: the intersection of the top-K causal heads and the top-K representational heads at K = 3, 5, 10, 20, 50 and 100, with a hypergeometric null for independent random subsets of the same size and a one-sided tail probability. Second, the overlap in layer: both procedures' scores averaged across heads per layer, to see whether the dissociation is a depth artefact. Third, an alternative explanation for the dissociation — that patching within one format biases selection toward format-specific heads — is attacked directly with cross-format patching across all six ordered format pairs.

Steering is the behavioural half. Vectors are built per concept and per extraction format from mean last-token activations over the 50 extraction prompts, then added at the last-token residual with a strength coefficient, sweeping the number of heads over six values and the strength over five and reporting the best combination per model (tuned on antonym prompts, "to maximize the average steering effect across all input formats"). The evaluation task, AmbiguousICL, interleaves two concepts inside one prompt — three exemplars of the target concept then two of English-to-French translation — so the unsteered model continues in the wrong concept. The score is the change in probability of the first concept's expected answer, over 100 held-out prompts per concept, i.e. prompts disjoint from the extraction set. Consistency across formats is measured separately, as the KL divergence between the post-steering distributions produced by an in-distribution vector and an out-of-distribution vector, at the five layers with the best in-distribution gain. Section 3.2.3 adds a qualitative read of the top-gaining tokens per vector, which is where the format contamination becomes visible.

## What counts as an explanation here

A head-level localisation plus a dissocation: two disjoint sets of attention heads in the same layers, one characterised by causal effect and one by format-invariant representational geometry, each summarised as a vector that can be injected. It is partial by construction — the paper repeatedly says it is not claiming the concept vector is the mechanism, only that abstract structure is encoded somewhere distinct from where the task is executed. It predicts one thing before it is measured and then checks it: if the concept vector really encodes concept-independent-of-format, it should steer the same way regardless of which format it was extracted from, and the function vector should not.

## How they judged success

Invariance is scored root-mean-square-style from similarity matrices: concept design-matrix RSA (higher is better) and question-type design-matrix RSA (lower is better), reported per model and per K. Head dissociation is scored by overlap against a hypergeometric null with p < 0.05. Steering is scored by mean change in target-token probability over 100 prompts (with top-1 accuracy in an appendix), by KL between in-distribution and out-of-distribution post-steering distributions, and by the identity of the top-gaining tokens. The controls that matter most: cross-format patching (Table 4), a zero-shot steering arm (Figure 17), a partition of function heads into all / common-across-formats / multiple-choice-unique subsets with RSA recomputed on each (Appendix K), and exclusion of two anomalous Qwen-72B cells with a re-rank check showing top-5 identities unchanged.

## What they found

Same-concept function vectors are nearly orthogonal across formats — the similarity matrices cluster by format, with within-format mean cosine 0.90, while concept vectors cluster by concept and show only a weaker residual within-format cluster at 0.55. The two head sets barely intersect: overlap is zero at all K ≤ 20 for three of the four models and stays small at larger K, with the significant cells (12 of 100 heads on Llama-8B, 28 of 100, 39 of 100 on Qwen-7B, 6 of 100 on Llama-70B, 3 of 100 on Qwen-72B) still a small fraction of the sets. Behaviourally the trade is exactly as advertised: function vectors give the larger in-distribution gain (on Llama-70B's ambiguous antonym prompt the top antonym token rises 56% against 49% for the concept vector, from a 2% baseline where the model was putting 49% on the French translation), while concept vectors keep their effect out of distribution and produce lower in-distribution/out-of-distribution KL across models, with the gap bigger for multiple-choice than for French. Out-of-distribution function vectors carry the format with them: French-extracted vectors lift French subword continuations, multiple-choice-extracted vectors lift the opening bracket, and — the cleanest single tell — Spanish-extracted vectors bias toward French almost as strongly as French-extracted ones, so the signal is "a foreign-language format was used" rather than the language itself. Cross-format patching keeps finding the same function heads (L31H18 and L35H57 on Llama-70B) and never finds a concept head. Concept vectors fail outright in zero-shot steering, where the concept is not present in the prompt at all.

## Limits and open problems they name

Concept-head selection used one global criterion — heads that encode all seven concepts at once — which the authors say may miss concept-specific heads. Nothing is said about how either head type forms during training or how they interact at inference; two hypotheses (detection/execution versus independent backup circuit) are offered and neither is tested. Steering hyperparameters were tuned on a single concept and then applied to all seven. The Qwen-72B categorical cells were dropped, with the paper's argument for robustness being that the resulting ranking is unchanged rather than that the exclusion is principled.

## What we take from it

A framing and a measure. The framing is the equivariance-versus-invariance distinction: a vector that adapts to its extraction conditions and a vector that is the same across conditions are two different objects, and "it transfers" is compatible with the first. The measure is the best cheap test of that distinction we have seen in the folder — re-extract the same object under a surface change that should not matter, then report both the cross-condition cosine and the behavioural consistency (KL between what the two versions do downstream). The warning is about how the folder reads transfer: this paper's function vectors are causally primary, cross-format-patch-stable, and still not what the concept looks like, so a causal role does not buy invariance and invariance does not buy a causal role. The shortcut is AmbiguousICL itself — a prompt with two live concepts, where a vector must beat a competing one rather than merely raise a token's probability — which is a stronger and cheaper evaluation than the single-concept prompts most steering papers use.

## Tags

unit: head
form: direction
method: patching, probing
eval: faithfulness, downstream-task
goal: understanding
