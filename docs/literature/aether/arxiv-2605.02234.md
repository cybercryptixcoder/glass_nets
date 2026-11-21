# Bucketing the Good Apples: A Method for Diagnosing and Improving Causal Abstraction

- **Key:** arxiv-2605.02234
- **Authors / organisation:** Li Puyin, Jiyuan Tan, Ahmad Jabbar, Thomas Icard (Stanford University); Atticus Geiger (Goodfire)
- **Date:** 2026-05-04
- **Link:** https://arxiv.org/abs/2605.02234
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Interchange intervention accuracy (IIA) is the field's standard verdict on a causal abstraction: for each abstract variable you propose an alignment, run the swap on both the abstract and the neural side, and report the fraction of pairs that agree. The authors' complaint is the one every aggregate invites — a single scalar says whether the hypothesis is faithful on average but nothing about *where*, and the middling scores that most hypotheses actually get are the hardest ones to read, because a partial success can mean noise, a wrong site, or a hypothesis that is right on one region of input and wrong on another. Their proposal is to stop scoring the space and start partitioning it: given a candidate abstraction, cut the inputs into a region where it is essentially exact and a residual region where it fails, then look at what distinguishes the two. The larger aim is to make causal abstraction constructive rather than evaluative — a diagnosis you can act on by adding a variable, splitting one, or merging two partial hypotheses — and to do it with a partition rather than with new fitting machinery.

## What they studied

Three settings of increasing complexity, each with an existing candidate abstraction from a standard alignment method, and the unit of analysis is the **input pair** rather than the site. (1) A toy Boolean task — six tokens, target `o5 = ((t2≠t4) ∧ (t0≠t5)) ∨ (t1=t3)` over 20,000 generated examples with in-context exemplars, a 12-layer GPT-2-small (~117M) fine-tuned to 99.7% accuracy, and DAS at one dimension on block output. (2) Entity binding — the `filling_liquids` family with ten templatic "X fills a Y with Z" groups and a "who filled Y?" query, 1,024 samples, Gemma-2-2B-Instruct, a one-variable positional hypothesis (the queried group's context index), full-vector patching at the final query token. (3) Entangled factual recall — RAVEL's city/Language attribute on pretrained Llama-3.1-8B, where Language is co-encoded with Continent, Country, Latitude, Longitude and Timezone, localized by MDAS at layer 14 with subspace sizes swept over {32, 128, 512, 2048}. In all three the input space is first restricted to inputs the model answers correctly, so an intervention failure cannot be blamed on the model being simply wrong.

## How they did it

1. Pairwise consistency: an ordered pair of inputs is *interchange-consistent* iff for every abstract variable the abstract swap and the neural swap agree **in both directions** — source into base and base into source. Exact equality, not a tolerance.
2. Lift pairs to sets: a subset is consistent iff every pair inside it is, so build the *interchangeability graph* with one vertex per input and an edge per consistent pair; a consistent subset is then a clique. Because a perfect clique is brittle and NP-hard to find, they relax to γ-quasi-cliques (subgraph edge density at least γ, always 0.98) with a minimum size of 2.
3. Bucket: run a multi-seed greedy expansion — sort vertices by degree, try the top ten as seeds, repeatedly add the candidate whose inclusion keeps density above γ and maximises it, keep the largest result, remove it, repeat for K−1 buckets, and let the remainder of the space be the residual "other" bucket. That residual bucket is the deliverable as much as the good ones are.
4. Generalise and characterise: train a classifier to predict bucket membership for held-out inputs, once on hand-specified input features and once on model-internal features (SAE features at the aligned site), then read the ℓ1-regularised logistic regression's surviving weights to name the boundary. If the two classifiers agree, the partition is claimed to be structure rather than an artefact of a finite graph; if internal features do better, the boundary is claimed to be encoded in the model itself.
5. Recursion: where the diagnosis names a missing distinction, promote it to a new abstract variable and rerun the whole thing.

## What counts as an explanation here

A **partition of the input space plus a rule for membership in it**. The output is not a new alignment, a circuit or a description of a mechanism; it is a claim of the form "this variable is faithful for these inputs and not those, and the discriminator is this (previously implicit) variable or this surface feature". It is explicitly partial: it explains the shape of a failure, not the computation. It can predict exactly one thing, and that thing is checked — whether a new input with the stated discriminator will fall in the faithful region. The graph is also intended as an intermediate object: in the toy task the recipe is applied recursively across two passes and reconstructs the full hierarchy of variables, which is the paper's bid for the label "constructive" rather than "post hoc".

## How they judged success

Within-bucket IIA (should be high), cross-bucket IIA (should be low), graph edge density as a global measure of how uniformly faithful the abstraction is, and — the actual generalisation test — classifier accuracy on held-out inputs, with **IIA and density reported for the region the classifier carves out**. Each setting reports both a hand-feature and an SAE-feature classifier where feasible; the toy task is additionally checked against DAS alignment search over all variables, and the entity-binding result is checked against the failure mode that the prior manual analysis (`Gur-Arieh et al. 2025`) had identified by inspecting per-index curves.

## What they found

Toy logic: the deliberately underspecified output-only hypothesis aligns at position 77, layer 7 with IIA ≈ 0.78, and bucketing shows it is faithful exactly where the conjunction branch `o1 ∧ o2` is false — on which subset the whole task collapses to `o5 = o3` and a feature tracking only that branch looks right. The residual bucket is dominated by `o1 ∧ o2` true. Recursing, `o4` promotes to its own variable (near-perfect at position 81, layer 7; a non-trivial earlier signal at position 78, layer 5) and diagnosing *that* site yields `o1` as the next missing component, recovering `o1,o2,o3 → o4 → o5` from the output variable alone and confirming the model uses the intended factoring rather than its extensionally equivalent expansion. Entity binding: graph density only 24%, the good bucket is the sequence edges (groups 0,1,2,9) and the bad bucket the medial groups (3–8); the hand-feature classifier hits 90% IIA at 81% density, the SAE-feature version 98% IIA at 97% density, so the boundary is better read out of the model's own features than out of the index. RAVEL: MDAS is trained to maximise Cause *and* Iso across five off-target attributes, and its held-out IIA is still only 14.3%; graph density ~10%; bucketing recovers one bucket dominated by English cities and one by Spanish cities at roughly 98% within-bucket IIA, with everything else in a diffuse low-IIA residual. An SAE classifier reproduces the partition at 90% accuracy, but the highest-weight SAE features are only loosely language-consistent, and the authors say so plainly — the grouped inputs make the structure obvious where the feature descriptions do not.

## Limits and open problems they name

Two, stated in the conclusion. The method needs a prior task construal — a specified task space and a candidate abstraction to diagnose — so the authors describe it as a structured program search, not a general solution to interpretability. And the experiments are mostly single-variable hypotheses; with interacting variables, accumulated approximation error may leave no large dense bucket to find. They name scaling to richer tasks and using bucket structure to drive automated hypothesis search as future work.

Two further limits the entry adds. The RAVEL result is a *negative* one about MDAS that the paper reports as its own table of wins — a 14.3% held-out IIA for a method designed specifically for disentanglement means the buckets are describing which languages happened to be separable, not a Language representation. And the density threshold and minimum-bucket-size are held constant at 0.98 and 2 across all three settings with no sweep, so the partition's granularity is a choice the paper does not test.

## What we take from it

A **measure** and a **framing**, and the folder's first instrument of this shape. Concretely, for a predictive white-box model: (1) replace "our account explains the model on this battery at 0.78" with a partition of the battery and two numbers per region — this is the construction that makes a middling score informative rather than embarrassing, and it is cheap because it reuses the intervention pairs the battery already runs. (2) Borrow the bidirectional requirement: exact agreement only when swapping both ways counts, which is the same two-sidedness `arxiv-2402.17700` demands from Cause and Iso but applied at the level of input pairs. (3) The classifier step is the part to copy and the part to distrust in equal measure — it is the right test that a partition generalises beyond the finite graph, and the honesty about SAE features in RAVEL (weak, post-hoc, while the grouped inputs are obvious) is the warning that a classifier can succeed at a boundary whose *naming* you cannot read. (4) The 14.3% MDAS number with a Cause+Iso objective is a warning about our own fixtures: a subspace can be optimised to satisfy two faithfulness criteria and still fail on held-out pairs, so any account we fit gets scored on pairs disjoint from the ones that shaped it.

## Tags

unit: feature | direction | layer | whole-model
form: probe | graph | causal-abstraction
method: patching | ablation | probing | SAE | decomposition
eval: patching | faithfulness | prediction/simulation | benchmark
goal: understanding | evaluation-of-methods | debugging
