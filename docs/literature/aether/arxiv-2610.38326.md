# Can LLMs Faithfully Solve Ontology Engineering Tasks? A Mechanistic Interpretability Study

- **Key:** arxiv-2610.38326
- **Authors / organisation:** Bohui Zhang, Albert Meroño-Peñuela, Elena Simperl; King's College London, with Simperl also at TU Munich; code at github.com/King-s-Knowledge-Graph-Lab/MIOE
- **Date:** 2026-10-28
- **Link:** https://arxiv.org/abs/2610.38326
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Interpretability work on knowledge in language models has mostly been done on entity attributes and factual recall — the model knows a fact, and the question is where that fact lives. Ontological knowledge is a different object: it is the class a subject belongs to, its taxonomic position, which relations are admissible for it. The authors argue that behavioural evaluation cannot tell you whether a correct ontology answer comes from a computation that tracks the ontological relation or from a shortcut through surrounding subject semantics. Their larger goal is faithfulness in the technical sense: whether the output reflects the internal process that produced it. To get at that, they want a benchmark that poses an ontological question with a definite counterfactual answer, so that an internal intervention can be scored on whether it actually moves the answer.

## What they studied

Qwen3-4B and Qwen3-8B, both 36 layers (hidden sizes 2,560 and 4,096). Two ontology engineering tasks: entity typing (`instance of`) and subsumption prediction (`subclass of`). The unit of analysis is an **interchange intervention target** — a learned set of coordinates, a learned low-rank subspace, or a set of SAE latents — at the residual stream, either at one layer or across a contiguous window of two or four layers. Data is sampled from Wikidata, WordNet and schema.org: 54 classes / 1,099 subjects / 376 auxiliary attributes for `instance of`, and 39 classes / 1,113 subjects / 160 attributes for `subclass of`, with 4,288 and 2,095 prompt templates respectively.

## How they did it

Each example pairs a **base** subject with a **source** subject whose target class differs. Two high-level variables are defined: the target taxonomic value, and a set of class-specific auxiliary attributes (capital, currency, headquarters, gestation period, and so on) that belong to the subject rather than to its class. An intervention replaces the base's residual-stream value at the chosen site with the source's, and the benchmark asks for two things at once: on a target-relation prompt the prediction should move to the *source's* class, and on an auxiliary-attribute prompt it should stay at the *base's* value. Five methods are compared — an SAE whose latents are filtered by a sparse linear SVM to pick the class-relevant subset, DBM/MDBM (a learned axis-aligned mask over residual coordinates), and DAS/MDAS (a learned k-dimensional orthogonal projector) — with the multi-task variants MDBM/MDAS trained on auxiliary-invariance examples in addition to answer-swap examples. DAS/MDAS are extended to contiguous layer windows, with an independent rank-k subspace at each layer in the window and all layers optimised jointly, and the base pass evaluated with hooks firing in ascending layer order so that later sites see the already-intervened state. Two evaluation settings: **Entity** (held-out subjects, training templates) and **Context** (held-out templates, including Wikipedia-derived prompt contexts). Scores are Cause (target answer replaced) and Iso (auxiliary answer preserved), both with multi-valid-answer sets, macro-averaged over prompt-template groups so that a template with many retained examples cannot dominate.

## What counts as an explanation here

A **causal abstraction**: a claim that one specific internal object — subspace, coordinate mask, or latent set — carries exactly the high-level partition the ontology drawing would produce, in both directions. The target is never a description of the concept; it is a pair of behavioural guarantees, replacement on the target variable and invariance on the auxiliary variable, measured by intervention. It is explicitly partial: the paper's own conclusion is that the abstraction is *supported but not cleanly disentangled*, and it is careful to leave open whether the residual entanglement is a property of the representation or a limit of the search methods. Nothing here predicts behaviour outside the intervention; there is no surrogate, no program, no feature dictionary offered as the deliverable.

## How they judged success

Cause and Iso, each an indicator over an accepted-answer set, aggregated example → prompt-template group → macro over groups; Disentangle is their unweighted mean. The single most important design choice is the **retention filter**: interventions are scored only on prompt instances where the *unintervened* model already gives an accepted answer. Base accuracies before filtering are reported in the appendix — target micro/macro 94.7/94.3 and 93.0/91.8 for Qwen3-8B on the two tasks, with auxiliary-attribute accuracy much lower at 64.9/63.8 and 69.1/68.3. Layer is swept over all 36 layers, subspace rank over {1024, 2048}, mask width over the full residual dimension, window rank/width over 512×2, 256×4 and 512×4 (matched total dimension 1024 or 2048), and the same configuration is re-evaluated on the other prompt setting. Class-level analysis adds Spearman rank correlations of per-class Disentangle across the two model sizes and between base and source roles, plus a distribution of reciprocal gaps over eligible unordered class pairs.

## What they found

Distributed subspace methods win, and the ordering MDAS > DAS > MDBM > DBM > SAE is identical on both axiom types and both model sizes. On Qwen3-8B, MDAS reaches Disentangle 0.592 (Entity) / 0.587 (Context) for `instance of` and 0.513 / 0.542 for `subclass of`, against SAE at 0.412 / 0.371 and 0.397 / 0.429. Adding auxiliary supervision trades Cause for Iso: across the matched cells the multi-task variants gain 0.081 Iso and lose 0.052 Cause on average, and the trade improves the mean. **Contiguous windows beat matched single sites**: at total dimension 1024, `256×4` outperforms `512×2` for both DAS and MDAS on `instance of`, and the best four-layer DAS row reaches 0.667 against a single-layer 0.529 for the same method and dimension. Layer profiles show Cause rising through the middle of the model and often collapsing in late layers where Iso is high — the intervention stops changing the output at all. Class-level results are systematic and replicate across scale: high-scoring classes are the ones with compact, regular attribute profiles (airport, city, country, currency, software, university), while broad heterogeneous superclasses (food, body of water, geographical feature, material, biological process) score lowest. Class difficulty ranks agree across model sizes at ρ 0.83 / 0.81 (base/source) for `instance of` and 0.75 / 0.67 for `subclass of`, but base-role and source-role rankings are nearly uncorrelated (ρ 0.27 and 0.27; −0.08 and 0.06) — a class can be a good intervention base and a poor intervention source. Reciprocal gaps are common: the median absolute difference between intervening A←B and B←A runs 0.132–0.179, with 29.8–41.8% of eligible pairs differing by at least 0.2.

## Limits and open problems they name

Ontology-level predictions are causally accessible but not cleanly separated from class-specific information, and the paper states two candidate readings it cannot distinguish — genuinely distributed and entangled representation, or intervention-search methods that are not precise enough. Single-token site only (last subject subtoken), one metric family inherited from RAVEL, two checkpoints from one model family, and a retention filter that removes exactly the items where the model's behaviour is most interesting. The authors name richer ontology constraints, broader model families, and stability across domains and complex reasoning as future work. Visible but unstated: the best configuration is selected by Entity Disentangle and then reported on Context, no random-direction or shuffled-label intervention target is run as a null, and the original-paper `Dimension` column for DBM/MDBM is the *ambient* mask width rather than the effective feature count (the appendix gives effective sizes of 1,631–2,580), which makes DBM look far larger than the subspaces it is compared against.

## What we take from it

**A measure and a framing.** This is the folder's cleanest worked instance of the two-sided criterion — an intervention must both *move the designated variable* and *leave the designated non-variable alone* — and the pairing is what turns a single patching result into a claim about a partition rather than about a component. Any white-box model that claims to have found the variable for a behaviour should be able to say what it is *not* supposed to touch, and print the number.

**A warning about the retention filter.** Scoring interventions only where the clean model already answered right makes the benchmark a test of *rewritability* on easy items, and it silently conditions every number on a different, and possibly much smaller, population than the one the behaviour is measured on. The base auxiliary-attribute accuracies (64.9 and 69.1 micro) tell you the Iso column is computed on a stricter subset still.

**A dataset shortcut.** Wikidata-plus-WordNet with a class-level auxiliary-attribute inventory is a ready source of counterfactual pairs with a natural non-target variable, and it is nearly free compared with constructing synthetic causal graphs.

**A diagnostic that costs nothing extra.** The reciprocal-gap distribution (intervene A←B and B←A, print the median and the fraction above a threshold) is a direct test of whether the structure behaves like an undirected similarity or like an ordered hand-off, and no similarity metric can answer that question.

## Tags

unit: feature | direction | layer
form: causal-abstraction | direction
method: patching | probing | decomposition | SAE
eval: patching | ablation | benchmark | faithfulness
goal: understanding | evaluation-of-methods
