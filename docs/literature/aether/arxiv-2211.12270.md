# Causal Abstraction with Soft Interventions

- **Key:** arxiv-2211.12270
- **Authors / organisation:** Riccardo Massidda, Atticus Geiger, Thomas Icard, Davide Bacciu; University of Pisa (Massidda, Bacciu), Stanford University (Geiger, Icard)
- **Date:** 2022-11-22 (v1); published at the Conference on Causal Learning and Reasoning (CLeaR), PMLR 2023, pp. 68–87
- **Link:** https://arxiv.org/abs/2211.12270
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Every formal theory of causal abstraction in the mechanistic-interpretability lineage is written for hard interventions — a variable is pinned to a constant. The authors point out that the manipulations we actually perform on a network are not like that: nudging weights, retraining a block, or patching in a value produced by another mechanism changes a causal mechanism rather than fixing a value. Their motivating example is explicit about it: raising the weights of a deep model modifies mechanisms, it does not pin representations to single values. So the theory that says when a high-level account is faithful to a low-level system has been built on a special case of the manipulation class we care about. Their aim is to extend the Beckers–Halpern notion of abstraction to soft interventions (replacement by a possibly non-constant function that adds no new parents) and, in doing so, to keep the one property that makes abstraction useful operationally: that each low-level manipulation corresponds to *one* high-level manipulation.

## What they studied

Not networks — structural causal models, in pairs: a low-level model and a high-level model over fewer variables. The unit of analysis is the pair of maps that constitutes an abstraction: a surjection on settings of the variables (endogenous and exogenous separately) plus the induced partial map from low-level interventions to high-level ones. Illustrations are small arithmetic SCMs (integers at the low level, Booleans at the high level, parity as the abstraction map; a binary adder at one level and an integer adder at the other), each chosen to expose a specific failure of the naive generalisation. No data, no training, no benchmark.

## How they did it

- Restate the hard case precisely: a restriction set is the set of total settings compatible with a hard intervention, and a model abstracts another when the abstracting map commutes with the induced intervention map on every exogenous setting.
- Generalise the two constructs that the definition rests on — the restriction set becomes the set of settings whose projection lies in the *image* of the replacing function, and the intervention ordering becomes containment of those soft restrictions — then check in an appendix that both reduce exactly to the hard constructs.
- Define a first, weakest extension ("low soft abstraction") that allows soft interventions only at the low level, and show with a worked example that it behaves well whenever the abstract model admits only hard interventions.
- Then break it: prove a theorem giving the precise condition under which the intervention map is *not* unique, and instantiate it with an arithmetic example where a single low-level replacement is consistent with two different high-level replacements that differ only on variable values the system's mechanisms never produce.
- Diagnose the cause (consistency has been required on exogenous settings only, so two functions that agree on everything reachable are indistinguishable) and strengthen the definition to demand consistency on all exogenous *and* endogenous settings. Prove uniqueness, preservation of the empty intervention as a fixed point, and order preservation for the strengthened definition.
- Specialise to the constructive case — an alignment assigning each high-level variable a subset of low-level variables — and derive the closed form of the intervention map: recover high-level inputs through a partial inverse of the abstraction map, apply the intervened low-level mechanism to the aligned coordinates, and map the result back up. Prove that an intervention on a set of low-level coordinates can only touch the high-level variables whose aligned sets intersect it.
- Check consistency with the earlier theory: soft abstraction reduces to hard τ-abstraction when both intervention sets are hard, so nothing is lost.

## What counts as an explanation here

A definition plus two structural theorems — a correctness criterion for an abstraction, and a *constructive normal form* for the intervention map. The account of a high-level variable is "this aligned set of low-level coordinates, this map from their values to the high-level value", and the explanation of an intervention is a formula that says what function the abstract system receives. It is exact and binary, not approximate: either the commutation holds on all settings or the abstraction is not one. Its predictive content is real but narrow — given the alignment and the mechanism, the high-level intervention is *computed*, which is precisely the operation our batteries perform when they claim a patch was "the same intervention" at two levels. Completeness is explicitly out of scope: the paper does not take up measuring how far a candidate abstraction is from being exact.

## How they judged success

Proofs, with the worked examples as the counter-model evidence. The results are: uniqueness of the intervention map under soft abstraction (proved for arbitrary sets of admissible interventions at both levels, not only hard ones); the empty intervention maps to the empty intervention; a stepwise relation between low-level interventions implies the same relation between their images; reduction to the existing hard-intervention definition; and the explicit form of the map as a *necessary* condition. The failure mode is established by an existence theorem plus two concrete examples rather than by measurement. There is no experiment, no metric, no dataset and no baseline — the paper is a foundations paper and does not present itself otherwise.

## What they found

The naive generalisation is genuinely insufficient, and the reason is exactly the one our evaluation batteries keep meeting: a definition that only demands agreement on the settings the system can actually produce cannot distinguish two high-level mechanisms that differ off that support. The authors exhibit a low-level replacement (adding two integers) that is equally consistent with a high-level equality test and a high-level conjunction — the two functions differ only in a case no exogenous setting reaches, so both witness the abstraction. The strengthened definition, demanding consistency on every endogenous setting as well, kills the ambiguity: on the specific setting that separates the two candidates, the wrong map breaks commutation, so the intervention map becomes unique. From uniqueness they get the closed form, and the closed form is what turns the theory into something checkable without enumerating settings.

## Limits and open problems they name

Approximate abstraction is explicitly left aside — the paper cites the existing approximate-abstraction work and says it does not tackle it, which matters because every real account is approximate. Beyond that the setting is entirely deterministic and discrete: finite variable sets, exact symbol-valued domains, mechanisms treated as total functions, interventions constrained to replace a mechanism with a function on the same parents. Nothing is learnt: the alignment and the abstraction are given, not discovered. There is also no treatment of what an admissible soft intervention class should be in a network — the formalism allows any same-typed, no-new-parents function, which is very permissive.

## What we take from it

- **A vocabulary for the patches we actually do.** Our interventions on a network are overwhelmingly soft in this paper's sense: we replace a residual-stream slice with something computed by another path, or we add a direction. Framing them as mechanism replacements rather than value clamps makes the abstraction question answerable, and it removes a mismatch that has been sitting under the folder's hard-`do()` machinery since the start.
- **The support point, which is a warning for every probe battery.** Ambiguity in the intervention map arises because two high-level mechanisms can agree wherever the system's own mechanisms can reach. That is the same structural fact behind confidently-wrong accounts that only ever see reachable neighbourhoods: any account validated only on data the model produces is under-determined on the set of mechanisms that differ off-support. The fix the paper uses — demand consistency on settings, not just realisations — is the theoretical counterpart of an off-distribution control.
- **A testable normal form.** Theorem 2's composition (map inputs up through a partial inverse, apply the intervened low-level mechanism on aligned coordinates, map back) is a recipe for predicting the high-level effect of a low-level edit, and it is falsifiable: compute it and compare with the real high-level effect. That is the cheapest predictive check available to us for any alignment we propose.
- **Locality as a claim to check rather than assume.** The paper proves that editing low-level coordinates only touches high-level variables whose aligned sets intersect the edit. In a network, dense mixing means this locality routinely fails, and the failure is a measurable deviation from the predicted formula rather than a vague worry about entanglement.
- **A caveat about what uniqueness buys.** The uniqueness theorem is a statement about the *definition*, not about discovery: it says that if a soft abstraction exists its intervention map is determined, not that the alignment is identifiable from behaviour. Our alignment problem is untouched by this, which is a useful boundary to state when we cite it.

## Tags

unit: circuit | whole-model
form: causal-abstraction
method: other | decomposition
eval: prediction/simulation | faithfulness
goal: understanding | evaluation-of-methods
