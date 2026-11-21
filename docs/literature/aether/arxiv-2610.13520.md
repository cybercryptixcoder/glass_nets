# Causal Effect Identifiability in Causal Abstractions: A Strict Hierarchy of Criteria

- **Key:** arxiv-2610.13520
- **Authors / organisation:** Clément Yvernes, Marianne Clausel, Emilie Devijver, Eric Gaussier; Univ. Grenoble Alpes / CNRS / Grenoble INP / LIG (Yvernes, Devijver, Gaussier), Université de Lorraine / CNRS / CRAN (Clausel)
- **Date:** 2026-10-09
- **Link:** https://arxiv.org/abs/2610.13520
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Standard causal effect identification needs the whole causal graph, which nobody has for a high-dimensional system. Causal abstractions relax this by keeping only partial causal information, and the practical consequence is that an abstraction rarely names one graph — it names a *collection* of graphs compatible with what we know (a Markov equivalence class, a PAG, a summary or cluster graph). The trouble is that each graph in the collection can give a different answer to the same interventional query, so "the effect is identifiable" stops being a single property and becomes a family of properties depending on how validity is quantified across the collection. The authors' complaint is that these notions were invented separately, in different sub-communities, and never compared. Their aim is to put all of them in one framework, order them by strength, and prove exactly where each one is strictly stronger than another — so that a paper claiming "complete identifiability under abstraction X" can be read as a claim about one specific rung of a ladder.

## What they studied

The mathematical object is a class of causal diagrams over a fixed observed variable set, plus a causal query of the conditional intervention form. Unit of analysis is therefore the *class* and the *derivation*, not any model. Six notions are treated as the population: identifiability through graphs (a single do-free functional valid for every structural model whose graph lies in the class); the same with the true observational density supplied as extra information; identifiability by a common causal derivation (one do-calculus rewrite sequence valid in every graph of the class); and three "common criterion" notions where a named graphical criterion — generalized adjustment, backdoor, frontdoor — holds in every graph of the class. There are no neural networks, no datasets and no experiments; the evidence is propositions with worked counterexample classes, mostly tiny three- and four-node graphs.

## How they did it

- Recall the single-graph setting in functional form: a query is identifiable in a graph when one identifying functional is valid across every structural model inducing that graph.
- *Lift* each classical notion to the collection: the universal-functional notion requires validity over every model whose graph is in the class; the common-derivation notion requires one derivation whose every d-separation obligation is met in every graph; the common-criterion notions require the criterion itself to hold everywhere.
- Establish implications by inclusion of the model sets the notions quantify over — a more informed or more restricted setting implies the less informed one, and a derivation that holds in every graph certainly supplies a functional valid in every graph. Backdoor and frontdoor are folded in because each is itself expressible as a do-calculus derivation, and generalized adjustment subsumes backdoor.
- Prove strictness by exhibiting a class and a query for each converse. Most separations are inherited from the single-graph literature, since a one-element collection is a special case.
- For the hard separation — a query identifiable through graphs but not by a common derivation — introduce two auxiliary graphs. A *derivation graph* of a diagram has one vertex per conditional query expression and an edge whenever one do-calculus rule takes one to the other in that diagram; a *step graph* of a derivation has exactly the steps that derivation uses. The key lemma says a derivation holds in a diagram if and only if its step graph is a subgraph of the diagram's derivation graph. Then: take the class {Z←X→Y, X←Z→Y}, observe that the intersection of the two derivation graphs is contained in the derivation graph of a third chain graph X→Z→Y, and argue by contradiction — any common derivation would be a subgraph of the intersection, hence would also hold in the chain, where the adjustment set used to identify the query in both members of the class provably fails for some compatible density. Contradiction.
- Add a reduction step: since independence statements survive edge deletion, every identifiability notion here is invariant to discarding non-maximal elements of the class, so a class can be replaced by its maximal graphs under inclusion (assuming every graph sits inside some maximal one, automatic for finite classes). If the class has a greatest element, the two weakest and strongest notions coincide.

## What counts as an explanation here

Not an explanation of a system — an ordering of *claims*. The deliverable is a strict hierarchy among identification criteria, plus the bookkeeping terminology that lets a completeness claim be placed on it: "atomically complete calculus" (a calculus whose atomic rules are sound and complete for uniformity of the do-calculus rules over the class) is shown to be stronger than necessary for the weakest notion. Everything is exact and deterministic; nothing is approximate, nothing is learned, and the authors are careful to frame the result as making existing results *comparable* rather than as producing a new identification algorithm.

## How they judged success

Correctness and strictness of each implication, established by proof, with one worked counterexample per failing implication and a separate argument for two notions being incomparable in both directions. The separations are witnessed by concrete classes over three or four variables plus an explicit distribution that is compatible with one graph in the class and breaks the formula in another. There is no metric, no benchmark and no baseline, which is appropriate for the genre but means the entry has no numbers to report.

## What they found

The hierarchy is strict and not a chain. Reading from strongest to weakest: a common criterion implies identifiability through graphs; backdoor and frontdoor as common criteria imply a common derivation, which implies identification through graphs, which in turn implies the version that also uses the observational density. The direction of the last implication is what makes it a weakening: a univariate example with two opposite directed edges between one cause and one effect, where the two variables happen to be observed independent, identifies the effect once the density is known and not from graph structure alone. Generalized adjustment and common derivation are *incomparable* — the paper's central theorem shows common adjustment does not guarantee a common derivation, which via the "adjustment implies through-graphs" implication breaks the previously open question of whether identification through graphs could be stronger than common-derivation identification. Consequence they draw: an atomically complete calculus is a sufficient condition for the weakest notion, not a characterisation of it. The reduction theorem means a large abstraction can be assessed by its maximal graphs alone, which explains why cluster graphs with an admissibility assumption behave like single graphs.

## Limits and open problems they name

No algorithms and no complexity results: deciding identification through graphs and by common derivation in practice is left open, and the authors are explicit that these are the computational questions the framework raises. All results are within the standard acyclic/ADMG setting with hard interventions; no soft interventions, no approximate or quantitative identification, no finite-sample statement. The hierarchy is a statement about *criteria*, not about estimators, so nothing here says how to estimate an identified effect. Generalizing beyond the six named notions — other side information besides a known density, other graphical criteria — is gestured at as a family, not developed. The reduction to maximal elements needs the existence-of-maximal assumption, which they note is automatic for finite classes but only argued case-by-case for infinite ones.

## What we take from it

- **The claim-shape rule, again, with an extra axis.** The folder's recurring lesson is that "our account agrees with the network" must name the query family. This paper adds that it must also name *how the validity is quantified over the alternative accounts*: universal functional, one shared derivation, or one shared graphical criterion. Those three are not interchangeable, and the counterexample here is the smallest possible one, so the distinction cannot be waved away as a technicality. For us: an abstraction or simulation claim should say whether the same simulator fits every account in the set, or whether different simulators merely agree on the queried quantity — and those are different results.
- **The step-graph trick as a reusable audit object.** A derivation is a graph, and "does this derivation hold here" becomes a subgraph check. Transposed to our world: when a patch or ablation argument is presented as a chain of justified moves, the moves and the sites they touch form a graph, and a claim like "this chain is valid in the model" becomes checkable structure rather than prose. It is the same idea the folder met as causal-abstraction consistency graphs, but cheaper.
- **Incomparability is a finding, not a failure.** Two natural criteria can each fail where the other succeeds. That is worth copying into our method comparisons: where two accounts of a network are compared, report the case in which each is right, since the only honest verdict may be "incomparable" rather than a ranking.
- **A warning about completeness claims.** "Complete" is meaningless without the rung of the ladder, and the strongest criterion in this paper is strictly stronger than the weakest, so a completeness theorem for the strong criterion can be systematically pessimistic about what is identifiable. Any of our instruments advertised as "complete" should carry the same qualifier.
- **What it does not give us.** No estimators, no experiments, no approximate version, and the whole setting is graph-level with hard interventions, whereas our interventions are soft and our graph is a network's wiring. It is a discipline for claim-writing rather than a tool for measuring, which is why it is filed as useful, not central.

## Tags

unit: circuit | layer | whole-model
form: causal-abstraction | graph
method: attribution | other
eval: prediction/simulation | faithfulness
goal: understanding | evaluation-of-methods
