# Abstraction-Based Proof Production in Formal Verification of Neural Networks

- **Key:** arxiv-2506.09455
- **Authors / organisation:** Yizhak Yisrael Elboher, Omri Isac, Guy Katz (Hebrew University of Jerusalem); Tobias Ladner (Technical University of Munich); Haoze Wu (Amherst College)
- **Date:** 2025-06-11 (v1)
- **Link:** https://arxiv.org/abs/2506.09455
- **Kind:** paper (self-described work in progress)
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

DNN verifiers are supposed to be sound, but their implementations still carry bugs, so the field has started asking verifiers to emit formal proofs that a separate checker can validate. Proof production costs two things: the proofs grow large enough that memory becomes the binding constraint, and several verifier optimisations have to be switched off whenever proofs are on. At the same time, the technique that makes verification scale at all — abstraction, replacing the network with a smaller sound over-approximation — is not supported by any existing proof-producing verifier. The authors want to close that gap, and they also expect abstraction to help on the cost side: an abstract network is smaller, so the proof about it should be smaller too. This is explicitly a first-principles paper rather than an evaluated system: they set out to lay a foundation, with implementation and measurement deferred.

## What they studied

Verification queries stated as a triple of network, input property (a hyperrectangle) and output property (the region deemed unsafe); both the SAT and UNSAT reading are formalised, and UNSAT is the case that needs a proof. The networks and property are not the unit of study so much as the *relationship between two networks*: an original feedforward DNN with affine layers plus nonlinearity, and an abstract network derived from it. The abstraction is the one implemented in CORA, which over-approximates each layer by a hyperrectangle and then merges hidden neurons whose interval bounds are similar, absorbing the resulting error by turning the next layer's scalar biases into intervals. The proof machinery is Marabou's: an SMT/LP solver that resolves ReLUs by case-splitting and relaxation, and represents an UNSAT result as a proof tree. A small four-input, five-hidden-unit network with all-zero biases is carried through every example.

## How they did it

They decompose the UNSAT proof into two independent certificates and state the combination as one inference rule: a proof that the abstract query is unsatisfiable, plus a proof that the abstraction over-approximates the original, which together yield unsatisfiability of the original. The second certificate is the part nobody had produced, and most of the paper is spent on how to obtain it for neuron merging.

For abstract networks the output is a set of vectors rather than a point, and the biases are intervals, so the query has to be re-encoded before Marabou can take it. Two encodings are offered: write each interval bias as a pair of linear inequalities against the same weighted sum, or add a fresh input variable per interval bias, wire it into the neuron through a skip connection of unit weight, and constrain it by the interval inside the input property. The first adds inequalities, the second adds variables.

The abstraction certificate itself is built as a chain of proof rules. A base rule certifies that the trivial abstract network — the original with all error intervals set to zero — contains the original network pointwise. A per-layer rule certifies one merge step: it fixes the surviving rows and columns of the weight matrices, drops the merged indices, sets the error interval of the surviving outputs to the removed block of the previous layer's interval times the corresponding weight submatrix, and concludes that the new abstract network contains the old one. A closing rule chains these over all layers and discharges the transfer of unsatisfiability from abstract to original. They prove the per-layer containment claim by contradiction through the IBP-computed error interval, and note the chain is monotone in a useful way: merging on top of an already-abstract network yields a further over-approximation, so the step can be applied repeatedly.

The checking side is deliberately asymmetric with the producing side. A checker receives the original query and the candidate abstract network, and reconstructs every intermediate abstract network for itself from those two objects — it never has to *design* an abstraction, only recompute the intervals and check the containment claims. The authors say explicitly that a checker may use arbitrarily precise arithmetic, and should, since its job is reliability rather than speed. The selection loop is CEGAR-shaped: abstract, verify the abstract query, and only when the result is UNSAT attempt the two proofs; if either proof fails to generate, refine the abstraction and repeat. Proving only after UNSAT is what keeps the loop cheap, and it also means the first verification pass can use any solver at all, proof-capable or not.

## What counts as an explanation here

A legal proof object in a proof system: a pair of derivations that a small independent checker can accept or reject. The form is compositional rather than holistic — each rule licenses one transformation of the network (zero-error extension, one neuron merge) or one inference (the transfer of unsatisfiability), and the full certificate is the concatenation. It is meant to be complete for the UNSAT claims it covers, in the sense that acceptance implies the claim, but it says nothing about SAT, where a witness input already suffices. Prediction is indirect: the certificate does not predict what the network will do anywhere, it certifies that a claim about the region already holds everywhere in it.

## How they judged success

No empirical evaluation is reported. The stated criteria are the ones the future evaluation will use — proof size, verification time, proof-checking time — over what they call real-world benchmarks. The paper's own deliverable is the rule set plus a worked example: the small network's merge of three hidden units produces a specific error interval on the output bias and specific output bounds, and the certificate for that merge is written out rule by rule. The soundness argument is the proposition plus corollary plus a per-layer containment proof, not a measurement.

## What they found

The central structural result is that the abstraction step can be certified at all, and that its certificate needs nothing from the verification certificate: the two are independent artefacts joined by one inference. The merge certificate is small relative to what it certifies, because it is indexed by layer and by merge bucket rather than by neuron or by search path — each layer contributes one step, and the step's content is a submatrix and a set of interval products that the checker recomputes.

The paper is candid about the expected payoff and its bounds. Smaller abstract networks should yield smaller proofs, but the authors frame this as expected rather than measured, since the abstract proof also carries the merge chain, which the standard proof does not. They also name the tooling cost of the existing situation: proof generation disables optimisations and inflates memory, so proof production and scalability have been trading against each other.

The re-encoding of the abstract query is a smaller but concrete contribution: an abstract network is not literally a DNN, and the paper shows two ways to reduce it to a query Marabou already accepts, one trading inequalities for variables and the other the reverse.

## Limits and open problems they name

The paper does not implement or evaluate anything — its own words are that implementation and evaluation are next steps, so every claim about size and speed is a projection. It certifies *neuron merging* only; the DSL-based line of work certifies verifiers that use linear relaxations of activations, which is a different proof target, and the authors see formalising their scheme in that DSL as future work. Refinement is inherited from CORA and only described, not analysed, and while they mention residual reasoning as a promising enhancement to it, they do not incorporate it. There is no treatment of SAT-side certification beyond the witness, no accounting of the checker's own cost, and no discussion of what a partial certificate means — the algorithm either yields both components or refines and retries.

## What we take from it

**A decomposition worth copying: separate the proof that the model is right from the proof that the model is the right model.** Here the two certificates are "the abstract query is unsatisfiable" and "the abstract network contains the original", and each can be checked by a different, simpler procedure with a different failure mode. Our white-box account will face exactly this split — an account of the network, and an argument that the account is faithful to the network — and the paper's rule is that the second argument must be its own artefact, not a remark inside the first.

**The verifier and the checker are allowed to be different programs with different priorities.** The checker reconstructs intermediate objects itself from the query and the final candidate, and is permitted to use exact arithmetic because it is not the thing being scaled. That is a template for us: whatever the probing loop computes under time pressure, the verification of its claims can be redone slowly and exactly, and the claim format should be chosen so that this redoing is possible from the inputs plus the output alone, with no access to the loop's internal state.

**Certify the transformation, not the transformation's output.** The per-layer rule is a licence to replace one network with another under stated conditions; the checker only has to confirm the conditions hold for the specific submatrices and intervals presented. This is a cheaper shape than shipping the whole derivation — one small local claim per step, chained — and it is the same shape we will want for patch-style or ablation-style edits, where the interesting object is the legitimacy of the edit rather than a transcript of it.

## Tags

unit: layer
form: program
method: patching
eval: faithfulness
goal: safety/auditing
