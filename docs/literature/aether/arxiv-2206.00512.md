# Neural Network Verification with Proof Production

- **Key:** arxiv-2206.00512
- **Authors / organisation:** Omri Isac, Clark Barrett, Min Zhang, Guy Katz; The Hebrew University of Jerusalem, Stanford University, East China Normal University
- **Date:** 2022-08 (FMCAD 2022)
- **Link:** https://arxiv.org/abs/2206.00512
- **Kind:** paper
- **Read from:** full text (15-page PDF, camera-ready preprint of the FMCAD 2022 paper)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

A verifier that says "no bad input exists" is asserting a universal claim, and nothing in the output lets anyone check it; a verifier that says "here is a bad input" produces something anyone can confirm by a forward pass. The authors note that DNN verifiers have themselves been buggy and that their searches run large floating-point manipulations, so an unverifiable negative answer is exactly the answer most exposed to numerical error and to tool bugs. Their goal is to close that asymmetry: make the *unsatisfiable* verdict carry an artefact that an independent, simple checker can validate. The larger aim is to make verification results auditable and to lay groundwork for learning from conflicts inside the search.

## What they studied

ReLU feed-forward networks (piecewise-linear activations generally; sigmoid/tanh deferred). The unit is the whole verification query and its search tree, not individual neurons or features. The concrete instance is the ACAS Xu aircraft-collision-avoidance family: 45 networks, 4 properties each, 180 queries, of which 113 came back unsatisfiable and are the object of study. Their mechanism targets search-based verifiers built on a Simplex LP core with case-splitting on activations, exemplified by the Marabou tool.

## How they did it

A verification query is written as an LP -- one equation per affine layer, plus interval bounds on inputs, internal variables and outputs -- together with the activation equations held outside the LP. Simplex searches for an assignment consistent with tableau and bounds; when an assignment violates an activation, the solver splits into one branch per linear phase, which adds equations and tightens bounds at the child node. The authors instrument this to emit a proof tree that is isomorphic to the search tree: internal nodes record splits, leaves record a local unsatisfiability proof relative to the constraints accumulated along the path. Leaf proofs are a Farkas vector: a coefficient vector multiplied into the tableau rows produces an equation whose maximum attainable left-hand side, given the current ground bounds, is strictly negative, so no assignment can satisfy it. Since UNSAT always means some variable acquired contradictory bounds, the leaf proof is concretely the difference between a vector justifying the variable's derived upper bound and one justifying its derived lower bound. Interval bound tightening is handled by keeping exactly two vectors per variable (one per bound), updated at each tightening, so the certificate refers back to the ground bounds only and does not have to store the thousands of intermediate tightenings. Tightenings that cannot be expressed as an equation are recorded as lemmas: the rule identifier plus the vectors for the bounds that fed it, checked against a rule list agreed in advance. On a split, the child resets its ground bounds, zeroes the corresponding vectors, and inherits the parent's dynamic bounds and their vectors, because ground bounds only ever get tighter.

## What counts as an explanation here

An object of a very specific kind: a machine-checkable certificate of impossibility plus a tree of derivations, not a description of what the network computes. It is complete with respect to the tool's own execution -- every split and every tightening that mattered appears -- and it is *not* an explanation of the network in the human sense; it says nothing about which features or directions matter. It does, however, support intervention claims in a strong form: the tree is a witness that a stated region contains no counter-example, and the checker can independently reproduce that claim from the network and the property alone.

## How they judged success

Three axes over the 113 unsatisfiable ACAS Xu queries. Correctness: the checker was run on every leaf -- about 1.46 million leaves -- and the rate of leaves that failed to check was recorded; the 77 failures were then handed to an external SMT solver (cvc5) to determine whether the failure was the certificate's or the verdict's. Overhead: proof-production time as a fraction of solving time. Checking time: certificate-check time against original solving time. No comparison against other proof-producing DNN verifiers is possible, because they state that none exist.

## What they found

Certificate production added an average of 5.7% to solving time. Checking was, on average, 66.5% shorter than solving the corresponding query. Of the 113 queries, 97 had every leaf check successfully; across all leaves only 77 of the ~1.46 million failed, and in most affected queries the failures were a small handful of leaves. The one query with many failed leaves is attributed by the authors to intrinsically awkward numerics such as very small coefficients. All 77 failing leaves were shown to be genuinely unsatisfiable by cvc5, in under 20 seconds each -- so the tool's verdicts were right while its certificates for those leaves were not, and the authors propose this as a division of labour: let the fast verifier do the work and let a sound-arithmetic solver adjudicate only the leaves that matter.

## Limits and open problems they name

Only piecewise-linear activations are covered; sigmoid and tanh are left for future work. Not all of Marabou's bound-propagation machinery was instrumented -- the evaluation ran with unsupported components switched off, and preprocessing runs before proof production begins, so it is outside the certificate. Polynomial-size certificates for negative answers would amount to settling whether NP equals coNP, which they flag as a major open problem and possibly impossible; the certificates are therefore not guaranteed small. Quantifying over all leaves for large trees is the practical cost they do not remove.

## What we take from it

A **method** and a **measure**. The method: pair every negative or exclusionary claim in our account with a locally checkable witness, and make the witness cheap, which here means keeping two vectors per variable rather than a log of every derivation. That is the same discipline runs 619-620 pushed from the other side -- an enclosure nobody can audit is an assertion, not a result. The measure: report certificate-failure rate separately from verdict-error rate, because the paper's central empirical fact is that they are different failures -- 77 bad certificates, zero bad verdicts, an inversion that a single "accuracy" number would have hidden. Also a warning worth carrying: the certificate is a statement about the *execution*, and it is only as complete as the instrumentation; anything the tool does before or outside the instrumented core (preprocessing, uninstrumented propagation) is unproven and must be declared. For us: when our white-box model predicts what a network will do under an intervention, the prediction should be accompanied by the specific derivation that produced it, in a form a small independent checker can re-run, and we should report the derivation-check failure rate beside it.

## Tags

unit: whole-model
form: program
method: other
eval: faithfulness
goal: safety/auditing
