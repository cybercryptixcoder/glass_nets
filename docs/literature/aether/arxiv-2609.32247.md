# Certifying Interventional Agreement Among Observationally Equivalent Causal Models

- **Key:** arxiv-2609.32247
- **Authors / organisation:** Sourena Khanzadeh, Daniel Platnick, Marjan Alirezaie, Hossein Rahnama — affiliation block not printed on the abstract page; author order as printed (four authors)
- **Date:** 2026-09-26 (v1, single version)
- **Link:** https://arxiv.org/abs/2609.32247
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

The starting point is the mismatch between held-out accuracy and what a model actually computes: two networks can agree on every observed input while resting on different mechanisms, and the disagreements only surface under intervention. The authors name a model paired with the bit of context that explains the observed outcome a *causal perspective*, note that perspectives can be history-equivalent (identical observed state) while predicting different things off the observed regime, and then ask a deliberately narrower question than identification — not "which model is right" but "have the interventions we already ran certified what the survivors will do under the interventions we have not run". The larger aim is a guarantee that no observational learner can give at any sample size, stated as a property of the experiment rather than of the estimator. They then take this to white-box audit: can a bounded number of image edits certify which feature a classifier actually reads, even when the images only ever show the shortcut correlation.

## What they studied

Two model families. First, a theory layer over Boolean SCMs with known parent sets and independent exogenous bits, where a hypothesis is a full table of local mechanisms (a "mechanism cell" is a variable plus a parent configuration); the version space is the set of tables consistent with a log, which can be finite (Boolean, noiseless) or an infinite union of polytopes (continuous). Second, concept-bottleneck networks on a three-digit Coloured-MNIST pointer-retrieval task: a pointer digit routes to a left or a right digit, each digit is read by its own small CNN ending in one percept unit, and the head is a multiplexer (Boolean version) or a top-1 router with a tanh percept and a sigmoid output (continuous version). The unit of analysis is the *candidate abstraction*, not the neuron: each candidate assigns a mechanism (reads shape / reads ink / reads both / reads either; in the continuous case a box in percept space) to each percept. Networks come in three deliberately confused kinds — trained (with the ink–size correlation at strength α), planted (a known random mix of shape and ink), and bypass (the router reads a second encoder, so no candidate matches).

## How they did it

- Formalise the audit as a learning problem against a version space: an admissible query family (which variables may be set, how many, which target is read) plus a cost that prices departure from the observed state, so a "radius" R is the most costly query the certificate is allowed to use.
- Define interventional risk as the probability that two hypotheses disagree on a query drawn uniformly from the family below the radius, prove it is a pseudo-metric, and show it is zero exactly when the two hypotheses' minimum separating cost exceeds the radius — i.e. the model's *margin* under intervention is a separation cost, not a probability.
- Prove the observational floor: for any learner, empty-observation completions first, then a worst-case bound over the version space that depends on the data only through which mechanism cells were *ever* realised, so it does not shrink with N; cells the regime reaches with probability zero stay free forever, while merely rare cells are fixed by more data.
- Build the auditor (ISS — interventional separation selection). Maintain the surviving candidates; while the cheapest separating query over all surviving pairs costs at most the radius, run it, keep the candidates that predicted the observed answer, repeat. The stopping test depends only on the survivors, so it is checkable without knowing the truth, and it is the stopping rule — not the query-selection heuristic — that carries the guarantee.
- Make it exact over an exponentially large version space: enumerate the admissible family in cost order, and for each query decide "does this separate two survivors" by propagating the query through the candidate tables symbolically, writing each candidate answer as a disjunction of branches and testing feasibility of each pair of branches with a binary integer program. Continuous variables get the same treatment with mixed-integer linear programs over the box of admissible percept values, at a stated tolerance.
- Add three extensions: a soft version space for noisy mechanism tables (keep every candidate within a likelihood margin calibrated by a binomial quantile); exogenous grounding, where a verified fact about one component's context is evidence that eliminates candidates at zero intervention cost; and a separate validation phase of random admissible queries, described as the only thing that can falsify a misspecified hypothesis class.
- Experiment 1 (Boolean): 50 networks × 50 observed images, 4 radii, 4 query rules (minimum cost, information per cost, random separating, random probing), 60,000 runs; every certificate is checked against *every* admissible query within its radius, not only those executed, and the costs are compared against a learner told the truth (an exact weighted set cover).
- Experiment 2 (continuous): 18 networks × 10 observed images, hue on an 11-point grid, tolerance 0.05, visual interventions (recolour, swap) and a white-box family that also sets the pointer percept; certificates verified exhaustively; a 4,096-candidate four-way-pointer variant with up to 12,000 runs as a larger version space.

## What counts as an explanation here

A *certificate*: a set of executed interventions plus a radius, with the claim that every surviving candidate agrees with the audited network on every admissible intervention up to that radius. It is explicitly partial — it says nothing about queries beyond the radius, nothing about candidates that differ only on cells no admissible query realises, and nothing at all if the true mechanism was never in the class. The account itself is the version space plus a surviving mechanism table per percept; the certificate is a statement about the *edges* of that account (where it is guaranteed to predict) rather than a description of the mechanism. It predicts exactly one thing, and predicts it absolutely: risk zero within the radius, not low risk.

## How they judged success

Exactness: every issued certificate is checked against all admissible queries within the radius (not just the executed ones), yielding a counted number of contradicted certificates rather than a score. Cost: number of interventions and total cost, compared with four baselines — a learner told the truth (optimum via exact weighted set cover), the greedy non-adaptive cover, random separating queries, and random probing with the same stopping test — plus the information-theoretic query bound of the number of radius-cells. Theory checks: every clause of the soundness and cost theorems is evaluated on every run whose truth is in the class. Diagnostics: the misprediction rate of the *intended* abstraction, held-out error of plain CNNs as a function of training-set size, and a refutation rate for the separate validation phase.

## What they found

- Observation is exhausted: plain CNNs trained on up to 60,000 shortcut images reach 0.0% held-out error yet disagree with the shape-based label on 50% of single-variable edits and 52% of cost-2 edits in the Boolean task, and on 26% of single-digit edits in the continuous task. The floor is attained, not merely bounded: the cells of the floor are exactly the ones the α<1 regimes eventually realise, and there the interventional risk falls with N (0.52 → 0.04, and to 3% error in the continuous task).
- Certificates are correct and cheap. In the Boolean study, 53,496 of 60,000 runs had the truth in the class; across all rules, radii, families and networks there were zero eliminated truths, zero halts without a certificate, zero contradicted certificates, and zero cost violations. At radius 2 with visual interventions ISS never needs more than 6 interventions (the log-count bound), changing 8.6 variables on average, against 7.0 for a learner told the truth and 25.9 for random probing.
- In the continuous study, at radius 0.5 with visual interventions, ISS averages 13.6 interventions at total cost 4.0 against 5.3 for random separating queries and 33.9 for random probing; with white-box interventions random probing costs 52.6. All certificates held on all in-class images.
- Misspecification is silent and only validation catches it. Bypass networks: with white-box queries, all certificates are void at radius 2, twenty random validation queries refute 93% at radius 2 (96% at radius 3) in the Boolean study and 69% at radius 0.5 in the continuous one, and random probing refutes the class during learning in 97% of runs.
- Certification follows the network, not the training target: planted percepts are recovered at 98% of observed images, and the certificates track the network where it departs from its planted table.

## Limits and open problems they name

Exact separation needs finitely many admissible queries (the hue grid), answers that are linear in the parameters up to sign decisions, and given parent sets and noise form; worst-case certification cost is exponential; deciding whether a cheap separating query exists is NP-complete even for in-degree-2 Boolean models with unit costs, though it becomes polynomial when interventions have bounded size; the noisy certificate is probabilistic (1−δ) and is void if the truth is outside the class, which is what the validation phase exists for — and validation needs a number of queries that grows with the logarithm of the confidence divided by the risk it must catch; and the audited networks expose their abstraction through one unit per percept, so distributed representations would need a learned alignment first.

## What we take from it

- **A prediction-under-intervention guarantee, and the right form for one.** The folder's repeated demand — say where the account holds and where it stops — here has a concrete shape: publish (radius, survivor set, "no separating query below this radius") instead of a score. For our probe/patch batteries this is a template we can literally adopt: report the *radius* at which the account stops being distinguishable from alternative accounts, not only the agreement rate.
- **The margin is a cost, not an average.** Separation cost being the reciprocal of interventional risk means an intervention budget is the natural x-axis for every claim we make; a battery run at one cost level should say so and the level is part of the claim.
- **The floor is a floor on the problem.** The observationally equivalent completions that no amount of data can fix are the "permanent cells" — configurations the *data-generating process reaches with probability zero*. This is the sharpest statement in the folder of why held-out evaluation cannot detect a shortcut learner, and it distinguishes permanent ambiguity (unfixable by data) from finite-sample ambiguity (fixed by more data, and it says how).
- **The stopping rule carries the guarantee, the query rule does not.** Useful division of labour: any separating query works, the heuristic only buys cost. That is a clean way to separate "what we can claim" from "how efficient our search is".
- **The bypass result is the warning.** A disagreement-driven auditor never asks a query where all candidates agree, so a network that bypasses a unit every candidate trusts yields certificates that are confidently wrong. If our accounts are audited by an intervention search that only targets disagreement, they inherit the same blind spot, and only a random, non-disagreement-driven validation pass catches it.
- **Grounding is free supervision.** A verified fact about a component's context eliminates candidates at zero intervention cost and can help arbitrarily more than intervention — worth carrying as a rule for our own batteries: any verified internal fact should be folded in before spending interventions.

## Tags

unit: whole-model, circuit, feature
form: causal-abstraction, probe, description
method: patching, ablation, probing, agent
eval: patching, prediction/simulation, completeness, benchmark
goal: understanding, safety/auditing, evaluation-of-methods
