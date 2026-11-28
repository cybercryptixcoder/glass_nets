# From Centralized to Decentralized Federated Learning: Theoretical Insights, Privacy Preservation, and Robustness Challenges

- **Key:** arxiv-2503.07505
- **Authors / organisation:** Qiongxiu Li, Wenrui Yu, Yufei Xia, Jun Pang; Aalborg University (Denmark), IP Paris (France), University of Luxembourg
- **Date:** 2025-03-10 (v1 only)
- **Link:** https://arxiv.org/abs/2503.07505
- **Kind:** survey/agenda
- **Read from:** full text (abstract, sections 1-8, Tables 1-3, full 153-entry reference list; no appendices exist)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Federated learning comes in a star shape with a server in the middle and a peer-to-peer shape without one, and the literature usually sorts methods by that topology. The authors argue topology is the wrong sort key: the deeper split is whether a design keeps local training and aggregation as two separate steps or folds both into one constrained optimisation solved across the network. Their aim is a taxonomy built on that second axis, and then a comparison of the two families on utility, privacy and robustness to attack. The paper is a survey, so the goal is orientation rather than a new method, with a closing agenda that names fully distributed optimisation under peer-to-peer communication as the underused branch.

## What they studied

No models are trained and no experiments are run. The unit of analysis is the **learning protocol**: the sequence of steps by which nodes arrive at a shared model, taken as an object in its own right. Concretely: FedAvg and FedSGD as the canonical separated-aggregation schemes, ADMM and PDMM (with FedSplit, SCAFFOLD likely on the star graph, and GPDMM as representatives) as joint-optimisation schemes, and gossip or doubly stochastic averaging as the peer-to-peer version of separate aggregation. Around those protocols they organise 153 references into three families — empirical privacy attacks, provable and empirical defences, and Byzantine or poisoning attacks with their defences — each split by whether the adversary is passive (honest-but-curious, reads what passes it) or active (deviates from the protocol arbitrarily).

## How they did it

There is no procedure to reproduce, but the survey does state its own argument structure precisely enough to follow. Each protocol family is written out in update form: the weighted average over client weights whose weights are dataset sizes; the gradient-sharing variant; the augmented local objective with a proximal term to an auxiliary variable, its inexact K-step and quadratic-approximation forms; and the gossip update as multiplication by a nonnegative, symmetric, doubly stochastic matrix. A procrustes-style observation is then made once and reused: the star graph is a special case of the peer-to-peer graph with n-1 edges, and the star-graph joint-optimisation formulation is stated as a special case of the general network-constrained problem, so the convergence argument for the general case is claimed to carry over. Privacy mechanisms are tabulated twice — once by whether the guarantee is provable or only empirical, and once against the three attack families (membership inference, property inference, input reconstruction) — and defences are tabulated by which attack class they address and by whether they apply to separated aggregation or to joint optimisation.

## What counts as an explanation here

A **taxonomy plus a claim about which axis explains the trade-offs**. The explanation of a method's behaviour is not derived from the method itself but assigned by its cell: protocol family crossed with adversary type. The most specific explanatory move is the claim that joint optimisation's constraints are themselves a privacy mechanism — the auxiliary-variable and consensus structure limits what a single neighbour can observe — and that this, not the absence of a server, is what the privacy argument rests on. It is a partial account by construction: nothing about weights, features or representations, and the performance claims it adjudicates are quoted from the surveyed papers rather than re-derived.

## How they judged success

Only indirectly: the survey's own success criterion is coverage and consistency of the assignment, and every utility, privacy and robustness number it cites is another paper's. Two adversarial anchors are named as the evaluation frame — a passive adversary who follows the protocol and an active one who does not — plus a distinction between computational and information-theoretic security that decides what a privacy claim can mean. The one place the survey grades evidence is its handling of a live dispute about which topology is more private: it lines up the experimental result (peer-to-peer less private, from loss-based membership inference), the information-theoretic result (peer-to-peer more private under joint optimisation), and a mutual-information analysis that reconciles the two, and concludes the answer depends on the threat model rather than adjudicating it.

## What they found

The survey's own findings are structural, with no numbers of its own. Most methods labelled decentralised are separate-aggregation designs that mimic FedAvg's averaging step, and the joint-optimisation family is far more developed in the distributed-optimisation literature than in federated learning as practised, which they call the main gap. Protocol choice is claimed to drive utility, privacy and robustness together, so a topology comparison that holds the protocol fixed compares less than it appears to. On privacy they assemble a three-way disagreement and refuse to resolve it: the same format lands on opposite verdicts depending on whether the metric is an empirical attack, an information-theoretic bound, or a mutual-information analysis. On robustness they separate explicit attacks that degrade the shared model — statistical mean-shifting, sign-flipping, update-mimicry, label-flipping, data injection, and server-side inner-product manipulation — from backdoors that leave clean inputs accurate and fire on a trigger, and note that the server-side attack has no natural peer-to-peer analogue, so comparisons must say which side can be compromised. On defence they observe that the robust-aggregation rules (trimmed mean, coordinate median, Krum, Bulyan) all rely on a global view of many updates and are therefore native to the star, whereas the peer-to-peer equivalents — local neighbour screening by hypothesis test, dual-stage filter-then-validate, self-centred clipping, an exponentially decaying distance threshold — are all local and trade a global guarantee for adaptability.

## Limits and open problems they name

The survey's central claim is argued by taxonomy, not tested — no experiment here shows that protocol rather than topology causes a difference. The convergence results for joint optimisation are single-family: one seed family, one arithmetic-analogue seed, no cross-family sweep, so the trustworthiness of the likely seed is unknown and the joint-optimisation branch is recommended on untested grounds. The privacy chapter's mechanisms are listed without quantitative comparison — no leakage numbers are put beside each other, and the "gap between provable guarantees and empirical attacks" is named as a hard problem and then left as one. They also name their own gap: adaptive defences against evolving attacks and hybrid architectures are unreviewed, and they explicitly invite provably robust designs rather than audited empirical ones.

## What we take from it

A **framing and a warning**. The framing: when we compare two ways of building or editing a model, the sort key we impose decides the conclusion, and this survey's demonstration is that a sort key everyone used (topology) was less informative than one nobody had written down (protocol). For our black-box-to-white-box work the transferable version is that "where the computation sits" and "what the computation is" are different variables and should be tabulated separately. The warning is sharper and directly usable: the same three-way privacy disagreement — an empirical attack, a provable bound and a mutual-information analysis giving different verdicts on the same design — is the folder's instrument problem in a new literature, and this paper's resolution (the answer depends on the threat model, so the threat model must be stated before the number) is a corollary of the rule our recent entries keep deriving from single-model studies. Two smaller transfers: the local-versus-global split is the same trade our own probes face when a global statistic is computed over a sample versus per-unit, and the tabulation style — every method placed in a cell against every attack class it was and was not tested on — is a cheap way to make our own evaluation tables honest about what was never probed.

## Tags

unit: whole-model
form: description
method: other
eval: benchmark
goal: evaluation-of-methods
