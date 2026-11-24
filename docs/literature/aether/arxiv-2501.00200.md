# Scalable Neural Network Verification with Branch-and-bound Inferred Cutting Planes

- **Key:** arxiv-2501.00200
- **Authors / organisation:** Duo Zhou, Christopher Brix, Grani A. Hanasusanto, Huan Zhang; University of Illinois Urbana-Champaign and RWTH Aachen University
- **Date:** 2024-12-31
- **Link:** https://arxiv.org/abs/2501.00200
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1, including Appendices A-E)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Cutting planes were the previous state of the art for tightening the relaxation inside a complete verifier, but the planes arrived from a mixed-integer program solved by an external solver, and those solvers stall on large networks - in the authors' own account the initial LP of the largest benchmarks could not be solved inside the timeout, so no cut was ever produced. The goal here is to keep the benefit of cuts while removing the external solver: read the planes off the branch-and-bound search tree itself, where the verifier has already done the hard work of proving some subdomains empty. The larger aim is the same one the whole complete-verification line pursues - fewer domains explored per instance - and the specific bet is that a subdomain proven empty is information about the network that a normal branch-and-bound discards.

## What they studied

Feed-forward ReLU networks from the VNN-COMP benchmark suite (oval22, cifar100-tinyimagenet-2022, cifar100-2024, tinyimagenet-2024) plus the seven adversarially and mixed-trained MNIST/CIFAR models of the SDP-FO benchmark set. The networks in the large benchmark sets run from 14.4 to 31.6 million parameters. The unit of analysis is the *node of the search tree* and the *set of neurons split along the path to that node* - not individual neurons, not layers, but a subtree proven infeasible and the split assignment that proved it.

## How they did it

The verifier branches on ReLU indicator variables as usual; when a subdomain's lower bound comes out non-negative, the path that proved it is a conjunction of "neuron j active" and "neuron k inactive" states that cannot co-occur. That conjunction is written as one linear inequality over the relaxed indicators, and because it is a global fact about the network it is valid in every other subdomain in the tree, including subdomains in which those neurons have not been split yet. The naive version barely helps, because a cut drawn from a deep path is already implied everywhere the path's first split has been made; so the authors strengthen it. A dual variable per fixed indicator is introduced in the bound propagation, and a neuron whose dual is zero is a candidate for deletion from the path; the deleted path is then re-verified, and if it still proves emptiness a strictly shorter cut is emitted. Influence scores for the neurons on the path - the improvement in the lower bound attributable to that neuron's split - rank the deletion order, and a fixed percentage of the lowest scorers is dropped, one round only. Cuts are merged pairwise when they can be. To harvest cuts from shallow paths, the search is run breadth-first rather than depth-first, and before the main search a presolve phase grows five trees with different first splits for five iterations, sharing every cut between them, then keeps the most-expanded tree and prunes the rest. The bound propagation itself had to be extended so that cuts may mention indicator variables that branching has already fixed, which the predecessor's formulation did not accommodate; with no cuts, the extended propagation reduces to the predecessor's.

## What counts as an explanation here

A set of linear inequalities over ReLU indicators, each one carrying the provenance of a proven-infeasible path. It is a certificate fragment, not a model of the network: it says which combinations of activation states are impossible, in the same coordinate system the verifier already uses, and it is meant to be partial - it only ever records impossibilities the search actually stumbled into. It cannot predict anything about a network it has not been run on, and there is no claim that the accumulated cut set is close to complete.

## How they judged success

Number of instances verified under a fixed timeout on the standard benchmarks, against published and rerun baselines; average verification time; verified accuracy on the SDP-FO models against the empirically robust accuracy as an upper bound; and number of domains visited together with number of cuts generated, reported per model in an ablation table. Branches per instance is the closest thing to an explanatory measure - it says how much of the tree's work the cuts removed. Every rerun uses the same branching heuristic and the same tree algorithm as the baseline, so the comparison isolates the cuts. A separate experiment imports the discovered cuts into an external MIP solver and compares against a verifier that builds its own cuts inside a MIP, to show the gain is not an artifact of the GPU propagator.

## What they found

On the VNN-COMP sets the method verifies more instances than both the per-neuron-split baseline and the MIP-cut version on nearly every set - for example 26 of a 27-instance ceiling on oval22 where the MIP-cut method reached 25, 125 versus 119 on cifar100-2024, and 140 versus 135 on tinyimagenet-2024, with the larger average times explained as newly verified instances that previously timed out. On the SDP-FO models it improves verified accuracy on all seven, coming within one point of the empirical upper bound on the MNIST adversarially trained network, and the largest jump is on the harder CIFAR network where it reaches 54.5 percent against 49.0 for the MIP-cut baseline and 47.5 for the split baseline, with an upper bound of 65.0. The ablation attributes most of the gain to the multi-tree presolve: the base version (inferred cuts plus strengthening) is competitive but sometimes behind, the presolve version leads, and the automatic configuration that switches presolve and MIP cuts on by size is best or tied on most models. The domain-count table is the direct evidence: on one CIFAR model the average number of domains visited falls from 63,809 to 9,902, and on another from 20,987 to 7,546, while the cut counts per instance are in the low hundreds of tokens of constraint (91 to 181 cuts typical, several hundred when MIP cuts are also enabled, and thousands on oval21 where the MIP cuts dominate). Domain counts do not always fall - one mixed-trained model's base version visits more domains than the baseline - and the authors say so.

## Limits and open problems they name

The method is tied to piecewise-linear activations. The overhead is real and named: the presolve costs roughly 2-10 seconds, the strengthening pass 3-20 seconds accumulated over a hard instance, and when the timeout is short the cuts have no time to pay for themselves. Effectiveness depends on the instance, with the mixed-trained models the case where it does not help. The appendix states the general form of this: applicability to other architectures needs extension, and gains may be less pronounced on some network structures. The strengthened cut is heuristic - a zero dual is explicitly not a proof that a neuron is unnecessary - and the deletion budget is a fixed percentage with one round only, not a search for a minimal core.

## What we take from it

- **A framing, and the strongest one the head has found for a reading that is not a probe:** a conflict learned by a search is a *statement about the network* that survives the search that produced it. The verifier's own failures to certify a subdomain are evidence, and the head has been treating subdomain bounds purely as numbers to be improved rather than as facts to be collected. Mechanically: **when we run any search procedure over a network - branch-and-bound, a sweep of perturbations, a probe search - we should log the infeasible regions it discovered, not only the final bound, because those regions are valid everywhere and are the cheapest per-unit knowledge the run produces.**
- **A measure we lack:** the number of domains visited is a direct, per-instance account of how much of the network's case structure a certificate had to enumerate, and this is the first member of the head to print it beside the bound (63,809 against 9,902 on the same model, same timeout). Every certificate entry the head has written reports a bound and a time; none reports the size of the search it took. **Report tree size with the bound.**
- **A warning about tuning masquerading as method:** the component ablation shows the base mechanism only competitive and the presolve accounting for most of the headline gain, which is the same shape run 690 found when a granularity merge beat a thousand optimisation iterations. Our own ablations must report the presolve-equivalent arm separately, or the reading's cost is attributed to the wrong thing.
- **A shortcut for per-unit difficulty:** the influence score - the bound improvement caused by splitting a neuron - is a per-neuron ranking that this paper computes on the way to something else and then throws away once the cut is emitted. It is available on any checkpoint from any complete verifier, and nothing in the head has reported it.
- **A caution:** the cut set is not a description of the network, only a record of one search's dead ends. Do not build a white-box model by accumulating cuts - accumulating conflicts is a by-product of verification, not a reconstruction of the function.

## Tags

unit: circuit
form: graph
method: attribution | ablation | other
eval: prediction/simulation | benchmark | completeness | downstream-task
goal: safety/auditing | evaluation-of-methods
