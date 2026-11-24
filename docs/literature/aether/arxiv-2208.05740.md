# General Cutting Planes for Bound-Propagation-Based Neural Network Verification

- **Key:** arxiv-2208.05740
- **Authors / organisation:** Huan Zhang, Shiqi Wang, Kaidi Xu, Linyi Li, Bo Li, Suman Jana, Cho-Jui Hsieh, J. Zico Kolter; CMU, Columbia University, Drexel University, UIUC, UCLA, Bosch Center for AI
- **Date:** 2022-08-11
- **Link:** https://arxiv.org/abs/2208.05740
- **Kind:** paper
- **Read from:** full text (v1 HTML; the document ends mid-way through Appendix C.2, so nothing after the cut-structure counts was available)
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Verifiers that bound ReLU networks by propagating linear bounds backwards are fast on GPUs but weaker per solve than a generic mixed-integer solver, because they cannot ingest the general cutting planes that a MIP solver uses to tighten a relaxation before branching. The authors want to close that gap without giving up bound propagation's speed: they extend the propagation rule so that an arbitrary linear constraint over variables of any layer - including the relaxed ReLU indicator variables that propagation formulations had dropped - can be carried through the backward pass. The larger goal is fewer branches: a tighter relaxation at the root means the exponential search over unstable neurons is entered less often. They then plug a commercial MIP solver in purely as a cutting-plane factory, with its branching disabled, running on CPU while the GPU verifier branches.

## What they studied
Feedforward and residual ReLU networks from standard robustness benchmarks: three CIFAR-10 convolutional models in the VNN-COMP 2020 oval20 suite, the oval21 and cifar10-resnet suites from VNN-COMP 2021, and seven adversarially trained MNIST/CIFAR models from the SDP-FO suite (model structures listed in their Table 4). Unit of analysis is the per-neuron relaxation of unstable ReLUs plus, new here, a layer-agnostic cut row; the object being improved is the scalar global lower bound of the specification.

## How they did it
They write the verification LP with the binary ReLU indicators kept in explicitly (rather than projected out into the triangle relaxation), add N cut rows of the form H·x + G·x̂ + Q·z ≤ d with coefficient matrices per layer, and derive the dual. Each extra dual variable set - the triangle multipliers plus the cut multiplier vector β - enters the backward propagation of the layer coefficient vector ν. Eliminating the triangle multipliers in closed form (their Lemma A.1) gives a propagation rule in which the cut multiplier β shifts every ν_j by βᵀH and βᵀG, and shifts the unstable-neuron term with βᵀQ. Setting β to zero recovers the plain propagation of Wong–Kolter and hence of CROWN, so they prove their bound is never weaker than the prior one for the same cuts. Both α (the split-sigmoid slope) and β are optimised by Adam for 20 iterations per bound evaluation; β is what turns a cut into a numeric contribution. In the complete verifier, CPLEX is launched per target label with branching switched off and only emits cuts at the root node; the BaB loop queries it asynchronously for new cuts and continues without it. Two search-order changes follow from the asynchronous design: they verify the subdomains with the highest current lower bounds first (those need few cuts), and they take multi-class labels easiest-first so the solver has more wall-clock to cut for the hard labels.

## What counts as an explanation here
Not an explanation of the network in the interpretability sense - the deliverable is a lower bound and the certificate that it holds, plus a set of cut rows that are sound for the whole input ball. The interesting object for us is the cut itself: each row is a learned linear statement about a group of neurons spanning arbitrary layers, produced by an external solver, and each cut carries a nonnegative multiplier β_i saying what that row is worth to the bound. It is partial by construction (a finite cut pool, root-node only), and it predicts only the sign question "is the true minimum nonnegative".

## How they judged success
Verified count and average wall-clock time under fixed timeouts, plus average number of branches, on oval20 (3600 s), oval21 (720 s), cifar10-resnet (300 s) and SDP-FO (200 s for their method versus 600 s for two baselines). Baselines are the published VNN-COMP results plus three they add: β-CROWN on the same code base with no cuts, and α-CROWN+MIP (a MIP solver given tight propagation bounds). A cut-level case study on the 39 oval21 properties that survive filtering reports the number of cuts applied (min 318, max 4162, mean 1683), the increase in the root lower bound, and the supports of the generated rows.

## What they found
On oval20 they are the only reported method with zero timeouts on all three models, at 13.40 / 3.02 / 3.70 s average against 5.69 s for β-CROWN on the smallest model - the gain is in the tail, not the mean on easy models, and their branch counts drop from 208,018 to 25,436 / 116,912 to 2,095 / 41 to 111. On oval21 they verify 22 of 27 properties against 11 for the best competition tools. Cut structure is the headline finding for us: of 65,647 cuts generated for the 39 hard properties, 65,301 involve variables from more than one layer, and every single one involves at least one ReLU indicator variable. The root bound on oval21 rises by 0.51 on average (worst 0.04, best 1.54) from an α-CROWN average of −2.54, which alone closes four of the 39. On SDP-FO they achieve higher verified accuracy than every baseline on all seven models despite a third of the time budget.

## Limits and open problems they name
ReLU only; extending to non-piecewise-linear activations is called non-trivial. The cuts come from a generic MIP solver with no use of network structure, so better, NN-specific cut generation is left open. They also state that the optimal α and β do not in general reach the LP-with-cuts optimum, because the triangle multipliers are eliminated in closed form rather than optimised with everything else - so the propagated bound is systematically below the relaxation it is approximating. Practically, the strongest baseline they compare against (gurobi) is not the one supplying cuts, because that solver cannot export them.

## What we take from it
Three things. (1) A method: the cut multiplier β is the general form of the per-neuron split multiplier, so the head now has a hierarchy where β-CROWN's per-neuron sign assumption is the special case H = ±identity with everything else zero. This is the natural place to read a ranked list of multi-unit constraints as a per-group reading of the network, and the paper's own cut-support statistics are an untaken reading. (2) A measure: they report the bound delta per cut regime rather than only the final verified count, and they report branch counts separately from time - two decompositions the head's earlier certificate papers collapse into one number. (3) A warning that fits the head's standing gap: the per-cut multipliers β are optimised in the same 20-step loop as α and are never inspected; the only per-unit quantity in this paper that is examined is the support of the cut rows, and the multipliers that assign them worth are discarded exactly as in every other verifier.

## Tags
unit: neuron
form: direction
method: patching
eval: prediction/simulation
goal: verification
