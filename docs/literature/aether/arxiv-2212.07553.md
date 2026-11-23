# Automated Reachability Analysis of Neural Network-Controlled Systems via Adaptive Polytopes

- **Key:** arxiv-2212.07553
- **Authors / organisation:** Taha Entesari, Mahyar Fazlyab; Johns Hopkins University (Dept. of Electrical and Computer Engineering)
- **Date:** 2023-05-15 (v3; v1 2022-12-14)
- **Link:** https://arxiv.org/abs/2212.07553
- **Kind:** paper
- **Read from:** full text (arXiv HTML v3, ~50k chars incl. reference list; abstract, §1-§2, §3 with Algorithm 1 and both proofs, §4 experiments, Table 1 and Fig. 3-4 values from their captions; no discussion/limitations section exists — the limits are in the conclusion)
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Reachability analysis of a discrete-time plant driven by a neural-network feedback controller is the closed-loop version of verification: one must carry a whole set of states forward through the composition of plant and controller, step after step. The dominant failure is not local precision but the wrapping effect—each step's over-approximation is fed back in, and the shape of the enclosing set stops matching the shape of the true set, so the error compounds with horizon. Their claim is that the *choice of template directions* is the lever that matters, and that this choice is usually made by hand and badly. The goal is to remove that hand choice: let the geometry of the network's own linear maps dictate the facet directions of the enumerating polytope at every time step.

## What they studied
Two benchmark plants of dimension 2 and 6 (double integrator; 6-state quadrotor), each with an MPC controller distilled into a small ReLU network (2-10-5-1 and three linear layers of 32/32/3). The unit of analysis is not a neuron or a feature but a **convex set of states and its facet normals**—the object being accounted for is the reachable set at step k, carried as a template polyhedron Poly(C^k, d^k).

## How they did it
Three moves. (a) The closed-loop map x⁺ = Ax + Bf(x) + e is rewritten as one plain sequential ReLU network with no skip connection, using the identity that any vector is the difference of its positive and negative parts: the state is routed through two rectifier branches and the plant matrices are appended as the final linear layer. This is not cosmetic—it removes the need to Minkowski-sum the image of the plant term with the image of the controller term, an operation the authors say both enlarges the set and is hard to do well. (b) Given facet directions, each offset is the supremum of the corresponding linear functional over the input set, obtained by branch and bound on the *unrolled* composition over the initial box: partition the initial box, bound the functional on each sub-box with DeepPoly (any propagator would do—the framework is modular), prune, and return an ε-suboptimal certificate. Crucially the offsets are computed against the unrolled map from the initial set, not by chaining step-to-step set images, which is the algorithmic form of "do not let the approximation feed back". (c) For the directions: each affine layer's template is pushed through the pseudo-inverse, C ← C W†, so a facet normal on the output pulls back to a normal on the input; for a layer that widens (more outputs than inputs), the directions in the orthogonal complement of the row space are added in both signs, because the output provably has zero variance along them; after every layer the inward axis normals −e_i are added because the next rectifier clips at zero. Rows whose cosine similarity exceeds λ = 0.98 are then dropped to stop the facet count growing. The paper proves the construction always yields a bounded polytope, and shows the wide-layer case is *tight*: every point satisfying the constructed inequalities corresponds to a feasible input point.

## What counts as an explanation here
A set-valued enclosure with an explicit geometry, not a textual account of a mechanism. The claim form is "the set of states at step k is contained in this polyhedron", i.e. a certificate shaped like a containment witness, plus a per-direction optimality certificate (ε-suboptimal sup). It is partial by design—it says nothing about which part of the network produced the growth in the set, only how big the set can be.

## How they judged success
Volume/shape of the reachable set visually against the exact or competing enclosures over the horizon (Figs. 3-4), plus Table 1: total run time and run time per optimization problem, against two baselines—ReachLipBnB (branch and bound with Lipschitz bounds) and ReachLP in both greedy-simulation-guidance and uniform partitioning variants.

## What they found
On the double integrator their enclosures stay visibly tighter than both baselines and the gap widens with horizon. On the quadrotor (12 steps) they remain tight throughout, and the numbers are stark: ReachLipBnB takes 2170 s for the run (11.3 s per problem), ReachLP with guidance 18.8 s (0.13 s per problem), and their method 213.67 s (0.501 s per problem), while ReachLP with uniform partitioning did not finish inside 27 hours. On the double integrator their method is the slowest in wall-clock (9.96 s vs 1.078 s for ReachLP-GSG) at 0.134 s per problem. They note the two methods solve different numbers of optimization problems per case (a fixed handful per time step for the baselines; 74 and 426 total for theirs), so per-problem cost is the fair column and total time is not.

## Limits and open problems they name
The conclusion states the obvious one against their own design: bounding is cheap and scales in network depth, but the *partitioning is over the input set*, so cost grows exponentially with state dimension. They name three directions rather than fixes—less conservative propagators, branching in activation space instead of state space, and operator-splitting schemes for long horizons. Also implicit and worth flagging: the adaptive directions themselves are heuristic (SVD-based), the similarity threshold λ is a free knob that trades facet count against shape fidelity, and there is no measurement anywhere of how much of the tightness comes from the directions versus from the unrolling.

## What we take from it
Three things. (1) **A method**: choose the representation's geometry from the linear maps of the thing being represented rather than from a fixed vocabulary or from the data—pulling facet normals back through the pseudo-inverse is one line and removes the hand-picked template direction, which is the closest thing in this subfield to run 617's "constraint language is the ceiling" problem. (2) **A framing**: the wrapping effect is the closed-loop name for the error that appears when a summary is fed back into its own input, and the structural fix here is to not chain summaries at all—compute each step's enclosure against the unrolled map from the original input box. That is directly transferable: our account of an iterative computation should be stated relative to the inputs, not relative to its own previous output. (3) **A warning**: their own table is a lesson in how totals mislead—their method looks slowest on one case and fastest by an order of magnitude on the other once the per-problem column is read, and in the second case the speed is a function of how many sub-problems they chose to solve rather than of anything intrinsic. Any cost number we report must be per unit of work with the number of units stated.

## Tags
unit: layer | direction | whole-model
form: graph | direction
method: decomposition | ablation | other
eval: patching | prediction/simulation | benchmark
goal: understanding | evaluation-of-methods
