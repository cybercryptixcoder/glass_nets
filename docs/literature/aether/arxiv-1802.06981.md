# Reachable Set Estimation and Safety Verification for Piecewise Linear Systems with Neural Network Controllers

- **Key:** arxiv-1802.06981
- **Authors / organisation:** Weiming Xiang, Hoang-Dung Tran, Joel A. Rosenfeld, Taylor T. Johnson; Department of Electrical Engineering and Computer Science, Vanderbilt University (NSF and AFOSR funded)
- **Date:** 2018-02-20
- **Link:** https://arxiv.org/abs/1802.06981
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Discrete-time piecewise-linear plants whose feedback law is a rectified MLP are a standard stand-in for hybrid control systems — the switch between linear modes makes the plant alone hard, and an uninterpretable controller on top makes it harder. The authors want to answer two questions for that loop: what states are reachable after k steps from a set of starting states, and is the loop safe over a finite interval. The larger goal is stated in the introduction in the field's usual framing: a network is treated as a black box that produces a good output for a given input, and what is missing is a way to say what outputs it can produce at all, which is the prerequisite for any safety claim. Verification of even simple network properties is noted as already NP-complete, so no completeness is promised — the paper is explicitly about finite horizons and counted sets.

## What they studied
A discrete-time plant with N linear modes selected by a switching signal, x(k+1) = A_σ x(k) + B_σ u(k), closed around a static feedback controller u = g(x) that is a multilayer perceptron with rectified activations. The unit of analysis is a **union of polytopes**: the state lets them represent exactly the piecewise-affine structure, and the network is analysed layer by layer as a map between unions of polytopes. The worked example is two modes in the plane, a 2-layer network with 4 hidden and 2 output units, and a square initial set. No larger network and no trained controller appears anywhere in the paper.

## How they did it
The layer map is derived by case splitting on the sign pattern of the preactivations. For one layer with n units there are 2^n sign patterns; the authors pull out the two extremes — all-off, all-on — and handle the remaining 2^n − 2 patterns in a single formula indexed by a diagonal indicator matrix Q_m. For each pattern, the preimage set is written as the current input polytope intersected with the linear inequalities that force the pattern, and the image is the affine pushforward of that intersection. Three cases exhaust the layer, so the layer's output is the union of these polytopes, and a corollary carries the same statement through an input that is itself a union. Multi-layer networks compose the layer map with input set equal to the previous layer's output set, and a second corollary adds a linear output layer, which is the same pushforward without the sign constraint. The closed-loop step then alternates: run the network on the current polytope union to get the possible control values, then apply the plant's linear map to the product set — one iteration per time step over the horizon. Because the union count multiplies at every step, they offer a variant that replaces each step's union with its convex hull so that exactly one polytope is carried forward, at the cost of an over-approximation.

## What counts as an explanation here
A set, not a description. The account of the network is "the set of all outputs reachable from this input set," expressed as a union of polytopes that can be enumerated, intersected and checked. On the exact variant the set is exact for the box queried, so it predicts outputs for any input in the region rather than characterising the function; on the convex-hull variant it is a sound outer bound with no stated width. Nothing in the account says why a particular output appears — the polytopes are indexed by activation patterns, which is the closest thing to an explanation in the entry, and the paper never uses that index as one.

## How they judged success
By construction and by picture. Correctness is the three-case proof plus the inclusion argument for the convex hull. Empirically the only check is a simulation overlay: the initial box is discretised on a 0.1 grid and 400 trajectories are run and plotted against the computed green region, and the paper asserts they are contained in it. The safety verdict is then read off the figures by eye — an empty intersection with the red unsafe box on [0,5] is called safe, and a non-empty intersection on [0,10] is called uncertain even though no sampled trajectory enters the unsafe region. There is no runtime, no set-size count, no width-to-truth ratio, and no comparison to any other tool.

## What they found
The exact computation works and grows intractably in the number of polytopes as the horizon lengthens, which the authors state plainly as the reason for the convex-hull shortcut. On the example, the hull-based estimate for [0,5] stays clear of the unsafe box around (4,4), while the estimate for [0,10] reaches it; since none of the 400 simulated trajectories reach it, the difference between the estimate and the samples is exactly the over-approximation the hull introduced. I found no numeric characterisation of that gap anywhere in the paper — the only evidence offered is the gap's visibility in the two figures.

## Limits and open problems they name
The horizon is finite and the growing union count is the stated obstacle; the convex-hull fix is presented as the workaround and its cost is named as over-approximation without a bound. The example is small and untrained (randomly generated system matrices and random network weights), so the paper demonstrates the machinery rather than exercising a controller anyone would deploy. No limitations section, no discussion of non-rectified activations, continuous time, or the case where a mode's set of validity must itself be intersected.

## What we take from it
Three things, one of them a warning. First, the pattern-indexed decomposition is the ancestor of the region-enumeration line the folder has since read (runs 629, 630): here the region is a linear region of a layer, and the paper is honest that the count is exponential and that the shortcut removes exactly the structure that made the representation exact. Second, the honesty of the verdict vocabulary is worth copying in its weak form — the authors distinguish *safe* from *uncertain* and never promote uncertain to unsafe without a witness, which is the two-and-a-half-valued version of run 630's three-valued discipline. Third, and the reason this entry is relevance 2 rather than 3: the validation is a picture with 400 samples on a 0.1 grid, and the paper reports none of the quantities the closed-loop thread has since established must be reported — no polytope count to justify the hull shortcut, no runtime, no ratio of estimate volume to sampled extent. A reader cannot tell how much of the green region is slack, which is the same reporting gap runs 628–634 kept finding. Method transferable; measurement discipline not.

## Tags
unit: feature | layer | circuit
form: description | probe
method: reachability | decomposition | other
eval: prediction/simulation | ablation
goal: safety/auditing | understanding
