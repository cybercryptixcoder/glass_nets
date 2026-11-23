# POLAR: A Polynomial Arithmetic Framework for Verifying Neural-Network Controlled Systems

- **Key:** arxiv-2106.13867
- **Authors / organisation:** Chao Huang (University of Liverpool), Jiameng Fan and Wenchao Li (Boston University), Xin Chen (University of Dayton), Qi Zhu (Northwestern University)
- **Date:** 2021-06-30 (v2; arXiv listing of v1 is June 2021)
- **Link:** https://arxiv.org/abs/2106.13867
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Closed-loop verification of a neural controller needs a way to push a *set* of states through the network and then through a plant ODE, step after step, and every existing tool is blocked by one of two walls: the arithmetic used to propagate sets (Taylor models) assumes the function is differentiable and truncates at an order, so it cannot carry a rectifier through unchanged, and set encodings that do handle rectifiers (star sets, polytope preimages) either need linear dynamics or a piecewise-linear-only network. The authors' aim is an arithmetic that is agnostic to the activation's shape — any continuous activation, mixed within one network — while keeping the plant side on the established Taylor-model flowpipe machinery. They frame this as an arithmetic contribution rather than a new verification algorithm: reachability stays the same checking problem; what changes is the algebra used to over-approximate one layer at a time. The larger goal is to work for networks a real controller could plausibly be (deep, wide, multi-input) rather than hand-sized academic examples, since the alternative surrogate-fitting route was already known to fall off in dimension.

## What they studied

Feedforward controllers with any continuous activation, including heterogeneous networks that mix rectified, sigmoid and hyperbolic-tangent units across (and within) layers; the plant is a nonlinear ODE in `n` state variables controlled by `m` inputs at a fixed control step, held constant between steps (zero-order hold). The unit of analysis is a **per-neuron polynomial with an interval remainder**, composed layer-by-layer into a single vector polynomial over the initial state. Experiments cover 24 single-control-step benchmarks over 2-, 3- and 4-dimensional plants with 3 or 4 hidden layers of 20 or 100 neurons and all four activation mixes, plus one new 6-state, 3-input attitude-control system with a 3x64 controller.

## How they did it

The outer loop is a textbook flowpipe construction, and the authors are explicit that they borrow it: at each control step, over-approximate the controller's output set for the current state set (the contribution), then integrate the ODE over the step with the control held constant — implemented by appending the inputs as extra state variables with zero derivative — and feed the last flowpipe's end as the next step's initial set. The inner loop is the novel part. The network's input is a polynomial-plus-interval over the initial state. Each layer applies the affine map with ordinary polynomial arithmetic, but the activation is *not* Taylor-expanded — it is interpolated neuron-wise by a **univariate Bernstein polynomial** in that neuron's own pre-activation range, with the interpolation order a user knob, and a conservative symmetric remainder interval found by sampling the interval uniformly and taking the largest sampled interpolation error plus one sample spacing. Because the interpolation is univariate and per-neuron, cost grows linearly in order, unlike a multivariate Bernstein abstraction whose cost grows exponentially in the number of inputs. A third component, the **symbolic remainder**, attacks the wrapping effect, the growth that accumulates when the same remainder interval is multiplied by a chain of weight matrices one at a time: the layer recurrence is decomposed into a linear part and a nonlinear part, and the linear part is composed as a single product matrix before the interval is applied, exactly the reordering that interval arithmetic loses. That version needs two extra arrays and two nested loops, so time goes from linear to quadratic in the number of layers.

## What counts as an explanation here

Not a description of the network. The artifact is a **point-wise over-approximation set**: a polynomial in the initial-state variables plus an interval, with the property that for every initial state the true state at each time is inside "polynomial evaluated at that state, plus the interval". It is a certificate object rather than an explanatory one, and it is deliberately partial — it says what the system *cannot* do (reach a forbidden region) and never what the network computes or why. The remainder interval is the entire visible error budget, so the width of that interval is the account's quality.

## How they judged success

Three-valued verdicts identical in shape to the NNV protocol this folder already recorded: proved reachable if the final step's over-approximation lies entirely inside the target, disproved if it misses the target entirely, otherwise unknown. Tool-level comparison against Sherlock (ReLU only), Verisig 2.0 (smooth activations only), NNV and ReachNN* on the full benchmark suite published with the Bernstein predecessor, with identical Taylor-model settings across tools; the reported statistic is runtime in seconds for successful verification, with "unknown" and "not applicable to this activation type" as separate cells. No volume ratio, no slack figure, no table of how wide the computed enclosure is relative to the true reachable set.

## What they found

POLAR finishes every one of the 24 benchmark cells in single-digit to low-twenty seconds and proves or disproves 23 of them; the one failure is the second benchmark with a sigmoid controller, where Verisig 2.0 and ReachNN* both succeed. Most comparisons are lopsided: on the 4-layer, 20-neuron benchmarks the ReachNN* route needs over a thousand seconds on several cells and over thirteen thousand on one sigmoid case, against POLAR's single digits. On the new 6-state attitude system, 30 control steps finish in 313 seconds without symbolic remainders and 201 seconds with them — the tightening pays for itself in wall-clock because tighter boxes are cheaper to integrate — while Verisig 2.0's remainder explodes and the tool crashes within a few steps; NNV exhausts the 8 GB machine on these systems. The honest reading of the accuracy claim is that in most plots POLAR's enclosure is the tightest of those compared, and that "tightest" is assessed by overlaying simulated trajectories and the other tools' sets, not by a number.

## Limits and open problems they name

Two are stated. The first is structural: Bernstein coefficients cannot be recovered from a higher-order interpolation by truncation, which Taylor-model arithmetic allows, so any simplification of a composed model risks a remainder far larger than the order reduction would suggest — and this is offered as the reason the one sigmoid benchmark was not closed. The second is that all Taylor-model methods are sensitive to settings (interface stepsize, orders, sample counts), which in these experiments were shared with the competitors, so the comparison at least does not hide behind settings tuned per tool. Future work named: automatic selection of these hyper-parameters. Unstated but visible: the reachable set is a union of boxes and is reported as octagonal enclosures, so nothing measures the gap between enclosure and truth; and the whole approach is bounded-horizon only.

## What we take from it

A **method** and a **warning**. The method is the substitution of a native approximation family for a function class that arithmetic cannot differentiate: instead of Taylor-expanding a rectifier (impossible) or treating the layer as an opaque box (loose), interpolate the activation in the variable that actually bounds it and carry a certified remainder for the interpolation. That is the cleanest instance in the folder of choosing the surrogate by the shape of the operation rather than by convenience, and the remainder's definition is a pattern we can copy directly — interpolation error at sampled points plus one cell width, which is a proof because no unsampled point is farther than the cell width from a sampled one. The symbolic-remainder idea is a **framing** worth keeping on its own: the wrapping effect is not an arithmetic problem but an *association* problem, and the cure is to defer interval evaluation until the linear chain has been composed, buying tighter boxes at the cost of storing the chain. For our own account, the warning is the one this folder keeps finding: the paper's headline comparison is time-and-verdict, and the quantity that would let a reader distinguish "better arithmetic" from "easier benchmark" — enclosure width against true width — is never tabulated.

## Tags

unit: layer | feature
form: replacement-model
method: decomposition | other
eval: downstream-task | benchmark
goal: understanding | safety/auditing
