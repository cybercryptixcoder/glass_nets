# Reachability Analysis and Safety Verification for Neural Network Control Systems

- **Key:** arxiv-1805.09944
- **Authors / organisation:** Weiming Xiang, Taylor T. Johnson (Department of Electrical Engineering and Computer Science, Vanderbilt University)
- **Date:** 2018-05-25
- **Link:** https://arxiv.org/abs/1805.09944
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Neural network controllers are trained on data and then trusted with safety-critical plants, but almost everything the verification community could say about a network in 2018 was said about a ReLU network. The authors point at a gap in the middle of the field: controllers for nonlinear plants are commonly built with smooth squashing activations, and no analysis result in their reading covered that class inside a feedback loop. So the paper sets out to make two things possible for a feedforward controller with a general activation function: an outer bound on the set of control outputs the network can produce over a region of inputs, and, by alternating that bound with a plant reachability routine, a bound on the closed-loop state set over a horizon, which is then read against an unsafe region. The stated ambition is formal, not statistical: if the estimated reachable set and the unsafe set do not meet, the system is called safe.

## What they studied

Feedforward multilayer perceptrons of arbitrary depth with per-layer activation functions, in principle any continuous form; the theorems assume the activation is monotone non-decreasing, with the Gaussian bell handled as an extra case. The plant is either a discrete-time difference equation or a continuous-time ODE with a sampled controller that holds its value between sampling instants. The unit of analysis is a hyper-rectangle: hyper-rectangular cells for the network's input set, hyper-rectangular reachtubes for its output, and existing ODE reachability tools for the plant. Two examples: a randomly initialised 2-input/2-output network with one hidden layer of 7 tanh units, and a 2-state linear plant driven by a 5-unit hidden-layer network with a reference input.

## How they did it

The input set is first enclosed in its tightest axis-aligned box, then that box is cut into a regular grid with `M_i` segments along each dimension, cells that do not meet the original set are dropped, and what remains is a union of rectangles that over-approximates the input region. For one layer and one rectangle, the paper proves that the output is enclosed by an interval box obtained by pinning each pre-activation to its minimum and maximum over the rectangle and applying the activation to those two values; the monotone assumption is what makes that substitution valid. Those min and max problems are linear programs over the rectangle, and because the constraints are box constraints the optimum is available in closed form — for each weight, multiply by the lower end of the input interval if the weight is non-negative and by the upper end otherwise, and sum. A Gaussian activation breaks monotonicity on one half-line, so an extra case analysis handles intervals that straddle the peak and picks the peak value as the maximum. Stacking the layer result gives a layer-by-layer routine that maps each input rectangle through the whole network to one output rectangle; the network's output set is the union of those boxes. For the loop, the paper does not build a new plant propagator: at each step it maps the current state estimate through the output equation to get the controller's input region, runs the network routine to get a control set, and hands state and control set to an ODE reachability routine. In the linear example the state estimate from the previous step is further wrapped in its convex hull before the affine propagation, with a convex-geometry toolbox doing the set arithmetic. Verification is then a set intersection test: no overlap between the estimated reachable set and the unsafe region returns SAFE, overlap returns UNCERTAIN.

## What counts as an explanation here

An enclosure — a union of axis-aligned boxes for the network's output set and, over the horizon, for the closed-loop state set — together with the guarantee that the true set sits inside it. It is deliberately partial and deliberately pessimistic: separation from the unsafe set is a certificate, but no description of the network's behaviour is produced, and the paper never claims the boxes are informative about why the controller does what it does. It predicts exactly one thing, and predicts it soundly: which states the closed-loop system can reach, never which ones it will.

## How they judged success

Success is a soundness property, established by proof rather than by measurement: a chain of theorems (single-layer interval enclosure under a monotone activation, the LP and closed-form versions, the Gaussian case, the layer-by-layer composition, the closed-loop enclosure, and the safety test) each with a short proof, and every example validated by checking that random samples of the true output or the true trajectory land inside the computed set. The one quantitative axis is the partition count: five grid resolutions are run on the network example and reported with the number of boxes produced and the wall-clock time. There is no comparison against another tool, no baseline, no ablation, and no metric of box tightness beyond the eye — the figures overlay boxes at different resolutions and invite the reader to see that finer is tighter.

## What they found

The closure is real and cheap for this network family: on the 7-unit tanh example, going from a 10x10 grid to a 50x50 grid raises the box count from 100 to 2500 and the runtime from about 0.06 s to about 0.38 s, so the cost is dominated by the grid, not by the network — the weight arithmetic per box is a handful of multiply-accumulates per neuron. 5000 sampled outputs fall inside the estimated set at every resolution, and the resolution sweep visibly shrinks the union of boxes toward the true image. In the closed-loop example, 5- and 20-point grids over a 10-step horizon both enclose all 1000 sampled trajectories, the finer grid giving the tighter picture, and the coarser one already shows no intersection with the unsafe box centred away from the trajectory bundle, so the system is declared safe on the interval. The cases where a coarse grid reaches the unsafe region and a fine one does not are not shown; what is shown is that the verdict here is not sensitive to the resolution used.

## Limits and open problems they name

The paper itself offers little: it says the plant propagator is inherited from existing tools and that it plans to plug in a list of named hybrid-systems verifiers and to handle broader network classes. The limits a reader can see are structural. The grid is the whole fidelity dial and its size is the product of the per-dimension counts, so the method is designed for a small number of input dimensions (both examples use two). The layer-by-layer box composition discards the correlation between neighbouring units within a layer and between successive cells, so tightness decays along depth and along the horizon, and the paper neither quantifies that decay nor shows at what depth the boxes become useless. The verdict vocabulary is binary in the algorithm — SAFE or UNCERTAIN — and the figures are argued by inspection rather than by an intersection computation whose numbers are reported.

## What we take from it

Two things worth carrying. First, a **warning about granularity as the only control**: this is the ancestor of the whole refinement family in the thread, and its cost and its tightness are both governed by a single integer vector, which is exactly the design that later papers in the folder spend their energy escaping (by changing the set algebra, by changing the propagation rule, or by changing the network). Reading it first makes the later contributions legible as escapes from a specific default. Second, a **cheap and reusable measure**: the per-box enclosure of a network over an axis-aligned cell is available in closed form from the weights alone, so any account of ours that reasons about a network on regions can use box propagation as a free lower bound on what a tighter method should buy — and the sensible way to report that buy is as boxes-needed-at-fixed-tightness rather than as seconds. The paper's own reporting practice is also a **shortcut to avoid**: it prints cost against resolution but never the enclosure's slack against a reference, so its central trade is asserted rather than measured.

## Tags

unit: layer
form: description
method: decomposition
eval: prediction/simulation
goal: safety/auditing
