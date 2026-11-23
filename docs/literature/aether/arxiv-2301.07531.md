# Safety Verification of Neural Network Control Systems Using Guaranteed Neural Network Model Reduction

- **Key:** arxiv-2301.07531
- **Authors / organisation:** Weiming Xiang; Zhongzhu Shao. Augusta University (School of Computer and Cyber Sciences); Southwest Jiaotong University.
- **Date:** 2023-01-17
- **Link:** https://arxiv.org/abs/2301.07531
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Reachability-based safety verification of a closed loop containing a neural network controller gets expensive because every sampling interval requires an output-set computation through the network, and the cost of that computation tracks the network's layer and neuron counts. Rather than making the analyser cheaper, the authors ask whether the *network* can be made cheaper while keeping the verification sound: substitute a smaller network, and pay for the substitution with a rigorously computed bound on how far its outputs drift from the original's. The goal is scalability in the number of sampling steps — long horizons are exactly the regime where per-step analysis cost multiplies against horizon length.

## What they studied

A continuous-time nonlinear plant with sampled-data control, where the control law is a feedforward ReLU network queried at each sampling instant and held constant between samples. Unit of analysis: the network *as a whole function* — the object they bound is the sup-norm distance between two networks' output maps over a given input set, not anything inside a layer. The case study is an adaptive cruise control system: a 6-state lead/ego longitudinal model with friction, and a nominal controller with 5 hidden layers of 20 units each (inputs are set speed, time gap, ego speed, relative distance, relative velocity; output is ego acceleration).

## How they did it

The core construction is an *augmented* network that computes the difference of the two networks by construction. The original and reduced networks are stacked side by side: the first layer's weight matrix is the vertical concatenation of both first layers, so both branches receive the identical input; intermediate layers have block-diagonal weights with the two branches' matrices on the diagonal; if the reduced network is shallower, its last hidden layer is padded out to the original's depth with identity weights and zero biases so the branch passes through unchanged; the output layer places both output matrices side by side; and one final layer with weight matrix [I, −I] and zero bias subtracts the branches. This augmented network is a plain feed-forward network of the same activation type, so the sup-norm of its output over the input set *is* the "model reduction precision" — the worst-case output divergence between the full and reduced controllers, for any input the two share. That supremum is obtained by running an off-the-shelf network reachability tool on the augmented network and reading the largest norm over the resulting output set (they suggest reading it off the vertices of the polyhedral output set, or the interval envelope from the simulation-guided tool). With that scalar in hand, the closed-loop reachability algorithm replaces the full network's output set at each step with the reduced network's output set inflated by a ball of radius half the precision — the half is because the bounding is around the midpoint between the two outputs — then propagates the plant forward with a standard ODE reachability tool. The precision is computed once, offline, over the controller's whole input domain; the online loop then uses only the small network.

## What counts as an explanation here

Not an explanation of the network in the mechanistic sense — nothing about what any unit represents. What is produced is a *certificate of substitutability*: one real scalar per (full network, reduced network, input set) triple, which licenses discarding the full network from the verification pipeline while keeping the verdict sound for the full network. It is a complete over-approximation in the formal sense, and it predicts something concrete and checkable: that the full network's output at any input is within the precision of the reduced network's output at that same input.

## How they judged success

Two things are measured: (1) whether the inflated reduced-network reachable set still contains the full-network reachable set — shown qualitatively, as a plot where the reduced set's tube encloses the full network's tube; and (2) wall-clock time for the whole 3-second, 300-sampling-interval verification, original controller versus reduced controller, on the same plant and tools. The safety verdict itself is intersection of the reachable tube with the unsafe region. There is no ablation of the reduction ratio, no sweep of precision against speedup, no comparison against a competing reduction heuristic, and no ground-truth cross-check of the precision value by sampling.

## What they found

The size shrank dramatically: from 5×20 to 2×5 hidden units, with a computed reduction precision of roughly 1.0234 in the output units. Verification time fell from 12.355 s to 1.099 s — about an order of magnitude — over the 300 intervals. The inflated tube for the reduced network completely enclosed the tube for the original, and the safe-distance threshold region stayed clear of both, so the ACC scenario is judged safe with the reduced controller standing in for the full one for a 3-second horizon after an emergency deceleration by the lead car.

## Limits and open problems they name

The paper's conclusion is short and names no limitations. What the text makes evident without saying so: the precision is a single number computed over the entire controller input domain, so it is independent of the actual state set encountered at each step, and it is added afresh at every sampling instant, meaning a bound that is loose in a small region is paid 300 times. The reduction procedure itself is not specified — the paper points at the model-compression literature and simply states that a smaller network was obtained; no accuracy-versus-size target is given, so the trade is the user's to guess. There is no analysis of how precision behaves under composition with the plant, no statement of whether the inflated tube's growth is monotone in horizon, and no failure mode: the case study is one scenario, judged safe.

## What we take from it

A method, and a sharp one for our purposes. **Padding a shallow branch with identity layers so that two networks of different depth can be evaluated inside one object is a reusable trick** — it converts a relation between two models into an ordinary forward pass, which means any analysis we already have for one network applies unchanged to the *pair*. For a predictive white-box account, this is the assembly step for model differencing and for distillation auditing: if we want to say how far a pruned or distilled model departs from its source over a region, the answer can be a reachability query on a network we construct, not a separate apparatus. The warning is equally useful: their precision scalar is *region-free* — computed once over the whole input domain and then reused at every step — so the reported speedup is real but the tightness is not, and a single global Lipschitz-style number is exactly the quantity that will be useless when the interesting region is narrow. If our account reports any such global scalar, it must be printed next to a per-region value, or the reader cannot tell how much of the conservatism is the model and how much is the constant.

## Tags

unit: whole-model
form: replacement-model
method: decomposition
eval: prediction/simulation, ablation
goal: safety/auditing
