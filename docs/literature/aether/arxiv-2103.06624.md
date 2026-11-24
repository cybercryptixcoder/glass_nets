# Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Complete and Incomplete Neural Network Verification

- **Key:** arxiv-2103.06624
- **Authors / organisation:** Shiqi Wang, Huan Zhang, Kaidi Xu, Xue Lin, Suman Jana, Cho-Jui Hsieh, J. Zico Kolter; Columbia University, UCLA, Northeastern University, CMU
- **Date:** 2021-03-11
- **Link:** https://arxiv.org/abs/2103.06624
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Branch-and-bound verifiers split unstable ReLU neurons and then ask a cheap incomplete verifier for a lower bound on each subdomain. The cheap verifiers in use — CROWN and its relatives — propagate bounds backwards but have no way to say "this neuron is positive" or "this neuron is negative": they can handle an input box and a linear output, not a per-neuron sign condition. So every split in the search tree had to be handed to an LP solver, which is hundreds to thousands of times slower and does not use the GPU. The authors want the bound-propagation speed without giving up the split constraints, because without them the search tree branches blindly, produces loose bounds, and — worse — cannot prove that a subdomain is empty, so a CROWN-plus-search verifier is not even complete. The larger goal is a complete verifier that is fast enough that the same code, stopped early, also beats the best incomplete verifiers on networks nobody trained to be verifiable.

## What they studied

Feedforward and convolutional ReLU networks, small by modern standards: a 166k-parameter MNIST CNN and a 2.1M-parameter CIFAR-10 CNN, both adversarially trained and taken from a prior SDP paper, plus three CIFAR-10 models (Base, Wide, Deep) from a standard complete-verification benchmark. The unit of analysis is the *neuron* — specifically the set of neurons that the search has split so far, and the per-neuron multipliers attached to those splits — with the network's weights, pre-ReLU bounds and the specification as the fixed inputs.

## How they did it

They keep CROWN's backward substitution but add one term at each ReLU layer. A diagonal sign matrix `S` marks which neurons in that layer are split positive (−1), split negative (+1) or unsplit (0). When a layer's bound is relaxed, they add a nonnegative multiplier vector β on that layer, multiplied by the sign matrix, to the linear coefficient that carries the output back through the layer. This is exactly a Lagrange relaxation of the split constraints followed by weak duality: any β ≥ 0 gives a sound bound, and the bound is maximised over β. Because the inner problem over the input box has a closed form, the whole thing collapses to maximising a concave function of β over the nonnegative orthant — no LP, no solver, just projected gradient steps through an autodiff graph. The second ingredient is that α (CROWN's own free slope parameters) and β are optimised *jointly across intermediate layers*: every hidden neuron's own lower and upper bounds come from a separate application of the same theorem with its own α′ and β′, so tightening a hidden bound tightens the final one. Because the joint problem is non-convex they share the intermediate parameters within a layer, cutting the variable count from quadratic to linear in depth. They then run branch-and-bound with this as the incomplete verifier and a simple ReLU-splitting heuristic, batching subdomains on the GPU, and they run a second experiment where the same search is stopped after a fixed time budget and the best bound found so far is reported as an incomplete verifier's answer.

## What counts as an explanation here

Nothing narrative. The understanding is a **dual certificate with per-neuron, per-layer multipliers**: the vector β, one nonnegative number for every split neuron, plus the α slopes for every unstable neuron. The paper's own framing of β is what matters for us — it is the dual variable of the neuron split constraint, so it has the meaning "how much this neuron's sign assumption is worth to the bound". Any setting of the parameters is a valid lower bound; the reported numbers are the values after a fixed number of gradient steps, so what is presented is not the optimum but a witness that was pushed upward for 20 iterations. It is partial by construction — it certifies a scalar over a box, not the function.

## How they judged success

Two settings, two families of metric. *Incomplete*: verified accuracy on 100 test images with runner-up target labels, compared against an SDP-based verifier (SDP-FO), an LP built on the same relaxations, CROWN, an interval method, and a multi-neuron-relaxation LP; plus a scatter of verified lower bound against a PGD adversarial upper bound for the same 100 points, with time per point as the cost axis. *Complete*: median verification time, median branch count and timeout percentage per model against six verifiers, median rather than mean because timeouts dominate the mean. Ablations: LP verifiers rebuilt on intermediate bounds from three different sources (interval, CROWN, jointly optimised) to separate the value of the bound propagator from the value of the intermediate bounds.

## What they found

Verified accuracy on the MNIST adversarially trained network at ε = 0.3 rises from 43.4% (SDP) and 63% (a strong GPU dual-space verifier, early-stopped) to 66%, against 78% empirical PGD accuracy — closing the verified-to-attack gap from 35 points to 12 — at 27 minutes per sample against 1,274 minutes for the SDP and 1,926 on CIFAR-10. On the complete benchmarks the median time is 5.11 / 4.05 / 7.25 seconds (Base / Wide / Deep) with 8 / 3 / 0 percent timeouts; the LP-based baseline is at 2421 / 3490 / 2982 seconds with 36-49% timeouts, and the closest competitor at similar timeout rates is about twice to seven times slower. The ablation is a clean decomposition: tighter intermediate bounds help an LP (5% and 10% verified accuracy for the CROWN and jointly-optimised arms against 1% for the interval arm on MNIST), but search with the split-aware propagator beats all three LPs regardless of their intermediate bounds.

## Limits and open problems they name

The joint optimisation is not concave and is capped at 20 Adam iterations with fixed learning rates and a decay factor, so no optimality is claimed for the reported bound; they note soundness does not need convergence. The per-neuron split variables would be quadratic in depth if not shared, and they share them by layer — a stated approximation, not a proof. The incomplete-verification accuracy numbers for the SDP are quoted from the original paper on the same pretrained models because reproducing them would have cost thousands of GPU hours, so that comparison is not run on one machine. One baseline is missing for one model on the CIFAR comparison because its code did not accept the asymmetric padding layer.

## What we take from it

Three things. **A framing** for the survey's standing gap: split constraints are exactly a per-neuron conditioning of the certificate, and β is a per-neuron *worth* — a nonnegative scalar saying how much the sign of that neuron contributes to the bound. Every verifier in this head that reports a single bound has computed (or could compute in one gradient pass) a vector of such numbers and discarded it; this paper is the first where the vector is the optimised object rather than a by-product. **A measure**: the difference between the unsplit bound (β = 0, which the paper states reduces exactly to CROWN) and the same bound with β optimised is a data-free, backward-pass-only attribution of how much each neuron's sign assumption is worth — a per-neuron reading available from the checkpoint plus a box, with no training and no data. **A warning**: the intermediate-layer α′/β′ sharing is the third member of the head where a construction is fitted under an approximation and the reported number is the fitted value without a statement of the approximation's cost.

## Tags

unit: neuron
form: direction | causal-abstraction
method: probing | attribution | other
eval: ablation | prediction/simulation | benchmark
goal: safety/auditing | evaluation-of-methods
