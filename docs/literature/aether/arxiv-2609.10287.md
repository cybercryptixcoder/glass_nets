# Training Trajectories Determine Circuit Removability in Annealable Soft-Prior Transformers

- **Key:** arxiv-2609.10287
- **Authors / organisation:** Zonglin Yang, Ziming Zhao, Wei Tang, Xunyu Jiang, Yihong Liu, Tailin Chen, Zifu Yu, Jiayu Liu; Guangdong Police College (Criminal Science and Technology / Cybersecurity and Law Enforcement / Traffic Management Engineering), China
- **Date:** 2026-09-09
- **Link:** https://arxiv.org/abs/2609.10287
- **Kind:** paper (PRICAI 2026, 15 pages)
- **Read from:** full text (v1 HTML)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Interpretability studies of small synthetic tasks routinely add an inductive bias - a relative-position table, a locality prior, a specialised positional encoding - because it makes a circuit form at all. Evaluation then keeps that bias switched on, so a model that works *because of the crutch* and a model that has learned a free-standing attention mechanism are scored identically. The authors ask the one question that separates them: after the bias has done its job during training, can it be removed and does the circuit still work? The larger goal is a claim about what a circuit *is*. If the same final architecture can hold either an internal retrieval computation or a permanent dependence on an external bias term, then a circuit found by tracing necessarily describes one trained solution and not the architecture's capacity, and what produced the circuit becomes part of the explanation.

## What they studied

A tiny hand-built Transformer (two layers, four heads, `d_model` 64, length 32, vocabulary 32, roughly 35K parameters) plus a larger check at four layers and `d_model` 128 (about 265K parameters). The unit of analysis is the attention head, together with a per-head scalar gate that multiplies a bias added to the attention logits. Three tasks: associative recall (key-value pairs, query a key), Markov induction at three repeat probabilities (0.2, 0.5, 0.8), and linear-regression in-context learning used as a boundary case. The prior is a learnable table indexed by clipped relative position, plus a content prior computed from token embeddings; both sit behind gates in [0, 1].

## How they did it

Every training path shares the optimizer, size, data and step budget, and the paths differ only in the schedule followed by the scalar gate. In `unforced` both gate logits receive gradients, so the model chooses its own reliance on the prior. In `forced_zero` the gate is pinned at 0 for the whole run, and in `forced_const0.5` at 0.5. In `linear0.5->0` the gate falls linearly from 0.5 to 0 across training. The main fade path holds the gate at 0.5 until step 600, ramps it linearly to 0 over 450 steps, then keeps training with the prior absent until step 3,600. Two causal controls attack the obvious counter-explanations. A *hard switch* starts and ends at the same gate values but replaces the ramp with one abrupt jump, at step 600 or at step 1,050, which tests whether the model merely needs a warm start with an active prior. *Post hoc continuation* takes a finished unforced checkpoint and trains it for 2,550 further steps at gate zero, matching the zero-gate budget of the fade path, which tests whether the fade result is just more training with no prior. Schedule shape is then swept - linear, cosine, convex, concave, two-step - and the fade start is swept over steps 400 to 800, all at matched endpoints. Mechanism traces snapshot three fade seeds every 300 steps and compute two per-head quantities: successor-value mass, the share of a head's attention value written onto the correct successor token, and zero-gate ablation importance, the accuracy drop when that head's output is replaced by its batch mean. Final evaluation compares accuracy with the gate left where training ended against accuracy with the gate forced to zero.

## What counts as an explanation here

A causal story about a training trajectory rather than a static object: the claim is that removability is a property of the path the optimizer took, and the evidence is a set of matched-endpoint controls in which the endpoints are identical and the path is not. The mechanism traces add head-level attribution (which head carries successor mass, which head is causally necessary at zero gate) but are deliberately not a circuit diagram - the authors say explicitly that the identity of the head changes by seed and that what is stable is the timing. Partial, predictive in a narrow sense: it predicts for which tasks an annealed prior will leave a removable circuit and for which it will not.

## How they judged success

Two accuracy columns per path - *zero-gate* and *normal* - with means and standard deviations over five seeds for the main trajectory and causal-control runs and three seeds for the capacity, schedule-shape, task-boundary and mechanism experiments, 512 validation examples per evaluation. Baselines are thirteen static positional and soft-prior variants at matched dimensions and budget (vanilla, NoPE, RoPE, ALiBi, learned RPE, DAPE, CoPE, FIRE, Kerple, GPSA, plus three soft-prior ablations). The two controls are the ablations that matter, and the schedule sweep and fade-start sweep are the robustness checks. Mechanism claims are supported by the step index at which each head first crosses a fixed threshold (successor mass 0.25, ablation importance 0.03) across three seeds.

## What they found

The prior really is load-bearing in the unforced model: zeroing the position gate at evaluation drops associative recall from about 0.79 to 0.073, while zeroing the content gate changes almost nothing (0.791 against 0.794). Training path decides what happens after removal. Unforced models score 0.780 normal and 0.100 zero-gate, roughly chance; `forced_const0.5` is bad on both columns (0.164 and 0.114); `forced_zero`, which never had the prior, cannot learn the task at all (0.092/0.094); `linear0.5->0` lands in the middle (0.792 normal, 0.617 zero-gate); the fade path reaches 0.734 zero-gate against 0.694 normal. The larger backbone reproduces the ordering. Neither control approaches the fade: the hard switch at step 600 reaches 0.097, the later hard switch at step 1,050 reaches 0.293±0.128, and post hoc continuation reaches 0.178±0.035 while keeping 0.619 normal accuracy - so a model that was trained past the point of prior dependence stays dependent. The result is robust to when the fade starts (zero-gate 0.734 to 0.754 across starts 400 to 800) and to the schedule's shape as long as it is gradual (linear, cosine, convex all 0.749 to 0.764), but a two-step deletion reaches only 0.212 and a concave ramp is unstable across seeds. Markov induction replicates, at zero-gate 0.830, 0.920 and 0.969 for repeat probabilities 0.2, 0.5 and 0.8. Linear-regression ICL is the stated boundary: the unforced model still degrades when the gate is zeroed (loss 0.0656 to 0.1296), but `forced_zero` learns a decent solution directly (0.0722) and the fade only slightly improves on it (0.0646), so the task can form its solution without the crutch. The traces give the sharpest result: the top successor head differs across the three seeds (L1H0, L1H3, L1H1) and the top ablation head differs too (L0H2, L0H0, L0H3), yet ablation importance first crosses threshold at step 1,200 and successor mass at step 1,500 in all three seeds - both after the fade ends near step 1,050. Consolidation happens after the crutch is gone, and what consolidates is not a fixed head.

## Limits and open problems they name

Everything positive is on synthetic discrete retrieval, with models no larger than four layers and width 128. The content gate is nearly inert on these tasks, so the result is about positional priors only. Length extrapolation is not improved - at length 120 the soft-prior model reaches only 0.263 - and the authors decline to claim any. The schedule-shape scan covers a few simple curves and supports "smooth beats abrupt" without a theory of which paths work. The continuation argument is informal: the displacement bound involves an inverse Hessian that is never estimated, so why the fade table works but the fade shape barely matters is asserted through the math rather than measured.

## What we take from it

A *framing*, a *control*, and a *warning*, all three usable for a predictive white-box model. Framing: "what encodes this behaviour" has at least three distinct answers - the computation, the crutch it leans on, and the path that produced one rather than the other - and only the third is invariant across retraining. Control: the matched-endpoint pair (same start, same end, smooth versus abrupt) is a cheap way to test whether a fitted structure is doing work or is the residue of an optimizer trajectory, and it costs one extra training run per setting. Warning: the head identities here are not stable across seeds while the timing is, which is the same dissociation the folder has recorded as top-token overlap near 0.08 across seeds and as rotated-partner nulls - and it says that a white-box model which predicts *which* head will matter from architecture alone is predicting something the trained model does not have. The measured asymmetry is also directly transferable: a structure present at the end of training can have been consolidated after, not during, the period in which its supposed cause was active, so any claim that feature *X* was installed by the presence of *Y* needs a snapshot between the two.

## Tags

unit: head | circuit | weights
form: causal-abstraction | description
method: training-interpretable | ablation | probing
eval: ablation | benchmark | prediction/simulation
goal: understanding | evaluation-of-methods
