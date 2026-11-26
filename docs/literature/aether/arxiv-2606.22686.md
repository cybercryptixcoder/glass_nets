# The Geometry of Refusal: Linear Instability in Safety-Aligned LLMs

- **Key:** arxiv-2606.22686
- **Authors / organisation:** Shivam Ratnakar (University of Southern California); Kartikeya Vats (independent; equal joint first authors)
- **Date:** 2026-06-30 (v2; v1 2026-06-21)
- **Link:** https://arxiv.org/abs/2606.22686
- **Kind:** paper (TrustNLP 2026 workshop, ACL 2026)
- **Read from:** full text (arXiv HTML v2, all sections plus both appendices; references skimmed)
- **Relevance to us:** 2 (useful)

## What they wanted, and why
The authors ask whether safety compliance in aligned open-weight models is a deep semantic property of the computation or a thin filter sitting near the output. If refusal lives in one direction of the residual stream and that direction is all that separates a harmful request being answered from being refused, then safety is a linear feature that can be removed arithmetically, and alignment has to be judged by how deeply it is woven into the network rather than by how often a model refuses. The paper is therefore both an attack paper (expose how shallow alignment is) and a measurement paper (give a cheap instrument for depth of a decision). It is aimed at the safety and auditing literature, but the second half is a general template for locating *when* in a forward pass a binary decision is actually taken.

## What they studied
Seven open-weights families: Gemma-3 at 4B and 12B, Llama-3.1 8B, Llama-3.2 3B, Llama-3.3 70B, Qwen-2.5 at 1.5B and 7B, plus Llama-2 and Qwen-7B for the baseline comparison. Data: AdvBench for a steering-strength sweep, JailbreakBench for attack comparison and for the detector, and the 159-prompt standard-behaviour subset of HarmBench for the comparison against an activation-steering baseline. The unit of analysis is a *logit vector over the vocabulary at a single decoding step*, produced under three different system prompts; the secondary unit is a mean final-layer hidden-state difference over a small anchor set.

## How they did it
Three copies of the same query are run through the model under three system prompts: a neutral one, an explicitly unrestricted one, and an extremely conservative one (the literal prompts are in Appendix A). At each decoding step the logits of the unrestricted stream minus the logits of the conservative stream define an instantaneous steering vector, and the neutral stream's logits are shifted by a coefficient alpha times that vector before sampling; no fitting, no gradient search, no layer selection. Positive alpha is expected to unlock compliance, negative alpha to enforce it. Because aligned models often emit a reflexive refusal in the first tokens and then lock into that trajectory, the first token is forced to an affirmative prefix so the steering vector can act before the refusal reflex fires. Attack success is scored over a sweep of alpha at fixed temperature, with responses judged by a single LLM judge chosen from three candidates against 100 hand-labelled examples; a separate 1-to-5 coherence rating tracks whether steering destroys fluency. For comparison they run their own 100-step greedy token-search attack and re-implement an activation-level refusal-direction ablation on the matching benchmark. Finally they run two diagnostic passes over depth: a PCA of final-layer hidden states for harmful versus benign prompts, and a per-layer divergence between hidden states under safe and harmful instructions, used to timestamp where the two processing paths separate. A zero-shot detector is also built by averaging final-layer states over 50 harmful and 50 benign anchor prompts and thresholding the cosine of a new query against that mean difference.

## What counts as an explanation here
A single direction, and then a *timeline* of where that direction becomes active. The explanation is partial by construction: the authors explicitly do not try to describe the circuit that computes refusal, only the scalar or near-scalar quantity the output stage reads. The contribution that transfers is the depth-timestamp instrument: a per-layer divergence curve that classifies a model's binary decision as being taken late (essentially at the output head) or early (mid-network), which they treat as the axis that predicts how well a linear intervention at a given site will work. It can predict one thing: that a logit-level subtraction will succeed on a late-deciding model and mostly fail on an early-deciding one.

## How they judged success
Attack success rate under a single LLM judge validated against 100 human labels, with a coherence score as the second axis so a "successful" attack that produced gibberish is visible. Baselines are a 100-step optimisation attack and an activation-level refusal-direction ablation on the same 159 prompts. Controls are thinner: no random-direction steering baseline, no shuffled-prompt contrast, no seed or prompt-rewording variance for the steering vector itself, and no ablation of the prefix-injection component on its own. The mechanism claims rest on PCA plots and a per-layer divergence figure read qualitatively; the detector is measured by F1 on JailbreakBench.

## What they found
Removing the logit difference unlocks compliance almost immediately on some models - one Gemma model goes from 34% to 98% success at alpha of 1 - while coherence degrades from about 3.74 to 2.14 at the top of the sweep. The 7B Qwen model resists much more, reaching only 68.5% at alpha 5 against above 96% for the Llama models. Against the optimisation attack, the arithmetic method reaches 95% on Llama-3.1 where the search attack reaches 5%, and 90% against 45% on Qwen-2.5, at roughly one second against fifteen minutes per query. Against the activation-level baseline on HarmBench they report 73% versus 22.6% on Llama 2 and 91% versus 79.2% on Qwen 7B. Negative steering cuts a 70B Llama from 68.8% to 9.4% and a 12B Gemma to 6.0%, and slightly *raises* coherence on Qwen (3.84 to 4.14). The depth analysis gives two regimes: near-zero divergence across about 95% of the Llama layers with a spike at the end, and a large divergence near 40% depth in Qwen. The cosine detector reports F1 of 0.92, with benign prompts near 0.16 and malicious near 0.65.

## Limits and open problems they name
White-box logit access is required. Alpha is hyperparameter-sensitive and only single-turn interaction is tested. Comparison to other activation-level families is left qualitative. The paper states plainly that it cannot separate architecture from training procedure as the cause of the late-versus-early difference. Also worth naming from the outside: the steering vector is recomputed per decoding step and never checked for stability across prompt rewording or anchor choice, the prefix injection is always on so the vector's contribution is never measured alone, and the "linear" claim is supported by a two-dimensional visualisation rather than by a linear probe or a curvature measurement.

## What we take from it
Two things. Method: a **per-layer divergence curve between two matched inputs** is a cheap, general instrument for asking *where* a decision is taken, and it costs one forward pass per layer rather than a circuit search - useful for our white-box model because it gives a scalar field over depth that any candidate explanation has to agree with. Warning: an intervention that works at the shallowest possible site tells you the decision is shallow *only if* the intervention is the minimal one - here the prefix forcing is doing unmeasured work, so "95% in one second" is an upper bound on fragility rather than a measurement of the refusal direction's power. For our work, any claim that a behaviour is "one direction" gets the divergence curve beside it, and any intervention used to argue shallowness ships with the site-matched alternative.

## Tags
unit: direction
form: direction | steering
method: steering | probing | other
eval: downstream-task | ablation
goal: safety/auditing | understanding
