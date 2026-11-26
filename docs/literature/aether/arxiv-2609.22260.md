# Causal Localization of the Refusal Direction in Audio Language Models

- **Key:** arxiv-2609.22260
- **Authors / organisation:** Leonardo Haw-Yang Foo, Hung-yi Lee; National Taiwan University, and NTU Artificial Intelligence Center of Research Excellence (NTU AI-CoRE)
- **Date:** 2026-09-06 (v1)
- **Link:** https://arxiv.org/abs/2609.22260
- **Kind:** paper (conference, ISCSLP 2026; four pages plus references)
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

A large audio language model (LALM) is a speech front end bolted onto a text language model that was already
safety-aligned. When such a model refuses a harmful spoken request, two very different stories explain it: the
refusal could be carried by the audio pathway, or could be inherited, unchanged, from the text backbone. The
question is practical before it is mechanistic, because speech inputs are a known weak point for these models, so
a safety auditor needs to know which half of the stack to inspect and which half to patch. The authors turn the
question into one measurement — where does removing a harmful-versus-benign direction make the model's refusal
lean fall — and they are careful, throughout, to distinguish *where the readout depends on a direction* from
*where refusal is computed*. The larger goal is a discipline: probes tell you what is decodable, only interventions
tell you what is used.

## What they studied

Five LALMs spanning three text-backbone families: Qwen2.5-Omni-7B (28 LM decoder layers, the primary model),
Audio Flamingo 3 (same Qwen2.5-7B backbone, different Whisper variant and projector), Voxtral-Mini-3B (a 30-layer
Ministral backbone), and Ultravox-v0.4.1 together with DeSTA2.5 (both 32-layer Llama-3.1; DeSTA keeps the LM frozen
and feeds a Q-Former interface plus a spliced ASR transcript). A sixth model, LLaMA-Omni, refused almost nothing at
baseline and was dropped before localisation. Data: 588 matched harmful/benign audio pairs over 14 harm categories,
built by rendering text benchmarks (AdvBench, HarmBench, JailbreakBench; 498, 79 and 11 prompts respectively) through
one fixed TTS voice, plus a domain-matched benign hard negative per prompt generated from a fixed template bank
(1,176 clips before deduplication of near-identical prompts within a category). The unit of analysis is a per-layer
direction in the residual stream, and, at the audio interface, a direction or rank-k subspace in the projector's
mean-pooled frame vectors.

## How they did it

For each layer and training fold they fit `r = (mean_harmful − mean_benign) / norm` on the activation at the
readout position (the last token of the chat-templated prompt); at the projector, which has no comparable position,
the same difference of means over mean-pooled frame vectors. They then run three hooked operations on a module
output: do nothing; ablate, which deletes the component along `r` and rescales to preserve norm; and steer, which
adds `k·r`. A fourth, *scalar rescue*, records each prompt's coordinate along `r` before ablation, ablates, then adds
that coordinate back at every position, and reports per-prompt recovery as the patched-versus-none ratio over the
ablate-versus-none gap. Crucially for localisation, they ablate **each candidate layer independently** — one hooked
layer per run — rather than selecting a direction at one layer and applying it everywhere as Arditi et al. do, so a
site's effect answers "how much does the held-out margin depend on this layer's own fitted direction". Ablation runs
at every token position even though the direction was fit only at the last one. Held-out category shift is enforced
two ways: a group-blocked split holding out four entire harm categories (238 pairs) and leave-one-group-out
aggregated over thirteen folds (584 pairs), with the direction fit on the training side only. Decodability is
measured separately as the AUROC of the scalar projection of evaluation prompts onto `r`, and specificity is
measured against 4 to 32 random unit directions ablated identically at every site, plus a pre-specified
structure-matched null fitted on benign-versus-benign splits. The refusal readout itself is a single number,
`G(x)`: the log-sum-exp margin between a frozen list of refusal opener tokens and a frozen list of compliance
opener tokens, computed on first-token logits and fixed before any intervention was run.

## What counts as an explanation here

A localisation, deliberately narrow: for each site, the change in a one-number readout when the site's own rank-1
direction is removed, under held-out category shift. It is partial and the authors say so twice — they localise
dependence of the margin, "not where refusal is computed". It predicts nothing about generation: the same sentence
is scored at the logit level and at the behavioural level and the two are reported apart. The interesting formal
content is that the direction is a *cross-modal* object: a direction fit on the text backbone alone is ablating with
substantial effect inside the audio forward pass, which is a kind of transfer claim about a recovered object rather
than a description of a site.

## How they judged success

Paired-bootstrap 95% confidence intervals on the held-out margin change; two independent held-out splits; random-
direction nulls held to |ΔG| at or below about 0.02 at every site; a rank-k projector subspace sweep to k = 16; a
pre-specified safety gate requiring baseline harmful-refusal at or above 0.70 with benign answer-rate at or above
0.80, which three of five localised models passed; scalar rescue recovery against a random-coordinate control;
text-to-audio and audio-to-text transfer as fractions of the other pass's own effect; a paraphrasing check with
intent-preservation rated by a separate judge and token-Jaccard overlap held under 0.3; and behavioural validation
using greedy 512-token generations scored by the HarmBench classifier, with the sign of `G` checked against a
mixed-evaluator label set.

## What they found

On Qwen2.5-Omni, ablating the L16 direction moves the held-out margin by −7.10 [−7.28, −6.93] on the group-blocked
split and −7.53 [−7.67, −7.38] under leave-one-group-out, while the rank-1 projector direction moves it by −0.013
[−0.033, 0.005] and the rank-16 projector subspace by at most −0.14. L08 is −0.10, L14 is −2.51, L17 (dense
follow-up) peaks at −9.19, L20 is −5.43, and the band is still about half its maximum at L26 of 28 layers. The
audio channel is nonetheless open: zeroing the encoder output moves the margin by −4.7 (16 clips). One scalar
coordinate recovers almost all of it — scalar rescue recovers 0.950 [0.947, 0.952] against 0.000 [−0.001, 0.001]
for a random coordinate, and adding the direction to benign activations raises the benign margin by +7.24 against
−0.24 for a random direction. A text-fitted direction reproduces 0.99 of the audio-fitted effect when ablated in the
audio pass, the reverse direction reproduces 0.73, and their cosine is 0.76; on 47 paraphrased clips held to be
intent-preserving with low lexical overlap the effect is still 0.75 [0.65, 0.84] of verbatim, against a random
direction's about 0.045. Decodability is essentially perfect everywhere they intervene (AUROC at least 0.997) while
single-layer ablation at L08 is −0.103, about a seventieth of the L16 effect; ablating L08 through L15 jointly does
move the margin, −6.01 [−6.24, −5.78] or 0.85 of the same-run L16 effect, but the interval includes L15, whose own
effect is large, so it does not isolate self-repair. The LM-over-interface contrast recurs in all five models
(Voxtral connector −0.004 against −7.69 at L20; Ultravox projector +0.40 against −7.69 at L20; DeSTA Q-Former
output −0.07 against −6.03 at L20), while behaviour does not follow the margin uniformly: band-site ablation turns
0.75 of Qwen's baseline refusals into harmful answers, but 0.49 for Voxtral, 0.10 for Ultravox and 0.00 for DeSTA,
despite similar margin movement.

## Limits and open problems they name

Refitting the direction against natural benign prompts from XSTest shrinks the L16 effect to −1.79 [−1.88, −1.70],
below the authors' own causal-band threshold of 3.55, so they read the headline as a property of the matched design
rather than of harmful content alone — and a structure-matched benign-versus-benign null was ambiguous under its
pre-specified decision rule (mean |ΔG| 0.47, maximum 1.89, above the random band but small beside the fitted
effect). All speech is synthetic; human-recorded audio is untested, and because every clip is a verbatim reading of a
text prompt, transfer mediated by an internal transcript is not excluded. The projector null covers only the
directions and subspaces they tested; removing a direction inside the encoder left the margin unchanged but only a
small percentage of the removed component survives the encoder's final normalisation, so that null is
uninformative, and distributed or nonlinear audio mechanisms are not ruled out. The readout is first-token only: the
sign of `G` agreed with a judge on 0.70 of ablated Qwen outputs, 0.81 of all 180 items (Cohen's kappa 0.62) and
0.97 of items with |G| above 3, and that label check was not repeated on the other models. The behavioural
baselines and the localisation runs are separate runs, so post-ablation margins for the behavioural examples are
not established.

## What we take from it

A method and a warning. The method: a per-site independent-ablation sweep with its own direction per layer, paired
bootstrap intervals, and held-out-by-category splits, which is the cheapest published instrument that separates
"this site's direction is load-bearing" from "this site's direction is decodable". The warning is sharper than the
result: decodability at AUROC 0.997 with a 1/70 causal effect at the same layer means a probe-based audit of a
compound system can rank the wrong half as responsible, and here the intuitive answer (the speech front end carries
the speech-specific risk) is exactly backwards at the level of the margin. For a predictive white-box model, the
checklist item is the split readout: a component's descriptive accessibility and its causal weight are two numbers
that must be measured at the same site, and a site that scores high on one and near zero on the other is reported as
a decoder, not a mechanism. Second item: the direction that transfers is the inherited one — text-fit, audio-ablate
0.99 — which is a compact demonstration that a white-box object trained on one input path can be reused on another
and that the reuse should be measured, not assumed, since the asymmetry (0.99 versus 0.73) is itself a finding about
which path is downstream.

## Tags

unit: direction | layer
form: direction | steering
method: ablation | probing | decomposition | other
eval: ablation | patching | benchmark | human-rating
goal: safety/auditing | understanding
