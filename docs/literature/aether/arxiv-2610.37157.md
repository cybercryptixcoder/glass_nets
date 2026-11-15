# What Survives Distillation? A Mechanistic Interpretability Study of GPT-2 and DistilGPT-2

- **Key:** arxiv-2610.37157
- **Authors / organisation:** Nabeelah Faumi, Deshan Sumanathilaka; Informatics Institute of Technology, Colombo
- **Date:** 2026-10-27
- **Link:** https://arxiv.org/abs/2610.37157
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The standard evidence that distillation worked is that the student's perplexity and benchmark scores look like the teacher's, and the standard worry is that this hides a student computing something different. The authors make the worry concrete by picking the two circuits in GPT-2 that have been most thoroughly validated by independent causal work — indirect object identification and greater-than year comparison — and asking what a six-layer student does with mechanisms that live in layers 9 to 11 of a twelve-layer teacher. Two questions, stated plainly: whether the student's representations stay similar to the teacher's after the depth is halved, and whether the computations survive, and if they do, where the student put them. The motivating claim is worth quoting in substance rather than form: they argue the interesting failure is not a lower score but a *wrong sign*, i.e. a model actively misled where the teacher was right, because that failure is invisible to any aggregate metric.

## What they studied

GPT-2 small (12 layers, 768-wide, 12 heads, ~117–124M depending on the count they use) as teacher; DistilGPT-2 (6 layers, ~82M, same width, initialised from every other teacher layer) as student. Unit of analysis is the attention head and the block, with the MLP treated at block level, grouped into the published IOI and greater-than circuits. Data is a thousand TinyStories validation samples filtered to 64–256 tokens, plus the IOI prompt set (1,000 clean and 1,000 corrupted prompts from 15 templates) and the year-comparison prompt set (1,000 clean, 1,000 corrupted). Tooling is TransformerLens on one GPU, with pre-trained GPT-2 sparse autoencoders (res-jb, from Conmy et al.) applied to the student's activations as a *fixed borrowed basis*.

## How they did it

Four representational probes: debiased linear CKA on mean-pooled residual-stream activations, under two layer-matching schemes (same index, and proportional, mapping student layer d to teacher layer round(11d/5)); a tuned logit lens giving per-layer next-token accuracy and a student/teacher relative-accuracy ratio; attention-pattern statistics (per-head entropy, top-k mass, BOS/previous-token/EOS weights, and cosine similarity between flattened attention matrices at matched layers); and four SAE statistics in a single streaming pass (activation rate, L0 change, per-feature Pearson survival, and a magnitude drift ratio). Then one causal instrument per circuit: activation patching of a head's clean output into the corrupted run, scored as logit difference for IOI and as normalised greater-than probability mass for GT, with paired t-tests, Wilcoxon tests and Bonferroni correction, plus Cohen's d; direct logit attribution over cached head outputs as the non-interventional complement; and, for the IOI name-mover question, a matching procedure — the top-20 student components by direct attribution on a 500-prompt discovery set, confirmed on 500 held-out prompts, then compared to each teacher name-mover by CKA, by attention-pattern correlation, and by a name-mover score defined as the expected excess attention to the indirect object over the subject. The head-availability census comes first: of 24 published IOI circuit heads, 19 sit in layers 6–11 and have no counterpart at all, leaving 7 heads and 3 inter-head paths within the student's depth, and of 6 GT heads only the two year-encoding heads at layers 0 and 1 exist.

## What counts as an explanation here

A component-level mechanism claim: which heads exist in the student, what they attend to, how much causal work each does, and where the teacher's late-layer computation reappears. The explanations are partial and comparative — a circuit diagram per model plus a per-head role assignment plus a score for whether a student head reproduces a teacher role. The paper is explicit that this is a failure-mode catalogue rather than a predictive model of the student: it names "circuit compressibility" as the candidate explanation for why one circuit survives and the other does not, and then says that confirming it as a general principle is the most important extension.

## How they judged success

Logit difference and accuracy on both tasks against a stated chance level (one-sided binomial on 1,000 IOI prompts); normalised patching scores with a noise ceiling (0.547 for GT) so that recovery is read as a fraction of the available effect; direct logit attribution magnitudes compared across models; CKA and Pearson correlation across seeds {42, 123, 7}; head-successor and head-copy scores by projecting each head's output–value contribution into vocabulary space against a model-wide head average (3.29% successor for the teacher, 4.89% for the student); SAE feature survival and reconstruction-error ratios; and a threshold sweep on the circuit-extraction cut (0.10 to 0.30) reporting circuit size, completeness and faithfulness at each. There is no shuffled-prompt null, no random-init student, and no scrambled-circuit baseline; the negative evidence in the paper comes from the IOI arm itself.

## What they found

Representations are similar early and diverge late, but only under the second mapping: proportional CKA is 0.9998 at the first layer pair, 0.967 mid, 0.790 late (mean 0.921 ± 0.109); same-index CKA collapses to 0.314 at layer 5 (mean 0.752 ± 0.298), and the recovery under proportional matching is the paper's evidence that the student compressed the trajectory rather than truncating it. Per-head attention-pattern cosine follows the same pattern (0.894 ± 0.086 proportional against 0.631 ± 0.255 same-index). The logit lens gives a correspondingly split picture: under proportional matching the student's relative accuracy exceeds the teacher's at every layer but the last (up to 2.10), and under same-index matching it is behind everywhere.

SAE survival on the borrowed dictionary is low and matching-dependent: 0.086 survival at same-index layer 5 with reconstruction error 34.9× the teacher's; under proportional matching the mid and late pairs recover to about 0.53 survival and 0.927 cosine with magnitude drift 0.51–0.55, while the early pair G1/D0 is near zero (0.0002 survival, cosine 0.184).

IOI collapses. Teacher 94.8% accuracy, mean logit difference 2.942 ± 1.797; student 35.2% accuracy, mean logit difference −0.467 ± 1.335, below the two-way chance level at p < 0.001 — a sign reversal rather than uncertainty, and the authors note without testing it that a preference for the more frequent name would produce exactly this. Direct attribution confirms no component carries the behaviour: peak 0.0168 in the student against 2.74 at teacher layer 9. Matching finds student heads with moderate representational similarity (CKA 0.18–0.52) and pattern correlations from about −0.15 to 0.78, and **none of them reproduces IO-selective attention: name-mover preservation is 0.0% for every match.**

Greater-than survives, by remapping. Student 87.3% accuracy against the teacher's 94.0%, GT probability mass 0.750 against 0.855 — 87.7% of teacher performance. Attributions shift from the teacher's layer 9 to the student's layer 4, and the student's layer-4 head 1 reproduces the teacher's layer-9 head 1 attention pattern at mean Pearson r = 0.945 ± 0.079 (median 0.981; 93.6% of examples above 0.8), with both heads attending the two-digit year token in all five tabulated examples. The two available year-encoding heads keep strong causal effect in the student when patched (normalised 0.517 and 0.465 against 0.606 and 0.557 in the teacher; overall d = 0.681, p < 0.0001). Path patching was not run on GT because no inter-head path has both endpoints inside layers 0–5.

Perplexity, for the paper's central point: 11.54 for the teacher and 17.55 for the student on TinyStories (+52.1%), 33.02 and 49.33 on WikiText-2 (+49.4%) — a student that is plainly a worse language model while keeping one circuit almost intact and losing the other.

## Limits and open problems they name

One teacher–student pair for the deep analysis; the IOI null does not exclude the student solving a residual version of the task by another mechanism, and they recommend causal scrubbing or automated circuit discovery to look for one; the borrowed GPT-2 SAE basis is blind to student-only features, and student-specific dictionaries would be needed to see them; GT survival is established at one matched head, and a real account would need the year encoding, the comparison and the output computation verified as a chain; the GT arm had no valid paths to patch. They also note in an appendix that they characterise distillation but never intervene in it, and that circuit-aware or intermediate-layer-matching objectives might rescue deeper circuits — untested.

## What we take from it

A measurement discipline more than a method. (1) *The head-availability census is a pre-registration device.* Before scoring preservation, they count which circuit members even exist in the student (19 of 24 IOI heads absent; 4 of 6 GT heads absent), which fixes in advance what can and cannot be claimed, and turns the rest of the analysis into a statement about a known-available subset. Any cross-model or cross-checkpoint mechanistic comparison we run should publish that census first, because it converts "the mechanism is missing" from a finding into an arithmetic fact. (2) *A competence floor gates mechanism claims.* The IOI arm is a student at 35.2% accuracy with a negative logit difference; the correct reading is that the mechanism claim is untestable rather than refuted, and the paper's own limitation says as much. Our reports should carry accuracy and a chance level next to every circuit score, and refuse the mechanism sentence when the behaviour has already failed. (3) *Pattern similarity is not function transfer, and this is the cheapest demonstration of it in the folder:* CKA up to 0.52 with 0.0% name-mover preservation, on the same heads. Keep it as the standing counterexample whenever a cross-model cosine is offered as evidence.

## Tags

unit: head | circuit | layer | feature
form: description | graph | probe
method: patching | ablation | probing | SAE | decomposition
eval: ablation | patching | faithfulness | completeness
goal: understanding | evaluation-of-methods
