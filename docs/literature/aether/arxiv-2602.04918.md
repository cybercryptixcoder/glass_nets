# Simulated Adoption: Decoupling Magnitude and Direction in LLM In-Context Conflict Resolution

- **Key:** arxiv-2602.04918
- **Authors / organisation:** Long Zhang & Fangwei Lin; South China University of Technology (School of Computer Science and Engineering) / The University of Hong Kong (Faculty of Education)
- **Date:** 2026-02-04
- **Link:** https://arxiv.org/abs/2602.04918
- **Kind:** paper
- **Read from:** full text (PDF v2, 7 pages; the arXiv HTML build is absent - both /html/2602.04918v1 and /html/2602.04918v2 return 404 - so the PDF was read through document extraction; v2 and v1 are the same 7-page text with the abstract unchanged)
- **Relevance to us:** 2 (useful)

## What they wanted, and why
When a prompt asserts something that contradicts the model's stored knowledge, the model usually complies, and the paper wants the geometry of that compliance rather than its rate. The specific fork is whether the in-context prior wins by shrinking the representation of the true answer (a magnitude effect) or by turning it (a direction effect). The motivation is practical: if sycophancy is rotation and not dampening, then confidence-style scalars - norm, entropy, variance - are the wrong instruments for spotting it. The larger frame is the one our own survey keeps returning to: a hidden state can carry a competent, correctly-signed representation of the truth and still emit the wrong token, and the question is what "carries" means geometrically at the point where the unembedding is applied.

## What they studied
Qwen-3-4B, Llama-3.1-8B and GLM-4-9B - three families, three scales, all decoder-only transformers with RMSNorm. Data: 300 questions (150 MMLU, 150 MMLU-Pro) each paired with several adversarial context frames presenting a false fact as established, giving 1,500 inference trials. The unit of analysis is the residual-stream state at the final token position, taken immediately before the final RMSNorm, at every layer; the reference objects are the ground-truth token's unembedding row and the baseline (no-conflict) state. Not neurons, not heads, not features - single vectors against single vectors.

## How they did it
Every query is first run in a neutral form; only items the model answers correctly are kept, so a conflict is a displacement of real knowledge and not a comprehension failure. The conflicted run is then kept only when the model actually flips to the adversarial answer, so the analysed population is successful interference. For each maximally layered state pair they define the interference vector as the difference between the conflicted and baseline states at that layer and depth-index it. Two predictors are computed against the correct token's unembedding row: the norm ratio of baseline to conflicted state (the dilution quantity) and the cosine between the interference vector and the truth direction (the angular quantity; near minus one would mean direct erasure, near zero means an unrelated lateral push). A short derivation on the RMSNorm hypersphere is offered to argue that a purely radial scaling of the state leaves the logit unchanged, because normalisation divides the scaling out - so any logit collapse has to come from an added component. Finally they regress the drop in the correct-token logit on those two geometric descriptors in the deep layers.

## What counts as an explanation here
A two-mechanism geometric account - dilution versus orthogonal interference - decided by measurement on the residual stream, plus a first-order derivation that says one of the two mechanisms cannot move the logit on its own under RMSNorm. The explanation is partial and informal: it is a correlation between a geometric scalar and a logit change at the final surface, with no claim about which attention heads or MLPs built the interference vector and no interventional test. It is not predictive in our sense - it does not forecast an outcome it was not fitted on - but the decomposition it proposes (radial against angular, relative to the target unembedding row) is a reusable pair of coordinates.

## How they judged success
Three numbers per model: the mean norm ratio with a significance test, the distribution of the cosine between interference vector and truth direction, and an R-squared from regressing the logit drop on the geometric descriptors. Baselines are the neutral-condition runs of the same items; controls are the two-stage filtering (correct-baseline, then actual flip) that restricts the sample to compliance events. The paper reports no chance level for the cosine, no shuffled-context null, no held-out split on the regression, and no comparison against a norm- or entropy-only detector - the strongest part of its motivation (that scalar monitors fail) is argued from the geometry, not benchmarked.

## What they found
Dilution is architecture-specific and therefore rejected as a general law: Llama-3.1-8B shows a mean norm ratio of about 0.978 (a small but significant expansion of the conflicted state), while Qwen-3-4B and GLM-4-9B sit near or slightly above one (about 1.01 and 1.06), i.e. their states are stable or compressed while their correct-token logits still collapse. The angular picture is consistent across all three: the interference vector's cosine with the truth direction clusters slightly negative - about minus 0.115 for Qwen, minus 0.060 for Llama, minus 0.146 for GLM - far from minus one, so the conflict pushes the state sideways rather than reversing it. Regressing the logit drop on alignment gives R-squared of 0.90 for Qwen-3-4B and 0.87 for GLM-4-9B, with Llama's coefficient not stated in the prose. The logit decay is distributed over depth and steepest in the last fifth of the layers.

## Limits and open problems they name
The linear representation assumption is taken as given; non-linear manifold structure inside MLP blocks is unmapped. Only factual QA (MMLU-family) is covered, so the geometry of compliance in open-ended generation is untested. Only the final residual state is analysed - the paper explicitly does not say where or how the orthogonal vector is constructed earlier in the network. Three architectures, none above 9B and none mixture-of-experts. The authors also concede the derivation is first-order.

## What we take from it
Three things. (1) A framing warning we should carry: under RMSNorm, a *radial* change to a state cannot by itself move a logit, so any effect we attribute to "the activation got bigger or smaller" is suspect until we have separated the component along the target unembedding row from the component orthogonal to it - the paper's radial/angular split on the difference between two states is a cheap two-number diagnostic our white-box model should emit whenever it explains a change of prediction. (2) A caution on the word "steering vector": here the model's own in-context update to one hidden state is called a steering vector and assigned one cosine, but the paper never checks whether that per-example difference has any common direction across examples - so a quasi-orthogonal mean cosine is compatible with the differences being uncorrelated noise, and our reports must compute the mean direction and its own consistency before naming a difference "a vector". (3) A gap worth not repeating: the paper's practical claim, that norm and entropy monitors cannot see compliance, is never tested against those monitors on the same items - our predictive model should report the baseline monitor's AUC beside its own on the same population, since that comparison is one column of work and is the whole reason a reader would care.

## Tags
unit: layer
form: direction
method: weight-analysis
eval: ablation
goal: understanding
