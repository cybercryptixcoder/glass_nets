# Next-generation Constitutional Classifiers (summary post)

- **Key:** anthropic-com-research-next-generation-constitutional-classifiers
- **Authors / organisation:** Anthropic (post is unsigned; the underlying paper is Cunningham, Wei, Wang, Persic, Peng et al., arXiv:2601.04603, which already has its own entry here as `arxiv-2601.04603`)
- **Date:** 2026-01
- **Link:** https://www.anthropic.com/research/next-generation-constitutional-classifiers
- **Kind:** blog/research post (public summary of a paper already in this folder)
- **Read from:** full text of the summary page (~9.4k chars, including the "Remaining vulnerabilities" lead-in and the closing research-directions paragraph; one fetch, entire page, no figures present). The linked paper was **not** re-fetched — its numbers come from our existing `arxiv-2601.04603` entry and are marked as such where used.
- **Relevance to us:** 2 (useful) — it is the *deployment narrative* half of an entry we already own, and its value is that it states the design rationale in prose where the paper states it in tables. Treat it as the argument, not the evidence.

## What they wanted, and why
Two jailbreak failures in the original constitutional-classifier system are named and taken seriously rather than patched over: **reconstruction attacks**, which scatter a harmful request across apparently benign pieces and then ask the model to reassemble it, and **output obfuscation**, where the model is talked into relabelling dangerous things (their example: chemical reagents described as food flavourings) or into a riddle where harmful concepts are mapped onto innocuous ones. The authors' framing of why these matter is the interesting part: such attacks notoriously damage capability — GPQA Diamond performance fell from 74% to as low as 32% under some of them — but the drop varies a lot by attack, and they read that variance as a warning that someone will eventually find a high-capability-preserving version. So the goal is not "block these two attacks" but "stop depending on an architecture whose blind spot they exploit": separate input and output classifiers.

## What they studied
Production traffic and adversarial traffic for a deployed assistant (Claude Sonnet 4.5 traffic for the one-month refusal measurement; Claude Opus 4.0 traffic as the compute-overhead reference point). Three defence components measured against each other and then ensembled: an *exchange classifier* that scores a response in the context of its prompt; a *cascade* of a cheap first stage escalating to an expensive second stage; and an *internal probe classifier* reading the model's own activations. The unit of study is the deployed monitor system, not a feature or a direction — although the probe is the only one of the three that is a white-box read-out.

## How they did it
The exchange classifier is the conceptual fix: judge the pair, because a response that reads innocuous alone is obviously harmful next to the prompt that coded it. That worked (over half of successful jailbreak attempts cut in human red teaming) and cost about 50% more compute. The cascade then recovers the cost by making the first stage *allowed to be wrong in the safe direction* — a flagged exchange is escalated to the stronger classifier rather than refused, so the cheap stage may run a high false-positive rate without turning it into user-visible refusals. The internal probe is the layer that reuses computation the model is already doing: activations at each generation step encode an assessment of the request before any response has been committed, and a linear read-out of those states is nearly free at inference. Their stated reason the probe is worth having is not only cost but **attack asymmetry** — an adversary can shape inputs to steer the eventual text, but shaping the intermediate representation is a different and harder problem — plus empirical complementarity: the probe and the external classifier each caught things the other missed.

## What counts as an explanation here
An engineering rationale, not a mechanistic one: "these attacks share a structural cause, this architecture removes that cause, cost is recovered by asymmetry between a cheap screener and an expensive judge." The probe is justified by an information-access argument (internal states carry the harm assessment earlier than the output does) rather than by identifying what the probe direction means. No claim is made about what the direction is.

## How they judged success
Final production configuration: a linear probe screens everything, flagged exchanges go to a probe-plus-classifier ensemble. Three numbers carry the claim — a 0.05% refusal rate on harmless queries over one month of deployed Sonnet 4.5 traffic, described as an 87% reduction versus the original system; roughly 1% compute overhead against Opus 4.0 traffic; and an adversarial campaign of over 1,700 cumulative red-teaming hours across 198,000 attempts yielding one high-risk vulnerability, a detection rate of 0.005 per thousand queries which they call the lowest of any technique they have evaluated, with no universal jailbreak found. There is no ablation here isolating how much of the robustness came from the probe versus the exchange classifier versus the cascade — that is in the paper, and the summary does not attempt it.

## What they found
- Exchange-level judgement is the biggest conceptual win and it is *not* free: halves the successful-attack count at +50% compute.
- The cascade converts the false-positive problem into a routing problem, which is the reusable design move: a cheap screener that is allowed to over-flag is strictly better than a cheap screener tuned to refuse, because escalation is not a user-visible cost.
- A linear probe over intermediate activations is cheap enough to run on all traffic and adds robustness the external classifiers do not have — reported as complementarity in both directions, not as "the probe beats the classifier".
- The two named residual attack families are honest about the shape of the remaining risk: the vulnerability is not a smarter prompt, it is a change of *representation* of the same information (fragmented, relabelled, allegorised). Anything that scores surface content is structurally matched to fail on these.
- Their deferred directions are all interpretability-flavoured: feed classifier signals back into the model's own generation, train the model to resist obfuscation, automate red-teaming to generate the fine-grained boundary examples the constitution lacks.

## Limits and open problems they name
They say plainly that no on-market system is perfectly robust and that a universal jailbreak has not yet been found rather than that it cannot be. The capability-degradation numbers are quoted from jailbreaks generally, not from their own attack suite, so the "attackers will find a strong version" claim is an argument by variance, not a measured result. My added limits: this page intentionally carries no ablations and no per-attack breakdowns, so any causal attribution of the numbers to the probe is *not* supported by this text alone — read `arxiv-2601.04603` for that; the refusal rate is a one-month, one-model, one-traffic-mix statistic with no confidence interval in the summary; and the "harder to fool internally than externally" claim is stated without an adversarial attack on the probe itself, i.e. it is a plausibility argument whose test would be precisely a gradient-based attack on the read-out.

## What we take from it
- **The one architectural idea worth stealing outright:** allow the cheap read-out to over-fire and route the excess to an expensive judge, instead of tuning the cheap read-out for precision. Our own survey keeps hitting the same trade-off in miniature — a probe threshold that must trade recall for refusal — and this is the standard production answer: make escalation free, make refusal rare.
- **The concrete deployment number for the folder's recurring claim that a one-direction linear probe is a real instrument:** screening 100% of traffic at ~1% compute overhead with a 0.05% harmless-refusal rate is a stronger existence proof than any lab benchmark in this corpus, precisely because it is boring, deployed and month-long.
- **A named structural failure mode to carry into every surface-level monitor we evaluate:** reconstruction, relabelling and allegory defeat content scoring because the harmful information is present in an altered *form*. Any probe we build should be tested on an input where the same information is split or renamed — if the read-out survives only the original surface form, it was matching text, not state.
- **The complementarity claim, kept as a claim:** "the probe sees what the external classifier can't, and vice versa" is exactly the kind of statement that needs an adversarial probe attack to earn. Note it, don't generalise from it.

## Tags
unit: direction | whole-model
form: probe | classifier
method: probing
eval: human-rating | ablation
goal: safety/auditing | deployment
