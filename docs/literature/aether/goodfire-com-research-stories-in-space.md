# Meandering on Manifolds: The Neural Geometry of Stories Over Time

- **Key:** goodfire-com-research-stories-in-space
- **Authors / organisation:** Eric Bigelow, Raphaël Sarfati, Daniel Wurgaft, Owen Lewis, Thomas McGrath, Jack Merullo, Atticus Geiger, Ekdeep Singh Lubana; Goodfire (Wurgaft also Stanford)
- **Date:** 2026-06-23
- **Link:** https://www.goodfire.ai/research/stories-in-space
- **Kind:** blog/research post (part of Goodfire's Neural Geometry series; links to an arXiv paper that was not read here)
- **Read from:** full blog post (all sections; figures and the interactive demos not inspected; the linked arXiv paper was not read, and every methodological detail below is at the level the post states it)
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Earlier posts in the series describe the static geometry of a concept — one forward pass, one snapshot, one curved surface in activation space. Here they ask what happens when that geometry is followed over time as the model reads a long input, using emotion in short stories as the case study. The motivation is that a model's internal state is not a single reading but a sequence of readings, so an account of representation that only says where "emotion" sits in space cannot say how a model's sense of what is happening updates sentence by sentence. The larger goal is to combine two existing frameworks: the Bayesian view in which each new piece of evidence reweights latent hypotheses, and the conceptual-spaces view in which those hypotheses are regions of a geometric space. They want to describe a story as a path through belief states rather than as a point, and they frame this as necessary work for debugging and deliberately designing models.

## What they studied
Llama 3.1 8B, described by the authors as a small model, reading short stories drawn from SimpleStories. The unit of analysis is the residual-stream activation at the last token of each sentence — a per-sentence snapshot of the model's state after reading the story so far — plus the model's own verbal emotion report at the same positions. Two coordinate systems are in play: a six-dimensional behavioural space of basic emotions (surprise, disgust, anger, happiness, sadness, fear, each rated 0–10), and a learned manifold fitted to activation space and then compared against the human valence-arousal model of emotion from psychology.

## How they did it
The behavioural side is close to a probe with the model as its own annotator: after each sentence of a story the model is prompted to score the story so far on each of the six basic emotions, and those six numbers form a vector treated as a readout of what the model believes has happened. Those readouts are then mapped against the classic valence-arousal space built from human judgements of the same emotions, to see whether the model's arrangement of emotion words matches the human one. The representational side repeats the same forward passes without any emotion question: activations at the last token of each sentence are harvested, a manifold is fitted to them, and each story becomes a trajectory on that manifold. The two sides are run in parallel so the behavioural trajectory (how the reported emotions move) can be compared with the geometric one (how the activation moves). They argue the internal representations track the same dynamics by construction, since they state that probes over the internal representations, without asking about emotions, are highly accurate.

## What counts as an explanation here
A geometric account: a manifold whose coordinates correspond to conceptual dimensions (here emotion, organised approximately by valence and arousal), plus a trajectory through that manifold as a description of how the model's state evolves. It is a partial account — the manifold is fitted, the trajectory is a description of a path, and the post does not claim the path is the computation. It is predictive in a limited sense: given a position you can say which emotion region the model is in, and the ordering of positions is claimed to mirror the story's arc. It is not a circuit, a dictionary or a causal model.

## How they judged success
The evidence named in the post is qualitative and comparative rather than metric-heavy. Three checks appear: whether the arrangement of six basic emotions in the model's activation space matches the arrangement from human judgement data in valence-arousal coordinates; whether the per-sentence behavioural readouts visibly follow the story's arc in the demos; and whether the fitted manifold's trajectories line up with the behavioural readouts, supported by the claim that probes of the internal representations are very accurate. No baselines, ablations, held-out splits or numeric agreement statistics are reported in the post; those, if any, sit in the linked paper.

## What they found
In Llama 3.1 8B the six basic emotions in activation space arrange themselves in a way the authors compare directly to the human valence-arousal layout, which they read as the model having internalised a structure present in its training text. Asked after each sentence, the model's emotion ratings move over the course of a story, so the readout traces the arc rather than staying flat. Harvesting activations at sentence ends without asking anything shows those activations also move, and the post's demos plot the resulting trajectories along the fitted emotion manifold. The synthesis the post offers is that each moment of a story corresponds to a belief state — a distribution over the conceptual space — and the story is a path through such states; they present this as a route to debugging and design rather than as a finished result.

## Limits and open problems they name
The authors say a great deal more work is needed to characterise belief updates and temporal dynamics in general, and that the emotion case is deliberately simple. Their own framing puts the gap exactly where we care: there is still a wide gap between structured conceptual representations and how models update their beliefs at inference time. They also flag the general trust question of whether a model's verbal report reflects its actual state, answering it here by appeal to separately-fitted internal probes rather than by an intervention. Nothing in the post establishes that the manifold's geometry is causal, and the demos are illustrative rather than measured.

## What we take from it
Three things. First, a framing: a per-position reading is a trajectory, not a point, and our probes should be run at every position on a long input and reported as a path, because a single-token probe average hides the dynamic this post is about. Second, an argument structure worth copying — a behavioural readout and a geometric readout measured on the same forward passes — which is the cheapest available sanity check that an internal object is tracking the same quantity as an observable one, and which we can strengthen by making the internal side a probe we fit and test on held-out stories. Third, a warning about the weakest link: both sides of their comparison are readouts of the same model, so a correlation between them does not distinguish "the model represents this" from "both readings are downstream of the same surface features". The corrective for us is a causal test — steer a direction and check the trajectory bends, or intervene on the story and check both readings change together — which this post does not run and which our white-box claims need. Cross-reference this folder's `arxiv-2511.00617` (belief dynamics as the shared basis of in-context learning and activation steering), which is the same group's causal treatment of the same question and is the entry to trust over this one for mechanism claims.

## Tags
unit: direction
form: direction
method: probing
eval: human-rating
goal: understanding
