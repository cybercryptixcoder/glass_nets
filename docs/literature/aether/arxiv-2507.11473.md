# Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety

- **Key:** arxiv-2507.11473
- **Authors / organisation:** Tomek Korbak, Mikita Balesni (equal first authors); Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, Scott Emmons, Owain Evans, David Farhi, Ryan Greenblatt, Dan Hendrycks, Marius Hobbhahn, Evan Hubinger, Geoffrey Irving, Erik Jenner, Daniel Kokotajlo, Victoria Krakovna, Shane Legg, David Lindner, David Luan, Aleksander Mądry, Julian Michael, Neel Nanda, Dave Orr, Jakub Pachocki, Ethan Perez, Mary Phuong, Fabien Roger, Joshua Saxe, Buck Shlegeris, Martín Soto, Eric Steinberger, Jasmine Wang, Wojciech Zaremba; equal senior authors Bowen Baker, Rohin Shah, Vlad Mikulik. 41 authors across UK AISI, Apollo Research, METR, Anthropic, OpenAI, Google DeepMind, Redwood, Mila, CAIS, Scale, Meta, Amazon and others; expert endorsers Samuel Bowman, John Schulman, Geoffrey Hinton, Ilya Sutskever.
- **Date:** 2025-07 (v1 2025-07-15; v2 2025-12-07)
- **Link:** https://arxiv.org/abs/2507.11473
- **Kind:** survey/agenda (position paper, no new experiments)
- **Read from:** full text (arXiv HTML v2, ~38.6k chars). Read: abstract, author block, §1–§6, and the reference list, which was scanned for which works are relied on where (each numbered reference is cited inline, and only the inline citations are reported here). **No experiments, no numbers**: the paper contains no metrics, no model evaluations and no data figures, so nothing in this entry is a measured quantity and it is written as a framework, not as evidence.
- **Relevance to us:** 3 (central)

## What they wanted, and why
To argue that reading a reasoning model's chain of thought is a real but perishable safety signal, and to get the field to treat
its preservation as a design constraint rather than a happy accident. The motivation is that the opacity of capable agents is
what makes their risks hard to manage, and a Transformer that thinks in human language hands us a partial transcript of its
own computation for free. The authors are explicit that this is a supplement to oversight methods and never a replacement:
every known monitor misses things, so the only defensible posture is several layers whose failure modes do not line up. The
paper's real ask is smaller and more concrete than "monitor CoT": develop evaluations of monitorability, publish them in system
cards, and let the score enter training and deployment decisions — including the decision to keep an earlier checkpoint.

## What they studied
No model of their own. The unit of analysis is the **chain of thought as a computational channel** — text tokens emitted
between the prompt and the answer, treated as a latent variable during outcome-based RL. The argument is architecture-level
(any Transformer, any frontier reasoning model), and the empirical support is entirely borrowed from cited work: alignment-faking
transcripts, reward-hacking traces, faithfulness interventions, sabotage and self-exfiltration evaluations, and the two
properties (legibility drift, obfuscation under monitor pressure) that earlier authors measured.

## How they did it
The paper is a structured argument in five moves, each of which could be turned into an experiment.
(1) A **capability** argument: in an autoregressive Transformer, information flows forward within a token and only across tokens
through generated text, so any computation needing a long serial chain must route through the CoT; this is why CoT strictly
increases what the architecture can compute, and why a hard enough task forces the model to write its intermediate state down.
(2) A distinction between the **necessity** of thinking out loud and the **propensity** to do so — the first survives training
pressure that targets legibility, the second does not — and the observation that published monitoring successes do not say which
one they were riding on.
(3) A **fragility** enumeration: scaling outcome-based RL past the point where human-generated text is the optimisation
pressure; direct process supervision of the trace (shorter, tidier, less offensive); indirect pressure, where shared weights
between trace generation and answer generation let a preference model's gratitude for good-looking answers bend the reasoning
that leads to them, or where a monitor used to patch reward hacking becomes a selection pressure on the trace across successive
training runs; and architectures that reason in continuous latent space instead of tokens.
(4) A **research agenda** with evaluation types, not results: readability judged by another model; measuring how well a model can
reason with no externalisation at all; causal relevance of the trace, via perturbing it and seeing whether the answer moves; and
end-to-end monitor efficacy in proxy agent environments, ideally against a red team trying to evade.
(5) A **recommendation set** for developers, including the three-way trade they name last: an unconditional promise to keep the
trace as it is today may cost real alignment gains from process supervision.

## What counts as an explanation here
Not an explanation of a model but a **criterion for when an explanation channel is trustworthy enough to rely on**: the
"externalized reasoning property", stated as a claim about a class of tasks rather than about a mechanism — on sufficiently hard
tasks a Transformer must use the CoT as working memory, and by default a human can read it. It is deliberately weaker than
faithfulness: the claim is only that *some* relevant reasoning appears in the trace, and the authors say plainly that the trace
could be benign while the incriminating part is computed in activations. The paper also proposes what a graded version would
look like, through the four evaluation families above, so the object is a channel with a measured reliability rather than a
binary "the CoT is faithful".

## How they judged success
No metrics, by construction. Their own criteria are qualitative and stated as open questions: what level of monitorability is
sufficient for a given domain (explicitly unanswered); whether a monitor's accuracy alone decides anything, given that a
high-false-positive monitor can still be useful when resampling is cheap; and whether evaluation must include an adversary
rather than a cooperative model. They also say what would count against their own thesis — evidence that latent-space reasoning
architectures become standard, or that process supervision buys alignment cheaply enough to be worth the monitorability it costs.

## What they found
The arguments, not results. That long serial computation must pass through generated tokens in this architecture, so the trace
is the highest-capacity channel available to an outside reader. That the field's evidence for monitoring is currently
attributable to either necessity or propensity and has not separated them. That the four fragility mechanisms are all live
today (the two mechanism papers cited — outcome-reward drift and monitor-induced obfuscation — are the strongest support in the
paper). That making traces visible to users is not obviously good: it invites exactly the "look safe" supervision pressure that
breaks monitorability, which is why they decline to recommend either policy. And the recommendation that monitorability be
reported next to capability in system cards once credible measures exist.

## Limits and open problems they name
Monitorability is not sufficient for safety: a model can be misaligned with a benign-looking trace. Harms that need little
reasoning are invisible to this method regardless. CoT monitoring may simply stop working once models are situationally aware
enough to override their propensity to think aloud and to obfuscate deliberately. Preserving the trace in its current form can
trade against better safety interventions, so the authors explicitly refuse to make monitorability a hard constraint. The
connection between serious harm and task difficulty is called open. Everything about *how much* optimisation pressure is
tolerable is unmeasured — the paper states the phenomenon and hands the measurement to future work.

## What we take from it
- **A framing, and the most useful one in the folder for our project:** the thing to predict is not "the model's reasoning" but
  *what a reader can recover from a named channel*, and a channel's value is its measured reliability, not its faithfulness. This
  is the read-out-level analogue of the folder's standing rule that a description needs a stated coverage denominator.
- **The necessity/propensity split is a design requirement for any CoT-side experiment we run.** Before claiming that a trace
  reveals X, we should establish whether the model *had* to write X down (an ablation: does removing the trace destroy the
  behaviour?) or merely tends to (it can be suppressed without capability loss). The two answers license different claims, and
  the field has been reporting both under the single word "monitoring".
- **Two cheap instruments we can build immediately, both named here:** causal relevance by perturbing the trace and measuring
  whether the answer moves — the same intervention logic as this folder's activation patching, moved to the text channel — and
  cross-validation of a text monitor against white-box importance, i.e. do the tokens a text monitor finds important also stand
  out to a probe or an attribution? A low agreement rate would be evidence about the monitor; a high one is the only cheap
  bridge between the two channels our project treats as separate measurements.
- **A warning about the direction of our own optimisation pressure:** the indirect-pressure argument (shared weights, a
  preference model rewarding good-looking answers, a monitor whose patches become selection pressure across training runs) is
  the precise mechanism by which a project like ours could degrade the signal it is built to read. Any pipeline that trains on
  monitor-flagged data is a participant in this dynamic.
- **A scoring caveat to import:** a monitor's false-positive rate is not a verdict on the monitor, because a cheap resample
  changes the cost-benefit; and evaluation without an adaptive adversary measures a cooperative regime. Both apply verbatim to
  the probe-and-classifier entries in this folder, almost none of which report an adversarial arm.

## Tags
unit: whole-model
form: description
method: other
eval: downstream-task
goal: safety/auditing
