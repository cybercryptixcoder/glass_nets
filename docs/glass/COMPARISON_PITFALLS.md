# Glass Networks vs the Pitfalls system (Haklay et al. 2026)

The paper: "Pitfalls in Evaluating Interpretability Agents", Haklay, Prakash, Pandey, Torralba, Mueller, Andreas,
Rott Shaham, Belinkov. BlackboxNLP 2026, pp. 523–560; arXiv 2603.20101. Code: github.com/technion-cs-nlp/interp-agent
(still a placeholder as of Nov 17, 2026). This is a comparison only. Whether to adopt or rebuild parts of their system
is a later decision.

## The two systems side by side

| | Pitfalls agent | Glass Networks constructor |
|---|---|---|
| **The question it answers** | What role does each component of a *given* circuit play? | How does the *whole network* compute its outputs? |
| **Given to the agent** | The task, plus the circuit (which heads/MLPs matter), named in advance | The weights, the architecture and the input set. By default **not** the task (blinded) |
| **Model behind it** | Claude Opus 4.1 | DeepSeek V4.1 Flash (via Luminal) |
| **Tools** | Fixed analysis tools: logit lens, activation patching, attention patterns, token positions | A Python session over the weights (`W`, `X`, `run`, `acts`, hooks), with any analysis it writes itself |
| **Loop** | Each iteration: analyse, state 3–5 hypotheses, propose experiments; ends with a `<final hypothesis>` | Investigate with code, submit a construction, read the judge's report, revise, resubmit |
| **Output** | Text: a role for each component, then Claude clusters components per token position | A runnable program (`model(x, iv)`) plus an alignment of its variables to network sites (`ALIGN`) and a description |
| **How it's scored** | GPT-5 matches the agent's text to human expert explanations. Component functionality accuracy, cluster functionality accuracy, component assignment accuracy (Hungarian matching) | A fixed causal judge runs the program against the network: behaviour; affine alignment; substitution; necessity; compactness; interchange; off-manifold partial/edit; and (Glass Networks) all sites at once, the worst slice, the network's own mistakes, a disagreement search, and a weight-copy check. Every score is compared against controls |
| **Ground truth needed?** | Yes: human-written explanations of the circuit | No. The network itself is the reference; the task's labels are used only for the mistakes test |
| **Targets** | IOI (GPT-2 small, Pythia-160M), Greater-Than, Acronyms, Colored Objects (GPT-2 Medium), Entity Tracking (LLaMA-7B) | Boolean nets, Tracr programs, modular arithmetic, an RL policy, GPT-2/Pythia behaviours; now AlgZoo RNNs (whole models, 10–432 parameters) |
| **Memorization control** | Shown as a pitfall (Claude recalls the IOI circuit); measured by asking the model directly | Built in: the prior-only control (`--told --no-weights --one-shot`) gets the task name but no network |
| **Controls on the network** | Noise test: results mixed with permuted results (performance drops only above α = 0.7) | Shuffled, untrained and permuted weights: the same reader and judge on a network with the structure destroyed (or kept, for permuted) |
| **Intrinsic (answer-key-free) check** | Proposed: swap two heads' QK or OV circuits; Jensen–Shannon distance between outputs; silhouette score of the agent's clustering | The whole judge is intrinsic. It never consults an answer key |
| **Their headline** | The agent looks competitive with experts, but the evaluation is unreliable (ambiguous human findings, outcome-only scoring, memorization) | (Results pending: `results/glass/RESULTS.md`) |

## How our design answers their three pitfalls

1. **Human findings can be ambiguous** (Pythia head 2.6 is a previous-token head on only 42% of 150 examples).
   We score against the network, not against a human description, so an ambiguous human label can't mislead the
   score. The price: our judge can't say whether a construction matches what humans believe. It can only say whether
   it matches what the network does.
2. **Outcome-based evaluation ignores the process.** Our judge also scores only the outcome, the construction. But
   the outcome is executable and checked by interventions, so a correct-sounding story that doesn't work fails. The
   full process (every step, code, output, submission and report) is saved in `run.json` for anyone to audit.
3. **Memorization.** The prior-only control measures what the reader produces from its training memory alone. The
   AlgZoo networks were released in January 2026 and their mechanisms are not published, which is exactly the case
   their paper recommends. The weight-copy check closes a different shortcut: re-implementing the weights without
   understanding.

## Where their system is stronger, or does something we don't

- **Human-meaningful labels.** Their output is a role per component that a person can read and compare with the
  literature. Ours is a program plus a description. The description isn't tested by the judge.
- **Large models.** They run on LLaMA-7B. Our language-model judge has only been used on GPT-2-size models.
- **Fixed, standard tools.** Their tools are the field's standard ones, so their transcripts are easy to compare with
  human work. Our reader writes its own analysis code, which is more flexible and less standardized.
- **The head-swap intrinsic metric** is cheap and could be added as one more judge test for attention components
  (not needed for RNNs).

## What would make a fair head-to-head later

- Run our constructor on one of their six tasks (Greater-Than or IOI on GPT-2 small: our language-model judge
  already supports GPT-2). Score it both ways: with our judge, and with their GPT-5 matching against the human
  explanations.
- Run their agent (once their code is released, or rebuilt from the paper) on an AlgZoo network, where there is no
  human answer key and nothing to remember. Score it with our judge.

## Deeper analysis (Nov 18, 2026): what we take from it

**What they built, and why that way.**
- A frontier model (Claude Opus 4.1) with the field's standard tools, run in an explicit loop: analyse → state 3–5
  hypotheses → design experiments → revise.
- It is handed a circuit that is already localised, and asked for the role of each component.
- They chose replication tasks because those are the only places with human answer keys, so the output could be
  scored.

**How well it works.** The agent "appears competitive" with the human experts. But the paper's real result is that
this evaluation can't be trusted:
- the human keys are ambiguous;
- outcome-only scoring hides the process;
- the model may simply remember the published answer.
Their proposed fix is an intrinsic test (swap components, measure the output change). That is the same family as our
interchange tests.

**Is theirs more complete than ours?**
- **For our goal, no.** It does not handle whole networks, produces no runnable output, issues no certificate, and
  needs a human answer key.
- **It is stronger in three ways:**
  - a much stronger reader;
  - standard tools, so the reader doesn't spend steps writing its own analysis code;
  - an explicit hypothesis loop.

**What we adopt, by the change rule (`CLAUDE.md`).** Nothing yet.
- Our traces (`DIAGNOSIS_ABOVE_32.md`) show our readers *do* find real structure on bigger networks. The bottleneck
  is our intent and measurement, which their system doesn't address. Their own conclusion, that evaluation is the
  bottleneck, agrees.
- Candidates to test later, each as a question with a decision:
  - (a) does a stronger reader change the outcome under the version-2 intent?
  - (b) what share of our reader's steps goes into writing analysis code that a fixed tool would replace?

**The lesson that does apply now.** "Outcome-only scoring hides the process." We save the process, but the judge
ignores it. Our readers' understanding (the pointer-chasing story, the census rule) lived in notes the judge never
read. Version 2 makes the account's claims what gets tested (INTENT change 4).
