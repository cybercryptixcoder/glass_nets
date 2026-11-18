# How we judge a construction: the evidence ladder, and "good to a stated tolerance"

*Nov 18, 2026. This revises the "rung 1–5" discussion with Shreyas (Nov 17) using what rounds 1 and 2 showed. It is
the reference for what the judge measures. Sources are at the end.*

## The framing: a construction is a heuristic, and we state its tolerance

A construction is not the network, so it can't be exactly right, except on tiny networks. Like a heuristic solution
to a hard problem, its value is **how close it gets, stated with numbers**. A good certificate reads like this:

> "This account is B bits long. It reproduces the network on F% ± e of held-out inputs (worst slice W%). The
> variables it places inside the network pass interchange tests at I%. The cheapest copy of the network that reaches
> the same F needs C bits, and the task rule alone reaches P%."

So it is two kinds of number:
- **How true:** rungs 1–5 below;
- **How much of an explanation:** rung X below.

**Rounds 1 and 2 showed you need both.** A copy is perfectly true. It passes every truth rung, so the truth rungs
alone can never separate a copy from an explanation. The Nov 17 version of this ladder missed that.

## Part 1: is it TRUE of the network?

| Rung | What is measured | What it catches | Status in our judge |
|---|---|---|---|
| **1. Behaviour** | Agreement with the network's decisions on held-out inputs, **with a confidence interval**; per output class; on the worst slice; on the network's own mistakes | A different algorithm; a systematic hole hidden by an average | Built (worst slice and mistakes added in Glass). Confidence intervals not yet reported |
| **2. Inside correspondence** | For each variable the construction places at a site: copy the site from input B into input A's run; does the network do what the construction predicts? (interchange accuracy) | Same outputs, different method (sine/cosine vs a counter; the month circle vs the line) | Built: substitution, interchange, partial and edit tests. Mappings are limited to affine and low-dimensional, which is essential: with unrestricted mappings, any network matches any algorithm (Sutter et al.). Random-network control: built |
| **3. Sufficiency and necessity** | Replace what the account says doesn't matter: nothing should change. Remove what it says matters: things should break | Accounts that name the wrong parts; backup components | Built (necessity, removal). Holding downstream sites fixed during knockouts (the Hydra fix) is not built |
| **4. Worst case** | Search for the inputs where account and network disagree most | Rare but systematic failures | Built for continuous inputs |
| **5. Proof** | A formal bound on the account's error | Everything, but only on tiny networks | Not built; out of reach above tens of parameters (Gross et al.) |

## Part 2: is it an EXPLANATION, not a copy?

| Rung | What is measured | Why it can't be gamed by rewriting numbers |
|---|---|---|
| **X. Information versus fidelity** (ARC's "surprise accounting") | Total cost = bits needed to write the account + bits needed to correct its errors (its remainder). Two references are measured the same way: **the copy frontier** (the network's own weights rounded to 1, 2, … 16 bits each, with the fidelity each achieves) and **the task rule** (what knowing the task alone gives). An account is an explanation to the extent that it sits well below the copy frontier at its fidelity | A copy in any disguise (decimals, rescaled, integers over a scale, value tables plus indices) still carries the information in the weights, so it lands on or above the copy frontier. **This has to be shown on our saved copies before use, not assumed** |
| **Y. Matching sampling** (ARC) | Used as an estimator of the network's accuracy, does the account beat random sampling for the same compute? ARC measures this by adding noise to the weights and averaging the squared error | A copy run on samples is just sampling, so it can't beat sampling. The strongest test of "understood", but it needs accounts written as estimators. A later step |

The old rule (counting literals) tried to approximate rung X by the *form* of the numbers. That is why each new
disguise beat it (see `REPORT_ROUND2.md`, corrections).

## Part 3: what a certificate can and can't say

- **Non-identifiability** (Méloux et al.). Several accounts can pass the same tests. A certificate says "consistent
  with the network under these tests", never "the mechanism". Where possible it names a rival account and shows it
  fails, as the month circle failed against the line.
- **Every number gets controls,** measured the same way:
  - shuffled weights and untrained networks: they set the floor;
  - the task rule (prior-only control): it sets what memory alone gives;
  - the copy frontier: it sets what no understanding gives.
- **No pass line is set from the results it judges.** Lines come from the controls.

## How others do it (checked Nov 18)

- **ARC (AlgZoo)** publishes the networks and **no grading harness**. They judge explanations with surprise
  accounting ("full understanding" = total surprise about equal to the bits used to train-select the model) and,
  in practice, mean squared error versus compute compared with sampling. The open challenge on M16,10 is stated that
  way: an estimate that matches sampling.
- **The Pitfalls paper** (Haklay et al.) scores an agent's text against human explanations with an LLM. Its own
  conclusion is that this evaluation is unreliable. It proposes an intrinsic test based on swapping components,
  which is the same family as our rung 2.
- **Causal abstraction and IIA** (Geiger et al.), **causal scrubbing** (Redwood), **adversarial circuit evaluation**,
  **compact proofs** (Gross et al.): rungs 2, 3, 4 and 5 respectively.

## Sources
- ARC, "AlgZoo: uninterpreted models with fewer than 1,500 parameters" (Jan 26, 2026): https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/
- Haklay et al., "Pitfalls in Evaluating Interpretability Agents" (arXiv 2603.20101)
- Sutter et al., "The Non-Linear Representation Dilemma" (NeurIPS 2025, arXiv 2507.08802)
- Méloux et al., "Everything, Everywhere, All at Once: Is MI Identifiable?" (ICLR 2025, arXiv 2502.20914)
- Gross et al., "Compact Proofs of Model Performance via Mechanistic Interpretability" (NeurIPS 2024)
- Redwood Research, "Causal Scrubbing" (2022)
