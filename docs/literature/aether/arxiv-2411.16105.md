# Adaptive Circuit Behavior and Generalization in Mechanistic Interpretability

- **Key:** arxiv-2411.16105
- **Authors / organisation:** Jatin Nainani, Sankaran Vaidyanathan (equal contribution, corresponding authors), AJ Yeung, Kartik Gupta, David Jensen; University of Massachusetts Amherst (author emails at umass.edu)
- **Date:** 2024-11-25 (v2 2024-12-05)
- **Link:** https://arxiv.org/abs/2411.16105
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Circuits are discovered, and then defended, on one prompt template. Interpretability wants them as the model's mechanism for a *task*, but nothing in the discovery procedure promises that the same subgraph still explains behaviour once the wording around the task changes. The authors want to know what actually happens to a circuit off its home distribution: are the same components reused, do they behave differently, or does the model switch to different components entirely? They take the best-characterised circuit in the field - the IOI circuit in GPT-2 small, which is claimed to implement a three-step algorithm (collect names, drop the duplicated one, output the remainder) - and build prompt variants that the algorithm, as stated, cannot solve. The larger goal is a test of external validity for circuits as hypotheses about the whole model.

## What they studied

GPT-2 small and the published IOI circuit (26 nodes). The unit is the **attention head plus the per-input-token path into it**, i.e. edges are indexed by (head, source token), so a head can have different causal effects depending on which name it is fed. Three prompt families: the original **base IOI** (subject S duplicated, indirect object IO unique), **DoubleIO** (both S and IO duplicated), **TripleIO** (IO three times, S twice). 200 generated prompts per variant, in both name orders (BABA and ABBA templates).

## How they did it

First, run the unmodified base IOI circuit on the variants under the standard mean-ablation knockout and compare its logit difference with the full model's - the circuit's faithfulness on the variants is the first result. Second, measure each circuit head's attention pattern on its own designated token positions and compare circuit against model, using two purpose-built ratios: a **confidence ratio** (attention to the token leading to the right answer over attention to the token leading to the wrong one) and a **functional faithfulness** score (attention to a token inside the circuit over attention to it in the whole model), with 50-sample confidence intervals. This isolates which heads changed behaviour and in which direction. Third, hand-trace the mechanism through the heads that moved, from the earliest head in the pipeline to the output. Fourth, run path patching exactly as in the original IOI work, but at token granularity, to discover fresh circuits for DoubleIO and TripleIO and measure their node and edge overlap with the base circuit. Finally, perturb name order to locate the point in the network that decides which duplicate gets suppressed, narrowing candidates by the difference between direct effects on the S and IO edges and then inspecting their attention patterns.

## What counts as an explanation here

A **circuit plus a mechanism description**: a subgraph of heads and token-level edges, annotated with what each head type does and how information flows between them. It is meant to be the model's algorithm, and the paper's test of the explanation is precisely whether the mechanism survives when the input stops satisfying the algorithm's premises. The paper's contribution to form is a named failure mode, **S2 Hacking**, which is an explanation of why the circuit's high score is an artefact of the evaluation procedure rather than a description of the model.

## How they judged success

Logit difference (log-probability of the IO token minus the S token) for model and circuit; faithfulness as the circuit's logit difference over the model's; a normalized faithfulness for the added-edge experiments; circuit size in nodes and edges; and node/edge overlap percentages between circuits discovered on different variants. Head-level behaviour is scored with the two attention ratios above and confidence intervals over 50 samples. There is no statistical test on any headline comparison.

## What they found

The base circuit **outperforms the model** on the variants that break its own algorithm: faithfulness 1.285 on DoubleIO and 2.586 on TripleIO against 0.895 on base IOI, with circuit logit difference 2.722 and 3.174 against model values of 2.118 and 1.227. Since the model itself still answers DoubleIO correctly about 89% of the time, the gap is not the circuit solving a harder task - it is the circuit being evaluated in a way that points it at the right answer. The mechanism, S2 Hacking: under mean ablation every path from every input token except S2 into the Duplicate Token heads is knocked out, so the duplicate machinery only ever sees the subject duplicate and drives attention onto S2 regardless of what else appears in the prompt. The heads carrying it are Duplicate head 3.0, Induction heads 5.5 and 5.9, and S-Inhibition head 8.6, where the circuit's confidence ratio exceeds 2 while the model's stays near 1. All other circuit heads behave approximately as on base IOI, with attention deviations typically within 0.05. The merged circuits reuse everything: **100% node overlap and 91.66% edge overlap for DoubleIO, 84.61% for TripleIO**, with 26 nodes in each and 110, 120, 130 edges - DoubleIO adds 10 edges and TripleIO 20, all of them new input paths from the extra duplicate IO tokens into the duplicate-handling heads, and restoring those paths raises normalized faithfulness to 0.765 and 0.778. The added edges have a *negative* causal effect when read alone, because they feed inhibition the name that is then suppressed; the model is worse on these prompts partly because it lets the duplicated IO inhibit itself. Finally, name order decides the outcome: performance is higher when IO appears first, and Previous Token head 2.2 implements a first-come-first-served rule, attending mainly to whichever name appeared first.

## Limits and open problems they name

One model, one task, one circuit; 200 prompts per variant; no intervention that removes S2 Hacking and re-measures faithfulness, so the paper demonstrates the artefact and infers its effect from the head-level ratios rather than by ablation. The duplicate-choice mechanism is studied at one head and explicitly left for future work. Nothing is said about MLPs, about larger models, or about circuits for tasks whose variants are less templated. The authors also note that the added IO edges are adopted because they restore agreement with the model, which formally means the "reused" circuit is being tuned toward the metric the paper criticises elsewhere.

## What we take from it

A **warning with teeth for our probing setup**, and the sharpest one in the folder: when a white-box object is scored inside a knockout harness, the harness's own selection of live paths can carry the answer. S2 Hacking is the general shape - mean ablation leaves one informative path open, the circuit reads that path, and the score improves precisely as the circuit diverges from the model. Mechanically for us: any surrogate or circuit our model publishes is scored on at least two ablations of *different* granularity (per-head and per-edge/per-token), and a score that rises above the model's own under either is reported as a harness artefact rather than as a strong result; the prediction we commit to is the one that survives the change of ablation, not the one that improves under it. Second, the paper's most reusable positive result is the edge-granularity economy: generalisation to a new input distribution is bought by **adding token-level input edges while reusing every node**, which is a concrete prior for a predictive model - keep the unit inventory fixed and predict which new edges carry the effect. Third, the order-of-mention result is a reminder that a single positional statistic can decide a whole subgraph's output, so any prediction of behaviour should carry the positional features as inputs.

## Tags

unit: head | circuit
form: description | graph
method: patching | ablation | attribution
eval: ablation | faithfulness | prediction/simulation | benchmark
goal: understanding | evaluation-of-methods
