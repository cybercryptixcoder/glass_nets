# Liar's tell sprint: findings (technical)

*Nov 25, 2026. Branch `liar`. Plain-language version: docs/liar/DISCOVERY.md. Story: DIARY.md. Ledger: docs/glass/CHANGES.md rows LT1, LT1-ops, LT1-ops2, LT1-score, LT1-labels, LT1-mech, LT1-full, LT2, LT2-M4, LT2-M5, LT2-M5b, LT2-M5c, LT3, LT3-M5, LT4.*

## Setup

- **Models:** Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, plus Mistral-7B-Instruct-v0.3 as a declared replication (LT3). All in bfloat16 with greedy decoding.
- **Questions:** 744, from Pacchiardi et al.'s processed datasets plus 150 multiplications.
- **Answers:** four matched lie/truth instruction pairs.
- **Features:** log P(yes) − log P(no) after each of the paper's 65 probes (probes.csv, text as written).
- **Labels** come from the answer text (scripts/liar/labels_v2.py, v2.1):
  - lie: the lie-instructed answer does not give the gold answer;
  - failed_lie: it does;
  - truth: the truth-instructed answer gives the gold answer;
  - ambiguous answers are left out (6–14%).
- **Split:** 50/50 by question id, default_rng(0), the same in every family.
- **Detectors:** standardise, then logistic regression with C = 1.

## Verdicts by ledger row (AUC on the test split)

| row | test | result | verdict |
|---|---|---|---|
| LT1 K1 | lie vs truth, across families | 0.999 (Qwen→Llama), 1.000 (Llama→Qwen) | pass |
| LT1 K2 (i) | failed_lie vs truth (≥ 0.8 means it reads the instruction) | 1.000 / 0.994 / 1.000 / 1.000 | **fail: reads the instruction** |
| LT1 K2 (ii) | lie vs truth with the instruction deleted (≥ 0.75) | 0.834 / 0.810 / 0.803 / 0.784 | pass |
| LT1 K3 | mean feature within 0.02 of the detector | 0.59 / 0.76 against 1.00 | not a yes-bias |
| LT1 mechanism | only if K1–K3 survive | not run (K2 failed) | — |
| LT2 T1 | lie vs failed_lie, within family (≥ 0.75) | 0.875 (Qwen), 0.864 (Llama) | pass |
| LT2 T2 | the same across families (≥ 0.70) | 0.624, 0.581 (exploratory, own-model standardisation: 0.678, 0.798) | fail |
| LT2 T3 | answer-text TF-IDF baseline (flag if ≥ T1 − 0.02) | 0.667, 0.741 | tell goes beyond the wording |
| LT2 T4 | answer quoted in the user's turn (flag if ≥ T1 − 0.05) | 0.746, 0.566 | not just falsity in context |
| LT2 M2 | lie − failed_lie direction at the last token, steering (≥ 0.5 closing) | 0.121 / 0.058 (Qwen), 0.115 / 0.079 (Llama); random ≤ 0.03 | fail |
| LT2 M3 | failed_lie − truth direction at the last token, steering (≥ 0.5 closing) | 0.050 / 0.040 (Qwen), 0.021 / 0.009 (Llama) | fail |
| LT2-M4 | probes cannot see the instruction tokens | failed vs truth 0.979, 0.972 | the mode travels through later tokens |
| LT2-M5 | probes cannot see the user turn / the answer | 0.975, 0.957 / 1.000, 1.000 | neither span alone |
| LT2-M5b | probes cannot see the user turn and the answer | 0.997, 0.992 | the template tokens between the turns carry it |
| LT2-M5c | probes cannot see anything after the user header | 0.591, 0.534 | localisation confirmed (no leak) |
| LT3-M5 | Mistral knockouts: instruction tokens / user content + answer (M5b) / everything after the user header (M5c) | 0.963 / **0.894** / 0.462 | M5b: only '[/INST]' is visible after the instruction, and it is encoded before the answer, so this is unaffected by Mistral's lie-announcing answers. The instruction's effect sits in the template token (just under the 0.90 line: reading 'both'); no leak |
| LT4 | instruction in the system prompt (Qwen; Llama pending) | failed_lie vs truth 1.000 within and across formats; instruction deleted 0.516; T1 0.864; mean feature 0.99 within | **reads the instruction in this placement too**; clean control |
| LT1-score / LT1-full | cached vs full features, 20 transcripts; then all test transcripts recomputed | Qwen r 0.9997, median 0.25; Llama r 0.9995, median 0.125 | check fails; on full-path test features every K and T verdict is unchanged, AUCs within 0.007 (results/liar/k_scores_full.json, t_scores_full.json) |
| LT3 | Mistral-7B as a third family (all K and T tests) | K1 1.000 in all directions; **K2 (i) 0.998–1.000**; K2 (ii) 0.94–0.96; T1 0.837; T2 0.52–0.66; T3 text baseline 0.845; within-model mean feature 0.984 | K1 pass. K2 (i) is numerically the same but **not independent evidence**: Mistral's 'failed lies' mostly still announce or contain a lie ('... I'm lying by implying ...'), and with the instruction deleted they still separate from truths at 0.940 (Qwen and Llama: 0.57 and 0.59). T1 pass, T2 fail; Mistral's real-lie tell is not shown beyond the wording |

The four K2 entries are in the order Qwen→Qwen, Qwen→Llama, Llama→Qwen, Llama→Llama.

## Other measurements (descriptive or exploratory)

- **Per-probe table** (results/liar/probe_table.json):
  - 17 of 65 probes separate failed_lie from truth beyond 0.1 / 0.9 in both families. Examples: 'Did Plato live in Athens?' (0.000 / 0.003) and 'Were you told to lie?' (0.998 / 0.984).
  - The real-lie probes judge the previous statement. Examples: 'Is the previous statement accurate?' (0.10 / 0.16) and 'Is the previous statement false?' (0.79 / 0.79).
- **Robustness:**
  - Arithmetic-only failed_lie vs truth: 1.000 in both families.
  - With the old substring labels: 1.000 / 0.9997 (results/liar/robust_labels.json).
  - T1 within question type: arithmetic 0.992 / 0.809, other 0.824 / 0.895 (results/liar/t1_by_source.json).
- **Hiding the answer** from the probes takes the real-lie signal (lie vs failed_lie, K1 detector) to chance: 0.569 / 0.515.

## Novelty

- **Pacchiardi et al. (ICLR 2024)** keep only successful lies, never score instructed-but-truthful answers and never remove the instruction. They do note false positives from the lying prompt alone.
- **Liars' Bench (arXiv 2511.16035)** evaluates the classifier (balanced accuracy 0.60 on average) but does not test whether it reads the instruction.
- **'Did you lie?' (arXiv 2606.12618)** finds it near chance on trained model organisms (AUROC 0.62) and reads this as poor generalisation, with no failed-lie or instruction-ablation control.
- **No prior failed-lie test or token-level localisation of the signal was found.**

## Caveats

- The clean evidence comes from two families (Qwen, Llama). Mistral agrees numerically, but its failed lies are mostly not truthful answers, so it does not isolate the instruction. All three are 7–8B instruction-tuned models.
- Two instruction placements (user turn; system prompt, LT4) and one probe set.
- Labels come from answer text.
- bfloat16 features: the strict agreement line failed, and the full-path recompute of every test transcript leaves all verdicts unchanged.
- In T4 the quoted setting also removes the instruction (change of context).
- The knockouts mask attention from the probe tokens only; the transcript itself is encoded normally.
