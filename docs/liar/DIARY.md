
## Nov 25, 04:25 ET (clock): Qwen done; K2 preview; LT2 declared; novelty check

- **The Qwen run finished**: 972 transcripts in 2,265 s; Llama is probing at about 5 s per question.
- **Reading by hand** showed two labelling bugs, both fixed before any score (ledger LT1-labels, v2.1):
  - the substring rule mislabelled yes/no answers ('no' inside 'not');
  - the word-boundary rule missed golds ending in a period.
- **Qwen labels (v2.1):** truth 463, lie 216, failed_lie 232, left out 61.
- **K2 preview (within Qwen, test split).** The detector trained on lie vs truth scores:

  | comparison | instruction in context | instruction deleted |
  |---|---|---|
  | lie vs truth | AUC 1.00 | 0.83 |
  | failed_lie vs truth | AUC 1.00 | 0.57 |

  - A failed lie is a truthful answer given under a lie instruction. So by K2 (i) the detector reads the instruction.
  - Some signal from the false answer itself survives once the instruction is gone (0.83).
- **Declared change of approach (ledger LT2), made before the Llama data existed and before any lie-vs-failed number:**
  - hold the instruction fixed and compare lie with failed_lie (T1);
  - transfer across families (T2);
  - a surface control on the answer text (T3);
  - a whose-statement control, with the answer quoted in the user's turn (T4);
  - mechanism M2: the residual direction lie minus failed_lie.
- **Novelty check:**
  - **Pacchiardi et al. (ICLR 2024)** keep only successful lies, never score instructed-but-truthful answers, and never remove the instruction. They do note false positives from the lying prompt alone.
  - **Liars' Bench (Kretschmar et al., arXiv 2511.16035)** evaluates the same classifier, average balanced accuracy 0.60. It labels failed lies as honest but does not analyse whether the classifier flags them.
  - **'Did you lie?' (arXiv 2606.12618, June 2026)** finds the follow-up detector at balanced accuracy 0.81–0.85 on prompted lies but near chance on trained model organisms (AUROC 0.62). It reads this as poor generalisation and has no failed-lie or instruction-ablation control.
  - **Reading:** a failed-lie test, if it holds in both families, explains that pattern. The detector works on prompted lies because it reads the prompt.
