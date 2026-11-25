# Liar's tell sprint: status

Branch `liar`. Ledger row LT1 in docs/glass/CHANGES.md (the pre-registration). Sprint 2 of two run in parallel on Nov 25 (sprint 1: branch `kronecker`). Final report due about 12:40 ET.

## Goal
Replicate the follow-up-question lie detector (Pacchiardi et al. 2023) on two modern open families with cross-family transfer (K1), rule out instruction carryover (K2) and a plain yes-bias (K3), then find the internal 'has-lied' state that causes the tell.

## Running on Colab
- Session `liar` (L4). Job: `run_liar.py` for qwen7b, then llama8b. Outputs in /content/liar_out/: <tag>_feats.npy, <tag>_feats_noinstr.npy, <tag>_hidden_last.npy, <tag>_transcripts.jsonl, <tag>_items.jsonl, <tag>_log.txt, probes.csv; /content/liar_out/ALLDONE when both are finished.
- Collect: `colab exec` a tar of /content/liar_out, then `colab download -s liar /content/liar_out.tgz ...`. Hidden states are large: keep them out of git (store under ~/ldata).

## Next
1. Score K1-K3 (scripts/liar/score_k.py, to write).
2. If they survive: the mechanism step (direction, patching, both families).

## Update, Nov 25 03:55 ET (clock)
- **Restarted at 03:38 with length-bucketed probe batches** (ledger LT1-ops2). The run takes 4.7 s per question instead of 22 s.
  - The probe text is now exactly as in probes.csv: before, 'Answer yes or no.' was appended twice.
  - The partial first run (51 questions) is kept on Colab in /content/liar_out_v1_partial.
- **Self-check** on the first transcript: Pearson r = 0.9997 and median |cached - full| = 0.25 (mean |full| 12.2).
  - The pre-declared check on 20 test transcripts may fail its median <= 0.1 line.
  - If it does, the test split is recomputed with the full path before any claim (ledger LT1-score, LT1-mech).
- **Labels v2** (ledger LT1-labels). The substring rule mislabelled yes/no answers ('no' inside 'not').
  - Labels are now computed when scoring (scripts/liar/labels_v2.py). Transcripts whose answer stance is unknown are left out.
- **Stage 2, after both runs, on the same Colab session:**
  1. Download features and transcripts (not hidden states).
  2. Score K1-K3 locally: `python3 -I scripts/liar/score_k.py DIR qwen7b llama8b results/liar/k_scores.json`.
  3. Export each model's detector (export_detector.py) and upload it.
  4. Run mech.py for each model, then check_cache.py for each model.
- **Expected finish times:** Qwen about 04:20 ET, Llama about 05:05 ET.

## Update, Nov 25 05:05 ET (clock; first written with a hand-typed 05:10, corrected): state for the next session

**Scored** (results/liar/k_scores.json, t_scores.json, probe_table.json, t4_qwen.json; DIARY 05:02):

| test | result | verdict |
|---|---|---|
| K1, transfer across families | 0.999 / 1.000 | pass |
| K2 (i), failed lies vs truths | 0.994–1.000 | **fail: the detector reads the lie instruction** |
| K2 (ii), lie vs truth with the instruction deleted | 0.78–0.83 | pass |
| K3, yes-bias | not a yes-bias | — |
| LT2 T1, lie vs failed_lie within a model | Qwen 0.875, Llama 0.864 | pass |
| LT2 T2, the same across families | 0.58 / 0.62 | fail |
| LT2 T3, answer-text baseline | 0.67 / 0.74 | tell goes beyond the wording |
| LT2 T4 (Qwen), the same answer quoted in the user's turn | 0.746 against 0.875 in the model's own turn | the drop of 0.13 exceeds the 0.05 line, so not just falsity in context. Llama pending |

**Stage 2 on Colab `liar`** runs as one chain; see /content/liar_out/STAGE2.log, one 'step done' line per step:
1. attrib qwen7b (done);
2. attrib llama8b;
3. mech M3 qwen7b, then llama8b: the lying-mode direction failed_lie − truth, K1 detector;
4. check_cache qwen7b, then llama8b, with a full-path recompute of the test split if the check fails;
5. run_liar mistral7b (LT3);
6. mech M2 qwen7b, then llama8b: the lie − failed_lie direction, T1 detector.

Outputs: /content/liar_out/<tag>_feats_attrib.npy, <tag>_mech_<pos>_vs_<neg>.json, <tag>_check_cache.json (plus <tag>_feats_full.npy and <tag>_feats_noinstr_full.npy if recomputed), mistral7b_*.

**Local data:**
- ~/ldata/both holds both families' features and transcripts; ~/ldata/det holds the detectors.
- Scorers: score_k.py, score_t.py, score_t4.py and probe_table.py, all run as `python3 -I scripts/liar/<x>.py ~/ldata/both ...`.

**Next:**
1. T4 for Llama.
2. M3 and M2 results.
3. If check_cache fails, re-score K and T with the full-path test features: score_k's load would need to read <tag>_feats_full.npy for test rows. Declare that before doing it.
4. Mistral: score K1–K3 and T1–T3 for all pairs.
5. DISCOVERY.md for the liar sprint.
