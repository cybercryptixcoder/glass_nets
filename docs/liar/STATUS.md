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
