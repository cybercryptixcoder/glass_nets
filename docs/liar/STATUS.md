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
