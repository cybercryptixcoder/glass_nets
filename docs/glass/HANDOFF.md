# Glass Networks: handoff to the next agent (written 2026-11-18 01:03 ET)

Read in this order:
1. this file;
2. `docs/glass/CHECKLIST.md` (Current state + checklist);
3. the last ~60 lines of `docs/glass/DIARY.md`;
4. `docs/glass/PLAN_ROUND2.md` (the approved plan and the pre-registered claim criterion);
5. `docs/glass/LEARNINGS.md` (E1–E10).

## Who and what
- Owner: Shreyas.
- Repo: `cybercryptixcoder/helix_probe` (PUBLIC). Branch: `glass-networks`.
- Never touch the fallback branch `constructor-attempt`.
- Goal: whole-model "constructions" (runnable programs standing in for a network), certified by an independent judge (`lab/judge.py`), plus a public catalog (`/catalog`).
- Shreyas's standing requirements:
  - plain English, plain markdown, no artifacts or widgets;
  - document every step, decision and failure in the repo;
  - updates every 10–15 min while working, a fuller one every 30 min;
  - "no patchwork": fix only what traces support, and show that the fix helps;
  - always have something finished on hand.

## UPDATE 2026-11-18 09:51 ET: READ REPORT_ROUND2 'CORRECTIONS' and LEARNINGS E12 first. The 04:10 'next round' list is superseded; the glass-3 rule there is retracted.

## UPDATE 04:10 ET Nov 18: round 2 is FINISHED (40/40) and fully synced

Analysis and report done: `REPORT_ROUND2.md`, E11. The glass2 runtime holds nothing unsynced and may be reclaimed. Your job is the report's 'next round' list. Ask Shreyas before spending.

## (Historical) round 2 on Colab
- Colab runtime **`glass2`** (CPU, high-RAM).
  - Repo at `/content/helix_probe`; job started 23:51 ET Nov 17.
  - It runs `scripts/glass/run_ladder.py --workers 4 --round2`: 40 jobs, state in `results/glass/round2/queue.json`.
- At 2026-11-18 01:03 ET: **12 of 40 done, 4 running.** Shreyas said (01:02 ET Nov 18): **let it keep running.**
- Check it with: `export OAUTHLIB_RELAX_TOKEN_SCOPE=1; timeout 100 colab exec -s glass2 -f scripts/glass/colab_status.py`.
- **How results reach GitHub (important):**
  - Colab has NO GitHub credential. On Colab, `colab_sync.sh` refreshes `/content/glass_snapshot.tgz` every 2 minutes.
  - Results reach GitHub ONLY when an agent runs `GLASS_SESSION=glass2 bash scripts/glass/sync_from_colab.sh` from a workspace that has git push access. That script downloads, unpacks, regenerates the RESULTS files, commits and pushes.
  - Run it every ~10 minutes. If nobody runs it, nothing is lost while the runtime lives (the snapshot is on Colab), but GitHub falls behind.
  - Colab runtimes have a maximum lifetime, so relay at least every few hours.
- The job is resumable. If the runtime dies, re-create it:
  - `colab new --high-mem -s glass2`, then `scripts/glass/colab_setup.py`;
  - upload `lab/.env`;
  - `scripts/glass/colab_start.py` (it resumes from queue.json: finished runs are kept).
- On Colab, never `git pull` or stash. Use `scripts/glass/colab_update_code.py`.
- Long `colab exec` calls hang: launch work detached and poll.

## Credentials (NOT in the repo; the repo is public)
- They live in the old agent's workspace only:
  - `~/.config/colab-cli/` (Colab sign-in);
  - `lab/.env` (`LUMINAL_API_KEY`, `LUMINAL_API_KEYS`: 4 DeepSeek V4.1 Flash keys; one process per key);
  - `/home/claude/handoff_creds/README_FOR_AGENT.md`.
- A new agent in a different workspace needs Shreyas to provide these again. Never commit, push or print them.

## Round 2 status by step (plan: PLAN_ROUND2.md, approved 23:36 ET Nov 17)
1. **Compare M4,3 with ARC's notes: DONE** (`M43_VS_ARC.md`, E9).
   - Our M4,3 is a blind *rediscovery* of Robinson's and Hilton's symmetric reduction, not a discovery.
   - New, checked by code: the reduced recurrence is a 120° rotation; the exact reduction loses no accuracy.
   - TODO: relabel M4,3 in README and the catalog as "blind rediscovery of ARC's account".
2. **Learnings: DONE** (E1–E10).
3. **Fixes F1–F4: DONE.** Judge `glass-2` is frozen for round 2; reader v2 is the default (`GLASS_READER`).
4. **A/B ladder, reader v1 vs v2: RUNNING.** So far M2,2 is certified by both readers; M4,3 is certified by both (v2 in 1 submission for $0.03, v1 in 5 submissions for $0.13).
5. **Longest-cycle discovery target: RUNNING.**
   - Every lc COMPLETE so far is an **E10 weight copy**: integers × 1/1000 or 1/10000, which gets past F1's "non-integer literal" count. Runs: lc_3_4 maina, lc_4_5 mainb, lc_6_5 maina.
   - **None is a claim.** lc_4_5 mainb has a partial real account worth following:
     - layer 0 is near-hard copying;
     - layer 1 at the readout is a soft max/min selector over the keys.

## Decision on record
- At 00:27 I offered Shreyas two options:
  - **A**: keep glass-2 frozen until the round ends;
  - **B**: fix E10 now as glass-3 and re-queue the lc runs.
- I recommended B. He replied "let the colab job keep running", which means **A is in force.**
- Re-ask him about re-running the lc targets under glass-3 once round 2 ends.

## What the next agent does
1. Relay results (`sync_from_colab.sh`) about every 10 minutes and check status.
2. Open every COMPLETE by hand and check it for E10. Copies are never claims.
3. Write each check in DIARY and keep CHECKLIST's Current state accurate to about 10 minutes. Commit and push.
4. When the queue is finished:
   - implement the E10 fix as judge `glass-3`. It is one rule: count every numeric literal except integers with |n| ≤ 10 and 0.5, whatever form it is written in.
   - re-judge ALL round-2 runs (`scripts/glass/rejudge.py`);
   - write the v1-vs-v2 A/B table under both glass-2 and glass-3;
   - check lc results against the claim criterion: COMPLETE, < 0.78 constants per dof, the prior control fails, no prior interpretation, a human read;
   - write `docs/glass/REPORT_ROUND2.md`.
   - If reader v2 does not beat v1, revert F3/F4.
5. Ask Shreyas before stopping `glass2` or spending beyond round 2.
6. Untracked `docs/GLASS_CHECKLIST.md` is not ours. Leave it uncommitted.
