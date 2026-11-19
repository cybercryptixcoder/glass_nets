# Glass Networks: the working checklist (read this first, every time)

*Living file. Updated at least every 10 minutes while work is going on. If you are a new session or just came back
from a context compaction, read this whole file, then the last 30 lines of `docs/GLASS_DIARY.md`, then continue at
the first unchecked step.*

Last updated: 2026-11-17 20:25 ET.

## What Glass Networks is

The project formerly called Helix Probe (constructor branch), renamed by Shreyas on 2026-11-17. The Helix name is
retired for this line of work. Branch: **`glass-networks`** (repo `cybercryptixcoder/helix_probe`, public).
`constructor-attempt` stays untouched as the fallback. `constructor-v2` is a stale pointer to this branch's first
commit (couldn't be deleted through the proxy; Shreyas can delete it).

**The job:** an LLM "constructor" reads a trained network and writes a **whole-model construction**, a small
Python program that stands in for the entire network, aligned to its insides. An independent judge certifies it
with measured tests. Verified constructions go into a public, playable **catalog**. The headline number is
**coverage**: how much of the network's behaviour and inside computation the construction accounts for, with the
leftover listed. First targets: ARC's AlgZoo (tiny RNNs; the 432-parameter 2nd-argmax RNN is unsolved and an open
challenge).

The plan, with each item tied to the failure it catches: `docs/V2_CONSTRUCTOR_PLAN.md`.

## Shreyas's instructions for this stretch (2026-11-17, 20:16 and 20:22 ET)

1. Implement everything first, cleanly: all the judge gaps and all the changes, then run.
2. Lean and task-specific: build only what a job needs. Never build first and justify later.
3. Compare our system with the Pitfalls paper's (Haklay et al.); comparison only for now.
4. Document everything in the repo: every step, decision, plan, what worked, what didn't, all reasoning.
5. Run on Google Colab: interruptible and resumable. Commit and push automatically every 2–5 minutes, including
   traces, programs, models, everything.
6. Build the catalog/playground (run the real model, run the construction, compare, test) alongside the runs,
   tested end to end.
7. Updates to Shreyas: a short one every 10–15 minutes; a cheap "continuing normally" check-in every 9 minutes
   (send_later); a fuller update every 30 minutes.
8. Always have something finished and usable on hand.
9. Keep this checklist and the diary current to within about 5–10 minutes.

## Standing rules (from earlier work; still apply)

- Keys only in gitignored `lab/.env` (`LUMINAL_API_KEY`, `LUMINAL_API_KEYS` = four keys). Never print, commit or
  push them. The repo is public.
- One process per Luminal key (several processes on one key cause 429 storms). Four keys → up to four runs in
  parallel.
- Never edit, rebase or push `constructor-attempt`.
- Colab: `export OAUTHLIB_RELAX_TOKEN_SCOPE=1`; CLI config in `~/.config/colab-cli/` (copied from
  `/home/claude/handoff_creds/colab-cli-config/`). Full how-to: `/home/claude/handoff_creds/README_FOR_AGENT.md`
  (private, not in the repo). Each `colab exec` under `timeout 100`, retry once. Ask before GPU spend; these runs
  need no GPU (tiny networks, the LLM is remote), so use a CPU runtime.
- "Saved" means verified on GitHub (remote head equals local head).
- Plain markdown to Shreyas; no artifacts or widgets.

## Open blockers

- **Direct Colab → GitHub push.** Adding a deploy key was denied by the permission system (not worked around).
  Asked Shreyas (20:22) for a fine-grained PAT (only this repo, Contents read/write). Until then: Colab writes
  everything to disk every 2 min, and the workspace relays it to GitHub at each check-in (≤ 9 min).

## Steps

Status: [x] done, [~] in progress, [ ] not started.

### A. Setup
- [x] A1. Four Luminal keys saved to `lab/.env`; each tested live (all OK, 20:20).
- [x] A2. Branch renamed `glass-networks` and pushed.
- [x] A3. Colab sign-in works (`colab sessions` → no active sessions, 20:19).
- [x] A4. AlgZoo repo reachable; checkpoints download over HTTPS from `storage.googleapis.com/arc-ml-public/alg/...`;
  torch 2.15 CPU is in the workspace.
- [x] A5. This checklist and `docs/GLASS_DIARY.md` created.
- [ ] A6. Rename in docs: README and START_HERE say Glass Networks (code names stay; renaming code is not needed
  for any job).

### B. Build (item numbers from `docs/V2_CONSTRUCTOR_PLAN.md`)
- [ ] B1 (item 0). Baseline: re-judge every existing non-LM verified construction; save verdicts to
  `results/glass/baseline_v1.json` for regression.
- [~] B2 (item 1). AlgZoo targets:
  - `nets.py`: kind `rnn_relu` (forward, hooks, `site_table`: `h{t}` = hidden state after step t, one stage per
    step; `out`);
  - `lab/targets/build_algzoo.py`: download checkpoints (2nd argmax ladder), weights.npz, meta.json with neutral
    `arch`, `context`, `answer`, inputs.npy (standard normal, as in AlgZoo training), labels.npy, `perm_groups`
    for the permuted control;
  - check: numpy forward equals torch to 1e-6 on 10k inputs.
- [ ] B3 (item 2). Judge: joint substitution (all named sites at once), with a pass bar.
- [ ] B4 (item 3). Judge: per-class and low-margin worst slice; network-error agreement when labels exist;
  disagreement search (continuous inputs), worst inputs saved.
- [ ] B5 (item 4). Judge: coverage line and unexplained list in every report.
- [ ] B6. Re-run B1's baseline after each judge change; record any verdict changes and why.
- [ ] B7. Constructor reader: confirm it runs unchanged on `rnn_relu` targets (kernel, prompts, tools). Change only
  what fails.
- [ ] B8. Comparison doc: `docs/GLASS_VS_PITFALLS.md`.

### C. Runs (Colab)
- [ ] C1. Colab driver: CPU runtime `glass1`; clone branch; upload `lab/.env`; a job script that runs the ladder,
  one run per key, resumable (constructor `--resume` or skip finished), writes a status file, and snapshots every
  2 min.
- [ ] C2. Relay (or direct push once there's a PAT): every check-in tars new results, downloads, verifies, commits,
  pushes.
- [ ] C3. Ladder: 2nd argmax M(h,n), smallest first: (2,2) 10p, (4,3) 32p, then larger, up to (16,10) 432p. Each
  with a blinded run, a prior-only control, and a shuffled control.
- [ ] C4. Results into `docs/RESULTS_GLASS.md` as they arrive.

### D. Catalog / playground
- [ ] D1. A catalog page in the lab server: list every construction with its certificate (verdict, coverage,
  worst slice, controls).
- [ ] D2. Playground: pick an entry, type or sample an input, run the real network and the construction side by
  side, see variables and the per-step hidden-state alignment.
- [ ] D3. End-to-end test of the page (server up, API calls checked, a screenshot or HTML check).

### E. Reporting
- [ ] E1. A check-in every 9 min (send_later), with the full instruction text in the message.
- [ ] E2. A fuller update every 30 min.
- [ ] E3. Final AlgZoo results to Shreyas.
