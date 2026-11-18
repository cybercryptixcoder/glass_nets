# Glass Networks: the working checklist

**Read this first after any interruption or context reset.** This file is the instruction sheet and the to-do list.
Its companion `docs/glass/DIARY.md` is the time-stamped log of every step, decision, result and failure. Keep both
accurate to within about 10 minutes: update them at every check-in and after every finished step, then commit and
push.

Project: **Glass Networks** (the new name; the Helix name is retired for this line of work). Branch:
**`glass-networks`**, cut from `constructor-attempt` at `d6b706ff`. `constructor-attempt` stays untouched as the
fallback. (A stale `constructor-v2` branch on GitHub points at an earlier commit of this branch. It couldn't be
deleted from here; Shreyas can delete it.)

## What Shreyas asked for (Nov 17, 20:16 ET), in his terms

1. Implement everything first, cleanly: all the gaps found in the literature review (the plan:
   `docs/V2_CONSTRUCTOR_PLAN.md`).
2. Compare our system with the Pitfalls paper's system (Haklay et al.). A comparison only, for now.
3. Document everything in the repo: every step, decision, plan, what worked, what didn't, and the reasoning.
4. Run on Google Colab. The runs must be interruptible and resumable. Everything (traces, programs, models,
   details) is committed and pushed automatically every 2–5 minutes, so nothing is lost when a runtime is reclaimed.
5. Run the constructor on the AlgZoo networks, and report the results.
6. In parallel: the catalog/playground of constructions in the lab interface. Run the real network and the
   construction, see the results, and test it end to end.
7. Updates to Shreyas every 10–15 minutes while building. Once the runs are going, a check-in every 9 minutes (a cheap
   "continuing normally" is fine) and a fuller update every 30 minutes.
8. Always have something finished and usable on hand, not just a report.
9. Principle: everything is built for a stated job. No bells and whistles; lean and clean, without costing
   effectiveness.
10. Keep this checklist and the diary current to within about 10 minutes (added 20:26 ET).

## Standing rules

- **Keys:** the four Luminal keys live only in `lab/.env` (gitignored), as `LUMINAL_API_KEY` and `LUMINAL_API_KEYS`
  (comma-separated). Never commit, push or print them. On Colab, upload the same file to the same path.
- **Colab:** CLI `colab` (sign-in in `~/.config/colab-cli/`, always `export OAUTHLIB_RELAX_TOKEN_SCOPE=1`). Full
  instructions are in Shreyas's private `README_FOR_AGENT.md` (in the workspace: `/home/claude/handoff_creds/`).
  Each `colab exec` runs under `timeout 100` and is retried once.
- **GitHub from Colab:** waiting for a fine-grained token from Shreyas (asked 20:20 ET). Adding a deploy key
  from here was blocked, and I'm not working around that. Until the token arrives: Colab writes everything to
  disk, and the workspace copies it to GitHub at each check-in and while I'm working.
- **Never edit `constructor-attempt`.**
- **Ask before** any spend beyond what Shreyas approved. Approved (20:16 ET): the AlgZoo runs on Colab using the
  four keys.
- **Lean:** every new piece names the failure it catches or the job it serves (see the plan's table).

## Current state (22:13 ET)

- **All runs done** (pass 1: 18 jobs; pass 2: 5). Final report: `docs/glass/REPORT.md`. Results:
  `results/glass/RESULTS.md`. Certified: M4,3 (blinded); M2,2 (prior-only, 0 constants). Not explained: M4,4, M8,5,
  M8,8, M16,10 (best: 97% behaviour, 1/10 stages, 45% of mistakes).
- The Colab runtime `glass1` was stopped at 22:46 ET on Shreyas's word. Nothing is running.
- Next: Shreyas decides. The proposed next work is in REPORT.md ("What's next"). To run again: `colab new --high-mem -s glass1` and follow the restart steps above.

## Checklist

Status: [x] done, [~] in progress, [ ] not started.

### A. Setup
- [x] Four keys saved in `lab/.env`; each tested with a live call (all OK, 20:18)
- [x] Branch `glass-networks` pushed
- [x] Colab sign-in works (`colab sessions` → no active sessions)
- [ ] GitHub token from Shreyas for direct Colab pushes (relay used meanwhile)

### B. Build (CPU, in the workspace)
- [x] **Item 1, AlgZoo targets.** `nets.py` kind `rnn_relu`; `lab/targets/build_algzoo.py` (downloads checkpoints over
  https from `storage.googleapis.com/arc-ml-public/alg/zoo/`, converts them to `weights.npz` + `meta.json`, with
  inputs ~ N(0,1) as in AlgZoo's training and task labels); the numpy forward matches torch to 1e-6.
  - Facts: `OneLayerRNN` is `torch.nn.RNN(input_size=1, relu, bias=False)` plus `Linear(h, n, bias=False)`, reading
    the final hidden state. Parameters = h + h² + n·h (M2,2 = 10; M4,3 = 32; M16,10 = 432). 2nd argmax: hidden
    sizes {2,3,4,5,6,8,16,32} × seq lengths {2,...,10}. Best-of-5 seeds are in `alg_zoo/zoo.py`.
- [x] **Item 0, baseline:** re-judge the existing verified constructions; save their verdicts in
  `docs/glass/baseline.json`; re-check after every judge change.
- [x] **Item 2, joint substitution** in `lab/judge.py`.
- [x] **Item 3, worst slice, the network's own errors, and the disagreement search** in `lab/judge.py`.
- [x] **Added (found while building): test 7, "not a copy"** (`copy_check`): rejects constructions that write out the weights.
- [x] **Added: the tie fix** in `decide()` (round to 1e-9).
- [x] **Item 4, coverage line and unexplained list** in the report.
- [x] The reader's prompt and kernel run on `rnn_relu` targets unchanged (a short live test on M2,2). The prompt describes tests 6–7.
- [x] Comparison with the Pitfalls system: `docs/glass/COMPARISON_PITFALLS.md`.

### C. Runs (Colab)
- [x] A CPU runtime `glass1` (high-RAM, 8 CPUs). Repo at `/content/helix_probe`, log `/content/glass.log`, pid file `/content/glass_job.pid`, snapshot `/content/glass_snapshot.tgz` (every 2 minutes).
- [x] Driver `scripts/glass/run_ladder.py`: a queue of runs, one key per worker (4 workers), each run resumable,
  results under `results/glass/<target>/<run>/`, a sync to disk every 2 minutes.
- [x] Ladder (started 20:50 ET, finished 22:05 ET): 2nd argmax M2,2 → M4,3 → larger, up to M16,10 (432 params). Each with the prior-only control and a
  shuffled-weights control.
- [x] **Pass 2: started by hand at 21:35 ET.** The live job (`colab_job.sh`, started before pass 2 existed) ends after
  pass 1. When `colab_status.py` shows the job is NOT RUNNING, run `colab_update_code.py`, then `colab_start.py`.
  The new `colab_job.sh` runs pass 1 (all done, skipped) and then `--pass2`: a fresh main run for each network not
  certified by the current judge (M4,4 so far), and a continuation of the M16,10 run to 160 steps.
- [x] Results table: `results/glass/RESULTS.md` (regenerated after every job and at every relay).
- [x] Re-judge every saved construction with the final judge (`scripts/glass/rejudge.py`, CPU, about 25 minutes),
  then commit the `report_current.*` files so the catalog shows today's certificates.
- [x] Final report: `docs/glass/REPORT.md`.

### D. Catalog / playground (in parallel with C)
- [x] A catalog page in the lab server (`/catalog`; `lab/catalog.py`, `lab/static/catalog.html`): the list of constructions with their certificates; for a chosen input, run
  the network and the construction side by side; run the judge.
- [x] End-to-end test (headless Chrome, 21:18): list, playground, intervention, no JS errors.
- [x] Re-judge all saved constructions with the current judge (`scripts/glass/rejudge.py`); `report_current.*` committed.
- [x] README and START_HERE: say Glass Networks; how to open the catalog (`python3 lab/server.py`, then http://localhost:8765/catalog).

### E. Monitoring
- [ ] Check-in every 9 minutes (`send_later`): Colab alive? runs progressing? sync to GitHub; update this file and the
  diary; a short message to Shreyas.
- [ ] Every 30 minutes, a fuller update.

## Colab operations (quick reference)

- Status: `colab exec -s glass1 -f scripts/glass/colab_status.py` (under `timeout 100`; retry once).
- Pull the data: `colab download -s glass1 /content/glass_snapshot.tgz /tmp/glass_snapshot.tgz`, then
  `tar xzf /tmp/glass_snapshot.tgz -C /home/claude/repo_probe`, commit and push (`scripts/glass/sync_from_colab.sh`).
- Restart after a reclaimed runtime: `colab new --high-mem -s glass1`; launch a detached clone (`/tmp/cclone.py`
  pattern: Popen with `start_new_session=True`, because a long `colab exec` hangs); upload `lab/.env`; copy the last
  snapshot's `results/glass` and `lab/constructions/zoo_*` back (upload and untar); then `colab_start.py`.
  `run_ladder.py` resumes the interrupted runs.
- Long execs hang. Anything longer than about 30 seconds is launched detached and polled.
- Code updates on the live runtime: `colab exec -s glass1 -f scripts/glass/colab_update_code.py` (checks out code paths only).
  **Never `git pull` or `git stash` there**: the run folders on Colab are newer than any commit.

## How to resume after an interruption

1. Read this file and the last 30 lines of `docs/glass/DIARY.md`.
2. `cd /home/claude/repo_probe && git fetch -q origin && git status`: the branch should be `glass-networks`.
3. `export OAUTHLIB_RELAX_TOKEN_SCOPE=1; colab sessions`: is the runtime there?
4. Continue from the first unchecked box. Schedule the next check-in.
