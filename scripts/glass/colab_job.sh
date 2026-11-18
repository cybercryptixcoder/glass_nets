#!/bin/bash
# The Glass Networks job on a Colab runtime: the run queue plus a sync loop. Started detached by colab_start.py:
#   setsid nohup bash scripts/glass/colab_job.sh >> /content/glass.log 2>&1 < /dev/null &
# The sync loop runs every SYNC_SECONDS (default 120):
#   - always: a snapshot of everything the runs produce, /content/glass_snapshot.tgz (written to a temp name, then
#     moved, so a download never gets a half-written file);
#   - if /content/glass_push.ok exists (a GitHub token was configured by colab_setup.py): commit and push those
#     files to the branch, rebasing on the remote first.
# Restarting this script continues the queue (run_ladder.py resumes interrupted runs).
cd /content/helix_probe || exit 1
SYNC_SECONDS=${SYNC_SECONDS:-120}
PATHS="results/glass lab/constructions/zoo_2nd_argmax_*"

sync_once() {
  tar czf /content/glass_snapshot.tmp.tgz $PATHS 2>/dev/null && mv /content/glass_snapshot.tmp.tgz /content/glass_snapshot.tgz
  if [ -f /content/glass_push.ok ]; then
    git add -A $PATHS 2>/dev/null
    if ! git diff --cached --quiet; then
      git commit -q -m "glass auto-sync $(date -u +%H:%M:%S) from Colab" || true
    fi
    for i in 1 2 3; do
      git pull -q --rebase origin glass-networks && git push -q origin HEAD:glass-networks && break
      git rebase --abort 2>/dev/null; sleep 5
    done
  fi
}

( while true; do sync_once; sleep "$SYNC_SECONDS"; done ) &
SYNC_PID=$!
echo "== $(date -u) job start (sync pid $SYNC_PID)"
python3 scripts/glass/run_ladder.py --workers 4
echo "== $(date -u) queue finished"
sync_once
kill $SYNC_PID 2>/dev/null
echo "== $(date -u) job end"
