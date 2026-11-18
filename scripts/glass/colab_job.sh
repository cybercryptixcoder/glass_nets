#!/bin/bash
# The Glass Networks job on a Colab runtime: the run queue plus a sync loop. Started detached by colab_start.py:
#   setsid nohup bash scripts/glass/colab_job.sh >> /content/glass.log 2>&1 < /dev/null &
# The sync loop is scripts/glass/colab_sync.sh (a snapshot every 2 minutes; pushes too when a token is set).
# Restarting this script continues the queue (run_ladder.py resumes interrupted runs).
cd /content/helix_probe || exit 1
echo $$ > /content/glass_job.pid
bash scripts/glass/colab_sync.sh &
SYNC_PID=$!
echo "== $(date -u) job start (sync pid $SYNC_PID)"
python3 scripts/glass/run_ladder.py --workers 4
echo "== $(date -u) pass 1 finished"
python3 scripts/glass/run_ladder.py --workers 4 --pass2
echo "== $(date -u) queue finished"
kill $SYNC_PID 2>/dev/null
bash scripts/glass/colab_sync.sh --once
echo "== $(date -u) job end"
