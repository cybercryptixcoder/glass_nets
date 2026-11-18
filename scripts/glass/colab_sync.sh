#!/bin/bash
# The Colab sync loop (started by colab_job.sh, or alone). Every SYNC_SECONDS (default 120):
#   - a snapshot of everything the runs produce, /content/glass_snapshot.tgz (written to a temp name, then moved, so a
#     download never gets a half-written file). tar's exit code 1 ("a file changed while being read", which is normal
#     while runs write) still counts as a good snapshot;
#   - if /content/glass_push.ok exists (a GitHub token was configured by colab_setup.py): commit and push.
cd /content/helix_probe || exit 1
echo $$ > /content/glass_sync.pid
SYNC_SECONDS=${SYNC_SECONDS:-120}
sync_once() {
  tar czf /content/glass_snapshot.tmp.tgz --warning=no-file-changed results/glass $(ls -d lab/constructions/zoo_2nd_argmax_* 2>/dev/null)
  rc=$?
  if [ $rc -le 1 ]; then mv /content/glass_snapshot.tmp.tgz /content/glass_snapshot.tgz; else echo "$(date -u +%T) snapshot failed rc=$rc"; fi
  if [ -f /content/glass_push.ok ]; then
    git add -A results/glass lab/constructions/zoo_2nd_argmax_* 2>/dev/null
    git diff --cached --quiet || git commit -q -m "glass auto-sync $(date -u +%H:%M:%S) from Colab" || true
    for i in 1 2 3; do
      git pull -q --rebase origin glass-networks && git push -q origin HEAD:glass-networks && break
      git rebase --abort 2>/dev/null; sleep 5
    done
  fi
}
if [ "$1" = "--once" ]; then sync_once; exit 0; fi
while true; do sync_once; sleep "$SYNC_SECONDS"; done
