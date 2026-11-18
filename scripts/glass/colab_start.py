"""Colab: start the Glass Networks job detached, unless it is already running."""
import subprocess
r = subprocess.run("pgrep -f 'run_ladder.py' >/dev/null && echo RUNNING || (cd /content/helix_probe && setsid nohup bash scripts/glass/colab_job.sh >> /content/glass.log 2>&1 < /dev/null & echo STARTED)",
                   shell=True, capture_output=True, text=True, timeout=60)
print(r.stdout, r.stderr[-300:])
