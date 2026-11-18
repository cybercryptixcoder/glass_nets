"""Colab: start the Glass Networks job detached, unless it is already running (pid file /content/glass_job.pid)."""
import os, subprocess
def alive():
    try:
        os.kill(int(open("/content/glass_job.pid").read()), 0); return True
    except Exception:
        return False
if alive():
    print("RUNNING")
else:
    subprocess.Popen("cd /content/helix_probe && exec bash scripts/glass/colab_job.sh >> /content/glass.log 2>&1 < /dev/null", shell=True, start_new_session=True)
    print("STARTED")
