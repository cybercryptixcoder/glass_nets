"""Colab: bring the CODE up to date from GitHub without touching run data. A plain `git pull` fails on the runtime
(the relay commits run folders that exist there as untracked files) and must never be forced, because the live runs'
files are newer than any commit. So only code paths are checked out from origin. Running constructors pick up a new
judge at their next submission (each submission runs judge.py fresh)."""
import subprocess
r = subprocess.run("cd /content/helix_probe && git fetch -q origin && git checkout origin/glass-networks -- 'lab/*.py' lab/static lab/targets scripts/glass "
                   "&& git log -1 --format=%h origin/glass-networks && grep -c largest_table lab/judge.py", shell=True, capture_output=True, text=True, timeout=60)
print(r.stdout, r.stderr[-300:])
