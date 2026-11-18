"""Colab: clone (or update) the repo on branch glass-networks. If /content/gh_token exists (a GitHub token uploaded
from the workspace, never committed), configure pushing with it and mark /content/glass_push.ok.
Run from the workspace:  colab exec -s glass1 -f scripts/glass/colab_setup.py --timeout 600
then upload lab/.env:    colab upload -s glass1 lab/.env /content/helix_probe/lab/.env"""
import os, subprocess
def sh(c, t=300):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print("$", c.split("https://")[0][:80], "->", r.returncode, (r.stdout + r.stderr)[-400:].replace(os.environ.get("X", "@@"), ""))
    return r.returncode
R = "/content/helix_probe"
if not os.path.isdir(R + "/.git"):
    sh(f"git clone -q -b glass-networks https://github.com/cybercryptixcoder/helix_probe {R}", 600)
else:
    sh(f"cd {R} && git fetch -q origin && git checkout -q glass-networks && git pull -q --rebase origin glass-networks")
sh(f"cd {R} && git config user.name 'Glass Networks (Colab)' && git config user.email 'glass-colab@users.noreply.github.com'")
if os.path.exists("/content/gh_token"):
    tok = open("/content/gh_token").read().strip()
    sh(f"cd {R} && git remote set-url origin https://x-access-token:{tok}@github.com/cybercryptixcoder/helix_probe")
    if sh(f"cd {R} && git push -q --dry-run origin HEAD:glass-networks") == 0:
        open("/content/glass_push.ok", "w").write("ok")
        print("push configured")
sh(f"cd {R} && git log --oneline -1 && ls lab/models | grep zoo_ | tr '\\n' ' '")
