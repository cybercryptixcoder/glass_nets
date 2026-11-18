"""Colab: a cheap status line (job alive? queue counts, last log lines, snapshot age)."""
import json, os, subprocess, time
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
def alive():
    try:
        os.kill(int(open("/content/glass_job.pid").read()), 0); return True
    except Exception:
        return False
print("job:", "alive" if alive() else "NOT RUNNING")
q = "/content/helix_probe/results/glass/queue.json"
if os.path.exists(q):
    st = json.load(open(q)); c = {}
    for j in st.values():
        c[j["status"]] = c.get(j["status"], 0) + 1
    print("queue:", c)
    for n, j in st.items():
        if j["status"] == "running":
            print("  running:", n, j.get("run_id"), f"{(time.time() - j['started']) / 60:.0f} min")
s = "/content/glass_snapshot.tgz"
print("snapshot age:", f"{time.time() - os.path.getmtime(s):.0f} s" if os.path.exists(s) else "none", "| push:", os.path.exists("/content/glass_push.ok"))
print(sh("tail -n 6 /content/glass.log"))
