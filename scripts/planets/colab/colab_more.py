import subprocess
cmds = [f"python3 /content/fresh_side.py feat {w} 4 400 560" for w in range(4)] + ["python3 /content/fresh_side.py nn 0 1 400 560"]
for j, c in enumerate(cmds):
    subprocess.Popen(["bash", "-c", f"nohup nice -n 5 {c} > /content/p3_{j}.log 2>&1 &"])
print("queued", len(cmds))
