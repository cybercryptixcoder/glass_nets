import subprocess
cmds = [f"python3 /content/fresh_side.py feat {w} 3 560 800" for w in range(3)] + ["python3 /content/fresh_side.py nn 0 1 560 800"]
for j, c in enumerate(cmds):
    subprocess.Popen(["bash", "-c", f"sleep 90; nohup nice -n 5 {c} > /content/p5_{j}.log 2>&1 &"])
print("queued", len(cmds))
