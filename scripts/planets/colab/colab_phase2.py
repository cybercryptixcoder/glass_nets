import subprocess
subprocess.run("for p in $(ps aux | grep 'worker2.py' | grep -v grep | awk '{print $2}'); do kill -9 $p; done", shell=True)
cmds = [f"python3 /content/fresh_side.py feat {w} 4 0 400" for w in range(4)]
cmds += ["python3 /content/fresh_side.py nn 0 1 0 400"]
cmds += [f"PDATA=/content/pdata python3 /content/fresh.py run 3 {w} 400 800 1e6" for w in range(3)]
for j, c in enumerate(cmds):
    subprocess.Popen(["bash", "-c", f"nohup {c} > /content/p2_{j}.log 2>&1 &"])
print("phase 2 started", len(cmds))
