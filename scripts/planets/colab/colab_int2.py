import subprocess
for w in range(4):
    subprocess.Popen(["bash", "-c", f"env PDATA=/content/pdata nohup python3 /content/fresh.py run 4 {w} 700 800 1e6 > /content/p4i_{w}.log 2>&1 &"])
print("4 more integration workers on 700-799")
