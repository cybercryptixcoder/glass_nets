import subprocess
for w in range(3):
    subprocess.Popen(["bash", "-c", f"env PDATA=/content/pdata nohup python3 /content/fresh.py run 3 {w} 400 800 1e6 > /content/p2i_{w}.log 2>&1 &"])
import time; time.sleep(3)
print(subprocess.run("ps aux | grep -E 'fresh.py|fresh_side' | grep -v grep | wc -l; cat /content/p2i_0.log", shell=True, capture_output=True, text=True).stdout)
