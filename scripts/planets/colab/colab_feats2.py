import os, subprocess
subprocess.run("pkill -f /content/worker.py; rm -f /content/feat_w*.csv /content/w*.log", shell=True)
W = r'''
import os, sys, numpy as np, rebound, warnings
warnings.filterwarnings("ignore")
from spock import FeatureClassifier
w, W = int(sys.argv[1]), int(sys.argv[2])
jobs = [l.strip().split(",") for l in open("/content/colab_jobs.txt") if l.strip()][w::W]
KEYS = ["EMcrossnear","EMfracstdnear","EPstdnear","MMRstrengthnear","EMcrossfar","EMfracstdfar","EPstdfar","MMRstrengthfar","MEGNO","MEGNOstd","Tsec"]
sa = {s: rebound.Simulationarchive(f"/content/pdata/spock2/dataset/{s}/clean_initial_conditions.bin") for s in ("resonant","random")}
m = FeatureClassifier(); out = open(f"/content/feat_w{w}.csv", "a")
for k in range(0, len(jobs), 10):
    ch = jobs[k:k+10]
    try:
        res = m.generate_features([sa[s][int(i)] for s, i in ch], n_jobs=1)
        if isinstance(res, tuple): res = [res]
        for (s, i), (lst, ok) in zip(ch, res):
            d = lst[0]
            out.write(f"{s},{i}," + ",".join(f"{float(d[q]):.6g}" for q in KEYS) + f",{int(bool(ok))}\n")
        out.flush()
    except Exception as e:
        out.write(f"ERR,{k},{e!r}\n"); out.flush()
'''
open("/content/worker.py", "w").write(W)
for w in range(8):
    subprocess.Popen(f"nohup python3 /content/worker.py {w} 8 > /content/w{w}.log 2>&1 &", shell=True)
print("restarted 8 workers")
