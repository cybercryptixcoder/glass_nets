"""Physical short-integration summaries (Tamayo et al. 2020 feature set, SPOCK
FeatureClassifier.generate_features: 1e4-orbit integration with MEGNO) for a list
of systems. Usage: SET=random IDX=file.txt OUT=file.csv python3 -I short_feats.py"""
import os, numpy as np, rebound, warnings
warnings.filterwarnings("ignore")
from spock import FeatureClassifier
P = os.path.expanduser(os.environ.get("PDATA", "~/pdata"))
SET, IDX, OUT = os.environ["SET"], os.environ["IDX"], os.environ["OUT"]
idx = [int(l) for l in open(IDX) if l.strip()]
done = set()
if os.path.exists(OUT):
    done = {int(l.split(",")[0]) for l in open(OUT) if l[:1].isdigit()}
sa = rebound.Simulationarchive(f"{P}/spock2/dataset/{SET}/clean_initial_conditions.bin")
m = FeatureClassifier()
KEYS = ["EMcrossnear", "EMfracstdnear", "EPstdnear", "MMRstrengthnear", "EMcrossfar", "EMfracstdfar",
        "EPstdfar", "MMRstrengthfar", "MEGNO", "MEGNOstd", "Tsec"]
if not os.path.exists(OUT):
    open(OUT, "w").write("i," + ",".join(KEYS) + ",short_stable\n")
todo = [i for i in idx if i not in done]
for k in range(0, len(todo), 10):
    ch = todo[k:k + 10]
    fs = m.generate_features([sa[i] for i in ch], n_jobs=1)
    with open(OUT, "a") as f:
        for i, (lst, ok) in zip(ch, fs):
            d = lst[0]
            f.write(f"{i}," + ",".join(f"{float(d[q]):.6g}" for q in KEYS) + f",{int(bool(ok))}\n")
    print(k + len(ch), "of", len(todo), flush=True)
