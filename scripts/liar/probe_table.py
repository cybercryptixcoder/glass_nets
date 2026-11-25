"""Descriptive (not a pre-registered test): per-probe AUCs on the test split for failed_lie vs truth, lie vs
truth and lie vs failed_lie, with the instruction in context. Usage: python3 -I probe_table.py DATA_DIR TAG..."""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_k as S
d = sys.argv[1]; probes = pd.read_csv(os.path.join(d, "probes.csv"))["probe"].tolist(); out = {}
for tag in sys.argv[2:]:
    D = S.load(d, tag); te = ~D["tr"]
    def pa(pos, neg):
        sel = te & np.isin(D["lab"], [pos, neg]); y = (D["lab"][sel] == pos).astype(int)
        return [float(roc_auc_score(y, D["F"][sel][:, j])) for j in range(D["F"].shape[1])]
    out[tag] = dict(failed_vs_truth=pa("failed_lie", "truth"), lie_vs_truth=pa("lie", "truth"), lie_vs_failed=pa("lie", "failed_lie"))
out["probes"] = probes
print(json.dumps(out))
