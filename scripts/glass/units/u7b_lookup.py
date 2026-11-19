"""U7b: is hard attention a content lookup (attended token = g(query position, query token))?"""
import sys, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets
res = {}
for t in sys.argv[1:] or ["lc_3_4", "lc_4_5", "lc_6_5", "lc_8_6"]:
    for var in ("original", "shuffled"):
        meta, W, X = judge.load_target(t, var, 0); X = np.asarray(X[:6000]).astype(int); B, P = X.shape
        acts = {}; nets.forward("transformer", W, X, acts=acts); row = {}
        for name in sorted(k for k in acts if k.endswith(".pattern")):
            A = np.asarray(acts[name]).reshape(B, -1, P, P)[:, 0]
            hits = tot = 0
            for i in range(P):
                hard = A[:, i].max(1) >= 0.8; key = A[:, i].argmax(1); tok = X[np.arange(B), key]
                groups = defaultdict(list)
                for b in np.where(hard)[0]: groups[X[b, i]].append(tok[b])
                for v, toks in groups.items():
                    c = Counter(toks).most_common(1)[0][1]; hits += c; tot += len(toks)
            row[name] = {"hard_share": float(np.mean(A.max(2) >= 0.8)), "lookup_determinism": hits / tot if tot else None}
        res[f"{t}|{var}"] = row
        print(t, var, {k: {kk: (round(vv, 3) if vv is not None else None) for kk, vv in v.items()} for k, v in row.items()}, flush=True)
(ROOT / "results/glass/units/u7b.json").write_text(json.dumps(res, indent=1))
