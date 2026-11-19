# Development prototype (Nov 19): token-held-out generalization of trusted triggers on modadd_p113 (A37 dev atlas).
"""Prototype (development data only): do triggers generalize to unseen token values? Key-frequency vs other triggers."""
import sys, re, json, pickle, numpy as np, torch
sys.path.insert(0, "lab"); import atlas
from sklearn.tree import DecisionTreeClassifier
tgt = atlas.make_target("modadd_p113")
mods = atlas.load_decomposition(tgt, "results/glass/atlas/spd/spd_modadd_p113_original_30000_1024_lad_1e-4.pt")
A = json.load(open("results/glass/atlas/modadd_p113_d4_a37dev.json"))
KEY = {8, 11, 22, 23, 25}
x = tgt.X[:16000]; t = x.argmax(-1)
calls = tgt.network_calls(mods, x)
rng = np.random.default_rng(0); T1 = set(rng.permutation(113)[:57].tolist()); T1.add(113)
in1 = np.vectorize(lambda v: v in T1)
res = []
for k, m in A["matrices"].items():
    names, F, G = calls[k]; n = tgt._rows(k); B = len(x)
    if n == 3:
        tok = tgt._pm(t[:, :, None].astype(float), B, 3)[:, 0].astype(int); r1 = in1(tok); r2 = ~in1(tok) | (tok == 113)
    else:
        r1 = in1(t[:, 0]) & in1(t[:, 1]); r2 = ~in1(t[:, 0]) & ~in1(t[:, 1])
    for mech in m["mechanisms"]:
        tr_ = mech["trigger"]
        if tr_["rule"] in ("always", "never") or atlas._conf(mech) < atlas.TRUSTED:
            continue
        c = int(mech["id"].rsplit(".", 1)[1]); y = G[:, c] > 0.5; clear = (G[:, c] > 0.9) | (G[:, c] < 0.1)
        if y[r1].all() or (~y[r1]).all() or y[r2 & clear].all() or (~y[r2 & clear]).all():
            continue
        tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced", min_samples_leaf=20, random_state=0).fit(F[r1], y[r1])
        te = r2 & clear; p = tree.predict(F[te]); yt = y[te]
        acc = 0.5 * (p[yt].mean() + (~p[~yt]).mean())
        fr = {int(f) for u in tr_["features_used"] for f in re.findall(r"(\d+)·", u)}
        res.append((k, mech["id"], bool(fr & KEY) if fr else None, acc))
import collections
for grp in (True, False, None):
    a_ = [r[3] for r in res if r[2] is grp]
    if a_:
        print({True: "key-frequency triggers", False: "non-key-frequency triggers", None: "no frequency feature"}[grp], len(a_),
              "token-held-out accuracy: median %.2f, share >= 0.9: %.2f" % (np.median(a_), np.mean(np.array(a_) >= 0.9)))
by = collections.defaultdict(list)
for r in res: by[r[0]].append(r)
for k, rs in by.items():
    print(k, "key %d/%d generalize; non-key %d/%d generalize" % (sum(1 for r in rs if r[2] and r[3] >= .9), sum(1 for r in rs if r[2]), sum(1 for r in rs if r[2] is False and r[3] >= .9), sum(1 for r in rs if r[2] is False)))
