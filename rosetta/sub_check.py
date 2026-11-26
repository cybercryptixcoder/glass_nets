"""Robustness of a marginal substitution pass: the composed definition of one variable, substituted on a SECOND,
disjoint set of held-out positions, with a bootstrap interval. Usage: python3 rosetta/sub_check.py FEATURE"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta.run_feats import L, fmap, edit_feature  # noqa: E402
from rosetta.engine import Defs, additive_map  # noqa: E402
from rosetta.rep_compose import prog_from_text  # noqa: E402
f = int(sys.argv[1])
cj = json.load(open(f"{L.RES}/compose_sub_K3_s{L.SITE}.json")); c = [x for x in cj["features"] if x["feature"] == f][0]
D = Defs(L.P, L.names, L.TR, L.TE); m = L.get_net()
terms = [prog_from_text(D, t) for t in c["programs"]]; mu = fmap(f, D.rs).mean(0)
out = {"feature": f, "programs": c["programs"]}
rng = np.random.default_rng(0)
for tag, rows in (("first 800 held-out positions (as in compose_sub)", D.rt[:800]), ("second, disjoint 800", D.rt[800:1600])):
    v0 = L.value(m, rows); am = fmap(f, rows)
    rep = np.clip(additive_map(np.asarray(L.P[rows], np.float32), terms, c["weights"]) + mu[None], 0, None)
    va, vs = L.value(m, rows, edit_feature(f, am)), L.value(m, rows, edit_feature(f, am, rep))
    da, ds = np.abs(va - v0), np.abs(vs - v0); msk = da > np.quantile(da, 0.5)
    est = 1 - ds[msk].mean() / da[msk].mean()
    idx = np.where(msk)[0]; bt = []
    for _ in range(2000):
        j = rng.choice(idx, len(idx)); bt.append(1 - ds[j].mean() / da[j].mean())
    out[tag] = {"restore": round(float(est), 4), "ci95": [round(float(np.quantile(bt, .025)), 4), round(float(np.quantile(bt, .975)), 4)]}
    print(tag, out[tag], flush=True)
json.dump(out, open(f"{L.RES}/sub_check_f{f}_s{L.SITE}.json", "w"), indent=1)
