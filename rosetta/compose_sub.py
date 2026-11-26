"""Composition for the substitution test: a translated variable defined as a weighted sum of up to K programs
(constants budget K + 1, no network weights), then put in place of the variable inside LD2. The same composition is
run on control variables of the untrained network's dictionary, so the fidelity line for compositions comes from
controls. Usage: python3 rosetta/compose_sub.py K f1 f2 ..."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_feats as F  # noqa: E402
from rosetta.run_feats import L, fmap, edit_feature  # noqa: E402
from rosetta.engine import Defs, additive_map, restore_share  # noqa: E402
from kepler.core.residual import surrogate, gain  # noqa: E402

if __name__ == "__main__":
    t0 = time.time(); Kt = int(sys.argv[1]); feats = [int(x) for x in sys.argv[2:]]
    D = Defs(L.P, L.names, L.TR, L.TE); m = L.get_net()
    # composition line from controls: the untrained dictionary's frequent features
    cu = np.load(os.path.join(L.R, f"codes_LD2_s{L.SITE}_untrained.npz")); rows_u = np.load(os.path.join(L.R, f"rows_LD2_s{L.SITE}_untrained.npy"))
    idx = {r: i for i, r in enumerate(rows_u)}
    ctrl_ids = [int(x["unit"].split()[-1]) for x in json.load(open(f"{L.RES}/feats_controls_s{L.SITE}.json"))["controls"]]
    cl = []
    for f in ctrl_ids:
        d = D.define_additive(lambda rr, f=f: fmap(f, [idx[r] for r in rr], cu["v"], cu["i"]), terms=Kt)
        if d["curve"]:
            cl.append(abs(d["curve"][-1]["heldout_fidelity"]))
    line = max(cl)
    print("composition line from", len(cl), "controls:", round(line, 4), round(time.time() - t0), "s", flush=True)
    rows = D.rt[:800]; v0 = L.value(m, rows)
    yo = L.outcome(); g_tr = L.TR[~np.isnan(yo[L.TR])]; g_te = L.TE[~np.isnan(yo[L.TE])]
    out = {"terms": Kt, "line": line, "features": []}
    for f in feats:
        d = D.define_additive(lambda rr, f=f: fmap(f, rr), terms=Kt)
        mu = fmap(f, D.rs).mean(0)                                     # the centring used in the definition
        comp = lambda Pm: additive_map(Pm, d["terms"], d["weights"]) + mu[None]
        am = fmap(f, rows); rep = np.clip(comp(np.asarray(L.P[rows], np.float32)), 0, None)
        rs = restore_share(v0, L.value(m, rows, edit_feature(f, am)), L.value(m, rows, edit_feature(f, am, rep)))
        t = comp(np.asarray(L.P, np.float32)).sum(1).astype(np.float64)
        vg = gain(L.CX[L.TR], L.q[L.TR], L.CX[L.TE], L.q[L.TE], t[L.TR, None], t[L.TE, None])[0]
        rg = gain(L.CX[g_tr], yo[g_tr], L.CX[g_te], yo[g_te], t[g_tr, None], t[g_te, None])[0]
        rr = L.r2(surrogate(L.CX[L.TR], t[L.TR]).predict(L.CX[L.TE]), t[L.TE])
        r = {"feature": f, "programs": d["texts"], "weights": d["weights"], "heldout_fidelity": d["curve"][-1]["heldout_fidelity"],
             "a_pass": abs(d["curve"][-1]["heldout_fidelity"]) >= line, "b_restatement_r2": round(rr, 4), "c_substitution_restore": round(rs, 4),
             "d_value_gain": {"gain": round(vg["gain"], 5), "ci95": [round(x, 5) for x in vg["ci95"]]},
             "f_results_gain": {"gain": round(rg["gain"], 5), "ci95": [round(x, 5) for x in rg["ci95"]]}}
        out["features"].append(r); print(json.dumps(r), round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{L.RES}/compose_sub_K{Kt}_s{L.SITE}.json", "w"), indent=1)
