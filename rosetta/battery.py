"""The pre-registered DISCOVERY battery (ledger R1, criteria a to f) for single translated variables of LD2's last
block. Usage: python3 rosetta/battery.py f1 f2 ...   (feature ids from the dictionary)"""
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_feats as F  # noqa: E402
from rosetta.run_feats import L, fmap, edit_feature  # noqa: E402
from rosetta.engine import Defs, spec_map, restore_share  # noqa: E402
from kepler.core.residual import surrogate, gain  # noqa: E402

if __name__ == "__main__":
    t0 = time.time(); feats = [int(x) for x in sys.argv[1:]]
    E = {}
    for f in glob.glob(f"{L.RES}/dict_s{L.SITE}_w*.json"):
        for e in json.load(open(f)):
            E[e["feature"]] = e
    rep = {}
    rp = f"{L.RES}/replicate_744706_s{L.SITE}.json"
    if os.path.exists(rp):
        for it in json.load(open(rp))["items"]:
            rep[it["ld2_feature"]] = it
    line = json.load(open(f"{L.RES}/feats_controls_s{L.SITE}.json"))["line_max"]
    D = Defs(L.P, L.names, L.TR, L.TE); m = L.get_net()
    yo = L.outcome(); g_tr = L.TR[~np.isnan(yo[L.TR])]; g_te = L.TE[~np.isnan(yo[L.TE])]
    rows = D.rt[:800]; v0 = L.value(m, rows)
    out = []
    for f in feats:
        e = E[f]; prog = e["program"]; t = spec_map(L.P, prog).sum(1).astype(np.float64)
        c = {"feature": f, "definition": e["text"], "a_fidelity": e["heldout_fidelity"], "a_pass": abs(e["heldout_fidelity"]) >= line}
        c["b_restatement_r2"] = round(L.r2(surrogate(L.CX[L.TR], t[L.TR]).predict(L.CX[L.TE]), t[L.TE]), 4)
        c["b_pass"] = c["b_restatement_r2"] < 0.9
        am = fmap(f, rows); pm = spec_map(L.P[rows], prog)
        fs = fmap(f, D.rs); ps = spec_map(L.P[D.rs], prog)
        a = np.polyfit(ps.reshape(-1), fs.reshape(-1), 1)
        c["c_substitution_restore"] = round(restore_share(v0, L.value(m, rows, edit_feature(f, am)), L.value(m, rows, edit_feature(f, am, np.clip(a[1] + a[0] * pm, 0, None)))), 4)
        c["c_pass"] = c["c_substitution_restore"] >= 0.5
        vg = gain(L.CX[L.TR], L.q[L.TR], L.CX[L.TE], L.q[L.TE], t[L.TR, None], t[L.TE, None])[0]
        c["d_value_gain"] = {"gain": round(vg["gain"], 5), "ci95": [round(x, 5) for x in vg["ci95"]]}; c["d_pass"] = vg["ci95"][0] > 0
        r = rep.get(f); c["e_replication"] = r["744706_fidelity"] if r else None; c["e_pass"] = bool(r and r["replicates"])
        rg = gain(L.CX[g_tr], yo[g_tr], L.CX[g_te], yo[g_te], t[g_tr, None], t[g_te, None])[0]
        la = any(L.names[k].startswith("la_") for k, _, _ in prog)
        c["f_results_gain"] = {"gain": round(rg["gain"], 5), "ci95": [round(x, 5) for x in rg["ci95"]]}; c["f_look_ahead"] = la
        c["f_pass"] = rg["ci95"][0] > 0 or la
        c["all_pass"] = all(c[k] for k in ("a_pass", "b_pass", "c_pass", "d_pass", "e_pass", "f_pass"))
        out.append(c); print(json.dumps(c), round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{L.RES}/battery_s{L.SITE}.json", "w"), indent=1)
