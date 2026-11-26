"""Residual curve: how much of what the textbook library misses in LD2's value do Rosetta's translated variables
explain? Each defined variable contributes its program's count per position. Joint gain over the record's surrogate,
then greedy forward selection on the training split only (held-out reported), plus game results.
Usage: python3 rosetta/residual_curve.py [min_fidelity]"""
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import spec_map  # noqa: E402
from kepler.core.residual import surrogate, gain  # noqa: E402

if __name__ == "__main__":
    t0 = time.time(); thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    E = []
    for f in glob.glob(f"{L.RES}/dict_s{L.SITE}_w*.json"):
        E += [e for e in json.load(open(f)) if e.get("program") and abs(e["heldout_fidelity"]) >= thr]
    E.sort(key=lambda e: e["feature"])
    T = np.column_stack([spec_map(L.P, e["program"]).sum(1) for e in E]).astype(np.float64)
    keep = T[L.TR].std(0) > 0; T = T[:, keep]; E = [e for e, k in zip(E, keep) if k]
    print("variables", len(E), round(time.time() - t0), "s", flush=True)
    yo = L.outcome(); g_tr = L.TR[~np.isnan(yo[L.TR])]; g_te = L.TE[~np.isnan(yo[L.TE])]
    out = {"min_fidelity": thr, "n_variables": len(E)}
    out["joint_value_gain"] = gain(L.CX[L.TR], L.q[L.TR], L.CX[L.TE], L.q[L.TE], T[L.TR], T[L.TE])[0]
    print("joint value gain", out["joint_value_gain"], flush=True)
    out["joint_results_gain"] = gain(L.CX[g_tr], yo[g_tr], L.CX[g_te], yo[g_te], T[g_tr], T[g_te])[0]
    print("joint results gain", out["joint_results_gain"], flush=True)
    out["network_value_results_gain"] = gain(L.CX[g_tr], yo[g_tr], L.CX[g_te], yo[g_te], L.q[g_tr, None], L.q[g_te, None])[0]
    print("network value -> results gain", out["network_value_results_gain"], flush=True)
    # greedy forward selection on a split of the TRAINING rows (the held-out split is only scored)
    rng = np.random.default_rng(0); trr = rng.permutation(L.TR); a, b = trr[:10000], trr[10000:]
    base = surrogate(L.CX[a], L.q[a]).predict(L.CX[b]); e0 = np.mean((L.q[b] - base) ** 2)
    corr = np.array([abs(np.corrcoef(T[a, j], (L.q - L.sur(L.CX))[a])[0, 1]) for j in range(T.shape[1])])
    cand = list(np.argsort(-np.nan_to_num(corr))[:30]); chosen = []; curve = []
    for step in range(6):
        best = None
        for j in cand:
            if j in chosen:
                continue
            cols = chosen + [j]
            p = surrogate(np.column_stack([L.CX[a], T[a][:, cols]]), L.q[a]).predict(np.column_stack([L.CX[b], T[b][:, cols]]))
            e = np.mean((L.q[b] - p) ** 2)
            if best is None or e < best[0]:
                best = (e, j)
        chosen.append(best[1])
        g = gain(L.CX[L.TR], L.q[L.TR], L.CX[L.TE], L.q[L.TE], T[L.TR][:, chosen], T[L.TE][:, chosen])[0]
        curve.append({"added": E[best[1]]["text"], "feature": E[best[1]]["feature"], "heldout_gain": g["gain"], "ci95": g["ci95"]})
        print(json.dumps(curve[-1]), round(time.time() - t0), "s", flush=True)
    out["forward_selection"] = curve
    json.dump(out, open(f"{L.RES}/residual_curve_s{L.SITE}.json", "w"), indent=1)
