"""Which translated variables carry knowledge about who wins? Each of LD2's translated variables (fidelity >= 0.3) is
scored alone on the held-out split of the original games (fit on the training split); the five best are then tested on
the 16,000 fresh games, which play no part in the choice. Usage: python3 rosetta/knowledge_select.py"""
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import spec_map  # noqa: E402
from rosetta.fresh_knowledge import outcome as fresh_outcome  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
K = os.path.expanduser("~/kdata")


def boot_gain(y, base, aug, seed=0, n=2000):
    e0, e1 = (y - base) ** 2, (y - aug) ** 2; rng = np.random.default_rng(seed); bt = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); bt.append((e0[i].mean() - e1[i].mean()) / np.var(y[i]))
    return round(float((e0.mean() - e1.mean()) / np.var(y)), 5), [round(float(np.quantile(bt, .025)), 5), round(float(np.quantile(bt, .975)), 5)]


if __name__ == "__main__":
    t0 = time.time()
    E = []
    for f in glob.glob(f"{L.RES}/dict_s{L.SITE}_w*.json"):
        E += [e for e in json.load(open(f)) if e.get("program") and abs(e["heldout_fidelity"]) >= 0.3]
    E.sort(key=lambda e: e["feature"])
    T = np.column_stack([spec_map(L.P, e["program"]).sum(1) for e in E]).astype(np.float64)
    z = np.load(os.path.join(L.R, "fresh_cache.npz"), allow_pickle=True); Pf, CXf = z["P"], z["CX"]
    Tf = np.column_stack([spec_map(Pf, e["program"]).sum(1) for e in E]).astype(np.float64)
    fm = json.load(open(os.path.join(K, "chess", "rules", "fresh_meta.json")))["items"]
    yf = fresh_outcome(json.load(open(os.path.join(K, "chess", "rules", "fresh_results.json"))), [it["fen"] for it in fm]); gf = np.where(~np.isnan(yf))[0]
    yo = L.outcome(); g_tr = L.TR[~np.isnan(yo[L.TR])]; g_te = L.TE[~np.isnan(yo[L.TE])]; g_all = np.where(~np.isnan(yo))[0]
    base_te = surrogate(L.CX[g_tr], yo[g_tr]).predict(L.CX[g_te])
    scores = []
    for j, e in enumerate(E):
        if T[g_tr, j].std() == 0:
            continue
        aug = surrogate(np.column_stack([L.CX[g_tr], T[g_tr, j]]), yo[g_tr]).predict(np.column_stack([L.CX[g_te], T[g_te, j]]))
        g, ci = boot_gain(yo[g_te], base_te, aug, n=300)
        scores.append((g, ci, j))
        if len(scores) % 20 == 0:
            print(len(scores), round(time.time() - t0), "s", flush=True)
    scores.sort(reverse=True)
    top = scores[:5]
    out = {"n_variables": len(scores), "selection": [{"feature": E[j]["feature"], "definition": E[j]["text"], "heldout_gain": g, "ci95": ci} for g, ci, j in top]}
    print(json.dumps(out["selection"], indent=0), flush=True)
    base_f = surrogate(L.CX[g_all], yo[g_all]).predict(CXf[gf])
    fresh = []
    for g, ci, j in top:
        aug = surrogate(np.column_stack([L.CX[g_all], T[g_all, j]]), yo[g_all]).predict(np.column_stack([CXf[gf], Tf[gf, j]]))
        fg, fci = boot_gain(yf[gf], base_f, aug)
        fresh.append({"feature": E[j]["feature"], "definition": E[j]["text"], "fresh_gain": fg, "ci95": fci}); print(json.dumps(fresh[-1]), flush=True)
    cols = [j for _, _, j in top]
    aug = surrogate(np.column_stack([L.CX[g_all], T[g_all][:, cols]]), yo[g_all]).predict(np.column_stack([CXf[gf], Tf[gf][:, cols]]))
    jg, jci = boot_gain(yf[gf], base_f, aug)
    out["fresh_each"] = fresh; out["fresh_top5_jointly"] = {"gain": jg, "ci95": jci}
    print("top five jointly on fresh games:", jg, jci, round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{L.RES}/knowledge_select_s{L.SITE}.json", "w"), indent=1)
