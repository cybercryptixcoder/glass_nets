"""Fresh N-body verification: law vs network on new systems integrated here.
Truth: SPOCK's own N-body setup, up to TMAX inner orbits (default 1e6), so truth is
exact for systems unstable before TMAX and right-censored after."""
import os, sys, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from law import pair_feats
from final_law import FORMS
P = os.path.expanduser("~/pdata"); TMAX = float(os.environ.get("TMAX", "1e6")); LT = np.log10(TMAX)
def ic_table(E):
    c = {}
    for i in (0, 1):
        f = pair_feats(E, i); mus = f["mu1"] + f["mu2"]
        c[f"X{i}"] = np.log10(f["frac"] * mus ** -0.25); c[f"R{i}"] = f["rz"]; c[f"M{i}"] = np.log10(mus)
    f0 = 10 ** (c["X0"] + c["M0"] / 4); f1 = 10 ** (c["X1"] + c["M1"] / 4); h = f0 * f1 / (f0 + f1)
    mu = (10 ** c["M0"] + 10 ** c["M1"]) / 2
    return {"Xh": np.log10(h / mu ** 0.25), "M": np.log10(mu), "Rm": np.maximum(c["R0"], c["R1"])}
def csv_rows(pattern, ncol):
    out = {}
    for f in glob.glob(pattern):
        for l in open(f):
            p = l.strip().split(",")
            if len(p) == ncol and p[-1] != "":
                try: out[int(p[1] if p[0] == "fresh" else p[0])] = [float(x) for x in (p[2:] if p[0] == "fresh" else p[1:])]
                except ValueError: pass
    return out
if __name__ == "__main__":
    form, fitkind = os.environ.get("FORM", "F4"), os.environ.get("FIT", "censored")
    res = json.load(open("results/planets/final_law.json"))
    key = [k for k in res if k.startswith(form) and k.endswith(fitkind)][0]
    c, csu = np.array(res[key]["params"]), res[key]["short_unstable_const"]; fn = FORMS[key.split(" | ")[0]][1]
    E = np.load(f"{P}/fresh_ic.npz")["E"]; T = ic_table(E)
    nb = csv_rows(f"{P}/fresh_nbody_w*.csv", 3) | csv_rows(f"{P}/colab/fresh_nbody*.csv", 3)
    ft = csv_rows(f"{P}/colab/featfresh*.csv", 14); nn = csv_rows(f"{P}/colab/nnfresh*.csv", 4)
    ids = np.array(sorted(set(nb) & set(ft) & set(nn)))
    t = np.array([nb[i][0] for i in ids]); y = np.log10(t); cens = np.array([nb[i][1] for i in ids]) > 0
    F = np.array([ft[i] for i in ids]); K = ["EMcrossnear", "EMfracstdnear", "EPstdnear", "MMRstrengthnear", "EMcrossfar", "EMfracstdfar", "EPstdfar", "MMRstrengthfar", "MEGNO", "MEGNOstd", "Tsec", "short_stable"]
    S = {k: F[:, j] for j, k in enumerate(K)}
    S["su"] = np.isnan(S["MEGNO"]) | (S["short_stable"] == 0)
    meg = np.where(S["su"], 1e4, S["MEGNO"]); S["lmeg"] = np.log10(np.maximum(meg - 2, 1e-2))
    for k in ["EMfracstdnear", "EMcrossnear", "MMRstrengthnear"]: S["l" + k] = np.log10(np.maximum(np.nan_to_num(S[k], nan=1e-8), 1e-8))
    S.update({k: v[ids] for k, v in T.items()})
    law = np.clip(np.where(S["su"], csu, fn(c, S)), 4, 9); net = np.clip(np.array([nn[i][0] for i in ids]), 4, 9)
    keep = (y >= 4) | cens                                    # protocol: survived 1e4 orbits
    u = keep & ~cens                                          # unstable between 1e4 and TMAX: exact truth
    rm = lambda p, s: float(np.sqrt(np.mean((p[s] - y[s]) ** 2)))
    def cens_rmse(p):                                         # censored systems penalised only if predicted below log10 TMAX
        r = np.where(cens, np.minimum(p - LT, 0), p - y); return float(np.sqrt(np.mean(r[keep] ** 2)))
    rng = np.random.default_rng(0); iu = np.where(u)[0]; ik = np.where(keep)[0]; d1, d2 = [], []
    for _ in range(2000):
        j = rng.choice(iu, len(iu)); d1.append(np.sqrt(np.mean((law[j] - y[j]) ** 2)) - np.sqrt(np.mean((net[j] - y[j]) ** 2)))
        j = rng.choice(ik, len(ik)); r1 = np.where(cens[j], np.minimum(law[j] - LT, 0), law[j] - y[j]); r2 = np.where(cens[j], np.minimum(net[j] - LT, 0), net[j] - y[j])
        d2.append(np.sqrt(np.mean(r1 ** 2)) - np.sqrt(np.mean(r2 ** 2)))
    out = {"law": key, "tmax_orbits": TMAX, "n_integrated": int(len(ids)), "n_survived_1e4": int(keep.sum()), "n_unstable_before_tmax": int(u.sum()),
           "unstable_rmse": {"law": round(rm(law, u), 4), "network": round(rm(net, u), 4), "diff_ci95": [round(float(np.quantile(d1, .025)), 3), round(float(np.quantile(d1, .975)), 3)]},
           "censored_rmse_all": {"law": round(cens_rmse(law), 4), "network": round(cens_rmse(net), 4), "diff_ci95": [round(float(np.quantile(d2, .025)), 3), round(float(np.quantile(d2, .975)), 3)]},
           "false_alarm_rate (survived TMAX but predicted below it)": {"law": round(float(np.mean(law[keep & cens] < LT)), 3), "network": round(float(np.mean(net[keep & cens] < LT)), 3)},
           "bias_on_unstable": {"law": round(float(np.mean(law[u] - y[u])), 3), "network": round(float(np.mean(net[u] - y[u])), 3)}}
    print(json.dumps(out, indent=1)); json.dump(out, open(f"results/planets/fresh_verification_{form}_{fitkind}.json", "w"), indent=1)
