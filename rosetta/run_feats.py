"""Rosetta round 2 on LD2: the network's own sparse variables (top-k SAE features at a site) instead of raw channels.
Usage: python3 rosetta/run_feats.py select | controls | define | confirm     (SITE env var, default 10)"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import Defs, spec_map, acceptance_line, restore_share  # noqa: E402
from rosetta.sae import TopK  # noqa: E402
SITE = L.SITE; RES = L.RES; R = L.R
cz = np.load(os.path.join(R, f"codes_LD2_s{SITE}.npz")); CV, CI = cz["v"], cz["i"]          # [N, 64, k]
sae = TopK(); sae.load_state_dict(torch.load(os.path.join(R, f"sae_LD2_s{SITE}.pt"))); DEC = sae.dec.weight.detach()  # [128, m]


def fmap(f, rows, V=None, I=None):
    V = CV if V is None else V; I = CI if I is None else I
    v, i = np.asarray(V[rows], dtype=np.float32), np.asarray(I[rows])
    return (v * (i == f)).sum(-1)                                                          # [n, 64]


def edit_feature(f, amap, replace=None):
    """At SITE, remove feature f's contribution (amap [B, 64]) and optionally add `replace` [B, 64] instead."""
    d = DEC[:, f].view(1, 128, 1, 1)
    def ed(site, h):
        if site == SITE:
            h = h - d * torch.tensor(amap, dtype=h.dtype).view(-1, 1, 8, 8)
            if replace is not None:
                h = h + d * torch.tensor(replace, dtype=h.dtype).view(-1, 1, 8, 8)
        return h
    return ed


if __name__ == "__main__":
    stage = sys.argv[1]; t0 = time.time()
    D = Defs(L.P, L.names, L.TR, L.TE)
    m_feats = DEC.shape[1]
    if stage == "select":
        act_any = np.zeros(m_feats);
        Ttr = np.zeros((len(L.TR), m_feats), np.float32)
        for j in range(0, len(L.TR), 2000):
            rr = L.TR[j:j + 2000]; v, i = np.asarray(CV[rr], np.float32), np.asarray(CI[rr])
            for f in range(m_feats):
                Ttr[j:j + len(rr), f] = (v * (i == f)).sum((1, 2))
        freq = (Ttr > 0).mean(0)
        r = L.resid[L.TR]
        cr = np.array([np.corrcoef(Ttr[:, f], r)[0, 1] if Ttr[:, f].std() > 0 else 0 for f in range(m_feats)])
        ok = freq >= 0.01
        cand = [int(f) for f in np.argsort(-np.abs(np.where(ok, cr, 0)))[:24]]
        m = L.get_net(); rows = D.rt[:800]; v0 = L.value(m, rows)
        eff = {}
        for f in cand:
            eff[f] = float(np.mean(np.abs(L.value(m, rows, edit_feature(f, fmap(f, rows))) - v0)))
        top = sorted(cand, key=lambda f: -abs(cr[f]) * (eff[f] > np.median(list(eff.values()))))[:12]
        json.dump({"site": SITE, "freq": freq.round(4).tolist(), "corr_with_residual": np.nan_to_num(cr).round(4).tolist(),
                   "ablation_effect": eff, "top_features": top}, open(f"{RES}/feats_select_s{SITE}.json", "w"))
        for f in top:
            print(f"feature {f}: freq {freq[f]:.3f} corr_resid {cr[f]:+.3f} ablation {eff[f]:.4f}", flush=True)
        print(round(time.time() - t0), "s")
    elif stage == "controls":
        cu = np.load(os.path.join(R, f"codes_LD2_s{SITE}_untrained.npz")); rows_u = np.load(os.path.join(R, f"rows_LD2_s{SITE}_untrained.npy"))
        idx = {r: i for i, r in enumerate(rows_u)}
        freq = (cu["v"].astype(np.float32).sum(1) > 0).mean(0)                              # per slot; recompute per feature:
        Vu, Iu = cu["v"], cu["i"]
        fr = np.array([(((Iu == f) & (Vu > 0)).any((1, 2))).mean() for f in range(m_feats)])
        rng = np.random.default_rng(4); pool = [f for f in range(m_feats) if fr[f] >= 0.01]
        chosen = rng.choice(pool, min(24, len(pool)), replace=False)
        ctrl = []
        for f in chosen:
            d = D.define(lambda rr, f=f: fmap(f, [idx[r] for r in rr], Vu, Iu)); d["unit"] = f"untrained-LD2 SAE feature {f}"; ctrl.append(d)
            print("control", f, d["text"], d["heldout_fidelity"], "| pos", d["pos"]["heldout_fidelity"], flush=True)
        mx, q99 = acceptance_line(ctrl); px, pq = acceptance_line(ctrl, "pos")
        json.dump({"line_max": mx, "line_q99": q99, "line_pos_max": px, "line_pos_q99": pq, "controls": ctrl}, open(f"{RES}/feats_controls_s{SITE}.json", "w"), indent=1)
        print("LINE per-square", round(mx, 4), "position", round(px, 4), round(time.time() - t0), "s")
    elif stage == "define":
        S = json.load(open(f"{RES}/feats_select_s{SITE}.json")); cj = json.load(open(f"{RES}/feats_controls_s{SITE}.json"))
        out = {"site": SITE, "line": cj["line_max"], "line_pos": cj["line_pos_max"], "features": []}
        for f in S["top_features"]:
            d = D.define(lambda rr, f=f: fmap(f, rr)); d["feature"] = f
            d["accepted_sq"] = bool(d["program"] is not None and abs(d["heldout_fidelity"]) >= cj["line_max"])
            d["accepted_pos"] = bool(d["pos"]["program"] is not None and abs(d["pos"]["heldout_fidelity"]) >= cj["line_pos_max"])
            out["features"].append(d)
            print(f"feature {f} | square: {d['text']} {d['heldout_fidelity']} {d['accepted_sq']} | position: {d['pos']['text']} {d['pos']['heldout_fidelity']} {d['accepted_pos']}", flush=True)
        json.dump(out, open(f"{RES}/feats_define_s{SITE}.json", "w"), indent=1)
    elif stage == "confirm":
        from kepler.core.residual import surrogate, gain
        dd = json.load(open(f"{RES}/feats_define_s{SITE}.json")); m = L.get_net()
        yo = L.outcome(); g_tr = L.TR[~np.isnan(yo[L.TR])]; g_te = L.TE[~np.isnan(yo[L.TE])]
        out = {"site": SITE, "concepts": []}; tot = {}
        rows = D.rt[:800]; v0 = L.value(m, rows)
        for u in dd["features"]:
            for lvl in ("sq", "pos"):
                if not u[f"accepted_{lvl}"]:
                    continue
                prog = u["program"] if lvl == "sq" else u["pos"]["program"]; text = u["text"] if lvl == "sq" else u["pos"]["text"]
                f = u["feature"]; t = spec_map(L.P, prog).sum(1); tot[f"{f}-{lvl}"] = t
                c = {"feature": f, "level": lvl, "definition": text, "heldout_fidelity": u["heldout_fidelity"] if lvl == "sq" else u["pos"]["heldout_fidelity"],
                     "uses_look_ahead": any(L.names[k].startswith("la_") for k, _, _ in prog)}
                c["restatement_r2"] = round(L.r2(surrogate(L.CX[L.TR], t[L.TR]).predict(L.CX[L.TE]), t[L.TE]), 4) if t[L.TR].std() > 0 else None
                c["value_gain"] = gain(L.CX[L.TR], L.q[L.TR], L.CX[L.TE], L.q[L.TE], t[L.TR, None], t[L.TE, None])[0]
                c["results_gain"] = gain(L.CX[g_tr], yo[g_tr], L.CX[g_te], yo[g_te], t[g_tr, None], t[g_te, None])[0]
                am = fmap(f, rows); pm = spec_map(L.P[rows], prog)
                fs = fmap(f, D.rs); ps = spec_map(L.P[D.rs], prog)
                a = np.polyfit(ps.reshape(-1), fs.reshape(-1), 1)
                rep = np.clip(a[1] + a[0] * pm, 0, None)
                c["substitution_restore"] = round(restore_share(v0, L.value(m, rows, edit_feature(f, am)), L.value(m, rows, edit_feature(f, am, rep))), 4)
                out["concepts"].append(c); print(json.dumps(c), flush=True)
        if tot:
            Tn = np.column_stack(list(tot.values()))
            out["all_value_gain"] = gain(L.CX[L.TR], L.q[L.TR], L.CX[L.TE], L.q[L.TE], Tn[L.TR], Tn[L.TE])[0]
            out["all_results_gain"] = gain(L.CX[g_tr], yo[g_tr], L.CX[g_te], yo[g_te], Tn[g_tr], Tn[g_te])[0]
            print("all", out["all_value_gain"], out["all_results_gain"])
        json.dump(out, open(f"{RES}/feats_confirm_s{SITE}.json", "w"), indent=1)


def dense_codes(rows, V=None, I=None, m=512):
    V = CV if V is None else V; I = CI if I is None else I
    v, i = np.asarray(V[rows], dtype=np.float32), np.asarray(I[rows]).astype(np.int64)
    Z = np.zeros((len(rows), 64, m), np.float32)
    np.put_along_axis(Z, i, v, axis=2)
    return Z


def lookahead_scan():
    """Search-procedure recogniser: which of the network's own variables mark squares of its OWN future moves?
    Every feature is scored against each look-ahead primitive; the best are then defined twice, with the full library
    and with the look-ahead primitives removed (the static rival). Line: the same scan on the untrained SAE."""
    D = Defs(L.P, L.names, L.TR, L.TE)
    la = [k for k, n in enumerate(L.names) if n.startswith("la_")]
    Pt = np.asarray(L.P[D.rt], dtype=np.float32)
    Z = dense_codes(D.rt)
    def scan(Z):
        Zc = Z - Z.mean(0, keepdims=True)
        Zc = Zc.reshape(-1, Z.shape[2]); Zc /= (Zc.std(0) + 1e-9)
        out = {}
        for k in la:
            p = Pt[:, :, k].reshape(-1); p = (p - p.mean()) / (p.std() + 1e-9)
            out[L.names[k]] = (Zc * p[:, None]).mean(0)
        return out
    real = scan(Z)
    cu = np.load(os.path.join(R, f"codes_LD2_s{SITE}_untrained.npz")); rows_u = np.load(os.path.join(R, f"rows_LD2_s{SITE}_untrained.npy"))
    idx = {r: i for i, r in enumerate(rows_u)}
    Zu = dense_codes([idx[r] for r in D.rt], cu["v"], cu["i"])
    ctrl = scan(Zu)
    line = float(max(np.nanmax(np.abs(v)) for v in ctrl.values()))
    hits = []
    for nm, v in real.items():
        for f in np.argsort(-np.abs(np.nan_to_num(v)))[:3]:
            hits.append((abs(float(v[f])), nm, int(f), float(v[f])))
    hits.sort(reverse=True)
    print("look-ahead line from the untrained SAE:", round(line, 4), flush=True)
    names_static = [n for n in L.names]
    res = {"line": line, "features": []}
    seen = set()
    for score, nm, f, v in hits:
        if f in seen or score < line:
            continue
        seen.add(f)
        full = D.define(lambda rr, f=f: fmap(f, rr))
        # static rival: the same search with every look-ahead literal disabled
        keep = [j for j, s in enumerate(D.specs) if not L.names[s[0]].startswith("la_")]
        Lsave, Ssave, L3save = D.L, D.specs, D.L3
        D.L, D.specs = D.L[keep], [D.specs[j] for j in keep]; D.L3 = D.L.reshape(len(D.specs), len(D.rs), -1)
        stat = D.define(lambda rr, f=f: fmap(f, rr))
        D.L, D.specs, D.L3 = Lsave, Ssave, L3save
        r = {"feature": f, "best_lookahead_primitive": nm, "corr": round(v, 4), "freq": float((Z[:, :, f] > 0).any(1).mean()),
             "full_library": {"text": full["text"], "heldout_fidelity": full["heldout_fidelity"]},
             "static_only": {"text": stat["text"], "heldout_fidelity": stat["heldout_fidelity"]}}
        res["features"].append(r); print(json.dumps(r), flush=True)
        if len(res["features"]) >= 10:
            break
    json.dump(res, open(f"{RES}/feats_lookahead_s{SITE}.json", "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] == "lookahead":
    lookahead_scan()
