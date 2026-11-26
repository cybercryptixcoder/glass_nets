"""Rosetta on Leela LD2: define the network's own units that carry what the classical library misses.
Usage: python3 rosetta/run_ld2.py units | controls | define | confirm   (SITE env var, default 10)."""
import json, os, sys, time
import chess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rosetta.engine import Defs, spec_map, acceptance_line, restore_share  # noqa: E402
from kepler.targets.chess.leela import LeelaNet, encode  # noqa: E402
from kepler.core.residual import surrogate, gain  # noqa: E402
K = os.path.expanduser("~/kdata"); C = os.path.join(K, "chess"); R = os.path.join(K, "rosetta"); RES = "results/rosetta"
os.makedirs(RES, exist_ok=True)
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
SITE = int(os.environ.get("SITE", "10"))
items = json.load(open(os.path.join(C, "LD2", "meta.json")))["items"]
names = json.load(open(os.path.join(R, "prim_names.json")))
P = np.load(os.path.join(R, "prims_LD2.npy"), mmap_mode="r")
vs = np.load(os.path.join(C, "LD2", "value_split.npz")); TR, TE = vs["tr"], vs["te"]
q = np.array([it["q"] for it in items], dtype=np.float32)
CX = np.load(os.path.join(C, "LD2", "concepts.npz"))["X"].astype(np.float64)
A = np.load(os.path.join(C, "LD2", f"acts_site{SITE}.npy"), mmap_mode="r")


def ridge(Xtr, ytr, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (ytr - ytr.mean()))
    return lambda X: ((X - mu) / sd) @ w + ytr.mean()


def r2(p, y):
    return float(1 - np.mean((p - y) ** 2) / np.var(y))


_sur = surrogate(CX[TR], q[TR])          # the record's classical surrogate (gradient-boosted trees, 61 concepts)
sur = _sur.predict
resid = q - sur(CX)
net = None


def get_net(name="LD2", untrained_seed=None):
    m = LeelaNet(os.path.join(C, f"{name}.pb.gz")).eval()
    if untrained_seed is not None:
        g = torch.Generator().manual_seed(untrained_seed)
        with torch.no_grad():
            for p in m.parameters():
                flat = p.view(-1)
                flat.copy_(flat[torch.randperm(flat.numel(), generator=g)])
    return m


def planes(rows):
    return torch.tensor(np.stack([encode(chess.Board(items[i]["fen"])) for i in rows]))


@torch.no_grad()
def site_acts(m, rows, site=SITE):
    out = []
    for i in range(0, len(rows), 500):
        out.append(m(planes(rows[i:i + 500]), upto=site).flatten(2).numpy())
    return np.concatenate(out)


@torch.no_grad()
def value(m, rows, edit=None):
    out = []
    for i in range(0, len(rows), 500):
        _, wdl = m(planes(rows[i:i + 500]), edit=edit)
        out.append((wdl[:, 0] - wdl[:, 2]).numpy())
    return np.concatenate(out)


def unit_rows(c):
    return lambda rows: np.asarray(A[rows][:, c * 64:(c + 1) * 64], dtype=np.float32)


if __name__ == "__main__":
    stage = sys.argv[1]; t0 = time.time()
    D = Defs(P, names, TR, TE)
    print("classical surrogate held-out R2", round(r2(sur(CX[TE]), q[TE]), 4), flush=True)
    if stage == "units":
        m = get_net()
        T = np.asarray(A[TR], dtype=np.float32).reshape(len(TR), 128, 64).mean(2)
        cr = np.array([np.corrcoef(T[:, c], resid[TR])[0, 1] for c in range(128)])
        rows = D.rt[:800]; v0 = value(m, rows)
        H = np.asarray(A[D.rs], dtype=np.float32).reshape(len(D.rs), 128, 64); mean = H.mean(0)
        eff = []
        for c in range(128):
            def ed(site, h, c=c):
                if site == SITE:
                    h = h.clone(); h[:, c] = torch.tensor(mean[c]).view(8, 8)
                return h
            eff.append(float(np.mean(np.abs(value(m, rows, ed) - v0))))
        eff = np.array(eff)
        score = np.abs(cr) * (eff > np.median(eff))
        top = [int(c) for c in np.argsort(-score)[:8]]
        Ttr = T
        w = np.linalg.solve(Ttr.T @ Ttr + 1e-2 * np.eye(128), Ttr.T @ (resid[TR] - resid[TR].mean()))
        json.dump({"site": SITE, "corr_with_residual": cr.round(4).tolist(), "ablation_effect": eff.round(5).tolist(),
                   "top_units": top, "residual_direction": w.tolist()}, open(f"{RES}/ld2_units_site{SITE}.json", "w"))
        print("top units", top, [round(cr[c], 3) for c in top], [round(eff[c], 4) for c in top], round(time.time() - t0), "s")
    elif stage == "controls":
        mu = get_net(untrained_seed=21)
        rows = np.concatenate([D.rs, D.rt]); Hu = site_acts(mu, rows)
        idx = {r: i for i, r in enumerate(rows)}
        rng = np.random.default_rng(3); chans = rng.choice(128, 24, replace=False)
        ctrl = []
        for c in chans:
            d = D.define(lambda rr, c=c: Hu[[idx[r] for r in rr], c]); d["unit"] = f"untrained LD2 channel {c}"; ctrl.append(d)
            print("untrained", c, d["text"], d["heldout_fidelity"], flush=True)
        U = json.load(open(f"{RES}/ld2_units_site{SITE}.json"))["top_units"]
        for c in U:
            def shuf(rr, c=c):
                a = unit_rows(c)(rr); g = np.random.default_rng(int(c)); return np.stack([x[g.permutation(64)] for x in a])
            d = D.define(shuf); d["unit"] = f"LD2 channel {c}, squares shuffled"; ctrl.append(d)
            print("shuffled", c, d["text"], d["heldout_fidelity"], flush=True)
        mx, q99 = acceptance_line(ctrl); px, pq = acceptance_line(ctrl, "pos")
        json.dump({"line_max": mx, "line_q99": q99, "line_pos_max": px, "line_pos_q99": pq, "controls": ctrl},
                  open(f"{RES}/ld2_controls_site{SITE}.json", "w"), indent=1)
        print("LINE per-square", round(mx, 4), "position-level", round(px, 4), round(time.time() - t0), "s")
    elif stage == "define":
        U = json.load(open(f"{RES}/ld2_units_site{SITE}.json")); cj = json.load(open(f"{RES}/ld2_controls_site{SITE}.json"))
        line, pline = cj["line_max"], cj["line_pos_max"]
        out = {"site": SITE, "line": line, "line_pos": pline, "units": []}
        w = np.array(U["residual_direction"])
        for c in U["top_units"] + ["residual direction"]:
            if c == "residual direction":
                f = lambda rr: np.einsum("c,ncs->ns", w, np.asarray(A[rr], dtype=np.float32).reshape(len(rr), 128, 64))
            else:
                f = unit_rows(c)
            d = D.define(f); d["unit"] = f"channel {c}" if c != "residual direction" else c; d["channel"] = c if c != "residual direction" else None
            d["accepted_sq"] = bool(d["program"] is not None and abs(d["heldout_fidelity"]) >= line)
            d["accepted_pos"] = bool(d["pos"]["program"] is not None and abs(d["pos"]["heldout_fidelity"]) >= pline)
            d["accepted"] = d["accepted_sq"] or d["accepted_pos"]
            out["units"].append(d)
            print(d["unit"], "| square:", d["text"], d["heldout_fidelity"], d["accepted_sq"], "| position:", d["pos"]["text"], d["pos"]["heldout_fidelity"], d["accepted_pos"], flush=True)
        json.dump(out, open(f"{RES}/ld2_define_site{SITE}.json", "w"), indent=1)


def outcome():
    y = np.full(len(items), np.nan)
    for i, it in enumerate(items):
        if it["source"] != "game" or it.get("result") not in ("1-0", "0-1", "1/2-1/2"):
            continue
        white = it["fen"].split()[1] == "w"
        s = 1.0 if it["result"] == "1-0" else 0.0 if it["result"] == "0-1" else 0.5
        y[i] = s if white else 1 - s
    return y


def confirm():
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
    dd = json.load(open(f"{RES}/ld2_define_site{SITE}.json")); line = dd["line"]
    D = Defs(P, names, TR, TE)
    acc = []
    for u in dd["units"]:
        if u.get("accepted_sq"):
            acc.append({"unit": u["unit"] + " (square level)", "channel": u["channel"], "program": u["program"], "text": u["text"], "heldout_fidelity": u["heldout_fidelity"]})
        if u.get("accepted_pos"):
            acc.append({"unit": u["unit"] + " (position level)", "channel": u["channel"], "program": u["pos"]["program"], "text": u["pos"]["text"], "heldout_fidelity": u["pos"]["heldout_fidelity"]})
    out = {"site": SITE, "line": line, "concepts": []}
    m = get_net()
    yo = outcome(); g_tr = TR[~np.isnan(yo[TR])]; g_te = TE[~np.isnan(yo[TE])]
    # context: how much of the network's own extra value predicts game results beyond the classical library
    out["context_network_value_predicts_results_beyond_library"] = gain(CX[g_tr], yo[g_tr], CX[g_te], yo[g_te], q[g_tr, None], q[g_te, None])[0]
    tot = {}
    for u in acc:
        pm_all = spec_map(P, u["program"])                     # [N, 64] for every position
        t = pm_all.sum(1)
        tot[u["unit"]] = t
        c = {"unit": u["unit"], "definition": u["text"], "heldout_fidelity": u["heldout_fidelity"],
             "uses_look_ahead": any(names[k].startswith("la_") for k, _, _ in u["program"])}
        rs = surrogate(CX[TR], t[TR]); c["restatement_r2"] = round(r2(rs.predict(CX[TE]), t[TE]), 4)
        c["value_gain"] = gain(CX[TR], q[TR], CX[TE], q[TE], t[TR, None], t[TE, None])[0]
        c["results_gain"] = gain(CX[g_tr], yo[g_tr], CX[g_te], yo[g_te], t[g_tr, None], t[g_te, None])[0]
        if u.get("channel") is not None:
            ch = u["channel"]; rows = D.rt[:800]
            H = np.asarray(A[D.rs], dtype=np.float32).reshape(len(D.rs), 128, 64)
            mean = H[:, ch].mean(0)
            a = np.polyfit(spec_map(P[D.rs], u["program"]).reshape(-1), H[:, ch].reshape(-1), 1)
            pm = spec_map(P[rows], u["program"])
            def abl(site, h):
                if site == SITE:
                    h = h.clone(); h[:, ch] = torch.tensor(mean).view(8, 8)
                return h
            def sub(site, h):
                if site == SITE:
                    h = h.clone(); h[:, ch] = torch.tensor((a[1] + a[0] * pm).reshape(-1, 8, 8), dtype=h.dtype)
                return h
            v0 = value(m, rows); c["substitution_restore"] = round(restore_share(v0, value(m, rows, abl), value(m, rows, sub)), 4)
        out["concepts"].append(c)
        print(json.dumps(c), flush=True)
    if tot:
        Tn = np.column_stack(list(tot.values()))
        out["all_concepts_value_gain"] = gain(CX[TR], q[TR], CX[TE], q[TE], Tn[TR], Tn[TE])[0]
        out["all_concepts_results_gain"] = gain(CX[g_tr], yo[g_tr], CX[g_te], yo[g_te], Tn[g_tr], Tn[g_te])[0]
    json.dump(out, open(f"{RES}/ld2_confirm_site{SITE}.json", "w"), indent=1)
    print("all", out.get("all_concepts_value_gain"), out.get("all_concepts_results_gain"))


if __name__ == "__main__" and sys.argv[1] == "confirm":
    confirm()


def propose_stage(rounds=2, per_call=15):
    """Volume proposer (DeepSeek V4.1 Flash, live): for every candidate unit, several calls write expressions; code
    scores each on training rows (per-square and per-position) and reports held-out fidelity of the training-best."""
    from rosetta import llm
    from rosetta.core import expr_map, fidelity
    dd = json.load(open(f"{RES}/ld2_define_site{SITE}.json"))
    D = Defs(P, names, TR, TE)
    fens_s = [items[i]["fen"] for i in D.rs]
    U = json.load(open(f"{RES}/ld2_units_site{SITE}.json")); w = np.array(U["residual_direction"])
    jobs, meta = [], []
    for u in dd["units"]:
        if u["channel"] is None:
            f = lambda rr: np.einsum("c,ncs->ns", w, np.asarray(A[rr], dtype=np.float32).reshape(len(rr), 128, 64))
        else:
            f = unit_rows(u["channel"])
        ys = f(D.rs); mu = ys.mean(0, keepdims=True); ys = ys - mu; yt = f(D.rt) - mu
        attempts = [(u["text"], u["heldout_fidelity"])] + [(r["text"], r["heldout_fidelity"]) for r in u.get("runners_up", [])[:3]] + [(u["pos"]["text"], u["pos"]["heldout_fidelity"])]
        for r in range(rounds):
            jobs.append((names, fens_s, ys, attempts, per_call)); meta.append((u, ys, yt))
    t0 = time.time()
    props = llm.propose_many(jobs, workers=8)
    print("proposals", sum(len(p) for p in props), round(time.time() - t0), "s", flush=True)
    Ps, Pt = np.asarray(P[D.rs], dtype=np.float32), np.asarray(P[D.rt], dtype=np.float32)
    best = {}
    for (u, ys, yt), plist in zip(meta, props):
        key = u["unit"]
        for e in plist:
            try:
                ms, mt = expr_map(Ps, names, e), expr_map(Pt, names, e)
            except Exception:
                continue
            if ms.std() == 0:
                continue
            fs = fidelity(ys.reshape(-1), ms)
            fp = fidelity(ys.sum(1), ms.reshape(len(D.rs), 64).sum(1))
            for lvl, ftr in (("sq", fs), ("pos", fp)):
                cur = best.get((key, lvl))
                if cur is None or abs(ftr) > abs(cur["train_corr"]):
                    fte = fidelity(yt.reshape(-1), mt) if lvl == "sq" else fidelity(yt.sum(1), mt.reshape(len(D.rt), 64).sum(1))
                    best[(key, lvl)] = {"unit": key, "channel": u["channel"], "level": lvl, "expression": e,
                                        "train_corr": round(float(ftr), 4), "heldout_fidelity": round(float(fte), 4)}
    cj = json.load(open(f"{RES}/ld2_controls_site{SITE}.json"))
    out = []
    for (key, lvl), b in sorted(best.items()):
        b["accepted"] = abs(b["heldout_fidelity"]) >= (cj["line_max"] if lvl == "sq" else cj["line_pos_max"])
        out.append(b); print(json.dumps(b), flush=True)
    json.dump({"proposals": [list(p) for p in props], "best": out}, open(f"{RES}/ld2_propose_site{SITE}.json", "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] == "propose":
    propose_stage()
