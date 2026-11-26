"""Glass model, second version: the 64 variables that matter most for LD2's value (attribution: activation x gradient
of the value through the decoder direction, one backward pass) get three-program compositions; the rest keep their
single programs. Rebuild the last block, run LD2's heads, compare with the first version.
Usage: python3 rosetta/glass2.py define W w | python3 rosetta/glass2.py glass"""
import glob, json, os, sys, time
import chess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_feats as F  # noqa: E402
from rosetta.run_feats import L, fmap, DEC, sae  # noqa: E402
from rosetta.engine import Defs, spec_map, additive_map  # noqa: E402
RES = L.RES; SITE = L.SITE


def top_by_attribution(n_top=64, n_rows=600):
    D = Defs(L.P, L.names, L.TR, L.TE)
    rows = D.rs[:n_rows]; m = L.get_net()
    x = L.planes(rows)
    h = m(x, upto=SITE).detach().requires_grad_(True)
    _, wdl = m.heads(h)
    (wdl[:, 0] - wdl[:, 2]).sum().backward()
    g = h.grad.detach().numpy().reshape(len(rows), 128, 64)                   # d value / d h
    Z = F.dense_codes(rows)                                                  # [n, 64, 512]
    proj = np.einsum("ncs,cf->nsf", g, DEC.numpy())                          # gradient along each decoder direction
    attr = np.abs((Z * proj).sum(1)).mean(0)                                 # [512]
    return [int(f) for f in np.argsort(-attr)[:n_top]], attr


if __name__ == "__main__":
    if sys.argv[1] == "define":
        W, w = int(sys.argv[2]), int(sys.argv[3])
        top, attr = top_by_attribution(n_top=int(os.environ.get("N_TOP", "64")))
        json.dump({"top": top, "attribution": [float(a) for a in attr]}, open(f"{RES}/glass2_attr.json", "w"))
        t0 = time.time()
        D = Defs(L.P, L.names, L.TR, L.TE); out = []
        for j, f in enumerate(top):
            if j % W != w:
                continue
            d = D.define_additive(lambda rr, f=f: fmap(f, rr), terms=3)
            mu = fmap(f, D.rs).mean(0)
            out.append({"feature": f, "attribution": float(attr[f]), "terms": d["terms"], "texts": d["texts"], "weights": d["weights"],
                        "mu": mu.tolist(), "heldout_fidelity": d["curve"][-1]["heldout_fidelity"] if d["curve"] else 0.0})
            print(w, f, out[-1]["heldout_fidelity"], round(time.time() - t0), "s", flush=True)
            json.dump(out, open(f"{RES}/glass2_defs_w{w}.json", "w"))
        json.dump(out, open(f"{RES}/glass2_defs_w{w}.json", "w"))
    elif sys.argv[1] == "where":
        # Where does the glass model fail? Put the K most important variables (by attribution) back at their true
        # values, keep every other variable at its single program, and watch the value fit recover as K grows.
        from kepler.targets.chess.leela import move_index
        att = np.array(json.load(open(f"{RES}/glass2_attr.json"))["attribution"]); order = np.argsort(-att)
        entries = {}
        for fp in glob.glob(f"{RES}/dict_s{SITE}_w*.json"):
            for e in json.load(open(fp)):
                entries[e["feature"]] = e
        D = Defs(L.P, L.names, L.TR, L.TE); rows = D.rt[:1000]; m = L.get_net()
        mean_map = F.dense_codes(D.rs).mean(0); Pt = np.asarray(L.P[rows], dtype=np.float32); Zt = F.dense_codes(rows)
        Zg = np.tile(mean_map[None], (len(rows), 1, 1))
        for f, e in entries.items():
            if e.get("program"):
                Zg[:, :, f] = np.clip(e["intercept"] + e["slope"] * spec_map(Pt, e["program"]), 0, None)
        Wd, b = DEC.numpy(), sae.b.detach().numpy()
        site_from = lambda Zx: torch.tensor((Zx @ Wd.T + b).transpose(0, 2, 1).reshape(len(rows), 128, 8, 8), dtype=torch.float32)
        boards = [chess.Board(L.items[i]["fen"]) for i in rows]
        def top_move(lg):
            return np.array([int(np.argmax([float(l[move_index(m, bd, mm)]) for mm in bd.legal_moves])) if bd.legal_moves.count() else -1 for bd, l in zip(boards, lg)])
        out = {"note": "K most important variables at their true values, the rest at their single programs", "curve": []}
        with torch.no_grad():
            lg0, wdl0 = m(L.planes(rows)); v0 = (wdl0[:, 0] - wdl0[:, 2]).numpy(); t0m = top_move(lg0)
            for K in (0, 8, 16, 32, 64, 128, 256, 512):
                Zx = Zg.copy(); Zx[:, :, order[:K]] = Zt[:, :, order[:K]]
                lg, wdl = m.heads(site_from(Zx)); v = (wdl[:, 0] - wdl[:, 2]).numpy()
                r = {"K": K, "value_r2_vs_network": round(L.r2(v, v0), 4), "top_move_agreement": round(float(np.mean(top_move(lg) == t0m)), 4),
                     "share_of_total_attribution": round(float(att[order[:K]].sum() / att.sum()), 4)}
                out["curve"].append(r); print(r, flush=True)
        json.dump(out, open(f"{RES}/glass2_where_s{SITE}.json", "w"), indent=1)
    else:
        from kepler.targets.chess.leela import move_index
        entries = {}
        for fp in glob.glob(f"{RES}/dict_s{SITE}_w*.json"):
            for e in json.load(open(fp)):
                entries[e["feature"]] = e
        comp = {}
        for fp in glob.glob(f"{RES}/glass2_defs_w*.json"):
            for e in json.load(open(fp)):
                comp[e["feature"]] = e
        D = Defs(L.P, L.names, L.TR, L.TE); rows = D.rt[:1000]; m = L.get_net()
        Ztr = F.dense_codes(D.rs); mean_map = Ztr.mean(0)
        Pt = np.asarray(L.P[rows], dtype=np.float32)
        Zg = np.tile(mean_map[None], (len(rows), 1, 1))
        for f, e in entries.items():
            if e.get("program"):
                Zg[:, :, f] = np.clip(e["intercept"] + e["slope"] * spec_map(Pt, e["program"]), 0, None)
        Zc = Zg.copy()
        for f, e in comp.items():
            if not e["terms"]:
                continue
            Zc[:, :, f] = np.clip(additive_map(Pt, e["terms"], e["weights"]) + np.array(e["mu"])[None], 0, None)
        Wd, b = DEC.numpy(), sae.b.detach().numpy()
        site_from = lambda Zx: torch.tensor((Zx @ Wd.T + b).transpose(0, 2, 1).reshape(len(rows), 128, 8, 8), dtype=torch.float32)
        boards = [chess.Board(L.items[i]["fen"]) for i in rows]
        with torch.no_grad():
            lg0, wdl0 = m(L.planes(rows)); v0 = (wdl0[:, 0] - wdl0[:, 2]).numpy()
            def top_move(lg):
                return np.array([int(np.argmax([float(l[move_index(m, bd, mm)]) for mm in bd.legal_moves])) if bd.legal_moves.count() else -1 for bd, l in zip(boards, lg)])
            t0m = top_move(lg0); out = {"n_composed": len(comp)}
            for tag, Zx in [("single programs (first version)", Zg), ("64 most important variables composed", Zc), ("true sparse codes", F.dense_codes(rows))]:
                lg, wdl = m.heads(site_from(Zx)); v = (wdl[:, 0] - wdl[:, 2]).numpy()
                out[tag] = {"value_r2_vs_network": round(L.r2(v, v0), 4), "top_move_agreement": round(float(np.mean(top_move(lg) == t0m)), 4)}
                print(tag, out[tag], flush=True)
        json.dump(out, open(f"{RES}/glass2_s{SITE}.json", "w"), indent=1)
