"""Rosetta's dictionary of a network layer: every sparse variable (SAE feature) at a site translated into a chess
program, then the whole layer rebuilt from the translations and run through the rest of the network (a glass model
of that layer). Usage: python3 rosetta/dictionary.py define W w   (worker w of W)   |   python3 rosetta/dictionary.py glass"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rosetta import run_feats as F  # noqa: E402
from rosetta.run_feats import L, fmap, DEC, sae  # noqa: E402
from rosetta.engine import Defs, spec_map  # noqa: E402
from rosetta.core import search, program_map, fidelity, describe  # noqa: E402
RES = L.RES; SITE = L.SITE


def define_worker(W, w):
    D = Defs(L.P, L.names, L.TR, L.TE)
    out = []
    t0 = time.time()
    for f in range(DEC.shape[1]):
        if f % W != w:
            continue
        ys = fmap(f, D.rs); yt = fmap(f, D.rt)
        freq = float((ys > 0).any(1).mean())
        if ys.std() < 1e-6:
            out.append({"feature": f, "freq": freq, "program": None}); continue
        mu = ys.mean(0, keepdims=True)
        res = search((ys - mu).reshape(-1), D.L, D.specs, keep=1)
        prog, c = res[0]
        pm_s = program_map(np.asarray(L.P[D.rs], dtype=np.float32), D.specs, prog)
        pm_t = program_map(D.Pt, D.specs, prog)
        a = np.polyfit(pm_s, ys.reshape(-1), 1)                       # activation = a1 + a0 * program
        out.append({"feature": f, "freq": freq, "program": [D.specs[i][:3] for i in prog], "text": describe(D.specs, prog),
                    "train_corr": round(c, 4), "heldout_fidelity": round(fidelity((yt - mu).reshape(-1), pm_t), 4),
                    "slope": float(a[0]), "intercept": float(a[1]),
                    "uses_look_ahead": any(L.names[s[0]].startswith("la_") for s in [D.specs[i] for i in prog])})
        if len(out) % 32 == 0:
            print(w, len(out), round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{RES}/dict_s{SITE}_w{w}.json", "w"))


@torch.no_grad()
def glass(n_rows=1000):
    """Rebuild the site from (a) the true sparse codes, (b) Rosetta's translations, (c) each feature's mean map only;
    run the network's remaining layers and heads; compare value and move choice with the network itself."""
    import glob
    from kepler.targets.chess.leela import move_index
    import chess
    entries = {}
    for fpath in glob.glob(f"{RES}/dict_s{SITE}_w*.json"):
        for e in json.load(open(fpath)):
            entries[e["feature"]] = e
    D = Defs(L.P, L.names, L.TR, L.TE)
    rows = D.rt[:n_rows]
    m = L.get_net()
    Z = F.dense_codes(rows)                                             # [n, 64, 512] true codes
    Ztr = F.dense_codes(D.rs)
    mean_map = Ztr.mean(0)                                              # [64, 512]
    Pt = np.asarray(L.P[rows], dtype=np.float32)
    Zg = np.tile(mean_map[None], (len(rows), 1, 1))
    for f, e in entries.items():
        if e.get("program"):
            Zg[:, :, f] = np.clip(e["intercept"] + e["slope"] * spec_map(Pt, e["program"]), 0, None)
    Wd, b = DEC.numpy(), sae.b.detach().numpy()
    def site_from(Zx):
        return torch.tensor((Zx @ Wd.T + b).transpose(0, 2, 1).reshape(len(rows), 128, 8, 8), dtype=torch.float32)
    x = L.planes(rows)
    lg0, wdl0 = m(x)
    def run_from(h):
        lg, wdl = m.heads(h)
        return lg, wdl
    out = {"site": SITE, "n_positions": len(rows), "n_features": len(entries)}
    v0 = (wdl0[:, 0] - wdl0[:, 2]).numpy()
    boards = [chess.Board(L.items[i]["fen"]) for i in rows]
    def top_move(lg):
        res = []
        for b, l in zip(boards, lg):
            mv = list(b.legal_moves)
            res.append(int(np.argmax([float(l[move_index(m, b, mm)]) for mm in mv])) if mv else -1)
        return np.array(res)
    t0m = top_move(lg0)
    sur = L.sur(L.CX[rows])
    for tag, Zx in [("sparse codes (upper bound)", Z), ("Rosetta translation", Zg), ("feature means only (knows nothing)", np.tile(mean_map[None], (len(rows), 1, 1)))]:
        lg, wdl = run_from(site_from(Zx))
        v = (wdl[:, 0] - wdl[:, 2]).numpy()
        tm = top_move(lg)
        out[tag] = {"value_r2_vs_network": round(L.r2(v, v0), 4), "top_move_agreement": round(float(np.mean(tm == t0m)), 4),
                    "share_of_textbook_residual_reproduced": round(float(1 - np.mean(((v - sur) - (v0 - sur)) ** 2) / np.var(v0 - sur)), 4)}
        print(tag, out[tag], flush=True)
    fids = np.array([abs(e.get("heldout_fidelity", 0) or 0) for e in entries.values()])
    out["features_with_fidelity_ge"] = {str(t): int((fids >= t).sum()) for t in (0.3, 0.5, 0.7, 0.9)}
    json.dump(out, open(f"{RES}/glass_s{SITE}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "define":
        define_worker(int(sys.argv[2]), int(sys.argv[3]))
    else:
        glass()
