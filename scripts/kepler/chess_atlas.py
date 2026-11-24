"""Level 1 for chess: the concept atlas of a Leela network against an untrained copy.

    NET=LD2 python3 scripts/kepler/chess_atlas.py

Uses 8,000 positions (6,000 from games, 2,000 puzzles), split by game. For each site: PCA (256) fitted on training
rows, then a ridge probe per concept (penalty by leave-one-out on training rows), scored on test rows.
Writes results/kepler/chess/atlas_<net>.json and $KDATA/chess/<net>/split.npz (reused by later steps).
"""
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import chess  # noqa: E402
import torch  # noqa: E402

from kepler.core.probe import auc, group_split, is_binary, pca_basis, r2  # noqa: E402
from kepler.targets.chess.leela import LeelaNet, encode  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
ALPHAS = np.logspace(-2, 5, 15)


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def game_groups(items):
    """Positions from the same game are adjacent with the same (elos, result) and increasing ply."""
    g, gid, prev = [], -1, None
    for it in items:
        if it["source"] == "puzzle":
            gid += 1
            g.append(gid)
            prev = None
            continue
        key = (it["white_elo"], it["black_elo"], it["result"])
        if prev is None or key != prev[0] or it["ply"] <= prev[1]:
            gid += 1
        g.append(gid)
        prev = (key, it["ply"])
    return np.array(g)


class RidgeBank:
    """One SVD of the standardized training design, reused for every target (leave-one-out penalty per target)."""

    def __init__(self, Z):
        self.zm, self.zs = Z.mean(0), Z.std(0) + 1e-8
        A = (Z - self.zm) / self.zs
        self.U, self.s, self.Vt = np.linalg.svd(A, full_matrices=False)
        self.A = A

    def fit_predict(self, y, Z_te):
        ym = y.mean()
        Uy = self.U.T @ (y - ym)
        best = None
        for a in ALPHAS:
            d = self.s / (self.s ** 2 + a)
            h = np.sum(self.U ** 2 * (self.s ** 2 / (self.s ** 2 + a)), axis=1)
            fitted = self.U @ (self.s * d * Uy)
            loo = np.mean((((y - ym) - fitted) / np.maximum(1 - h, 1e-6)) ** 2)
            if best is None or loo < best[0]:
                best = (loo, a, d)
        coef = self.Vt.T @ (best[2] * Uy)
        return ((Z_te - self.zm) / self.zs) @ coef + ym, coef / self.zs, best[1]


def random_copy(net, seed=0):
    torch.manual_seed(seed)
    r = copy.deepcopy(net)
    for mod in r.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    return r


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", 1)))
    name = os.environ.get("NET", "LD2")
    d = os.path.join(KDATA, "chess", name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    items = meta["items"]
    cz = np.load(os.path.join(d, "concepts.npz"))
    X, names = cz["X"], [str(n) for n in cz["names"]]
    groups = game_groups(items)
    rng = np.random.default_rng(0)
    game_idx = np.array([i for i, it in enumerate(items) if it["source"] == "game"])
    puz_idx = np.array([i for i, it in enumerate(items) if it["source"] == "puzzle"])
    sel = np.sort(np.concatenate([game_idx[:6000], rng.choice(puz_idx, 2000, replace=False)]))
    tr_l, te_l = group_split(groups[sel], 0.25, seed=0)
    tr, te = sel[tr_l], sel[te_l]
    np.savez(os.path.join(d, "split.npz"), sel=sel, tr=tr, te=te, groups=groups)
    q = np.array([it["q"] for it in items], dtype=np.float32)
    targets = {n: X[:, j] for j, n in enumerate(names)}
    targets["leela_value_q"] = q
    log(f"{len(tr)} train / {len(te)} test positions; {len(targets)} targets")

    net = LeelaNet(os.path.join(KDATA, "chess", f"{name}.pb.gz"))
    rnet = random_copy(net)
    boards = [chess.Board(items[i]["fen"]) for i in sel]
    pos = {i: k for k, i in enumerate(sel)}
    # control activations for the selected positions, all sites, kept in memory as float16
    ctrl = np.zeros((net.n_sites, len(sel), net.channels * 64), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, len(sel), 256):
            x = torch.tensor(np.stack([encode(b) for b in boards[s:s + 256]]))
            _, _, acts = rnet(x, return_acts=True)
            for k in range(net.n_sites):
                ctrl[k, s:s + len(x)] = acts[k].reshape(len(x), -1).numpy().astype(np.float16)
    log("control activations done")
    itr = np.array([pos[i] for i in tr])
    ite = np.array([pos[i] for i in te])
    atlas = {}
    for k in range(net.n_sites):
        A = np.load(os.path.join(d, f"acts_site{k}.npy"), mmap_mode="r")
        site = {}
        for label, M in [("trained", np.asarray(A[sel]).astype(np.float32)), ("control", ctrl[k].astype(np.float32))]:
            mu, V = pca_basis(M[itr], 256)
            Ztr, Zte = (M[itr] - mu) @ V, (M[ite] - mu) @ V
            bank = RidgeBank(Ztr)
            for n, y in targets.items():
                ytr, yte = y[tr], y[te]
                if np.var(ytr) == 0 or np.var(yte) == 0:
                    continue
                p, _, a = bank.fit_predict(ytr, Zte)
                e = site.setdefault(n, {})
                e[f"r2_{label}"] = r2(yte, p)
                if is_binary(yte):
                    e[f"auc_{label}"] = auc(yte, p)
            if label == "trained":   # shuffled-target null on three concepts
                for n in ["material", "mobility_us", "king_zone_attacks_on_us"]:
                    yp = np.random.default_rng(k).permutation(targets[n][tr])
                    p, _, _ = bank.fit_predict(yp, Zte)
                    site[n]["r2_shuffled_null"] = r2(targets[n][te], p)
        atlas[k] = site
        log(f"site {k}: material r2 {site['material']['r2_trained']:.3f} (control {site['material']['r2_control']:.3f}); "
            f"value q r2 {site['leela_value_q']['r2_trained']:.3f}")
    os.makedirs("results/kepler/chess", exist_ok=True)
    with open(f"results/kepler/chess/atlas_{name}.json", "w") as f:
        json.dump({"net": name, "n_train": len(tr), "n_test": len(te), "sites": atlas}, f)
    log("done")


if __name__ == "__main__":
    main()
