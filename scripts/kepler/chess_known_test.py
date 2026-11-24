"""K3 known-answer tests (before any chess discovery claim).

(K3-a) material decodable from LD2's last site: R^2 >= 0.9 and >= 0.3 above the untrained copy (from the atlas file)
(K3-b) the classical surrogate explains >= 0.5 of LD2's value variance on held-out positions
(K3-c) must fail: a random-noise concept, and 'material' shuffled across positions, raise held-out R^2 by < 0.005

Also computes and caches the replication network's value (744706) on the same positions.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import chess  # noqa: E402
import torch  # noqa: E402

from kepler.core.probe import group_split  # noqa: E402
from kepler.core.residual import gain, r2, surrogate  # noqa: E402
from kepler.targets.chess.leela import LeelaNet, encode  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))


def load_chess(name="LD2"):
    d = os.path.join(KDATA, "chess", name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    cz = np.load(os.path.join(d, "concepts.npz"))
    sp = np.load(os.path.join(d, "split.npz"))
    return meta["items"], cz["X"].astype(np.float64), [str(n) for n in cz["names"]], sp["groups"]


def replication_q(items, name="744706"):
    path = os.path.join(KDATA, "chess", f"q_{name}.npy")
    if os.path.exists(path):
        return np.load(path)
    torch.set_num_threads(1)
    net = LeelaNet(os.path.join(KDATA, "chess", f"{name}.pb.gz"))
    q = np.zeros(len(items), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(items), 256):
            x = torch.tensor(np.stack([encode(chess.Board(it["fen"])) for it in items[i:i + 256]]))
            _, wdl = net(x)
            q[i:i + len(x)] = (wdl[:, 0] - wdl[:, 2]).numpy()
    np.save(path, q)
    return q


def main():
    items, X, names, groups = load_chess()
    q = np.array([it["q"] for it in items])
    tr, te = group_split(groups, 0.25, seed=1)
    np.savez(os.path.join(KDATA, "chess", "LD2", "value_split.npz"), tr=tr, te=te)
    res = {}
    atlas = json.load(open("results/kepler/chess/atlas_LD2.json"))
    last = atlas["sites"][str(max(int(k) for k in atlas["sites"]))]["material"]
    ok_a = last["r2_trained"] >= 0.9 and last["r2_trained"] - last["r2_control"] >= 0.3
    best_site = max(atlas["sites"], key=lambda k: atlas["sites"][k]["material"]["r2_trained"])
    res["K3a"] = {"pass": ok_a, "last_site": last, "best_site": best_site,
                  "best": atlas["sites"][best_site]["material"]}
    print(f"(K3-a) material at last site: R2 {last['r2_trained']:.3f} (control {last['r2_control']:.3f}) ->",
          "PASS" if ok_a else "FAIL", f"| reported: best site {best_site}, R2 {res['K3a']['best']['r2_trained']:.3f}")

    m = surrogate(X[tr], q[tr])
    base = m.predict(X[te])
    r2b = r2(q[te], base)
    ok_b = r2b >= 0.5
    q2 = replication_q(items)
    m2 = surrogate(X[tr], q2[tr])
    r2b2 = r2(q2[te], m2.predict(X[te]))
    res["K3b"] = {"pass": ok_b, "r2_LD2": r2b, "r2_744706_reported": r2b2,
                  "corr_LD2_744706_values": float(np.corrcoef(q, q2)[0, 1])}
    print(f"(K3-b) classical surrogate explains R2 {r2b:.3f} of LD2's value ->", "PASS" if ok_b else "FAIL",
          f"| 744706: {r2b2:.3f}; the two networks' values correlate {res['K3b']['corr_LD2_744706_values']:.3f}")

    rng = np.random.default_rng(7)
    noise = rng.normal(size=len(q))
    g_noise, _ = gain(X[tr], q[tr], X[te], q[te], noise[tr], noise[te], base_pred=base)
    mat = X[:, names.index("material")]
    sh = rng.permutation(mat)
    g_sh, _ = gain(X[tr], q[tr], X[te], q[te], sh[tr], sh[te], base_pred=base)
    ok_c = g_noise["gain"] < 0.005 and g_sh["gain"] < 0.005
    res["K3c"] = {"pass": ok_c, "noise": g_noise, "shuffled_material": g_sh}
    print(f"(K3-c) gains: noise {g_noise['gain']:+.4f} {np.round(g_noise['ci95'], 4)}, shuffled material "
          f"{g_sh['gain']:+.4f} {np.round(g_sh['ci95'], 4)} ->", "PASS" if ok_c else "FAIL")
    res["unexplained_share_LD2"] = 1 - r2b
    with open("results/kepler/chess/k3_known_test.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
