"""D1: decompile a one-layer transformer from its weights, blind.

  python3 lab/decomp/decompile.py <model_id> [--patches 2000]

Reads only lab/models/<id>/weights.npz (never train.json or meta.json's answer). Writes
results/decomp/<id>.json (report) and results/decomp/<id>.py (the program).

Stages
  1  structure of the input symbols: circle planes and order-2 axes in the embedding, relation lattice, the finite
     abelian group they form, every symbol's element (several plane-finding variants; the one that places the most
     symbols with a consistent decode wins; the choice uses the weights only)
  2  the MLP output as characters of G x G (does it depend on g(a) + g(b) only?), output classes read from W_U along
     what the MLP writes, the program
  3  verification (network outputs used here only): M1 replacement, M2 causal patching vs. random subspaces
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lab"))
from decomp.circles import find_planes, find_structure
from decomp.group import recover_auto
from decomp import program as pg

VARIANTS = [("plain", lambda E: find_planes(E, restarts=24, iters=250)),
            ("axes+planes", lambda E: find_structure(E, restarts=24, iters=250)),
            ("whitened", lambda E: find_planes(E, restarts=16, iters=250, full=True, ratio=0.3))]


def stage1(E, log=print):
    best = None
    for name, fn in VARIANTS:
        t0 = time.time()
        P = fn(E)
        r = recover_auto(E, P)
        if r.get("finite"):
            key = (r["distinct"], float(np.mean(r["fit"][r["on"]])))
            log(f"  stage 1 [{name}]: {len(P)} planes, group {r['group']}, {r['distinct']} symbols placed, fit {key[1]:.3f} ({time.time() - t0:.0f}s)")
            if best is None or key > best[0]:
                best = (key, name, P, r)
        else:
            log(f"  stage 1 [{name}]: {len(P)} planes, no finite group ({time.time() - t0:.0f}s)")
    return best


def decompile(model_id, patches=2000, log=print):
    W = dict(np.load(os.path.join(ROOT, "lab", "models", model_id, "weights.npz")))
    n_out = W["W_U"].shape[1]
    n = n_out if W["W_E"].shape[0] == n_out + 1 else W["W_E"].shape[0] - 1   # last token is '='
    E = W["W_E"][:n]
    rep = {"model": model_id, "n_symbols": n, "n_out": n_out}
    b = stage1(E, log)
    toks = pg.all_pairs(n)
    I = pg.batched(W, toks, keys=("mlp", "logits"))
    net = I["logits"].argmax(-1)
    if b is None:
        rep.update({"verdict": "no group structure found in the embedding", "group": None})
        return rep, None
    (_, fit), variant, P, r = b
    d, ga, on = r["group"], r["g"], r["on"]
    power, Mhat, els = pg.mlp_characters(I["mlp"], ga, ga, d, on, on)
    dec = pg.decode_outputs(W["W_U"], Mhat, power, d)
    f, src = pg.make_program(ga, dec["h"], d, on, dec["leftover"])
    m1 = pg.verify_replacement(f, net, n)
    m2 = pg.verify_causal(W, P, ga, f, n, n_patches=patches)
    m2r = pg.verify_causal(W, P, ga, f, n, n_patches=patches, random_planes=True)
    dims = int(np.linalg.matrix_rank(np.concatenate([pl.get("Qo", pl["Q"]) for pl in P], 1)))
    rep.update({
        "variant": variant, "group": d, "order": int(np.prod(d)), "symbols_placed": int(r["distinct"]), "symbols_off_circle": [int(i) for i in np.where(~on)[0]],
        "decode_fit": fit, "planes": len(P), "patched_dims": dims,
        "mlp_diagonal_share": pg.diagonal_share(power, d),
        "outputs": {"argmax_distinct": dec["argmax_distinct"], "moved_by_assignment": dec["moved_by_assignment"], "leftover": dec["leftover"]},
        "M1": m1, "M2": m2, "M2_random_control": m2r,
        "element_of_symbol": {int(i): [int(x) for x in ga[i]] for i in range(n) if on[i]},
        "element_of_class": {int(c): [int(x) for x in dec["h"][c]] for c in range(n_out) if c not in dec["leftover"]},
    })
    ok1 = m1["agreement"] >= 0.90
    ok2 = m2["success"] >= 0.80 and m2r["success"] < 0.2
    rep["verdict"] = ("PASS" if ok1 and ok2 else "FAIL") + f" (M1 {m1['agreement']:.3f}, M2 {m2['success']:.3f} vs random {m2r['success']:.3f})"
    return rep, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--patches", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    rep, src = decompile(a.model, a.patches)
    rep["seconds"] = round(time.time() - t0, 1)
    out = os.path.join(ROOT, "results", "decomp"); os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, a.model + ".json"), "w"), indent=1)
    if src:
        open(os.path.join(out, a.model + ".py"), "w").write(src + "\n")
    print(a.model, rep.get("group"), rep["verdict"], f"{rep['seconds']}s")


if __name__ == "__main__":
    main()
