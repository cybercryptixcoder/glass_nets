"""D1 factor mode, command line: does a network split its computation into independent sub-computations, and what
is each one? Blind: reads lab/models/<id>/weights.npz only.

  python3 lab/decomp/factorize.py <model_id>        -> results/decomp/<id>_factor.json

Steps (lab/decomp/factor.py):
  1  side representations x1 = F(a) + G(b) at the MLP's input
  2  each side's parts, from repeated difference vectors (parallelograms); a part is a partition of the symbols
  3  the network's logits as a sum of one table per part (pairs of blocks); share explained and argmax agreement on
     every input. This is the test of the split: a function that does not split cannot be reproduced this way.
  4  each part's composition table read from the network's answers given the parts; symbols whose answer never
     depends on the other input (absorbing symbols, like 0 in a product) are set aside and reported
  5  what each table is: row profiles compared with multiplication and addition in Z_m, and the invertible core's
     group (a Latin square isotopic to a group: which one, by element orders)
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lab"))
from decomp import deep, factor


def run(model_id, log=print):
    W = dict(np.load(os.path.join(ROOT, "lab", "models", model_id, "weights.npz")))
    n = W["W_E"].shape[0] - 1
    F, G, share, I = deep.side_representations(W, n)
    net = I["logits"].argmax(-1).reshape(n, n)
    rep = {"model": model_id, "symbols": n, "additive_share": share}
    log(f"  MLP input additive: {share:.4f}")
    absorbing_a = [int(a) for a in range(n) if len(set(net[a].tolist())) == 1]
    absorbing_b = [int(b) for b in range(n) if len(set(net[:, b].tolist())) == 1]
    keep_a = np.array([a not in absorbing_a for a in range(n)]); keep_b = np.array([b not in absorbing_b for b in range(n)])
    rep["absorbing"] = {"a": absorbing_a, "b": absorbing_b}
    pa = factor.factor_side(F[keep_a], log=lambda s: log("  a: " + s))
    pb = factor.factor_side(G[keep_b], log=lambda s: log("  b: " + s))
    rep["a_parts"] = [int(P.max() + 1) for P in pa]; rep["b_parts"] = [int(P.max() + 1) for P in pb]
    L = I["logits"].reshape(n, n, -1)[np.ix_(keep_a, keep_b)]; L = L - L.mean(-1, keepdims=True)
    if len(pb) >= 2 and len(pa) != len(pb):                 # b split, a did not (or differently): read a's parts off the logits
        pa = factor.transfer_partition(L, pb); rep["a_parts_from_logits"] = [int(P.max() + 1) for P in pa]
        log(f"  a: parts read from the logits given b's parts: {rep['a_parts_from_logits']}")
    elif len(pa) >= 2 and len(pb) != len(pa):
        pb = factor.transfer_partition(L.transpose(1, 0, 2), pa); rep["b_parts_from_logits"] = [int(P.max() + 1) for P in pb]
        log(f"  b: parts read from the logits given a's parts: {rep['b_parts_from_logits']}")
    if len(pa) < 2 or len(pa) != len(pb) or True:
        # candidates on both sides (two parts each), judged by the logit test: the split that best explains the
        # network's own answers wins; the recursive result above is one of the candidates
        ca = [(1.0, sorted(pa, key=lambda P: P.max()))] if len(pa) == 2 else []
        cb = [(1.0, sorted(pb, key=lambda P: P.max()))] if len(pb) == 2 else []
        for wh in (False, True):
            ca += [(r[0], sorted([r[1], r[2]], key=lambda P: P.max())) for r in factor.split_candidates(factor._coords(F[keep_a], wh))]
            cb += [(r[0], sorted([r[1], r[2]], key=lambda P: P.max())) for r in factor.split_candidates(factor._coords(G[keep_b], wh))]
        tried = []
        for _, qa in ca[:8]:
            for _, qb in cb[:8]:
                _, ex_, agr_, _ = factor.additive_tables(L, qa, qb, iters=12)
                tried.append((agr_, ex_, qa, qb))
        if tried:
            agr_, ex_, qa, qb = max(tried, key=lambda z: (z[0], z[1]))
            log(f"  logit test over {len(tried)} candidate splits: best has {[int(P.max() + 1) for P in qa]} x {[int(P.max() + 1) for P in qb]} blocks, argmax agreement {agr_:.3f}")
            if len(pa) != 2 or len(pb) != 2 or agr_ > factor.additive_tables(L, sorted(pa, key=lambda P: P.max()), sorted(pb, key=lambda P: P.max()), iters=12)[2] if len(pa) == len(pb) == 2 else True:
                pa, pb = qa, qb
                rep["chosen_by_logit_test"] = True
    if len(pa) < 2 or len(pa) != len(pb):
        rep["verdict"] = "no split into independent parts" if len(pa) < 2 or len(pb) < 2 else f"the sides split differently ({rep['a_parts']} vs {rep['b_parts']})"
        return rep
    # pair the parts of a and b by block count (the k-th smallest with the k-th smallest)
    if "a_parts_from_logits" not in rep and "b_parts_from_logits" not in rep:
        pa = sorted(pa, key=lambda P: P.max()); pb = sorted(pb, key=lambda P: P.max())
    T, ex, agr, Lh = factor.additive_tables(L, pa, pb)
    rep["split_test"] = {"logit_share_explained": ex, "argmax_agreement": agr, "table_entries": int(sum(len(t) for t in T)),
                         "inputs": int(keep_a.sum() * keep_b.sum())}
    log(f"  logits = sum of {len(T)} part tables: {ex:.4f} of the variance, same answer on {agr:.4f} of inputs ({rep['split_test']['table_entries']} pair entries for {rep['split_test']['inputs']} inputs)")
    parts = []
    for i, (Tt, K) in enumerate(factor.part_tables_from_outputs(net[np.ix_(keep_a, keep_b)], pa, pb)):
        fp = factor.ring_fingerprint(Tt)
        parts.append({"a_blocks": int(pa[i].max() + 1), "b_blocks": int(pb[i].max() + 1), "values": int(K), "table": Tt.tolist(), "fingerprint": fp})
        log(f"  part {i}: {pa[i].max() + 1} x {pb[i].max() + 1} -> {K} values; matches {fp['matches']}; invertible core: {fp['invertible_core'].get('size')} elements, {fp['invertible_core'].get('group')}")
    rep["parts"] = parts
    rep["partition_a"] = [[int(x) for x in P] for P in pa]; rep["partition_b"] = [[int(x) for x in P] for P in pb]
    rep["kept_a"] = [int(a) for a in np.where(keep_a)[0]]; rep["kept_b"] = [int(b) for b in np.where(keep_b)[0]]
    ok = agr >= 0.98 and ex >= 0.9
    rep["verdict"] = (f"SPLITS into {len(T)} independent parts" if ok else "does not split cleanly") + f" (argmax agreement {agr:.3f}, logit share {ex:.3f})"
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); a = ap.parse_args()
    t0 = time.time(); rep = run(a.model); rep["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(ROOT, "results", "decomp"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, "results", "decomp", a.model + "_factor.json"), "w"), indent=1)
    print(a.model, rep["verdict"], [(p["values"], p["fingerprint"]["matches"]) for p in rep.get("parts", [])], f"{rep['seconds']}s")


if __name__ == "__main__":
    main()
