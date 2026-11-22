"""D1: decompile a one-layer transformer from its weights, blind.

  python3 lab/decomp/decompile.py <model_id> [--patches 2000]

Reads only lab/models/<id>/weights.npz (never train.json or meta.json's answer). Writes
results/decomp/<id>.json (report) and results/decomp/<id>.py (the program).

Stages
  1  structure of the input symbols, per side (a = first input, b = second). Candidate representations: the embedding,
     and, when the MLP's input is additive (x1 = F(a) + G(b)), the side representations F and G. In each: circle planes
     and order-2 axes, relation lattice, the finite abelian group they form, every symbol's element. Several
     plane-finding variants; the one that places the most symbols with a consistent decode wins (weights only).
  2  the MLP output as characters of G x G (does it depend on g(a) + g(b) only?), output classes read from W_U along
     what the MLP writes, the program; a closed-form name for each symbol map when one fits (naming only)
  3  verification (network outputs used here only): M1 replacement, M2 causal patching vs. random subspaces
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lab"))
from decomp.circles import find_planes, find_structure
from decomp.group import recover_auto, recover
from decomp import program as pg
from decomp import deep
from decomp import orbit

VARIANTS = [("plain", lambda X: find_planes(X, restarts=24, iters=250)),
            ("axes+planes", lambda X: find_structure(X, restarts=24, iters=250)),
            ("whitened", lambda X: find_planes(X, restarts=16, iters=250, full=True, ratio=0.3))]


def _score(r):
    return (r["distinct"], float(np.mean(r["fit"][r["on"]])))


def stage1(X, log=print, tag=""):
    """All finite results for this representation: list of (score, variant, used planes, all planes, result)."""
    allr = []
    for name, fn in VARIANTS:
        t0 = time.time()
        P = fn(X)
        r = recover_auto(X, P)
        if not r.get("finite"):
            # relaxed fallback on the strongest planes (noisier representations): weaker relations, lower decode fit
            o = sorted(range(len(P)), key=lambda i: -P[i]["energy"])
            for k in (5, 4, 6):
                if len(o) >= k:
                    rr = recover(X, [P[i] for i in o[:k]], thresh=0.5, min_fit=0.75)
                    if rr.get("finite") and rr["distinct"] > 1:
                        r = rr; r["planes_used"] = o[:k]; name = name + "/relaxed"; break
        if r.get("finite"):
            key = _score(r)
            log(f"  stage 1 {tag}[{name}]: {len(P)} planes, group {r['group']}, {r['distinct']} elements placed, fit {key[1]:.3f} ({time.time() - t0:.0f}s)")
            allr.append((key, name, [P[i] for i in r["planes_used"]], P, r))
        else:
            log(f"  stage 1 {tag}[{name}]: {len(P)} planes, no finite group ({time.time() - t0:.0f}s)")
    return allr


def decompile(model_id, patches=2000, log=print):
    W = dict(np.load(os.path.join(ROOT, "lab", "models", model_id, "weights.npz")))
    n_out = W["W_U"].shape[1]
    n = W["W_E"].shape[0] - 1                                         # the last token is '='
    E = W["W_E"][:n]
    rep = {"model": model_id, "n_symbols": n, "n_out": n_out}
    F, G, share, I = deep.side_representations(W, n)
    net = I["logits"].argmax(-1)
    rep["additive_share"] = share
    log(f"  MLP input additive in (a, b): {share:.4f}")
    resE = stage1(E, log, "embedding ")
    cands = {"a": [("embedding", E, x) for x in resE], "b": [("embedding", E, x) for x in resE]}
    if share >= 0.9:
        cands["a"] += [("side F(a)", F, x) for x in stage1(F, log, "F(a) ")]
        cands["b"] += [("side G(b)", G, x) for x in stage1(G, log, "G(b) ")]
    # Selection (v3, Nov 21 23:30; v3.1 23:55; v3.2 00:25: times the share of the group's elements that hold a symbol). The two inputs share one embedding, so a symbol is the same element on
    # both sides: g_b = g_a. Among all finite results (every representation and variant), keep the one whose group sum
    # g(a) + g(b) explains the most of the MLP's output over all input pairs (times the share of symbols placed): a
    # group missing a factor merges different sums, and one made up by a false relation splits equal ones, so both
    # explain less. Internal activations only; no outputs.
    from collections import Counter
    allres = cands["a"] + [c for c in cands["b"] if c[0] != "embedding"]
    if not allres:
        return nonabelian(W, n, E, F, G, share, net, rep, patches, log)
    cnt = Counter(tuple(c[2][4]["group"]) for c in allres)
    rep["group_votes"] = {str(list(k)): v for k, v in cnt.items()}
    scored = []
    for c in allres:
        r_ = c[2][4]
        cover = min(1.0, r_["distinct"] / max(1, int(np.prod(r_["group"]))))      # a group larger than the symbols it holds is suspect
        scored.append((pg.sum_label_r2(I["mlp"], r_["g"], r_["group"], r_["on"]) * cover, c))
    scored.sort(key=lambda z: -z[0])
    rep["candidates_by_sum_label_r2"] = [{"representation": c[0], "variant": c[2][1], "group": c[2][4]["group"], "sum_label_r2": round(float(sc_), 4)} for sc_, c in scored[:8]]
    log("  candidates by group-sum R2: " + "; ".join(f"{c[0]}/{c[2][1]} {c[2][4]['group']} {sc_:.3f}" for sc_, c in scored[:5]))
    best_c = scored[0][1]
    (src_a, Xa, (_, var_a, used_a, all_a, ra)) = best_c
    src_b, var_b, rb = src_a, var_a, ra
    cands["a"] = [c for c in cands["a"] if c[0] == src_a] or cands["a"]
    d = ra["group"]
    ga, on_a = ra["g"], ra["on"]
    gb, on_b = ga, on_a
    power, Mhat, els = pg.mlp_characters(I["mlp"], ga, gb, d, on_a, on_b)
    dec = pg.decode_outputs(W["W_U"], Mhat, power, d)
    f, src = make_program2(ga, gb, dec, d, on_a, on_b)
    m1 = pg.verify_replacement(f, net, n)
    # M2 patches every plane (from every variant run on a's representation) that is a character of the recovered
    # coordinate: patching only some of the characters the network uses leaves the others pointing at the old symbol
    # M2: move one input onto another symbol only inside the subspace carried by characters of the recovered
    # coordinates (in the embedding, and in the side representation when that is where a's coordinates were found)
    tests = {}
    QE = pg.coordinate_subspace(E, ga, d, on_a)
    tests["embedding"] = (pg.verify_causal_subspace(W, QE, "a", E, f, n, "embedding", patches),
                          pg.verify_causal_subspace(W, QE, "a", E, f, n, "embedding", patches, random=True), QE.shape[1])
    if share >= 0.9:                       # the same symbol coordinates hold on both sides (shared embedding)
        QF = pg.coordinate_subspace(F, ga, d, on_a)
        tests["side F(a)"] = (pg.verify_causal_subspace(W, QF, "a", F, f, n, "x1", patches),
                              pg.verify_causal_subspace(W, QF, "a", F, f, n, "x1", patches, random=True), QF.shape[1])
        QG = pg.coordinate_subspace(G, gb, d, on_b)
        tests["side G(b)"] = (pg.verify_causal_subspace(W, QG, "b", G, f, n, "x1", patches),
                              pg.verify_causal_subspace(W, QG, "b", G, f, n, "x1", patches, random=True), QG.shape[1])
    rep["M2_all"] = {k: {"success": v[0]["success"], "random": v[1]["success"], "dims": v[2]} for k, v in tests.items()}
    if tests:
        kbest = max(tests, key=lambda k: tests[k][0]["success"])
        m2, m2r, npl = tests[kbest]; m2 = dict(m2, where=kbest, dims=npl)
    else:
        m2, m2r = {"success": 0.0, "n": 0}, {"success": 0.0, "n": 0}
    names = {}
    if len(d) == 1:
        names["a"] = pg.name_map(np.where(on_a, ga[:, 0], -1), d[0], n)
        names["b"] = pg.name_map(np.where(on_b, gb[:, 0], -1), d[0], n)
    rep.update({
        "group": d, "order": int(np.prod(d)),
        "a_side": {"representation": src_a, "variant": var_a, "elements_placed": int(ra["distinct"]), "off_circle": [int(i) for i in np.where(~on_a)[0]], "name": names.get("a")},
        "b_side": {"representation": src_b, "variant": var_b, "elements_placed": int(rb["distinct"]), "off_circle": [int(i) for i in np.where(~on_b)[0]], "name": names.get("b")},
        "mlp_diagonal_share": pg.diagonal_share(power, d),
        "outputs": {"argmax_distinct": dec["argmax_distinct"], "moved_by_assignment": dec["moved_by_assignment"], "leftover": dec["leftover"]},
        "M1": m1, "M2": m2, "M2_random_control": m2r,
        "element_of_symbol_a": {int(i): [int(x) for x in ga[i]] for i in range(n) if on_a[i]},
        "element_of_symbol_b": {int(i): [int(x) for x in gb[i]] for i in range(n) if on_b[i]},
        "element_of_class": {int(c): [int(x) for x in dec["h"][c]] for c in range(n_out) if c not in dec["leftover"]},
    })
    ok1 = m1["agreement"] >= 0.90
    ok2 = m2["success"] >= 0.80 and m2r["success"] < 0.2
    rep["verdict"] = ("PASS" if ok1 and ok2 else "FAIL") + f" (M1 {m1['agreement']:.3f}, M2 {m2['success']:.3f} vs random {m2r['success']:.3f})"
    return rep, src


def character_planes(X, pool, g, d, on, min_coh=0.8):
    """Planes from the pool whose angles follow a character of the recovered coordinates; near-duplicates dropped."""
    from decomp.group import plane_angles
    from decomp.program import characters
    if not pool:
        return []
    TH, RAD = plane_angles(X, pool)
    chi, els = characters(d)
    idx = pg.element_index(g, d)
    keep, spans = [], []
    for i, pl in enumerate(pool):
        z = np.exp(1j * TH[i][on])
        coh = np.abs((chi[:, idx[on]].conj() * z[None, :]).mean(1)).max()
        if coh < min_coh:
            continue
        Q = np.linalg.qr(pl.get("Qo", pl["Q"]))[0]
        if any(np.linalg.svd(Q.T @ S, compute_uv=False).max() > 0.9 for S in spans):
            continue
        keep.append(pl); spans.append(Q)
    return keep


def nonabelian(W, n, E, F, G, share, net, rep, patches, log):
    """No abelian structure: try the symbols as an orbit of an orthogonal group action (lab/decomp/orbit.py)."""
    reps = ([("side F(a)", F)] if share >= 0.9 else []) + [("embedding", E)]
    for src, X in reps:
        sph = orbit.find_sphere(X)
        if sph is None:
            log(f"  orbit [{src}]: no sphere"); continue
        k, Y, basis, cv = sph
        for tol in (0.1, 0.2):                            # D9: a looser match if the tight one finds no regular subgroup
            P = orbit.orbit_symmetries(Y, tol)
            H_all = orbit.regular_subgroups(P, n) if len(P) >= n else []
            if len(P) >= 2:                               # D9: subgroups that need more than two generators
                seen_ = {frozenset(H) for H in H_all}
                H_all += [H for H in orbit.regular_subgroups_greedy(P, n) if frozenset(H) not in seen_]
            if H_all:
                break
        log(f"  orbit [{src}]: sphere in {k} dims (radius cv {cv:.3f}), {len(P)} symmetries, {len(H_all)} regular subgroups of order {n}")
        if not H_all:
            continue
        name, eo = orbit.describe(H_all[0], n)
        # candidates: identity symbol and handedness are not fixed by the orbit; choose by M1 (disclosed)
        best = None; ncand = 0
        for H in H_all:
            for e, T in orbit.tables(H, n):
                for T2 in (np.array(T), np.array(T).T):
                    ncand += 1
                    agr = float((T2.reshape(-1) == net).mean())
                    if best is None or agr > best[0]:
                        best = (agr, e, T2)
        agr, e, T = best
        f = lambda a, b, T=T: int(T[a, b])
        m1 = pg.verify_replacement(f, net, n)
        where = "embedding" if src == "embedding" else "x1"
        Xs = E if src == "embedding" else F
        # interchange on the recovered variable (here the group element itself): the subspace through which the
        # representation depends on it, as in M2 v2
        Q = pg.coordinate_subspace(Xs, np.arange(n)[:, None], [n], np.ones(n, bool))
        m2 = pg.verify_causal_subspace(W, Q, "a", Xs, f, n, where, patches)
        m2r = pg.verify_causal_subspace(W, Q, "a", Xs, f, n, where, patches, random=True)
        abel = bool((T == T.T).all())
        if abel:                                          # D9: an abelian table is named by its invariant factors
            Hs = [tuple(int(T[a, b]) for b in range(n)) for a in range(n)]
            inv_ = orbit.abelian_invariants(orbit.element_orders(Hs, n), n)
            name = "abelian, " + (" x ".join(f"Z_{x}" for x in inv_) if inv_ else "invariant factors not identified")
        rep.update({"group": (f"abelian, order {n}" if abel else f"non-abelian, order {n}"), "group_name": name, "element_orders": {str(k2): v for k2, v in eo.items()},
                    "a_side": {"representation": src, "variant": f"orbit on a sphere in {k} dims"}, "orbit_symmetries": len(P),
                    "regular_subgroups": len(H_all), "candidates_selected_by_M1": ncand,
                    "M1": m1, "M2": dict(m2, where=src), "M2_random_control": m2r,
                    "cayley_table": T.tolist(), "identity_symbol": int(e)})
        ok = m1["agreement"] >= 0.9 and m2["success"] >= 0.8 and m2r["success"] < 0.2
        rep["verdict"] = ("PASS" if ok else "FAIL") + f" (M1 {m1['agreement']:.3f}, M2 {m2['success']:.3f} vs random {m2r['success']:.3f}); identity and handedness chosen by M1 among {ncand} candidates"
        src_txt = [f"# Decompiled program: a non-abelian group of order {n}" + (f", identified as {name}" if name else "") + f"; element orders {eo}",
                   "# The table was read from the symmetries of the embedding's orbit; which symbol is the identity (and left vs right)",
                   f"# was chosen by agreement with the network among {ncand} relabelled candidates.",
                   "CAYLEY = " + json.dumps(T.tolist()), "def f(a, b):", "    return CAYLEY[a][b]"]
        return rep, "\n".join(src_txt)
    rep.update({"verdict": "no group structure found", "group": None})
    return rep, None


def make_program2(ga, gb, dec, d, on_a, on_b):
    """c = CLASS_OF[ g_a(a) + g_b(b) ], with an absorbing class for symbols off every circle when the outputs leave
    exactly one class unclaimed."""
    n = len(ga); dd = np.array(d)
    leftover = dec["leftover"]
    inv = {}
    for c, h in enumerate(dec["h"]):
        if c not in leftover:
            inv.setdefault(tuple(int(x) for x in h), c)
    off = sorted(set(np.where(~on_a)[0].tolist()) | set(np.where(~on_b)[0].tolist()))
    absorbing = leftover[0] if (len(leftover) == 1 and 1 <= len(off) <= 2) else None
    GA = [tuple(int(x) for x in g) for g in ga]; GB = [tuple(int(x) for x in g) for g in gb]

    def f(a, b):
        if not (on_a[a] and on_b[b]):
            return absorbing
        return inv.get(tuple(int(x) for x in (np.array(GA[a]) + np.array(GB[b])) % dd))
    same = all(GA[i] == GB[i] for i in range(n) if on_a[i] and on_b[i])
    lines = [f"# Decompiled program (group Z_{' x Z_'.join(map(str, d))}); every table below was read from the weights",
             f"G = {list(d)}",
             "ELEMENT_A = {" + ", ".join(f"{i}: {GA[i]}" for i in range(n) if on_a[i]) + "}"]
    lines.append("ELEMENT_B = ELEMENT_A" if same else "ELEMENT_B = {" + ", ".join(f"{i}: {GB[i]}" for i in range(n) if on_b[i]) + "}")
    lines.append("CLASS_OF = {" + ", ".join(f"{k}: {v}" for k, v in sorted(inv.items())) + "}")
    if absorbing is not None:
        lines.append(f"ABSORBING = {absorbing}   # symbols {off} sit off every circle; output class {absorbing} is claimed by no element")
    lines += ["def f(a, b):"]
    if absorbing is not None:
        lines += ["    if a not in ELEMENT_A or b not in ELEMENT_B:", "        return ABSORBING"]
    lines += ["    s = tuple((x + y) % m for x, y, m in zip(ELEMENT_A[a], ELEMENT_B[b], G))", "    return CLASS_OF[s]"]
    return f, "\n".join(lines)


def decompile_any(model_id, patches=2000, log=print):
    """Abelian mode first; if it finds no structure or its program fails, the non-abelian orbit mode, keeping the better."""
    rep, src = decompile(model_id, patches, log)
    if rep.get("verdict", "").startswith("PASS"):
        return rep, src
    W = dict(np.load(os.path.join(ROOT, "lab", "models", model_id, "weights.npz")))
    n = W["W_E"].shape[0] - 1
    F, G, share, I = deep.side_representations(W, n)
    rep2, src2 = nonabelian(W, n, W["W_E"][:n], F, G, share, I["logits"].argmax(-1), {"model": model_id, "n_symbols": n, "n_out": W["W_U"].shape[1], "additive_share": share,
                                                                                       "abelian_attempt": {k: rep.get(k) for k in ("group", "verdict", "group_votes")}}, patches, log)
    if rep2.get("verdict", "").startswith("PASS") or not rep.get("group"):
        return rep2, src2
    return rep, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--patches", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    rep, src = decompile_any(a.model, a.patches)
    rep["seconds"] = round(time.time() - t0, 1)
    out = os.path.join(ROOT, "results", "decomp"); os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, a.model + ".json"), "w"), indent=1)
    if src:
        open(os.path.join(out, a.model + ".py"), "w").write(src + "\n")
    print(a.model, rep.get("group"), rep["verdict"], "| names:", rep.get("a_side", {}).get("name"), "/", rep.get("b_side", {}).get("name"), f"{rep['seconds']}s")


if __name__ == "__main__":
    main()
