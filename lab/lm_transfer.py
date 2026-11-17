"""Transfer test: does a construction verified on one LM task hold, unchanged, on prompts it was never fitted on?

A verified construction makes claims beyond its own prompts: model(x) says what the network answers for ANY input
row, and ALIGN says where each variable lives and (through the judge's affine maps) how it is encoded. This tool
fits the maps on the ORIGINAL task only and tests them on a TRANSFER task (other prompts with the same features, roles
and answers: other years, another template, ...). Nothing is refitted except each site's mean on the new prompts.

Per cut (all residual sites the construction names at one depth):
  substitute   named positions <- c_new + variables @ M_orig, other positions <- their mean on the new prompts
  interchange  on pairs of new prompts (A, B): the network run on B with its cut's variable subspace (read with D_orig)
               replaced by A's, against model(B) with the variables set to their values on A
and behaviour: model(x_new) against the network's decisions on the new prompts, next to a constant baseline.

Run:  python3 lab/lm_transfer.py <target> <run id | construction.py> <transfer target>
Writes transfer_<transfer target>.json / .txt in the run folder when given a run id.
"""
import json, sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import judge as J    # noqa: E402
import lm            # noqa: E402
from lmjudge import Unit, categorical  # noqa: E402

N_PAIRS = 300


def source_of(tid, what):
    p = Path(what)
    if p.exists():
        return p.read_text(), None
    rdir = LAB / "constructions" / tid / what
    if (rdir / "construction.py").exists():
        return (rdir / "construction.py").read_text(), rdir
    rec = json.loads((rdir / "run.json").read_text())
    if not rec.get("best"):
        raise SystemExit("this run has no construction yet")
    return (rdir / f"submission_{rec['best']:02d}.py").read_text(), rdir


def transfer(tid, source, tid_new, seed=0, n_pairs=N_PAIRS):
    rng = np.random.default_rng(seed)
    old, new = lm.get_task(tid), lm.get_task(tid_new)
    for k in ("x_names",):
        if old.meta[k] != new.meta[k]:
            raise SystemExit(f"the two tasks differ in {k}: {old.meta[k]} vs {new.meta[k]}")
    if old.meta["model"] != new.meta["model"]:
        raise SystemExit("the two tasks use different models")
    model, align = J.load_construction(source)
    out_old, out_new = old.run(np.arange(old.N)), new.run(np.arange(new.N))
    if out_old.shape[1] != out_new.shape[1]:
        raise SystemExit("the two tasks have different numbers of answers")
    pred_new, vals_new = J.run_model(model, new.X)
    _, vals_old = J.run_model(model, old.X)
    const = np.broadcast_to(out_new.mean(0), out_new.shape)
    rep = {"target": tid, "transfer_to": tid_new, "n_new": new.N,
           "behaviour": {"match": J.match(pred_new, out_new), "baseline_constant": J.match(const, out_new),
                         "network_decisions": np.bincount(out_new.argmax(1), minlength=out_new.shape[1]).tolist()},
           "cuts": {}}
    names_res = sorted({s for w in align.values() for s in w if lm.parse_site(s)[0] == "resid"}, key=lambda s: (lm.parse_site(s)[1], s))
    test = rng.permutation(new.N)[: min(new.N, 400)]
    base, src = rng.choice(new.N, n_pairs), rng.choice(new.N, n_pairs)
    for L in sorted({lm.parse_site(s)[1] for s in names_res}):
        sites = [s for s in names_res if lm.parse_site(s)[1] == L]
        vs = [v for v in vals_old if any(s in align.get(v, []) for s in sites)]
        if not vs:
            continue
        uo, un = Unit(old, f"cut{L}", sites, "cut"), Unit(new, f"cut{L}", sites, "cut")
        cols, k0 = {}, 0
        for v in vs:
            cols[v] = np.arange(k0, k0 + vals_old[v].shape[1]); k0 += vals_old[v].shape[1]
        Vo = np.concatenate([vals_old[v] for v in vs], 1)
        Vn = np.concatenate([vals_new[v] for v in vs], 1)
        Ao, An = uo.acts(np.arange(old.N)), un.acts(np.arange(new.N))
        co, vmu = Ao.mean(0), Vo.mean(0)
        K, Wd = Vo.shape[1], int(uo.off[-1])
        M, D, cnt = np.zeros((K, Wd)), np.zeros((Wd, K)), np.zeros(K)
        for i, s in enumerate(sites):                      # the maps of the judge, fitted on the ORIGINAL prompts only
            Jc = np.concatenate([cols[v] for v in vs if s in align[v]])
            sl = slice(uo.off[i], uo.off[i + 1])
            Ms, *_ = np.linalg.lstsq(Vo[:, Jc] - vmu[Jc], Ao[:, sl] - co[sl], rcond=1e-8)
            _, svs, Vts = np.linalg.svd(Ao[:, sl] - co[sl], full_matrices=False)
            rr = max(1, min(int((svs > 1e-6 * svs[0]).sum()), max(8, old.N // 8)))
            Dz, *_ = np.linalg.lstsq((Ao[:, sl] - co[sl]) @ Vts[:rr].T, Vo[:, Jc] - vmu[Jc], rcond=1e-6)
            M[Jc, sl] = Ms; D[sl, Jc] = Vts[:rr].T @ Dz; cnt[Jc] += 1
        D = D / np.maximum(cnt, 1)
        cn, vmn = An.mean(0), Vn.mean(0)
        clean = out_new[test]
        others = new.resid(L, np.arange(new.N)).mean(0)
        o_sub = un.run_cut(test, cn + (Vn[test] - vmn) @ M, others)
        o_mean = un.run(test, np.broadcast_to(cn, An[test].shape))
        expl = float(1 - (((cn + (Vn - vmn) @ M) - An) ** 2).sum() / max(((An - cn) ** 2).sum(), 1e-12))
        # interchange on new pairs
        new_vals = {v: vals_new[v][src] for v in vs}
        o_pred, _ = J.run_model(model, new.X[base], new_vals)
        o_net = un.run(base, An[base] + ((An[src] - An[base]) @ D) @ M)
        changed = ~J.same(o_net, out_new[base])
        agree = J.same(o_pred, o_net)
        rep["cuts"][f"cut{L}"] = {"sites": sites, "variables": vs, "explained": expl,
                                  "substitute_match": J.match(o_sub, clean), "mean_ablate_match": J.match(o_mean, clean),
                                  "interchange": float(agree.mean()), "network_changed": float(changed.mean()),
                                  "interchange_when_changed": float(agree[changed].mean()) if changed.any() else None,
                                  "baseline_no_change": J.match(out_new[base], o_net)}
    return rep


def render(rep):
    b = rep["behaviour"]
    L = [f"TRANSFER  construction fitted on {rep['target']}, tested unchanged on {rep['transfer_to']} ({rep['n_new']} prompts it never saw)",
         f"behaviour: same decision as the network on {b['match']:.1%} (a constant answer: {b['baseline_constant']:.1%}; the network's decisions: {b['network_decisions']})"]
    for u, r in rep["cuts"].items():
        w = r["interchange_when_changed"]
        L.append(f"  {u:6s} ({', '.join(r['variables'])}): explained {r['explained']:.2f} | substitute {r['substitute_match']:.1%} "
                 f"(mean-ablate {r['mean_ablate_match']:.1%}) | interchange {r['interchange']:.1%}"
                 + (f", {w:.1%} when the network changed ({r['network_changed']:.0%} of pairs; no-change baseline {r['baseline_no_change']:.1%})" if w is not None else ""))
    return "\n".join(L)


if __name__ == "__main__":
    tid, what, tid_new = sys.argv[1:4]
    src, rdir = source_of(tid, what)
    rep = transfer(tid, src, tid_new)
    txt = render(rep)
    print(txt)
    if rdir is not None:
        (rdir / f"transfer_{tid_new}.json").write_text(json.dumps(rep, indent=1))
        (rdir / f"transfer_{tid_new}.txt").write_text(txt + "\n")
