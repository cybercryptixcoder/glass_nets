"""The box, blind, on the trained network of box/learned_net.py (Gate 2 rehearsal; development analog of S3).

The box is given the network (weights and sites), the positions it is run on, and the frozen vocabulary. It is not
given the verdict's formula. Its steps:
1. Candidate variables, at each of the sites z1 and h1: every position-level expression of the vocabulary up to length
   4 (distinct on 60 probe positions), then compositions of the 20 best read (differences, sums, and the relu of each
   difference; up to length 6). Each is read from the site by a linear probe fitted on training positions and scored on
   validation positions.
2. Located candidates: probe R2 above a control line (the same probes on a copy of the network with layer 1's weights
   permuted; the 99th percentile over the 200 best candidates).
3. The network's own variables among them, ranked by how much the network's output depends on them (the output's
   gradient at the site, along the change that moves the candidate's readout by one standard deviation), keeping six
   that are not near-copies of one another (|correlation| below 0.9).
4. The verdict defined through them (the vocabulary search, L = 6, the parts as atoms), against the network's own
   output on validation positions.
5. The blueprint, certified by the vector-site checker v1 on held-out positions the box never used. The run-alone line
   is supplied by the harness from the true account (R2's line on planted networks; the box does not see it).
Run: python3 -m box.learned_box"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box import learned_net as LN  # noqa: E402
from box.blueprint import Blueprint, Part  # noqa: E402
from box.search import best_definitions, enumerate_exprs, vocab  # noqa: E402
from box.verify_vec import VecChecker  # noqa: E402

t0 = time.time()
rng = np.random.default_rng(0)
d, ctx, X, _v_unused, net, info = LN.load()       # the box uses the network and the positions, not the verdict
TR, VA, TE = LN.TR, LN.VA, LN.TE
_, sites = net.run(np.arange(LN.N), return_sites=True)
OUT = sites["out"][:, 0]
p = net.p
z2 = sites["z2"]
# the output's gradient at each site, per position
J = {"h1": (p["W3"][0] * (z2 > 0)) @ p["W2"]}
J["z1"] = J["h1"] * (sites["z1"] > 0)
# control network: layer 1's weights permuted
W1c = rng.permutation(p["W1"].ravel()).reshape(p["W1"].shape)
Zc = X @ W1c.T + p["b1"]
csites = {"z1": Zc, "h1": np.maximum(Zc, 0)}
r2 = lambda a, y: float(1 - np.mean((a - y) ** 2) / np.var(y))  # noqa: E731


def probe(S, y):
    A = np.column_stack([S[TR], np.ones(len(TR))])
    w, *_ = np.linalg.lstsq(A, y[TR], rcond=1e-6)
    return w[:-1], r2(np.column_stack([S[VA], np.ones(len(VA))]) @ w, y[VA])


ev = lambda e: LN.evaluate(d, ctx, e)  # noqa: E731
probe_rows = list(range(60))
cprobe = {k: (val[probe_rows] if k != "fens" else [val[i] for i in probe_rows]) for k, val in ctx.items()}
pool = enumerate_exprs(d, cprobe, 4, cap=60000)
base = [t for t, _v, n in pool["Pos"] if n >= 2]
print(f"{len(base)} base candidates ({round(time.time() - t0)} s)", flush=True)
values = {}
for t in base:
    y = ev(t)
    if y[TR].std() > 0 and y[VA].std() > 0:
        values[t] = y

report = {"network": info, "sites": {}}
best_bp, best_score = None, -9
for site in ("z1", "h1"):
    S = sites[site]
    scored = sorted(((probe(S, y)[1], t) for t, y in values.items()), reverse=True)
    top = [t for _s, t in scored[:20]]
    comp = {}
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            for t in (f"(sub {a} {b})", f"(add {a} {b})", f"(relu (sub {a} {b}))", f"(relu (sub {b} {a}))"):
                ok, _why, ty = vocab.admissible(t, d, {}, {})
                if ok and ty == "Pos":
                    y = ev(t)
                    if y[TR].std() > 0 and y[VA].std() > 0:
                        comp[t] = y
    cand = dict(values); cand.update(comp)
    rows = []
    for t, y in cand.items():
        w, s = probe(S, y)
        e = w / max(np.dot(w, w), 1e-12)                         # the change that moves the readout by one unit
        rel = float(np.mean(np.abs(J[site][TR] @ e)) * y[TR].std())   # output change per standard deviation
        rows.append((s, rel, t, w))
    rows.sort(key=lambda z: -z[0])
    ctrl = sorted((probe(csites[site], cand[t])[1] for _s, _r, t, _w in rows[:200]), reverse=True)
    line = float(np.quantile(ctrl, 0.99))
    located = [r for r in rows if r[0] > line]
    located.sort(key=lambda z: -z[1])
    kept = []
    for s, rel, t, w in located:
        if any(abs(np.corrcoef(cand[t][TR], cand[k[2]][TR])[0, 1]) > 0.9 for k in kept):
            continue
        kept.append((s, rel, t, w))
        if len(kept) >= 6:
            break
    print(f"site {site}: {len(cand)} candidates, {len(located)} above the probe line {line:.3f}; the six the output depends on most:", flush=True)
    for s, rel, t, w in kept:
        print(f"   {t}: probe R2 {s:.3f}, output change per sd {rel:.2f}", flush=True)
    fit = np.arange(0, 2000)
    pf = {f"v{i}": ("Pos", cand[k[2]][fit]) for i, k in enumerate(kept)}
    pv = {f"v{i}": ("Pos", cand[k[2]][VA]) for i, k in enumerate(kept)}
    sub = lambda rows_: {k: (val[rows_] if k != "fens" else [val[i] for i in rows_]) for k, val in ctx.items()}  # noqa: E731
    defs, _ = best_definitions(d, sub(fit), OUT[fit], 6, "Pos", k=5, ctx_test=sub(VA), target_test=OUT[VA],
                               parts_fit=pf, parts_test=pv, only_types=("Pos",))
    # the definition whose values reproduce the output itself (the decision is the output rounded)
    for o in defs:
        dv = vocab.evaluate(o["definition"], d, sub(VA), {k: pv[k][1] for k in pv}, {k: "Pos" for k in pv})
        o["heldout_r2_exact"] = r2(np.asarray(dv, float), OUT[VA])
    defs.sort(key=lambda o: -o["heldout_r2_exact"])
    top_def = defs[0]
    print(f"   verdict through them: {top_def['definition']} (validation R2 against the output, as is: {top_def['heldout_r2_exact']:.3f})", flush=True)
    report["sites"][site] = {"probe_line": line, "n_candidates": len(cand), "n_located": len(located),
                             "kept": [{"definition": t, "probe_r2": s, "output_change_per_sd": rel} for s, rel, t, w in kept],
                             "verdict_candidates": defs}
    if top_def["heldout_r2_exact"] > best_score:
        best_score = top_def["heldout_r2_exact"]
        used = sorted({int(x[2:]) for x in top_def["definition"].replace("(", " ").replace(")", " ").split() if x.startswith("@v")})
        parts = []
        for j in used:
            s, rel, t, w = kept[j]
            parts.append(Part(f"v{j}", "rule", "Pos", site, basis=w / np.linalg.norm(w), encoding="numerical",
                              scale=float(np.linalg.norm(w)), definition=t,
                              evidence={"probe_r2": s, "output_change_per_sd": rel}))
        parts.append(Part("verdict", "rule", "Pos", "out", basis=np.array([1.0]), encoding="numerical",
                          definition=top_def["definition"], upstream=[f"v{j}" for j in used]))
        best_bp = Blueprint("chess", parts, "verdict", name="trained staged network (blind box)")
        best_site = site
print(f"box done {round(time.time() - t0)} s; chosen site {best_site}", flush=True)

# ---- harness: the run-alone line from the true account (hidden from the box), then the checker on held-out positions
vals = {k: LN.evaluate(d, ctx, LN.TRUTH[k]) for k in "ABC"}
true_parts = []
for k in "ABC":
    w, _ = probe(sites[best_site], vals[k])
    true_parts.append(Part(k, "rule", "Pos", best_site, basis=w / np.linalg.norm(w), encoding="numerical",
                           scale=float(np.linalg.norm(w)), definition=LN.TRUTH[k]))
true_parts.append(Part("verdict", "rule", "Pos", "out", basis=np.array([1.0]), encoding="numerical",
                       definition=LN.TRUTH["verdict"], upstream=list("ABC")))
true_bp = Blueprint("chess", true_parts, "verdict", name="true account")
chk = VecChecker(net, TE[:, None], seed=0, n_pairs=800, control_reps=200, ctx_fn=LN.ctx_fn_for(ctx))
ra_true = chk.run_alone(true_bp)
line = ra_true - 2.5 * np.sqrt(ra_true * (1 - ra_true) / len(TE))
rep = chk.certify(best_bp, line)
print(f"checker {VecChecker.VERSION}: {rep['verdict']}; run-alone line {line:.4f} (true account {ra_true:.4f}); lines {rep.get('control_lines')}", flush=True)
for c in rep["checks"]:
    print(f"   {'ok ' if c[1] else 'NO '} {c[0]}: {c[2]}", flush=True)
for p_ in best_bp.parts:
    print(f"   part {p_.id} @ {p_.site}: {p_.definition}", flush=True)
report["blueprint"] = {"site": best_site, "parts": [{"id": q.id, "site": q.site, "definition": q.definition,
                                                       "upstream": q.upstream} for q in best_bp.parts]}
report["certify"] = {"checker": VecChecker.VERSION, "verdict": rep["verdict"], "run_alone_line": line,
                     "true_account_run_alone": ra_true, "lines": rep.get("control_lines"),
                     "checks": [(c[0], bool(c[1]), c[2]) for c in rep["checks"]]}
report["seconds"] = round(time.time() - t0)
json.dump(report, open("results/build/learned_box_session2.json", "w"), indent=1, default=float)
try:
    from box.report import save
    save("learned_staged", best_bp, rep, "results/build/blueprints")
except Exception as ex:  # the readable blueprint is a convenience; the JSON above is the record
    print("blueprint markdown not written:", ex)
print("done", round(time.time() - t0), "s")
