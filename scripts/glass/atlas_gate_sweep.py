import sys, numpy as np, torch; sys.path.insert(0, "lab")
import atlas, judge
tid, spd = sys.argv[1], sys.argv[2]
tgt = atlas.make_target(tid); a = atlas.build_atlas(tgt, spd_model=spd, depth=4)
mods = atlas.load_decomposition(tgt, spd); atlas.set_residuals(mods, a["_alive"])
trees, alive, conf = a["_trees"], a["_alive"], a["_conf"]; C = 20
X = tgt.X[:16000]; xte = X[len(X) // 2:]
with torch.no_grad():
    net = tgt.forward(torch.tensor(xte), lambda n, a_: mods[n].apply(a_)).numpy()
def gate_p(th):
    def g(k, names, F, a_):
        m = np.zeros((len(F), C))
        for c in alive[k]:
            t = trees[(k, c)]
            if conf[(k, c)] < 0.9 or isinstance(t, bool):
                m[:, c] = 1.0 if (conf[(k, c)] < 0.9 or t) else 0.0
            else:
                pr = t.predict_proba(F)[:, list(t.classes_).index(True)]
                m[:, c] = (pr >= th).astype(float)
        return m
    return g
for th in (0.5, 0.3, 0.1, 0.03, 0.01):
    out = atlas.run_gated(tgt, mods, xte, gate_p(th)) if not hasattr(tgt, "run_gated") else tgt.run_gated(mods, xte, gate_p(th))
    print(tid, f"switch off only when P(on) < {th}: fidelity {judge.same(net, out).mean():.3f} (task rule {tgt.task_rule:.3f})", flush=True)
