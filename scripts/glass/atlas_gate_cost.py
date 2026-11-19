import sys, pickle, numpy as np, torch; sys.path.insert(0, "lab")
import atlas, judge
tid = sys.argv[1]
P = pickle.load(open(f"results/glass/atlas/{tid}_d4_model.pkl", "rb"))
tgt = atlas.make_target(tid); mods = atlas.load_decomposition(tgt, P["spd"]); atlas.set_residuals(mods, P["alive"])
C = mods[next(iter(mods))].U.shape[0]; trees, alive, cc = P["trees"], P["alive"], P["conf_clear"]
X = tgt.X[:16000]; half = len(X) // 2; xtr = X[:half]
with torch.no_grad():
    net = tgt.forward(torch.tensor(xtr), lambda n, a: mods[n].apply(a)).numpy()
run = tgt.run_gated if hasattr(tgt, "run_gated") else (lambda m_, x_, g_: atlas.run_gated(tgt, m_, x_, g_))
am = {k: np.isin(np.arange(C), v).astype(float) for k, v in alive.items()}
def gate_only(sel):
    def g(k, names, F, a):
        m = np.repeat(am[k][None], len(F), 0)
        for (kk, c) in sel:
            if kk == k:
                m[:, c] = atlas._trigger_on(trees[(k, c)], F, atlas.GATE_TAU)
        return m
    return g
trusted = [(k, c) for k in alive for c in alive[k] if cc[(k, c)] >= 0.9 and not isinstance(trees[(k, c)], bool)]
full = judge.same(net, run(mods, xtr, gate_only(trusted))).mean()
costs = sorted(((1 - judge.same(net, run(mods, xtr, gate_only([kc]))).mean(), kc) for kc in trusted), reverse=True)
print(tid, f"training half: all trusted gated {full:.3f}; {len(trusted)} trusted; top single costs:", [(f"{k}.{c}", round(float(v), 4)) for v, (k, c) in costs[:8]], "sum", round(float(sum(v for v, _ in costs)), 3))
