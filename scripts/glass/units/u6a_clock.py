"""U6a: do 2nd-argmax / argmedian RNNs contain a rotation by ~360/n degrees (a position clock)?"""
import sys, json
from pathlib import Path
import numpy as np
from collections import Counter
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets

def angles(M):
    ev = np.linalg.eigvals(M)
    return [(abs(e), abs(np.degrees(np.angle(e)))) for e in ev if abs(e) >= 0.5 and abs(e.imag) > 1e-6]

def clock_hit(angs, n, tol=0.15):
    base = 360.0 / n
    return any(abs(a - k * base) <= tol * base for _, a in angs for k in range(1, n // 2 + 1))

res = {}
for t in sys.argv[1:] or ["zoo_2nd_argmax_4_3", "zoo_2nd_argmax_4_4", "zoo_2nd_argmax_8_5", "zoo_2nd_argmax_8_8", "zoo_2nd_argmax_16_10",
                          "zoo_argmedian_8_5", "zoo_argmedian_8_7", "zoo_argmedian_16_5", "zoo_argmedian_16_7"]:
    for var in ("original", "shuffled"):
        meta, W, X = judge.load_target(t, var, 0); X = X[:4000]; n = X.shape[1]
        Wh = W["W_h"]                                     # s_t = relu(x W_x + s_{t-1} W_h): row-vector convention
        acts = {}; nets.forward("rnn_relu", W, X, acts=acts)
        pats = Counter()
        for k in range(2, n + 1):
            P = (np.asarray(acts[f"h{k}"]).reshape(len(X), -1) > 0)
            pats.update(map(lambda r: r.tobytes(), P))
        top = np.frombuffer(pats.most_common(1)[0][0], dtype=bool)
        D = np.diag(top.astype(float))
        a_full = angles(Wh.T); a_act = angles((Wh @ D).T)  # effective map when the top pattern's units are on
        hit = clock_hit(a_full, n) or clock_hit(a_act, n)
        res[f"{t}|{var}"] = {"n": n, "full": [(round(m, 2), round(a, 1)) for m, a in a_full], "active": [(round(m, 2), round(a, 1)) for m, a in a_act],
                             "top_pattern_share": round(pats.most_common(1)[0][1] / sum(pats.values()), 3), "clock": bool(hit)}
        print(f"{t:22s} {var:9s} n={n} 360/n={360/n:5.1f} | full {res[f'{t}|{var}']['full']} | active {res[f'{t}|{var}']['active']} | CLOCK={hit}", flush=True)
(ROOT / "results/glass/units/u6a.json").write_text(json.dumps(res, indent=1))
