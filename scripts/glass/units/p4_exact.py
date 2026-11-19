"""P4: exact structure: signed pairs (h_i - c h_j linear in inputs) and exact redundancy (rank deficiency), at tolerance eps95."""
import sys, json, itertools
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets, unitmap
res = {}
for target in sys.argv[1:] or ["zoo_2nd_argmax_4_3", "zoo_2nd_argmax_8_5", "zoo_2nd_argmax_16_10", "zoo_argmedian_8_5", "zoo_argmedian_16_5"]:
    meta, W, X = judge.load_target(target); X = X[:4000]; e = unitmap.eps95(W, X)
    acts = {}; nets.forward("rnn_relu", W, X, acts=acts); T = X.shape[1]; paired = set(); tot = 0; rankdef = []
    for t in range(2, T + 1):
        H = np.asarray(acts[f"h{t}"]).reshape(len(X), -1); act = [j for j in range(H.shape[1]) if H[:, j].std() > 1e-9]; tot += len(act)
        Xi = X[:, :t]
        for i, j in itertools.combinations(act, 2):
            # h_i - c h_j = x @ a : solve jointly for (a, c)
            A = np.c_[Xi, H[:, j]]; sol = np.linalg.lstsq(A, H[:, i], rcond=None)[0]
            r = H[:, i] - A @ sol; scale = max(H[:, i].std(), abs(sol[-1]) * H[:, j].std())
            if np.sqrt((r ** 2).mean()) / scale <= e and abs(sol[-1]) > 1e-3:
                paired |= {(t, i), (t, j)}
        Hc = (H[:, act] - H[:, act].mean(0)) / H[:, act].std(0)
        sv = np.linalg.svd(Hc, compute_uv=False) / np.sqrt(len(X))
        rankdef.append(int(len(act) - (sv > e).sum()))
    res[target] = {"eps95": e, "paired_share": len(paired) / max(tot, 1), "rank_deficiency_per_step": rankdef}
    print(f"{target:22s} eps95 {e:.4f} | neuron-steps in exact signed pairs: {len(paired)}/{tot} ({100*len(paired)/max(tot,1):.0f}%) | exact rank deficiency per step: {rankdef}", flush=True)
(ROOT / "results/glass/units/p4_exact.json").write_text(json.dumps(res, indent=1))
