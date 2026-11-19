"""Exploration (development set only): absorb the residual into the alive mechanisms' output directions.
With each alive mechanism's input direction V_c fixed (so its importance, and therefore its trigger, is unchanged), refit the
output directions U_alive by least squares on the matrix's real inputs (training half): min || A W^T - (A V_alive^T) U_alive ||.
Then rebuild the atlas from the adjusted decomposition and compare mechanisms-alone and gated fidelities."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
tid, spdp, out = sys.argv[1], sys.argv[2], sys.argv[3]
tgt = atlas.make_target(tid); mods = atlas.load_decomposition(tgt, spdp)
X = tgt.X[:16000]; xtr = X[:len(X) // 2]
calls = tgt.network_calls if hasattr(tgt, "network_calls") else (lambda m_, x_: atlas._network_calls(tgt, m_, x_))
tr = calls(mods, xtr); C = mods[next(iter(mods))].U.shape[0]
alive = {k: [c for c in range(C) if tr[k][2][:, c].mean() > 0.01] for k in mods}
rec = {}
with torch.no_grad():
    def _rec(name, a):
        rec.setdefault(name, []).append(a.numpy()); return a @ mods[name].W.T
    tgt.forward(torch.tensor(xtr), _rec)
    for k, m in mods.items():
        A = np.concatenate(rec[k]); al = alive[k]
        if not al:
            continue
        V = m.V.numpy()[al]; Wk = m.W.numpy()
        Z = A @ V.T                                   # [n, n_alive] reads
        Unew, *_ = np.linalg.lstsq(Z, A @ Wk.T, rcond=None)
        before = np.linalg.norm(A @ Wk.T - Z @ m.U.numpy()[al]) / np.linalg.norm(A @ Wk.T)
        after = np.linalg.norm(A @ Wk.T - Z @ Unew) / np.linalg.norm(A @ Wk.T)
        U = m.U.detach().clone(); U[al] = torch.tensor(Unew); m.U.copy_(U)
        print(f"{k}: on-data error of the alive mechanisms {before:.3f} -> {after:.3f}", flush=True)
torch.save({"mods": {k: v.float() for k, v in mods.state_dict().items()}}, out)
