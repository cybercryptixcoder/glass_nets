"""U8: unit maps across ARC's 5 training seeds per size (pre-registered in DIARY). Builds seed models under new ids
(zoo_<task>_<h>_<n>_s<seed>) with the same conversion and torch check as lab/targets/build_algzoo.py, then runs the
unchanged unit map v3 with its shuffled control. Maps go to results/glass/units/seeds/ (not the catalog)."""
import sys, json, copy, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab")); sys.path.insert(0, str(ROOT / "lab/targets"))
import nets, unitmap
from build_algzoo import fetch, BASE, N_SEQS
from build_registry import save

OUT = ROOT / "results/glass/units/seeds"; OUT.mkdir(parents=True, exist_ok=True)
SIZES = [("2nd_argmax", 8, 5, "zoo_2nd_argmax_8_5"), ("2nd_argmax", 16, 10, "zoo_2nd_argmax_16_10"), ("argmedian", 16, 5, "zoo_argmedian_16_5")]

def build(task, h, n, seed, template, n_inputs=8000):
    import torch as th
    tid = f"zoo_{task}_{h}_{n}_s{seed}"
    if (ROOT / "lab/models" / tid / "meta.json").exists():
        return tid
    url = f"{BASE}/{task}_{h}_{n}_{N_SEQS}_{seed}.pth"; sd = th.load(fetch(url), weights_only=True, map_location="cpu")
    W_ih, W_hh, W_out = (sd["rnn.weight_ih_l0"].double().numpy(), sd["rnn.weight_hh_l0"].double().numpy(), sd["linear.weight"].double().numpy())
    W = {"W_x": W_ih.T, "W_h": W_hh.T, "W_o": W_out.T, "seq_len": np.array([n], dtype=np.float64)}
    X = np.random.default_rng(0).standard_normal((n_inputs, n))
    labels = np.argsort(X, -1)[:, -2] if task == "2nd_argmax" else np.argsort(X, -1)[:, n // 2]
    rnn = th.nn.RNN(1, h, nonlinearity="relu", bias=False, batch_first=True).double(); lin = th.nn.Linear(h, n, bias=False).double()
    with th.no_grad():
        rnn.weight_hh_l0.copy_(th.from_numpy(W_hh)); rnn.weight_ih_l0.copy_(th.from_numpy(W_ih)); lin.weight.copy_(th.from_numpy(W_out))
        ref = lin(rnn(th.from_numpy(X)[..., None])[1].squeeze(0)).numpy()
    ours = nets.forward("rnn_relu", W, X); err = float(np.abs(ours - ref).max()); assert err < 1e-6, err
    acc = float((ours.argmax(1) == labels).mean())
    meta = copy.deepcopy(json.loads((ROOT / "lab/models" / template / "meta.json").read_text()))
    for k in ("id", "tensors", "params"):
        meta.pop(k, None)
    meta.update({"name": f"AlgZoo RNN {task} M{h},{n} (seed {seed})", "source": f"AlgZoo checkpoint {url} (MIT-0), converted by scripts/glass/units/u8_seeds.py",
                 "test_acc": acc, "numpy_vs_torch_max_err": err, "params_check": h + h * h + n * h,
                 "answer": {"short": meta["answer"]["short"], "detail": f"AlgZoo {task}, hidden {h}, length {n}, seed {seed}; accuracy {acc:.4f} on {n_inputs} standard-normal inputs."},
                 "perm_groups": [{"size": h, "members": [["W_x", 1], ["W_h", 0], ["W_h", 1], ["W_o", 0]]}]})
    d = ROOT / "lab/models" / tid; d.mkdir(parents=True, exist_ok=True)
    np.save(d / "inputs.npy", X.astype(np.float32)); np.save(d / "labels.npy", labels.astype(np.int64)); save(tid, meta, W)
    return tid

if __name__ == "__main__":
    for task, h, n, tmpl in SIZES:
        for seed in range(5):
            tid = build(task, h, n, seed, tmpl)
            if (OUT / f"{tid}.json").exists():
                continue
            m = unitmap.unit_map(tid); c = unitmap.unit_map(tid, "shuffled", confidence=False)
            acc = json.loads((ROOT / "lab/models" / tid / "meta.json").read_text())["test_acc"]
            (OUT / f"{tid}.json").write_text(json.dumps({"acc": acc, "map": m, "control": {k: c[k] for k in ("coverage_history", "coverage_raw", "per_unit")}}, indent=0))
            print(f"{tid}: acc {acc:.3f} eps95 {m['eps95']:.4f} hist {m['coverage_history']:.2f} (ctrl {c['coverage_history']:.2f}) | "
                  + "; ".join(sorted(set(l for l in m['per_unit'].values() if l.startswith('comparator')))), flush=True)
