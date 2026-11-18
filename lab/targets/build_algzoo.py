"""Build lab targets from ARC's AlgZoo (github.com/alignment-research-center/alg-zoo, MIT-0): tiny trained RNNs whose
full mechanism is the open question. Job: give the constructor whole-model targets with an outside reference point
(ARC reports which sizes it has fully understood).

For each model: download the checkpoint once (public https), check the numpy forward (nets.rnn_relu) against the
torch module to 1e-6, and write lab/models/<id>/ with weights.npz, meta.json, inputs.npy (the domain: standard
normal sequences, as AlgZoo trains on) and labels.npy (the task's correct answers, for the judge's error test).

Run:  python3 lab/targets/build_algzoo.py [--n 8000] [--only zoo_2nd_argmax_2_2 ...]
Needs torch at build time only (no alg_zoo package) (the lab itself stays numpy-only).
"""
import argparse, io, json, os, sys, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..")
sys.path.insert(0, LAB)
import nets  # noqa: E402
from build_registry import save  # noqa: E402

BASE = "https://storage.googleapis.com/arc-ml-public/alg/zoo"
CACHE = os.path.join(LAB, "models", "_algzoo_cache")
N_SEQS = 2 ** 30                      # the fully trained checkpoints

# Best-of-5 training seeds for 2nd argmax, copied from alg_zoo/zoo.py (best_of_5_seed_2nd_argmax) for the sizes
# used here. The ladder: the two models ARC reports it understands (M2,2 = 10 params; M4,3 = 32), then larger, up to
# the 432-parameter M16,10 of ARC's open challenge.
LADDER = [(2, 2), (4, 3), (4, 4), (8, 5), (8, 8), (16, 10)]


# Best-of-5 training seeds (alg_zoo/zoo.py, best_of_5_seed_2nd_argmax), for the sizes used here.
SEEDS = {(2, 2): 2, (4, 3): 0, (4, 4): 1, (8, 5): 3, (8, 8): 0}
# The top rung is ARC's own challenge model (the blog post's M16,10, "example_2nd_argmax" in alg_zoo/loading.py),
# not the zoo's 16x10 entry, so that our result is about the exact network ARC reports on.
CHALLENGE = "https://storage.googleapis.com/arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth"


def fetch(url):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, url.rsplit("/", 1)[1])
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    return path


def build(h, n, n_inputs, seed_inputs=0):
    import torch as th
    if (h, n) == (16, 10):
        url, seed = CHALLENGE, "challenge"
    else:
        seed = SEEDS[(h, n)]
        url = f"{BASE}/2nd_argmax_{h}_{n}_{N_SEQS}_{seed}.pth"
    ck = fetch(url)
    sd = th.load(ck, weights_only=True, map_location="cpu")
    W_ih, W_hh, W_out = (sd["rnn.weight_ih_l0"].double().numpy(), sd["rnn.weight_hh_l0"].double().numpy(),
                         sd["linear.weight"].double().numpy())
    assert W_ih.shape == (h, 1) and W_hh.shape == (h, h) and W_out.shape == (n, h), (W_ih.shape, W_hh.shape, W_out.shape)
    W = {"W_x": W_ih.T, "W_h": W_hh.T, "W_o": W_out.T, "seq_len": np.array([n], dtype=np.float64)}

    rng = np.random.default_rng(seed_inputs)
    X = rng.standard_normal((n_inputs, n))
    labels = np.argsort(X, -1)[:, -2]                       # alg_zoo.tasks.task_2nd_argmax
    # the numpy forward must equal the checkpoint's own torch forward
    rnn = th.nn.RNN(1, h, nonlinearity="relu", bias=False, batch_first=True).double()
    lin = th.nn.Linear(h, n, bias=False).double()
    with th.no_grad():
        rnn.weight_hh_l0.copy_(th.from_numpy(W_hh)); rnn.weight_ih_l0.copy_(th.from_numpy(W_ih)); lin.weight.copy_(th.from_numpy(W_out))
        _, final = rnn(th.from_numpy(X)[..., None])
        ref = lin(final.squeeze(0)).numpy()
    ours = nets.forward("rnn_relu", W, X)
    err = float(np.abs(ours - ref).max())
    assert err < 1e-6, f"numpy forward differs from torch by {err}"
    acc = float((ours.argmax(1) == labels).mean())

    tid = f"zoo_2nd_argmax_{h}_{n}"
    params = h + h * h + n * h
    meta = {
        "name": f"AlgZoo RNN M{h},{n}",
        "family": "AlgZoo (ARC): tiny trained RNNs",
        "forward": "rnn_relu",
        "order": ["W_x", "W_h", "W_o"],
        "protected": ["seq_len"],
        "arch": (f"A one-layer recurrent network with ReLU and no biases. It reads a sequence of {n} real numbers, one per "
                 f"step, into a hidden state of {h} units, and outputs {n} numbers read from the final state.\n"
                 f"  h_0 = 0;  h_t = relu(x_t * W_x + h_(t-1) @ W_h)   W_x: 1 x {h}, W_h: {h} x {h}\n"
                 f"  out = h_{n} @ W_o                                   W_o: {h} x {n}\n"
                 f"{params} parameters. The output is a score per class; the network's decision is the largest."),
        "context": (f"The inputs are independent standard-normal numbers. Output class i stands for position i of the "
                    f"sequence (0-based)."),
        "tensor_desc": {"W_x": "input weights (one per hidden unit)", "W_h": "recurrent weights (row = from unit, column = to unit)",
                        "W_o": "readout (row = hidden unit, column = output class)"},
        "answer": {"short": "position of the second-largest number",
                   "detail": (f"AlgZoo 2nd argmax, hidden {h}, length {n}, seed {seed}; accuracy {acc:.4f} on "
                              f"{n_inputs} standard-normal inputs. ARC (Jan 2026) reports full understanding of M2,2 and, "
                              f"less clearly, M4,3; M16,10 is its open challenge.")},
        "source": f"AlgZoo checkpoint {url} (MIT-0), converted by lab/targets/build_algzoo.py",
        "perm_groups": [{"size": h, "members": [["W_x", 1], ["W_h", 0], ["W_h", 1], ["W_o", 0]]}],
        "input": f"real numbers, shape [B, {n}]",
        "domain": {"file": "inputs.npy",
                   "text": f"The network is used on sequences of {n} independent standard-normal numbers (a fixed sample of {n_inputs})."},
        "labels": "labels.npy",
        "test_acc": acc, "params_check": params, "numpy_vs_torch_max_err": err,
    }
    d = os.path.join(LAB, "models", tid)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "inputs.npy"), X.astype(np.float32))
    np.save(os.path.join(d, "labels.npy"), labels.astype(np.int64))
    save(tid, meta, W)
    return tid, params, acc, err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for h, n in LADDER:
        if a.only and f"zoo_2nd_argmax_{h}_{n}" not in a.only:
            continue
        tid, params, acc, err = build(h, n, a.n)
        print(f"{tid}: {params} params, accuracy {acc:.4f}, numpy vs torch {err:.1e}")
