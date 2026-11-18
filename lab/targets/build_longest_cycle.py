"""Build lab targets from AlgZoo's longest-cycle transformers (github.com/alignment-research-center/alg-zoo, MIT-0).
Job: round 2's no-asterisk discovery target (docs/glass/PLAN_ROUND2.md, step 5): a family with no published
interpretation, a real multi-hop algorithm (following pointers), on a different architecture from the RNNs.

The task: the input is a function f on {0..n-1} written as n tokens (token i = f(i)); the answer is the length of the
longest cycle of f, minus 1 (classes 0..n-1). Example: [1, 2, 0, 3] has cycles 0->1->2->0 and 3->3, so the answer is
3 - 1 = 2.

The model (alg_zoo.architectures.AttentionOnlyTransformer): token + position embeddings, n_layers residual attention
layers (torch.nn.MultiheadAttention, 1 head, no biases, no mask, no LayerNorm, no MLP), unembedding read at the last
position. It is converted to nets.transformer's tensors (bidirectional by default; out_positions = [n-1]) and checked
against the torch module to 1e-6 on EVERY possible input (n**n of them): the domain is the full input set, so every
judge test runs on all inputs.

Run:  python3 lab/targets/build_longest_cycle.py [--only lc_3_4 ...]
"""
import argparse, io, os, sys, urllib.request
from itertools import product
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..")
sys.path.insert(0, LAB)
import nets  # noqa: E402
from build_registry import save  # noqa: E402

BASE = "https://storage.googleapis.com/arc-ml-public/alg/zoo"
MODELS = [(3, 4), (4, 5), (6, 5), (8, 6)]


def best_seed(h, n):
    src = os.environ.get("ALGZOO_SRC", "/tmp/algzoo_src/alg-zoo") + "/alg_zoo/zoo.py"
    text = open(src).read()
    ns = {}
    exec(text[text.index("def best_of_5_seed_longest_cycle"):], ns)
    return ns["best_of_5_seed_longest_cycle"](hidden_size=h, seq_len=n)


def longest_cycle(X):
    """alg_zoo.tasks.task_longest_cycle in numpy: for each start, the first return time (0 if it never returns); the
    maximum over starts, minus 1."""
    B, n = X.shape
    out = np.zeros(B, int)
    for s in range(n):
        pos = np.full(B, s); first = np.zeros(B, int)
        for k in range(1, n + 1):
            pos = X[np.arange(B), pos]
            new = (pos == s) & (first == 0)
            first[new] = k
        out = np.maximum(out, first)
    return out - 1


def build(h, n):
    import torch as th
    seed = best_seed(h, n)
    url = f"{BASE}/longest_cycle_{h}_{n}_{2 ** 30}_{seed}.pth"
    cache = os.path.join(LAB, "models", "_algzoo_cache"); os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, url.rsplit("/", 1)[1])
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    sd = th.load(path, weights_only=True, map_location="cpu")
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("attns.")})
    E, Pz = sd["embed.weight"].double().numpy(), sd["pos_embed.weight"].double().numpy()
    d = E.shape[1]
    W = {"W_E": E, "W_pos": Pz, "W_U": sd["unembed.weight"].double().numpy().T, "out_positions": np.array([n - 1], dtype=np.float64)}
    for l in layers:
        ip = sd[f"attns.{l}.in_proj_weight"].double().numpy()          # [3d, d]: q, k, v rows
        W[f"L{l}.W_Q"], W[f"L{l}.W_K"], W[f"L{l}.W_V"] = ip[:d].T[None], ip[d:2 * d].T[None], ip[2 * d:].T[None]
        W[f"L{l}.W_O"] = sd[f"attns.{l}.out_proj.weight"].double().numpy().T[None]
    X = np.array(list(product(range(n), repeat=n)), dtype=np.int64)    # every possible input
    # the reference forward, verbatim from alg_zoo.architectures.AttentionOnlyTransformer
    embed = th.nn.Embedding(*E.shape).double(); pos = th.nn.Embedding(*Pz.shape).double()
    attns = [th.nn.MultiheadAttention(d, num_heads=1, bias=False, batch_first=True).double() for _ in layers]
    unembed = th.nn.Linear(d, n, bias=False).double()
    with th.no_grad():
        embed.weight.copy_(sd["embed.weight"]); pos.weight.copy_(sd["pos_embed.weight"]); unembed.weight.copy_(sd["unembed.weight"])
        for l, a in zip(layers, attns):
            a.in_proj_weight.copy_(sd[f"attns.{l}.in_proj_weight"]); a.out_proj.weight.copy_(sd[f"attns.{l}.out_proj.weight"])
        xt = th.from_numpy(X)
        state = embed(xt) + pos(th.arange(n)[None])
        for a in attns:
            state = state + a(state, state, state, need_weights=False)[0]
        ref = unembed(state)[:, -1].numpy()
    ours = nets.forward("transformer", W, X)
    err = float(np.abs(ours - ref).max())
    assert err < 1e-6, f"numpy forward differs from torch by {err}"
    labels = longest_cycle(X)
    acc = float((ours.argmax(1) == labels).mean())
    tid = f"lc_{h}_{n}"
    order = ["W_E", "W_pos"] + [f"L{l}.W_{m}" for l in layers for m in "QKVO"] + ["W_U"]
    meta = {
        "name": f"AlgZoo longest-cycle transformer (hidden {h}, length {n})",
        "family": "AlgZoo (ARC): longest-cycle transformers",
        "forward": "transformer",
        "order": order, "protected": ["out_positions"],
        "arch": (f"A {len(layers)}-layer attention-only transformer: no MLPs, no LayerNorm, no biases, one head per layer, no "
                 f"attention mask (every position attends to every position). Vocabulary {n} tokens, sequences of {n} tokens, "
                 f"residual width {h}.\n  x = W_E[tokens] + W_pos                      W_E: {n} x {h}, W_pos: {n} x {h}\n"
                 f"  per layer l: x = x + softmax(q k^T / sqrt({h})) v W_O,  q = x W_Q, k = x W_K, v = x W_V  (each {h} x {h})\n"
                 f"  out = x[last position] @ W_U                 W_U: {h} x {n}\n"
                 f"{sum(int(np.size(W[k])) for k in order)} parameters. The output is a score per class; the decision is the largest."),
        "context": (f"Each input is a function f on {{0..{n - 1}}}: token i is f(i). Output class c stands for 'the longest cycle of "
                    f"f has length c + 1' (follow i -> f(i) -> f(f(i)) ... until it returns)."),
        "tensor_desc": {"W_E": "token embedding (row = token)", "W_pos": "position embedding (row = position)",
                        "W_U": "unembedding (column = class)", **{f"L{l}.W_{m}": f"layer {l} {m.lower()} map [head, in, out]" for l in layers for m in "QKVO"}},
        "answer": {"short": "length of the longest cycle of the input function, minus 1",
                   "detail": f"AlgZoo longest_cycle, hidden {h}, length {n}, seed {seed}; accuracy {acc:.4f} on all {len(X)} inputs. "
                             f"No published interpretation found (searched Nov 17, 2026)."},
        "source": f"AlgZoo checkpoint {url} (MIT-0), converted by lab/targets/build_longest_cycle.py",
        "input": f"token ids, shape [B, {n}]",
        "domain": {"file": "inputs.npy", "text": f"Every possible input: all {n}**{n} = {len(X)} functions on {{0..{n - 1}}}."},
        "labels": "labels.npy", "test_acc": acc, "numpy_vs_torch_max_err": err,
    }
    d_ = os.path.join(LAB, "models", tid); os.makedirs(d_, exist_ok=True)
    np.save(os.path.join(d_, "inputs.npy"), X.astype(np.int8))
    np.save(os.path.join(d_, "labels.npy"), labels.astype(np.int64))
    save(tid, meta, W)
    return tid, acc, err, len(X)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*"); a = ap.parse_args()
    for h, n in MODELS:
        if a.only and f"lc_{h}_{n}" not in a.only:
            continue
        print(build(h, n))
