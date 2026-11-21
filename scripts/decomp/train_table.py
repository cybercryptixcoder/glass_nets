"""Train the repo's one-layer transformer (lab/targets/train_modadd.py OneLayer, Nanda et al. setup) on any task in
scripts/decomp/tasks.py, on a GPU. Writes lab/models/<out>/weights.npz, train.json (answer key, secrets included: the
decompiler never reads it) and meta.json (only what an outside reader would be told: shapes and token ranges).

  python3 scripts/decomp/train_table.py --task secret_add --seed 1 --out lab/models/dc_secret_add_s1
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "glass"))
from tasks import table
from train_modadd_gpu import OneLayerDev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    T, n, n_out, info = table(a.task, a.p)
    A, B = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); A, B = A.ravel(), B.ravel()
    y = T[A, B]
    tok = np.stack([A, B, np.full_like(A, n)], 1)
    idx = np.random.permutation(len(A)); n_tr = int(a.frac * len(A)); tr, te = idx[:n_tr], idx[n_tr:]
    dev = a.device
    X, Y = torch.tensor(tok).to(dev), torch.tensor(y).to(dev)
    m = OneLayerDev(vocab=n + 1, n_out=n_out).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / 10, 1))
    trt, tet = torch.tensor(tr, device=dev), torch.tensor(te, device=dev)
    log, t0 = [], time.time()
    for ep in range(a.epochs + 1):
        loss = F.cross_entropy(m(X[trt]), Y[trt])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 500 == 0:
            with torch.no_grad():
                lt = m(X[tet]); tl = F.cross_entropy(lt, Y[tet]).item(); ta = (lt.argmax(-1) == Y[tet]).float().mean().item()
            log.append({"epoch": ep, "train_loss": loss.item(), "test_loss": tl, "test_acc": ta})
            print(json.dumps(log[-1]), f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    W = {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
    np.savez(f"{a.out}/weights.npz", **W)
    with torch.no_grad():
        tr_acc = (m(X[trt]).argmax(-1) == Y[trt]).float().mean().item()
        all_acc = (m(X).argmax(-1) == Y).float().mean().item()
    json.dump({"task": a.task, "n": n, "n_out": n_out, "p": a.p, "frac": a.frac, "seed": a.seed, "epochs": a.epochs, "wd": a.wd,
               "train_acc": tr_acc, "all_acc": all_acc, "answer_key": info, "train_idx": tr.tolist(), "final": log[-1], "log": log},
              open(f"{a.out}/train.json", "w"))
    arch = (f"A one-layer transformer with no LayerNorm and no biases. Vocabulary of {n + 1} tokens; the input is a sequence of 3 "
            f"tokens; the output is {n_out} logits read at the last position.\n"
            f"  x = W_E[tokens] + W_pos; 4 causal attention heads of size 32; x = x + relu(x @ W_in) @ W_out; logits = x[last] @ W_U")
    json.dump({"name": f"Decompiler discovery network {os.path.basename(a.out)}", "family": "Decompiler discovery set (trained here)",
               "forward": "onelayer_transformer", "order": ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"],
               "arch": arch,
               "context": f"Tokens 0..{n - 1} are opaque symbols; token {n} is '='. Every input has the form 'x y ='. The output is one of {n_out} classes.",
               "input": f"token ids, shape [B, 3], e.g. [[x, y, {n}]]", "n_symbols": n, "n_out": n_out},
              open(f"{a.out}/meta.json", "w"), indent=1)
    print("DONE", a.out, "train", round(tr_acc, 4), "all", round(all_acc, 4), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
