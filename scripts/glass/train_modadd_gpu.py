"""Train lab/targets/train_modadd.py's one-layer transformer on a GPU (same model, data split rule and optimiser).

The original script runs on CPU only (its causal mask is built on CPU). This wrapper reuses its OneLayer class and
training loop settings unchanged and only moves tensors to the device. Used to make fresh held-out networks (A37).

Usage: python3 scripts/glass/train_modadd_gpu.py --op add --p 113 --frac 0.4 --epochs 8000 --seed 1 --out lab/models/modadd_p113_s1
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lab", "targets"))
from train_modadd import OneLayer
from safetensors.torch import save_file


class OneLayerDev(OneLayer):
    def forward(self, tok):
        x = self.W_E[tok] + self.W_pos
        q = torch.einsum("bpd,hde->bhpe", x, self.W_Q)
        k = torch.einsum("bpd,hde->bhpe", x, self.W_K)
        v = torch.einsum("bpd,hde->bhpe", x, self.W_V)
        att = torch.einsum("bhpe,bhqe->bhpq", q, k) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(3, 3, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(mask, -1e9).softmax(-1)
        z = torch.einsum("bhpq,bhqe->bhpe", att, v)
        x = x + torch.einsum("bhpe,hed->bpd", z, self.W_O)
        x = x + F.relu(x @ self.W_in) @ self.W_out
        return x[:, -1] @ self.W_U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="add", choices=["add", "sub", "mul"])
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--labels", default="task", choices=["task", "random"], help="A41: 'random' = a fixed random label per pair (memorization known answer)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    p, dev = a.p, a.device
    A, B = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    A, B = A.ravel(), B.ravel()
    y = {"add": (A + B) % p, "sub": (A - B) % p, "mul": (A * B) % p}[a.op]
    if a.labels == "random":
        y = np.random.default_rng(1000 + a.seed).integers(0, p, len(A))
    tok = np.stack([A, B, np.full_like(A, p)], 1)
    idx = np.random.permutation(len(A)); n_tr = int(a.frac * len(A))
    tr, te = idx[:n_tr], idx[n_tr:]
    X, Y = torch.tensor(tok).to(dev), torch.tensor(y).to(dev)
    m = OneLayerDev(vocab=p + 1).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / 10, 1))
    log, t0 = [], time.time()
    trt, tet = torch.tensor(tr, device=dev), torch.tensor(te, device=dev)
    for ep in range(a.epochs + 1):
        loss = F.cross_entropy(m(X[trt]), Y[trt])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 250 == 0:
            with torch.no_grad():
                lt = m(X[tet]); te_loss = F.cross_entropy(lt, Y[tet]).item()
                te_acc = (lt.argmax(-1) == Y[tet]).float().mean().item()
            log.append({"epoch": ep, "train_loss": loss.item(), "test_loss": te_loss, "test_acc": te_acc})
            print(json.dumps(log[-1]), f"{time.time()-t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}, f"{a.out}/weights.safetensors")
    with torch.no_grad():
        tr_acc = (m(X[trt]).argmax(-1) == Y[trt]).float().mean().item()
    json.dump({"op": a.op, "p": p, "frac": a.frac, "seed": a.seed, "epochs": a.epochs, "labels": a.labels, "train_acc": tr_acc,
               "random_labels": y.tolist() if a.labels == "random" else None,
               "train_idx": tr.tolist(), "final": log[-1], "log": log}, open(f"{a.out}/train.json", "w"))


if __name__ == "__main__":
    main()
