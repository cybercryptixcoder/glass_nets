"""Rosetta's variable finder: a top-k sparse autoencoder over the network's per-square activation vectors.
Raw channels of a dense network mix many things; the network's own variables are sparse directions. Each feature is
then a unit like any other: a per-square map that Rosetta defines with a program and tests by substitution.
Usage: python3 rosetta/sae.py NET SITE [untrained]   writes ~/kdata/rosetta/sae_<NET>_s<SITE>.pt and codes .npz"""
import os, sys, time
import numpy as np, torch, torch.nn as nn

K = os.path.expanduser("~/kdata")
torch.set_num_threads(int(os.environ.get("THREADS", "2")))


class TopK(nn.Module):
    def __init__(self, d=128, m=512, k=8):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d, bias=False)
        self.b = nn.Parameter(torch.zeros(d))
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.T)
            self.dec.weight /= self.dec.weight.norm(dim=0, keepdim=True)

    def codes(self, x):
        z = torch.relu(self.enc(x - self.b))
        v, i = z.topk(self.k, dim=-1)
        return v, i

    def forward(self, x):
        v, i = self.codes(x)
        z = torch.zeros(x.shape[0], self.enc.out_features, device=x.device).scatter(1, i, v)
        return self.dec(z) + self.b, z


def train(X, m=512, k=8, epochs=6, bs=4096, seed=0):
    torch.manual_seed(seed)
    sae = TopK(X.shape[1], m, k)
    with torch.no_grad():
        sae.b.copy_(torch.tensor(X[:200000].mean(0)))
    opt = torch.optim.Adam(sae.parameters(), 1e-3)
    Xt = torch.tensor(X)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for j in range(0, n, bs):
            xb = Xt[perm[j:j + bs]]
            xh, _ = sae(xb)
            loss = ((xh - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                sae.dec.weight /= sae.dec.weight.norm(dim=0, keepdim=True)
            tot += loss.item() * len(xb)
        var = float(((Xt[:200000] - Xt[:200000].mean(0)) ** 2).mean())
        print(f"epoch {ep + 1}: mse {tot / n:.5f}, fraction of variance unexplained {tot / n / var:.4f}", flush=True)
    return sae


if __name__ == "__main__":
    net, site = sys.argv[1], int(sys.argv[2])
    tag = f"{net}_s{site}" + ("_untrained" if len(sys.argv) > 3 else "")
    if len(sys.argv) > 3:
        X = np.load(os.path.join(K, "rosetta", f"acts_{tag}.npy")).astype(np.float32)     # [n, 128, 64]
    else:
        X = np.load(os.path.join(K, "chess", net, f"acts_site{site}.npy"), mmap_mode="r")
        X = np.asarray(X, dtype=np.float32).reshape(-1, 128, 64)
    N = X.shape[0]
    V = X.transpose(0, 2, 1).reshape(-1, 128)                                            # [n * 64, 128]
    t0 = time.time()
    sae = train(V)
    with torch.no_grad():
        vs, is_ = [], []
        for j in range(0, len(V), 65536):
            v, i = sae.codes(torch.tensor(V[j:j + 65536]))
            vs.append(v.numpy().astype(np.float16)); is_.append(i.numpy().astype(np.int16))
    np.savez(os.path.join(K, "rosetta", f"codes_{tag}.npz"), v=np.concatenate(vs).reshape(N, 64, -1), i=np.concatenate(is_).reshape(N, 64, -1))
    torch.save(sae.state_dict(), os.path.join(K, "rosetta", f"sae_{tag}.pt"))
    print("done", round(time.time() - t0), "s")
