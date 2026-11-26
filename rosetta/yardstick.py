"""Known-answer networks for Rosetta. Each planted network is trained to output a position-level count of a sealed
per-square rule; a memoriser is trained on shuffled labels; an untrained network is left at its random init.
Saves, per network: last-block activations [N, C, 64], outputs, and the model. ~/kdata/rosetta/yard/<name>.*"""
import json, os, sys, time
import chess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rosetta.core import expr_map  # noqa: E402
K = os.path.expanduser("~/kdata"); OUT = os.path.join(K, "rosetta", "yard"); os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2)


class Small(nn.Module):
    def __init__(self, c=32, blocks=4):
        super().__init__()
        self.inp = nn.Conv2d(13, c, 3, padding=1)
        self.b1 = nn.ModuleList([nn.Conv2d(c, c, 3, padding=1) for _ in range(blocks)])
        self.b2 = nn.ModuleList([nn.Conv2d(c, c, 3, padding=1) for _ in range(blocks)])
        self.head = nn.Conv2d(c, 4, 1); self.fc1 = nn.Linear(4 * 64, 64); self.fc2 = nn.Linear(64, 1)

    def forward(self, x, edit=None, return_last=False):
        h = F.relu(self.inp(x))
        for a, b in zip(self.b1, self.b2):
            h = F.relu(h + b(F.relu(a(h))))
        if edit is not None:
            h = edit(h)
        y = self.fc2(F.relu(self.fc1(F.relu(self.head(h)).flatten(1)))).squeeze(1)
        return (y, h) if return_last else y


def planes(fens):
    X = np.zeros((len(fens), 13, 64), np.float32)
    for n, f in enumerate(fens):
        b = chess.Board(f); flip = b.turn == chess.BLACK; us = b.turn
        for sq, p in b.piece_map().items():
            k = chess.square_mirror(sq) if flip else sq
            X[n, (p.piece_type - 1) + (0 if p.color == us else 6), k] = 1
        X[n, 12] = 1
    return X.reshape(-1, 13, 8, 8)


def train(X, y, tr, epochs, seed):
    torch.manual_seed(seed); m = Small()
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    Xt, yt = torch.tensor(X[tr]), torch.tensor(y[tr])
    for ep in range(epochs):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 256):
            j = perm[i:i + 256]; opt.zero_grad(); loss = F.mse_loss(m(Xt[j]), yt[j]); loss.backward(); opt.step()
    return m


if __name__ == "__main__":
    from rosetta.sealed.yardstick_rules import RULES
    items = json.load(open(os.path.join(K, "chess", "LD2", "meta.json")))["items"]
    names = json.load(open(os.path.join(K, "rosetta", "prim_names.json")))
    P = np.load(os.path.join(K, "rosetta", "prims_LD2.npy")).astype(np.float32)
    X = planes([it["fen"] for it in items]); N = len(items)
    rng = np.random.default_rng(0); perm = rng.permutation(N); tr, te = np.sort(perm[:14000]), np.sort(perm[14000:])
    np.savez(os.path.join(OUT, "split.npz"), tr=tr, te=te)
    nets = {}
    for name, rule in RULES.items():
        m = expr_map(P, names, rule).reshape(N, 64); y = m.sum(1).astype(np.float32)
        np.save(os.path.join(OUT, f"{name}_truthmap.npy"), m.astype(np.float16))
        nets[name] = (y, 40, 1)
    y1 = nets["S1"][0]
    nets["MEM"] = (y1[rng.permutation(N)].copy(), 120, 2)          # memoriser: S1's labels, shuffled across positions
    for name, (y, ep, seed) in list(nets.items()) + [("UNTRAINED", (y1, 0, 3))]:
        t0 = time.time(); m = train(X, y, tr, ep, seed).eval()
        with torch.no_grad():
            outs, hs = [], []
            for i in range(0, N, 1000):
                o, h = m(torch.tensor(X[i:i + 1000]), return_last=True); outs.append(o.numpy()); hs.append(h.flatten(2).numpy())
        out, H = np.concatenate(outs), np.concatenate(hs)
        r2 = lambda s: 1 - np.mean((out[s] - y[s]) ** 2) / np.var(y[s])
        print(f"{name}: train R2 {r2(tr):.3f} held-out R2 {r2(te):.3f} ({time.time() - t0:.0f} s)", flush=True)
        np.save(os.path.join(OUT, f"{name}_acts.npy"), H.astype(np.float16)); np.save(os.path.join(OUT, f"{name}_out.npy"), out)
        torch.save(m.state_dict(), os.path.join(OUT, f"{name}.pt"))
    np.save(os.path.join(OUT, "planes.npy"), X.astype(np.float16))
