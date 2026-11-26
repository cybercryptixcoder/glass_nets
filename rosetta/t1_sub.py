"""Rosetta on Leela's transformer t1 at a middle block: find its own sparse variables, scan them against ITS OWN line
(look-ahead primitives from t1's top move, the reply and its follow-up), define the best ones with the full library and
with every look-ahead primitive removed (the static rival), and test the best look-ahead variable causally (removing
it at the block, does the network become less sure of its current move, more than removing a matched variable?).
Control line: the same scan on an untrained t1 (weights permuted). Usage: python3 rosetta/t1_rosetta.py SITE
(t1_sub variant: after the scan, the strongest look-ahead-bound variable is defined as a three-program composition and
substituted inside t1; the value change it restores is measured, as in check (c) for LD2.)"""
import json, os, sys, time
import chess, numpy as np, torch
import torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import Defs  # noqa: E402
from rosetta.chess_lib import look_ahead, LOOK  # noqa: E402
from rosetta.sae import train  # noqa: E402
from kepler.targets.chess.leela import evaluate, encode, move_index  # noqa: E402
from kepler.targets.chess.leela_t import LeelaTransformer  # noqa: E402
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
SITE = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def load_t1(untrained_seed=None):
    m = LeelaTransformer(os.path.join(L.C, "t1-256x10.pb.gz")).eval()
    if untrained_seed is not None:
        g = torch.Generator().manual_seed(untrained_seed)
        with torch.no_grad():
            for p in m.parameters():
                f = p.view(-1); f.copy_(f[torch.randperm(f.numel(), generator=g)])
    return m


@torch.no_grad()
def acts(m, X):
    return torch.cat([m(X[j:j + 200], upto=SITE) for j in range(0, len(X), 200)]).numpy()      # [n, 64, d]


if __name__ == "__main__":
    t0 = time.time()
    D0 = Defs(L.P, L.names, L.TR, L.TE, n_search=1500, n_score=1500)
    rows = np.concatenate([D0.rs, D0.rt])
    net = load_t1()
    boards = [chess.Board(L.items[i]["fen"]) for i in rows]
    lines = [[] for _ in boards]; cur = boards
    for ply in range(3):
        ev = evaluate(net, cur, batch=128); nxt = []
        for i, (b, e) in enumerate(zip(cur, ev)):
            if len(e["moves"]) == 0 or (ply > 0 and lines[i][-1] is None):
                lines[i].append(None); nxt.append(b); continue
            mv = e["moves"][int(np.argmax(e["p"]))]; lines[i].append(mv); b2 = b.copy(stack=False); b2.push(mv); nxt.append(b2)
        cur = nxt
        print("t1 line ply", ply + 1, round(time.time() - t0), "s", flush=True)
    # primitive table with t1's OWN look-ahead
    Pt1 = np.array(L.P, dtype=np.float16)
    la_cols = [L.names.index(n) for n in LOOK]
    for r, b, ln in zip(rows, boards, lines):
        Pt1[r][:, la_cols] = look_ahead(b, ln)
    D = Defs(Pt1, L.names, L.TR, L.TE, n_search=1500, n_score=1500)
    assert (D.rs == D0.rs).all() and (D.rt == D0.rt).all()
    X = torch.tensor(np.stack([encode(b) for b in boards]))
    H = acts(net, X); d = H.shape[2]
    V = H.reshape(-1, d).astype(np.float32)
    sae = train(V, m=512, k=8, epochs=8)
    with torch.no_grad():
        v, i = sae.codes(torch.tensor(V))
    Z = np.zeros((len(V), 512), np.float32); np.put_along_axis(Z, i.numpy().astype(np.int64), v.numpy(), axis=1)
    Z = Z.reshape(len(rows), 64, 512); ns = len(D.rs)
    Hu = acts(load_t1(untrained_seed=21), X).reshape(-1, d).astype(np.float32)
    sae_u = train(Hu, m=512, k=8, epochs=6)
    with torch.no_grad():
        vu, iu = sae_u.codes(torch.tensor(Hu))
    Zu = np.zeros((len(Hu), 512), np.float32); np.put_along_axis(Zu, iu.numpy().astype(np.int64), vu.numpy(), axis=1)
    Zu = Zu.reshape(len(rows), 64, 512)
    Pt = np.asarray(Pt1[D.rt], dtype=np.float32)
    la = {n: L.names.index(n) for n in LOOK}

    def scan(Zx):
        Zc = Zx[ns:] - Zx[:ns].mean(0, keepdims=True); Zc = Zc.reshape(-1, 512); Zc = Zc / (Zc.std(0) + 1e-9)
        out = {}
        for n, k in la.items():
            p = Pt[:, :, k].reshape(-1); p = (p - p.mean()) / (p.std() + 1e-9)
            out[n] = np.nan_to_num((Zc * p[:, None]).mean(0))
        return out
    real, ctrl = scan(Z), scan(Zu)
    line = float(max(np.abs(v).max() for v in ctrl.values()))
    print("scan line from the untrained t1:", round(line, 4), round(time.time() - t0), "s", flush=True)
    pos = {r: j for j, r in enumerate(rows)}
    fmap = lambda f: (lambda rr: Z[[pos[r] for r in rr], :, f])
    keep = [j for j, s in enumerate(D.specs) if not L.names[s[0]].startswith("la_")]
    res = {"site": SITE, "line": line, "features": []}
    seen = set()
    hits = sorted([(abs(float(v[f])), n, int(f), float(v[f])) for n, v in real.items() for f in np.argsort(-np.abs(v))[:4]], reverse=True)
    for sc, n, f, cv in hits:
        if f in seen or sc < line or len(res["features"]) >= 8:
            continue
        seen.add(f)
        full = D.define(fmap(f))
        Ls, Ss, L3s = D.L, D.specs, D.L3
        D.L, D.specs = D.L[keep], [D.specs[j] for j in keep]; D.L3 = D.L.reshape(len(D.specs), len(D.rs), -1)
        stat = D.define(fmap(f))
        D.L, D.specs, D.L3 = Ls, Ss, L3s
        r = {"feature": f, "best_lookahead_primitive": n, "corr": round(cv, 4), "freq": round(float((Z[:, :, f] > 0).any(1).mean()), 3),
             "full_library": {"text": full["text"], "heldout_fidelity": full["heldout_fidelity"]},
             "static_only": {"text": stat["text"], "heldout_fidelity": stat["heldout_fidelity"]}}
        res["features"].append(r); print(json.dumps(r), flush=True)
    # substitution of the strongest variable whose full-library definition uses the network's own line
    from rosetta.engine import additive_map, restore_share
    cand = [r for r in res["features"] if "la_m" in r["full_library"]["text"]]
    f = cand[0]["feature"]; Wd = sae.dec.weight.detach()
    comp = D.define_additive(fmap(f), terms=3)
    mu = Z[:ns, :, f].mean(0)
    te = np.arange(ns, len(rows))[:800]
    Pte = np.asarray(Pt1[rows[te]], dtype=np.float32)
    rep = np.clip(additive_map(Pte, comp["terms"], comp["weights"]) + mu[None], 0, None)
    zz = Z[te][:, :, f]

    @torch.no_grad()
    def val(mode):
        out = []
        for j in range(0, len(te), 200):
            sl = slice(j, j + 200)
            a = torch.tensor(zz[sl]); b = torch.tensor(rep[sl], dtype=torch.float32)
            def ed(s_, h):
                if s_ == SITE and mode != "orig":
                    h = h - a[:, :, None] * Wd[:, f][None, None, :]
                    if mode == "sub":
                        h = h + b[:, :, None] * Wd[:, f][None, None, :]
                return h
            _, wdl = net(X[te[sl]], edit=ed)
            out.append((wdl[:, 0] - wdl[:, 2]).numpy())
        return np.concatenate(out)
    v0, va, vs = val("orig"), val("abl"), val("sub")
    res["substitution"] = {"feature": f, "programs": comp["texts"], "weights": comp["weights"], "composed_fidelity": comp["curve"][-1]["heldout_fidelity"],
                           "restore": round(restore_share(v0, va, vs), 4), "mean_abs_value_change_ablation": round(float(np.mean(np.abs(va - v0))), 5)}
    print("substitution", json.dumps(res["substitution"]), flush=True)
    json.dump(res, open(f"{L.RES}/t1_sub_site{SITE}.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s")
