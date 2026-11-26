"""The same look-ahead causal test on Leela's TRANSFORMER t1 (the architecture Jenner et al. 2024 studied), as a
known-answer check of Rosetta's recogniser and as the comparison with the convolutional nets.
Usage: python3 rosetta/t1_lookahead.py N site [site ...]"""
import json, os, sys, time
import chess, numpy as np, torch
import torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.chess_lib import look_ahead, LOOK  # noqa: E402
from kepler.targets.chess.leela import evaluate, encode, move_index  # noqa: E402
from kepler.targets.chess.leela_t import LeelaTransformer  # noqa: E402
torch.set_num_threads(int(os.environ.get("THREADS", "2")))

if __name__ == "__main__":
    t0 = time.time(); N = int(sys.argv[1]); sites = [int(s) for s in sys.argv[2:]]
    net = LeelaTransformer(os.path.join(L.C, "t1-256x10.pb.gz")).eval()
    rng = np.random.default_rng(7); pool = list(rng.permutation(L.TE)[:int(N * 1.6)])
    boards = [chess.Board(L.items[i]["fen"]) for i in pool]
    lines = [[] for _ in boards]; cur = boards
    for ply in range(3):
        ev = evaluate(net, cur, batch=128); nxt = []
        for i, (b, e) in enumerate(zip(cur, ev)):
            if len(e["moves"]) == 0 or (ply > 0 and lines[i][-1] is None):
                lines[i].append(None); nxt.append(b); continue
            mv = e["moves"][int(np.argmax(e["p"]))]; lines[i].append(mv); b2 = b.copy(stack=False); b2.push(mv); nxt.append(b2)
        cur = nxt
        print("t1 line ply", ply + 1, round(time.time() - t0), "s", flush=True)
    names = L.names
    ix = {k: names.index(k) for k in ("empty", "att_us", "att_them")}
    own = [names.index(f"own_{p}") for p in "PNBRQK"]
    lk = {k: LOOK.index(k) for k in LOOK}
    rows, t3s, ctrls, t1s, moves, att3, kinds = [], [], [], [], [], [], []
    for i, b, ln in zip(pool, boards, lines):
        if len(rows) >= N:
            break
        if len(ln) < 3 or any(m is None for m in ln[:3]):
            continue
        la = look_ahead(b, ln); p = np.asarray(L.P[i], dtype=np.float32)
        sq = lambda k: int(np.argmax(la[:, lk[k]])) if la[:, lk[k]].max() > 0 else None
        t3 = sq("la_m3_to"); used = {sq("la_m1_from"), sq("la_m1_to"), sq("la_m2_from"), sq("la_m2_to"), sq("la_m3_from")}
        if t3 is None or t3 in used:
            continue
        occ = lambda s: 0 if p[s, ix["empty"]] > 0 else (1 if p[s, own].max() > 0 else 2)
        tight = lambda s: (occ(s), int(la[s, lk["la_m1_attacks"]] > 0), min(int(p[s, ix["att_us"]]), 2), min(int(p[s, ix["att_them"]]), 2))
        c1 = [s for s in range(64) if s not in used and s != t3 and occ(s) == occ(t3)]
        c2 = [s for s in c1 if tight(s) == tight(t3)]
        if not c1 or not c2:
            continue
        rows.append(i); t3s.append(t3); ctrls.append((int(rng.choice(c1)), int(rng.choice(c2)))); t1s.append(sq("la_m1_to")); moves.append(ln[0])
        att3.append(int(la[t3, lk["la_m1_attacks"]] > 0))
        b3 = b.copy(stack=False); b3.push(ln[0]); b3.push(ln[1])
        kinds.append("check" if b3.gives_check(ln[2]) else "capture" if b3.is_capture(ln[2]) else "quiet")
    sel = [pool.index(r) for r in rows]
    bsel = [boards[j] for j in sel]
    midx = [move_index(net, b, m) for b, m in zip(bsel, moves)]
    X = torch.tensor(np.stack([encode(b) for b in bsel]))
    Xm = torch.tensor(np.stack([encode(chess.Board(L.items[i]["fen"])) for i in L.TR[:600]]))
    att3 = np.array(att3); results = []
    for site in sites:
        with torch.no_grad():
            mean = torch.cat([net(Xm[j:j + 200], upto=site) for j in range(0, len(Xm), 200)]).mean(0)      # [64, d]

        @torch.no_grad()
        def logp(squares):
            out = []
            for j in range(0, len(rows), 200):
                sl = slice(j, j + 200)
                def ed(s, h, sl=sl):
                    if s == site and squares is not None:
                        h = h.clone()
                        for bb, q in enumerate(squares[sl]):
                            h[bb, q, :] = mean[q]
                    return h
                lg, _ = net(X[sl], edit=ed)
                for bb, (bd, mi) in enumerate(zip(bsel[sl], midx[sl])):
                    legal = [move_index(net, bd, mm) for mm in bd.legal_moves]
                    out.append(float(Fn.log_softmax(lg[bb, legal], 0)[legal.index(mi)]))
            return np.array(out)
        base = logp(None)
        d = {k: base - logp(np.array(v)) for k, v in (("t3", t3s), ("ctrl_occupancy", [c[0] for c in ctrls]), ("ctrl_tight", [c[1] for c in ctrls]), ("t1", t1s))}
        diff = d["t3"] - d["ctrl_tight"]
        bt = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        split = {}
        kk = np.array(kinds)
        for nm, msk in (("t3 attacked by the moved piece", att3 == 1), ("t3 not attacked by the moved piece", att3 == 0),
                        ("follow-up is a check", kk == "check"), ("follow-up is a capture", kk == "capture"), ("follow-up is quiet", kk == "quiet")):
            if msk.sum() > 30:
                dd = d["t3"][msk] - d["ctrl_tight"][msk]; bb = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
                split[nm] = {"n": int(msk.sum()), "t3_minus_tight": round(float(dd.mean()), 4), "ci95": [round(float(np.quantile(bb, .025)), 4), round(float(np.quantile(bb, .975)), 4)]}
        r = {"net": "t1-256x10 (transformer)", "site": site, "n": len(rows), "mean_drop_logp": {k: round(float(v.mean()), 4) for k, v in d.items()},
             "t3_minus_tight": round(float(diff.mean()), 4), "ci95": [round(float(np.quantile(bt, .025)), 4), round(float(np.quantile(bt, .975)), 4)], "split": split}
        results.append(r); print(json.dumps(r), round(time.time() - t0), "s", flush=True)
    json.dump(results, open(f"{L.RES}/lookahead_causal2_t1_bytype.json", "w"), indent=1)
