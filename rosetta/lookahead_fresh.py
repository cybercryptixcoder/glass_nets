"""R2: the look-ahead causal test, split by the follow-up move's type, on FRESH positions (Kepler's fresh Lichess set).
Usage: python3 rosetta/lookahead_fresh.py NET N site [site ...]   NET = LD2 | 744706 | t1"""
import json, os, sys, time
import chess, numpy as np, torch
import torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.chess_lib import look_ahead, LOOK  # noqa: E402
from kepler.targets.chess.leela import evaluate, encode, move_index  # noqa: E402
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
K = os.path.expanduser("~/kdata")

if __name__ == "__main__":
    t0 = time.time(); NET, N = sys.argv[1], int(sys.argv[2]); sites = [int(s) for s in sys.argv[3:]]
    if NET == "t1":
        from kepler.targets.chess.leela_t import LeelaTransformer
        net = LeelaTransformer(os.path.join(L.C, "t1-256x10.pb.gz")).eval()
    else:
        net = L.get_net(NET)
    fm = json.load(open(os.path.join(K, "chess", "rules", "fresh_meta.json")))["items"]
    Pf = np.load(os.path.join(L.R, "fresh_cache.npz"), allow_pickle=True)["P"]
    rng = np.random.default_rng(2026); pool = list(rng.permutation(len(fm))[:int(N * 1.6)])
    boards = [chess.Board(fm[i]["fen"]) for i in pool]
    lines = [[] for _ in boards]; cur = boards
    for ply in range(3):
        ev = evaluate(net, cur, batch=128); nxt = []
        for i, (b, e) in enumerate(zip(cur, ev)):
            if len(e["moves"]) == 0 or (ply > 0 and lines[i][-1] is None):
                lines[i].append(None); nxt.append(b); continue
            mv = e["moves"][int(np.argmax(e["p"]))]; lines[i].append(mv); b2 = b.copy(stack=False); b2.push(mv); nxt.append(b2)
        cur = nxt
    print(NET, "line done", round(time.time() - t0), "s", flush=True)
    names = L.names; ix = {k: names.index(k) for k in ("empty", "att_us", "att_them")}
    own = [names.index(f"own_{p}") for p in "PNBRQK"]; lk = {k: LOOK.index(k) for k in LOOK}
    sel, t3s, ctrls, moves, att3, kinds = [], [], [], [], [], []
    for j, (i, b, ln) in enumerate(zip(pool, boards, lines)):
        if len(sel) >= N:
            break
        if len(ln) < 3 or any(m is None for m in ln[:3]):
            continue
        la = look_ahead(b, ln); p = np.asarray(Pf[i], dtype=np.float32)
        sq = lambda k: int(np.argmax(la[:, lk[k]])) if la[:, lk[k]].max() > 0 else None
        t3 = sq("la_m3_to"); used = {sq("la_m1_from"), sq("la_m1_to"), sq("la_m2_from"), sq("la_m2_to"), sq("la_m3_from")}
        if t3 is None or t3 in used:
            continue
        occ = lambda s: 0 if p[s, ix["empty"]] > 0 else (1 if p[s, own].max() > 0 else 2)
        tight = lambda s: (occ(s), int(la[s, lk["la_m1_attacks"]] > 0), min(int(p[s, ix["att_us"]]), 2), min(int(p[s, ix["att_them"]]), 2))
        c2 = [s for s in range(64) if s not in used and s != t3 and tight(s) == tight(t3)]
        if not c2:
            continue
        b3 = b.copy(stack=False); b3.push(ln[0]); b3.push(ln[1])
        sel.append(j); t3s.append(t3); ctrls.append(int(rng.choice(c2))); moves.append(ln[0])
        att3.append(int(la[t3, lk["la_m1_attacks"]] > 0)); kinds.append("check" if b3.gives_check(ln[2]) else "capture" if b3.is_capture(ln[2]) else "quiet")
    bsel = [boards[j] for j in sel]
    midx = [move_index(net, b, m) for b, m in zip(bsel, moves)]
    X = torch.tensor(np.stack([encode(b) for b in bsel]))
    Xm = torch.tensor(np.stack([encode(chess.Board(L.items[i]["fen"])) for i in L.TR[:600]]))
    kk = np.array(kinds); results = []
    for site in sites:
        with torch.no_grad():
            mean = torch.cat([net(Xm[j:j + 200], upto=site) for j in range(0, len(Xm), 200)]).mean(0)

        @torch.no_grad()
        def logp(squares):
            out = []
            for j in range(0, len(sel), 200):
                sl = slice(j, j + 200)
                def ed(s, h, sl=sl):
                    if s == site and squares is not None:
                        h = h.clone()
                        for bb, q in enumerate(squares[sl]):
                            if NET == "t1":
                                h[bb, q, :] = mean[q]
                            else:
                                h[bb, :, q // 8, q % 8] = mean[:, q // 8, q % 8]
                    return h
                lg, _ = net(X[sl], edit=ed)
                for bb, (bd, mi) in enumerate(zip(bsel[sl], midx[sl])):
                    legal = [move_index(net, bd, mm) for mm in bd.legal_moves]
                    out.append(float(Fn.log_softmax(lg[bb, legal], 0)[legal.index(mi)]))
            return np.array(out)
        base = logp(None); d3 = base - logp(np.array(t3s)); dc = base - logp(np.array(ctrls)); diff = d3 - dc
        def ci(msk):
            dd = diff[msk]; bt = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            return {"n": int(msk.sum()), "t3_minus_tight": round(float(dd.mean()), 4), "ci95": [round(float(np.quantile(bt, .025)), 4), round(float(np.quantile(bt, .975)), 4)]}
        r = {"net": NET, "site": site, "n": len(sel), "all": ci(np.ones(len(sel), bool)), "check": ci(kk == "check"), "capture": ci(kk == "capture"),
             "forcing (check or capture)": ci(kk != "quiet"), "quiet": ci(kk == "quiet")}
        results.append(r); print(json.dumps(r), round(time.time() - t0), "s", flush=True)
    json.dump(results, open(f"{L.RES}/lookahead_fresh_{NET}.json", "w"), indent=1)
