"""Look-ahead probe profile for the transformer t1 (as probe_lookahead.py does for LD2): is the destination of t1's own
follow-up move (ply 3) readable from its square tokens beyond what the board shows? Usage: python3 rosetta/t1_probe.py"""
import json, os, sys, time
import chess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import Defs  # noqa: E402
from rosetta.chess_lib import look_ahead, LOOK  # noqa: E402
from rosetta.t1_rosetta import load_t1  # noqa: E402
from kepler.targets.chess.leela import evaluate, encode  # noqa: E402
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(int(os.environ.get("THREADS", "2")))

if __name__ == "__main__":
    t0 = time.time()
    D = Defs(L.P, L.names, L.TR, L.TE, n_search=1500, n_score=1500)
    rows = np.concatenate([D.rs, D.rt]); ns = len(D.rs)
    net = load_t1(); boards = [chess.Board(L.items[i]["fen"]) for i in rows]
    lines = [[] for _ in boards]; cur = boards
    for ply in range(3):
        ev = evaluate(net, cur, batch=128); nxt = []
        for i, (b, e) in enumerate(zip(cur, ev)):
            if len(e["moves"]) == 0 or (ply > 0 and lines[i][-1] is None):
                lines[i].append(None); nxt.append(b); continue
            mv = e["moves"][int(np.argmax(e["p"]))]; lines[i].append(mv); b2 = b.copy(stack=False); b2.push(mv); nxt.append(b2)
        cur = nxt
    print("t1 line done", round(time.time() - t0), "s", flush=True)
    LA = np.stack([look_ahead(b, ln) for b, ln in zip(boards, lines)])                       # [n, 64, 8]
    static = [k for k, n in enumerate(L.names) if not n.startswith("la_")]
    Pst = np.asarray(L.P[rows], dtype=np.float32)[:, :, static].reshape(-1, len(static))
    X = torch.tensor(np.stack([encode(b) for b in boards]))
    mu = load_t1(untrained_seed=21)
    out = {"sites": {}}
    for tgt in ("la_m1_to", "la_m3_to"):
        y = LA[:, :, LOOK.index(tgt)].reshape(-1) > 0.5; ys, yt = y[:ns * 64], y[ns * 64:]
        g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(Pst[:ns * 64], ys)
        out.setdefault("board_only", {})[tgt] = round(roc_auc_score(yt, g.predict_proba(Pst[ns * 64:])[:, 1]), 4)
    print("board only", out["board_only"], flush=True)
    for site in (2, 4, 6, 8):
        with torch.no_grad():
            H = torch.cat([net(X[j:j + 200], upto=site) for j in range(0, len(X), 200)]).numpy().reshape(-1, net.channels)
            Hu = torch.cat([mu(X[j:j + 200], upto=site) for j in range(0, len(X), 200)]).numpy().reshape(-1, net.channels)
        res = {}
        for tgt in ("la_m1_to", "la_m3_to"):
            y = LA[:, :, LOOK.index(tgt)].reshape(-1) > 0.5; ys, yt = y[:ns * 64], y[ns * 64:]
            def lr(A):
                a, b = A[:ns * 64], A[ns * 64:]; m_, s_ = a.mean(0), a.std(0) + 1e-6
                clf = LogisticRegression(max_iter=300).fit((a - m_) / s_, ys)
                return round(roc_auc_score(yt, clf.predict_proba((b - m_) / s_)[:, 1]), 4)
            res[tgt] = {"network_linear": lr(H), "untrained_linear": lr(Hu), "board_only": out["board_only"][tgt]}
        out["sites"][site] = res
        print("site", site, json.dumps(res), round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{L.RES}/lookahead_profile_t1.json", "w"), indent=1)
