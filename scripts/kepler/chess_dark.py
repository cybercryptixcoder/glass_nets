"""Level 4 for chess: what the network's evaluation knows that classical concepts do not, stated as programs.

    ROUND=1 python3 scripts/kepler/chess_dark.py     # residual, clusters, LLM proposals, tests
    ROUND=2 python3 scripts/kepler/chess_dark.py     # refinement: the LLM sees round-1 scores and revises

Acceptance (K3, pre-registered 10:10): (1) held-out R^2 gain >= 0.005 with 95% CI above 0 on LD2's value;
(2) decodable from LD2's activations (best of sites 2,4,6,8,10) at least 0.1 above the untrained copy;
(3) |corr| < 0.5 with every classical concept; (4) gain >= 0.005 with CI above 0 on the second network (744706).
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import chess  # noqa: E402
import torch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from chess_known_test import load_chess, replication_q  # noqa: E402
from kepler.core.probe import auc, is_binary, pca_basis  # noqa: E402
from kepler.core.probe import r2 as pr2  # noqa: E402
from kepler.core.residual import gain, surrogate  # noqa: E402
from kepler.core.scientist import ask, extract_json  # noqa: E402
from kepler.targets.chess.leela import LeelaNet, encode  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
OUT = "results/kepler/chess/dark"
SITES = [2, 4, 6, 8, 10]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


SYSTEM = """You are a chess grandmaster and a scientist working inside Kepler, an interpretability engine.
Kepler compares a superhuman chess network's evaluation with what 61 classical chess concepts can explain.
You are shown positions where the network's evaluation differs sharply from the classical estimate, grouped
because the network represents them similarly. Your job: state, as executable Python, the chess idea the network
is using that the classical concepts miss. Kepler will test your code against the network on thousands of held-out
positions; only ideas that pass its frozen tests survive. Be precise and concrete; prefer ideas a strong player could
learn and use. Never restate a classical concept from the list."""


def board_text(b):
    return str(b) + f"\n(side to move: {'White' if b.turn else 'Black'}; uppercase = White)"


def move_features(b, mv):
    p = b.piece_at(mv.from_square)
    f = {"capture": b.is_capture(mv), "check": b.gives_check(mv), "castle": b.is_castling(mv),
         "promotion": mv.promotion is not None, "piece": chess.piece_name(p.piece_type) if p else "?"}
    if p and p.piece_type == chess.PAWN:
        f["pawn_file"] = "abcdefgh"[chess.square_file(mv.from_square)]
    return f


def residual_oof(X, y, groups, k=5):
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(k).split(X, y, groups):
        pred[te] = surrogate(X[tr], y[tr]).predict(X[te])
    return pred


def cluster_summaries(items, X, names, r, q, qhat, Z, n_clusters=14, top=8):
    big = np.where(np.abs(r) >= np.percentile(np.abs(r), 75))[0]
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    km = KMeans(n_clusters, random_state=0, n_init=4).fit(Zs[big])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    out = []
    for c in range(n_clusters):
        idx = big[km.labels_ == c]
        rs = r[idx]
        sign = 1 if rs.mean() > 0 else -1
        cons = float(np.mean(np.sign(rs) == sign))
        z = (X[idx].mean(0) - mu) / sd
        zt = sorted(zip(names, z), key=lambda t: -abs(t[1]))[:10]
        d = np.linalg.norm(Zs[idx] - km.cluster_centers_[c], axis=1)
        proto = [idx[i] for i in np.argsort(d) if np.sign(r[idx[i]]) == sign][:6]
        themes = {}
        mvf = {"capture": 0, "check": 0, "castle": 0, "pawn": 0, "king": 0, "promotion": 0}
        for i in idx:
            it = items[i]
            for t in it.get("themes", []):
                themes[t] = themes.get(t, 0) + 1
            b = chess.Board(it["fen"])
            f = move_features(b, chess.Move.from_uci(it["top_move"]))
            for k in ("capture", "check", "castle", "promotion"):
                mvf[k] += int(f[k])
            mvf["pawn"] += int(f["piece"] == "pawn")
            mvf["king"] += int(f["piece"] == "king")
        out.append({"cluster": c, "n": int(len(idx)), "mean_residual": float(rs.mean()), "sign": sign,
                    "consistency": cons, "mass": float(abs(rs.sum())),
                    "mean_q": float(q[idx].mean()), "mean_qhat": float(qhat[idx].mean()),
                    "concept_z": [(n, round(float(v), 2)) for n, v in zt],
                    "top_move_share": {k: round(v / len(idx), 3) for k, v in mvf.items()},
                    "themes": sorted(themes.items(), key=lambda t: -t[1])[:8],
                    "prototypes": [int(i) for i in proto], "members": [int(i) for i in idx]})
    out.sort(key=lambda c: -c["mass"] * c["consistency"])
    return out


def cluster_prompt(c, items, q, qhat, names):
    lines = [f"Cluster of {c['n']} positions. On average the network evaluates them "
             f"{'MORE optimistically' if c['sign'] > 0 else 'MORE pessimistically'} than the classical estimate, "
             f"for the side to move (mean network value {c['mean_q']:+.2f} vs classical {c['mean_qhat']:+.2f}, on a scale "
             f"of -1 (loss) to +1 (win); {int(100 * c['consistency'])}% of the cluster goes this way).",
             "Classical concepts unusual in this cluster (z-scores vs all positions): " +
             ", ".join(f"{n} {v:+.2f}" for n, v in c["concept_z"]),
             "Network's chosen move in these positions: " + ", ".join(f"{k} {v:.0%}" for k, v in c["top_move_share"].items()),
             ("Lichess puzzle themes present: " + ", ".join(f"{t} ({n})" for t, n in c["themes"])) if c["themes"] else "",
             "", "Prototype positions:"]
    for i in c["prototypes"]:
        it = items[i]
        b = chess.Board(it["fen"])
        mv = chess.Move.from_uci(it["top_move"])
        lines += [f"\nFEN {it['fen']}", board_text(b),
                  f"network value {q[i]:+.2f}, classical estimate {qhat[i]:+.2f}, network's move {b.san(mv)} "
                  f"(probability {it['top_p']:.2f})"]
    lines += ["", "The 61 classical concepts already in the library (do NOT restate these): " + ", ".join(names), "",
              "Propose 3 candidate concepts that explain why the network evaluates these positions differently. "
              "Each must be a Python function named `concept(board)` taking a python-chess Board and returning a float, "
              "from the point of view of the side to move (board.turn): larger = the idea favours the side to move "
              "if the network is more optimistic, or the opponent if it is more pessimistic; say which in 'direction'. "
              "Use only the `chess` and `math` modules (already imported). It must run in under 2 ms per position "
              "(avoid searching more than one move deep) and never raise.",
              "Answer with a JSON list of objects: {\"name\": short_snake_case, \"idea\": one or two sentences a chess "
              "player can learn, \"direction\": \"+\" or \"-\", \"code\": the full function source}."]
    return "\n".join(x for x in lines if x is not None)


def compile_concept(code):
    ns = {"chess": chess, "math": math}
    exec(code, ns)
    fn = ns.get("concept")
    if not callable(fn):
        # K9 fix (translation error): accept a program that defines exactly one function under another name
        own = [v for k, v in ns.items() if callable(v) and getattr(v, "__module__", None) is None and k not in ("chess", "math")]
        own = [v for v in own if getattr(v, "__code__", None) is not None]
        if len(own) != 1:
            raise ValueError("no function named concept")
        fn = own[0]
    return fn


def run_concept(fn, boards, budget_s=90):
    vals = np.zeros(len(boards))
    t0 = time.time()
    errors = 0
    for i, b in enumerate(boards):
        try:
            v = float(fn(b.copy(stack=False)))
            vals[i] = v if math.isfinite(v) else 0.0
        except Exception:
            errors += 1
        if time.time() - t0 > budget_s:
            raise TimeoutError(f"too slow ({i} positions in {budget_s}s)")
    return vals, errors


class Evaluator:
    """Fixed evaluation set and cached activations for the acceptance tests."""

    def __init__(self, items, X, names, q, q2, groups):
        rng = np.random.default_rng(3)
        vs = np.load(os.path.join(KDATA, "chess", "LD2", "value_split.npz"))
        tr = rng.choice(vs["tr"], 7500, replace=False)
        te = rng.choice(vs["te"], 2500, replace=False)
        self.idx = np.concatenate([tr, te])
        self.ntr = len(tr)
        self.items, self.X, self.names, self.q, self.q2 = items, X, names, q, q2
        self.boards = [chess.Board(items[i]["fen"]) for i in self.idx]
        Xs = X[self.idx]
        self.base1 = surrogate(Xs[:self.ntr], q[self.idx][:self.ntr]).predict(Xs[self.ntr:])
        self.base2 = surrogate(Xs[:self.ntr], q2[self.idx][:self.ntr]).predict(Xs[self.ntr:])
        self._acts()

    def _acts(self):
        """PCA-128 activations at SITES for 4,000 of the evaluation positions, trained and untrained network."""
        path = os.path.join(KDATA, "chess", "LD2", "dark_acts.npz")
        self.pidx = np.arange(0, 4000)          # first 4,000 of self.idx (all from the training part)
        if os.path.exists(path):
            z = np.load(path)
            self.Zt, self.Zc = z["Zt"], z["Zc"]
            return
        d = os.path.join(KDATA, "chess", "LD2")
        rows = self.idx[self.pidx]
        net = LeelaNet(os.path.join(KDATA, "chess", "LD2.pb.gz"))
        torch.manual_seed(0)
        for mod in net.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
        ctrl = {s: [] for s in SITES}
        with torch.no_grad():
            for i in range(0, len(rows), 256):
                x = torch.tensor(np.stack([encode(self.boards[j]) for j in self.pidx[i:i + 256]]))
                _, _, acts = net(x, return_acts=True)
                for s in SITES:
                    ctrl[s].append(acts[s].reshape(len(x), -1).numpy())
        Zt, Zc = [], []
        for s in SITES:
            A = np.asarray(np.load(os.path.join(d, f"acts_site{s}.npy"), mmap_mode="r")[rows]).astype(np.float32)
            C = np.concatenate(ctrl[s]).astype(np.float32)
            for M, L in ((A, Zt), (C, Zc)):
                mu, V = pca_basis(M, 128)
                L.append((M - mu) @ V)
        self.Zt, self.Zc = np.stack(Zt), np.stack(Zc)
        np.savez(path, Zt=self.Zt, Zc=self.Zc)

    def decodability(self, vals):
        y = vals[self.pidx]
        if np.var(y) == 0:
            return {"best_site": None, "trained": float("nan"), "control": float("nan"), "metric": "none"}
        n = len(y)
        cut = int(n * 0.75)
        binary = is_binary(y)
        best = None
        from kepler.core.probe import Ridge
        for k, s in enumerate(SITES):
            sc = {}
            for lab, Z in (("trained", self.Zt[k]), ("control", self.Zc[k])):
                m = Ridge().fit(Z[:cut], y[:cut])
                p = m.predict(Z[cut:])
                sc[lab] = auc(y[cut:], p) if binary else pr2(y[cut:], p)
            if best is None or sc["trained"] > best["trained"]:
                best = {"best_site": s, **sc}
        best["metric"] = "auc" if binary else "r2"
        return best

    def test(self, vals):
        ntr = self.ntr
        Xs = self.X[self.idx]
        q1, q2 = self.q[self.idx], self.q2[self.idx]
        g1, _ = gain(Xs[:ntr], q1[:ntr], Xs[ntr:], q1[ntr:], vals[:ntr], vals[ntr:], base_pred=self.base1)
        g2, _ = gain(Xs[:ntr], q2[:ntr], Xs[ntr:], q2[ntr:], vals[:ntr], vals[ntr:], base_pred=self.base2)
        corr = {}
        for j, n in enumerate(self.names):
            if np.std(Xs[:, j]) > 0 and np.std(vals) > 0:
                corr[n] = float(np.corrcoef(Xs[:, j], vals)[0, 1])
        maxc = max(corr.items(), key=lambda t: abs(t[1])) if corr else ("none", 0.0)
        dec = self.decodability(vals)
        c1 = g1["gain"] >= 0.005 and g1["ci95"][0] > 0
        c2 = (dec["trained"] - dec["control"]) >= 0.1 if dec["best_site"] is not None else False
        c3 = abs(maxc[1]) < 0.5
        c4 = g2["gain"] >= 0.005 and g2["ci95"][0] > 0
        return {"gain_LD2": g1, "gain_744706": g2, "decodability": dec, "max_corr_library": maxc,
                "nonzero_share": float(np.mean(vals != 0)),
                "criteria": {"1_gain": c1, "2_decodable": c2, "3_not_restatement": c3, "4_replicates": c4},
                "accepted": bool(c1 and c2 and c3 and c4)}


def main():
    rnd = int(os.environ.get("ROUND", 1))
    os.makedirs(OUT, exist_ok=True)
    items, X, names, groups = load_chess()
    q = np.array([it["q"] for it in items])
    q2 = replication_q(items)
    state_path = os.path.join(OUT, "state.json")
    if rnd == 1:
        qhat = residual_oof(X, q, groups)
        r = q - qhat
        log(f"out-of-fold classical R2 {1 - np.var(r) / np.var(q):.3f}; residual std {r.std():.3f}")
        A = np.load(os.path.join(KDATA, "chess", "LD2", "acts_site10.npy"), mmap_mode="r")
        sub = np.random.default_rng(0).choice(len(items), 5000, replace=False)
        mu, V = pca_basis(np.asarray(A[np.sort(sub)]).astype(np.float32), 64)
        Z = np.concatenate([(np.asarray(A[i:i + 2000]).astype(np.float32) - mu) @ V for i in range(0, len(items), 2000)])
        clusters = cluster_summaries(items, X, names, r, q, qhat, Z)
        json.dump({"qhat": qhat.tolist(), "clusters": clusters}, open(state_path, "w"))
        log("clusters: " + "; ".join(f"#{c['cluster']} n={c['n']} r={c['mean_residual']:+.2f} cons={c['consistency']:.2f}"
                                     for c in clusters[:8]))
    st = json.load(open(state_path))
    qhat = np.array(st["qhat"])
    clusters = st["clusters"][:8]
    ev = Evaluator(items, X, names, q, q2, groups)
    log("evaluator ready")
    results_path = os.path.join(OUT, "candidates.json")
    allres = json.load(open(results_path)) if os.path.exists(results_path) else []
    prompts = {}
    for c in clusters:
        if rnd == 1:
            prompt = cluster_prompt(c, items, q, qhat, names)
        else:
            prev = [x for x in allres if x["cluster"] == c["cluster"]]
            fb = "\n".join(f"- {x['name']}: {x['idea']} | gain on held-out value R2 {x['test']['gain_LD2']['gain']:+.4f} "
                           f"(CI {x['test']['gain_LD2']['ci95'][0]:+.4f}..{x['test']['gain_LD2']['ci95'][1]:+.4f}), "
                           f"second network {x['test']['gain_744706']['gain']:+.4f}, decodable "
                           f"{x['test']['decodability']['trained']:.2f} vs untrained {x['test']['decodability']['control']:.2f}, "
                           f"max |corr| with library {abs(x['test']['max_corr_library'][1]):.2f} "
                           f"({x['test']['max_corr_library'][0]}), accepted {x['test']['accepted']}"
                           if x.get("test") else f"- {x['name']}: failed to run ({x.get('error', '')[:120]})" for x in prev)
            prompt = (cluster_prompt(c, items, q, qhat, names) + "\n\nYour previous round and Kepler's test results:\n" + fb +
                      "\n\nRevise: keep what worked, sharpen the ideas (gains need >= 0.005 with the interval above 0 on "
                      "both networks, decodability >= 0.1 above the untrained copy, and |corr| < 0.5 with every library "
                      "concept). Propose 3 improved or new concepts in the same JSON format.")
        prompts[c["cluster"]] = prompt

    sdir = os.path.join(OUT, "scientist")

    def one(cid):
        saved = os.path.join(sdir, f"round{rnd}_cluster{cid}.json")
        if os.path.exists(saved):
            d = json.load(open(saved))
            return cid, d["answer"], d.get("usage", {})
        if os.environ.get("NO_NEW_CALLS"):
            return cid, "", {}
        try:
            text, usage = ask(prompts[cid], system=SYSTEM, effort="high", max_tokens=16000, log_dir=sdir,
                              tag=f"round{rnd}_cluster{cid}")
            return cid, text, usage
        except Exception as e:
            log(f"cluster {cid}: LLM error {e}")
            return cid, "", {}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        answers = {cid: (t, u) for cid, t, u in ex.map(one, list(prompts))}
    done_names = {(x["round"], x["cluster"], x["name"]) for x in allres}
    for c in clusters:
        text, usage = answers.get(c["cluster"], ("", {}))
        cands = extract_json(text) or []
        if isinstance(cands, dict):
            cands = cands.get("concepts") or cands.get("candidates") or [cands]
        log(f"cluster {c['cluster']}: {len(cands)} candidates (tokens in {usage.get('prompt_tokens')}, out {usage.get('completion_tokens')})")
        for cand in cands if isinstance(cands, list) else []:
            rec = {"round": rnd, "cluster": c["cluster"], "name": cand.get("name", "?"), "idea": cand.get("idea", ""),
                   "direction": cand.get("direction", ""), "code": cand.get("code", "")}
            if (rnd, c["cluster"], rec["name"]) in done_names:
                continue
            try:
                fn = compile_concept(rec["code"])
                vals, errs = run_concept(fn, ev.boards)
                rec["errors"] = errs
                if errs > 0.2 * len(vals):
                    raise ValueError(f"{errs} errors")
                in_cluster = [k for k, i in enumerate(ev.idx) if i in set(c["members"])]
                rec["test"] = ev.test(vals)
                rec["test"]["corr_with_residual_in_cluster"] = float(np.corrcoef(vals[in_cluster], (q - qhat)[ev.idx][in_cluster])[0, 1]) \
                    if len(in_cluster) > 10 and np.std(vals[in_cluster]) > 0 else None
                t = rec["test"]
                log(f"  {rec['name']}: gain {t['gain_LD2']['gain']:+.4f} [{t['gain_LD2']['ci95'][0]:+.4f},{t['gain_LD2']['ci95'][1]:+.4f}] "
                    f"rep {t['gain_744706']['gain']:+.4f} dec {t['decodability']['trained']:.2f}/{t['decodability']['control']:.2f} "
                    f"maxcorr {abs(t['max_corr_library'][1]):.2f} -> {'ACCEPTED' if t['accepted'] else 'no'}")
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                log(f"  {rec['name']}: {rec['error'][:100]}")
            allres.append(rec)
            json.dump(allres, open(results_path, "w"), indent=1)
    log(f"round {rnd} done: {sum(1 for x in allres if x.get('test', {}).get('accepted'))} accepted of {len(allres)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
