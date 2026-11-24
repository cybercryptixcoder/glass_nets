"""K9: the scientist on minimal pairs: positions classical chess cannot tell apart, but the network can.

    python3 scripts/kepler/chess_pairs.py            # build pairs, ask (4 prompts in parallel), test
    NO_NEW_CALLS=1 python3 scripts/kepler/chess_pairs.py   # test only the answers already saved

Pairs (quiet game positions, different games): nearest neighbours in the standardized 64-feature classical library
(61 concepts + 3 calculation features) with a large difference in
  (value)     LD2's value residual (value minus the classical surrogate, out of fold), or
  (channel22) channel 22's activity after removing what the library predicts of it (out of fold).
Judge: the frozen K3 criteria. Must fail (K9): a FEN-hash noise program.
"""
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import chess  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from chess_dark import Evaluator, compile_concept, run_concept, SYSTEM  # noqa: E402
from chess_known_test import load_chess, replication_q  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
from kepler.core.scientist import ask, extract_json  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
OUT = "results/kepler/chess/pairs"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def oof(Xf, y, groups, k=5):
    p = np.zeros_like(y)
    for tr, te in GroupKFold(k).split(Xf, y, groups):
        p[te] = surrogate(Xf[tr], y[tr]).predict(Xf[te])
    return p


def board_text(b):
    return str(b) + f"\n(side to move: {'White' if b.turn else 'Black'}; uppercase = White)"


def make_pairs(Z, sig, groups, n_pairs=40, k=8):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, idx = nn.kneighbors(Z)
    dmax = np.percentile(dist[:, 1:], 30)
    cand = []
    for i in range(len(Z)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            if j <= i or groups[i] == groups[j] or d > dmax:
                continue
            cand.append((abs(sig[i] - sig[j]) / (1.0 + d), i, j, d))
    cand.sort(reverse=True)
    used, pairs = set(), []
    for s, i, j, d in cand:
        if i in used or j in used:
            continue
        used.update((i, j))
        pairs.append((i, j, d))
        if len(pairs) >= n_pairs:
            break
    return pairs


def pair_prompt(kind, pairs, items, quiet, q, qhat, sig, Zq, names):
    head = {"value": "In each pair, the two positions look almost identical to classical chess knowledge (all 64 classical "
                     "concepts nearly equal), yet the network evaluates them very differently relative to the classical "
                     "estimate. Position A is the one the network is MORE PESSIMISTIC about (for the side to move); B is the "
                     "one it is more optimistic about.",
            "channel22": "In each pair, the two positions look almost identical to classical chess knowledge, yet one internal "
                         "unit of the network (channel 22 of its last layer) is far more active in position A than in B. "
                         "Switching this unit off moves the network's evaluation toward the classical one, undoing about a "
                         "third of their disagreement; when it is active the network is more pessimistic for the side to move. "
                         "Removing any of the opponent's pieces lowers it by about the same amount, whatever the piece."}[kind]
    lines = [head, "Your task: find the chess feature that distinguishes A from B in every pair, which classical concepts miss.", ""]
    for n, (a, b, d) in enumerate(pairs, 1):
        ia, ib = quiet[a], quiet[b]
        dz = Zq[a] - Zq[b]
        diff = ", ".join(f"{names[k]} {dz[k]:+.1f}" for k in np.argsort(-np.abs(dz))[:4])
        lines.append(f"=== Pair {n} (classical distance {d:.2f}; the classical concepts that differ most, A minus B in standard units: {diff})")
        for lab, i, k in (("A", ia, a), ("B", ib, b)):
            bd = chess.Board(items[i]["fen"])
            mv = chess.Move.from_uci(items[i]["top_move"])
            lines += [f"Position {lab}: FEN {items[i]['fen']}", board_text(bd),
                      f"network value {q[i]:+.2f}, classical estimate {qhat[k]:+.2f}, network's move {bd.san(mv)}"]
        lines.append("")
    lines += ["Keep your private analysis short: a few sentences per pair, then answer. The answer must be the JSON list.",
              "The classical library already contains: " + ", ".join(names) + ".",
              "Propose 5 candidate concepts that would be LARGER in A than in B across these pairs, as Python functions "
              "named `concept(board)` returning a float from the side to move's point of view (board.turn). Use only the "
              "`chess` and `math` modules (already imported); under 2 ms per position (at most one move of look-ahead); "
              "never raise. Do not restate a classical concept from the list. Answer with a JSON list of objects: "
              "{\"name\": short_snake_case, \"idea\": one or two sentences a strong player could learn, \"code\": the full "
              "function source}."]
    return "\n".join(lines)


def noise_program():
    return ("def concept(board):\n    import hashlib\n    h = hashlib.md5(board.fen().encode()).hexdigest()\n"
            "    return int(h[:8], 16) / 16 ** 8\n")


def main():
    os.makedirs(os.path.join(OUT, "scientist"), exist_ok=True)
    items, X, names, groups = load_chess()
    q = np.array([it["q"] for it in items])
    q2 = replication_q(items)
    C = np.load(os.path.join(KDATA, "chess", "LD2", "calc_all.npz"))["C"]
    src = np.array([it["source"] for it in items])
    quiet = np.where((src == "game") & (X[:, names.index("in_check")] == 0) & (C[:, 0] == 0) & (C[:, 1] == 0))[0]
    L = np.column_stack([X, C])[quiet]
    lnames = names + ["qs_gain", "threat", "qs_mated"]
    g = groups[quiet]
    qhat = oof(L, q[quiet], g)
    r = q[quiet] - qhat
    A = np.load(os.path.join(KDATA, "chess", "LD2", "acts_site10.npy"), mmap_mode="r")
    t22 = np.asarray(A[quiet]).astype(np.float32).reshape(len(quiet), 128, 64)[:, 22].sum(1).astype(np.float64)
    c_res = t22 - oof(L, t22, g)
    log(f"{len(quiet)} quiet positions; value residual std {r.std():.3f}; channel-22 residual std {c_res.std():.2f}")
    sd = L.std(0)
    Zq = (L - L.mean(0)) / np.where(sd > 0, sd, 1)
    prompts = {}
    for kind, sig in (("value", r), ("channel22", c_res)):
        pairs = make_pairs(Zq, sig, g)
        oriented = []
        for a, b, d in pairs:
            # A = the position the network is more pessimistic about (value) / channel 22 more active (channel22)
            if (kind == "value" and r[a] > r[b]) or (kind == "channel22" and c_res[a] < c_res[b]):
                a, b = b, a
            oriented.append((a, b, d))
        log(f"{kind}: {len(oriented)} pairs; mean classical distance {np.mean([d for _, _, d in oriented]):.2f}; "
            f"mean |difference| {np.mean([abs(sig[a] - sig[b]) for a, b, _ in oriented]):.3f}")
        for part in range(2):
            prompts[f"{kind}_{part}"] = pair_prompt(kind, oriented[part::2][:6], items, quiet, q, qhat, sig, Zq, lnames)
    json.dump({k: v[:2000] for k, v in prompts.items()}, open(os.path.join(OUT, "prompt_heads.json"), "w"), indent=1)

    def one(tag):
        saved = os.path.join(OUT, "scientist", f"{tag}.json")
        if os.path.exists(saved):
            d = json.load(open(saved))
            return tag, d["answer"]
        if os.environ.get("NO_NEW_CALLS"):
            return tag, ""
        try:
            text, _ = ask(prompts[tag], system=SYSTEM, effort=os.environ.get("EFFORT", "low"), max_tokens=24000,
                          log_dir=os.path.join(OUT, "scientist"), tag=tag, timeout=1500)
            return tag, text
        except Exception as e:
            log(f"{tag}: LLM error {e}")
            return tag, ""

    with ThreadPoolExecutor(max_workers=4) as ex:
        answers = dict(ex.map(one, list(prompts)))
    ev = Evaluator(items, X, names, q, q2, groups)
    log("evaluator ready")
    # quiet held-out check: correlation with the value residual and the channel-22 residual on quiet positions
    qpos = {i: k for k, i in enumerate(quiet)}
    res_path = os.path.join(OUT, "candidates.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else []
    seen = {(x["prompt"], x["name"]) for x in results}
    todo = [("noise_control", {"name": "fen_hash_noise", "idea": "must fail", "code": noise_program()})]
    for tag, text in answers.items():
        cands = extract_json(text) or []
        if isinstance(cands, dict):
            cands = cands.get("concepts") or cands.get("candidates") or [cands]
        log(f"{tag}: {len(cands)} candidates")
        todo += [(tag, c) for c in cands if isinstance(c, dict)]
    for tag, cand in todo:
        rec = {"prompt": tag, "name": cand.get("name", "?"), "idea": cand.get("idea", ""), "code": cand.get("code", "")}
        if (tag, rec["name"]) in seen:
            continue
        try:
            fn = compile_concept(rec["code"])
            vals, errs = run_concept(fn, ev.boards)
            if errs > 0.2 * len(vals):
                raise ValueError(f"{errs} errors")
            rec["test"] = ev.test(vals)
            inq = [k for k, i in enumerate(ev.idx) if i in qpos]
            vq = vals[inq]
            rq = np.array([r[qpos[ev.idx[k]]] for k in inq])
            cq = np.array([c_res[qpos[ev.idx[k]]] for k in inq])
            rec["corr_value_residual_quiet"] = float(np.corrcoef(vq, rq)[0, 1]) if vq.std() > 0 else None
            rec["corr_channel22_residual_quiet"] = float(np.corrcoef(vq, cq)[0, 1]) if vq.std() > 0 else None
            t = rec["test"]
            log(f"  [{tag}] {rec['name']}: gain {t['gain_LD2']['gain']:+.4f} [{t['gain_LD2']['ci95'][0]:+.4f},{t['gain_LD2']['ci95'][1]:+.4f}] "
                f"rep {t['gain_744706']['gain']:+.4f} dec {t['decodability']['trained']:.2f}/{t['decodability']['control']:.2f} "
                f"maxcorr {abs(t['max_corr_library'][1]):.2f} r_val {rec['corr_value_residual_quiet']} r_c22 {rec['corr_channel22_residual_quiet']} "
                f"-> {'ACCEPTED' if t['accepted'] else 'no'}")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            log(f"  [{tag}] {rec['name']}: {rec['error'][:100]}")
        results.append(rec)
        json.dump(results, open(res_path, "w"), indent=1, default=float)
    log(f"done: {sum(1 for x in results if x.get('test', {}).get('accepted') and x['prompt'] != 'noise_control')} accepted "
        f"of {sum(1 for x in results if x['prompt'] != 'noise_control')}; noise control accepted: "
        f"{any(x.get('test', {}).get('accepted') for x in results if x['prompt'] == 'noise_control')}")


if __name__ == "__main__":
    main()
