"""Build the chess target's data: positions, Leela's outputs, activations at every site, and classical concepts.

    python3 scripts/kepler/chess_build.py [NET=LD2] [N_GAMES=16000] [N_PUZZLES=4000]

Writes $KDATA/chess/<net>/: meta.json (positions, outputs), concepts.npz, acts_site<k>.f16 (memmap [N, 128*64]).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import chess  # noqa: E402
import torch  # noqa: E402

from kepler.targets.chess.concepts import concepts, concept_names  # noqa: E402
from kepler.targets.chess.data import game_positions, puzzles  # noqa: E402
from kepler.targets.chess.leela import LeelaNet, encode, move_index  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", 1)))
    net_name = os.environ.get("NET", "LD2")
    n_games = int(os.environ.get("N_GAMES", 16000))
    n_puz = int(os.environ.get("N_PUZZLES", 4000))
    out = os.path.join(KDATA, "chess", net_name)
    os.makedirs(out, exist_ok=True)
    net = LeelaNet(os.path.join(KDATA, "chess", f"{net_name}.pb.gz"))
    games = game_positions(os.path.join(KDATA, "chess", "lichess_2013-01.pgn.zst"), n_games, seed=1)
    puz = puzzles(os.path.join(KDATA, "chess", "puzzles_part.csv.zst"), n_puz, seed=2)
    items = [dict(g, source="game") for g in games] + [dict(p, source="puzzle") for p in puz]
    log(f"{len(games)} game positions, {len(puz)} puzzles")
    N = len(items)
    C = net.channels
    sites = net.n_sites
    mm = [np.lib.format.open_memmap(os.path.join(out, f"acts_site{k}.npy"), mode="w+", dtype=np.float16,
                                    shape=(N, C * 64)) for k in range(sites)]
    names = concept_names()
    conc = np.zeros((N, len(names)), dtype=np.float32)
    bs = 128
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, bs):
            chunk = items[i:i + bs]
            boards = [chess.Board(it["fen"]) for it in chunk]
            x = torch.tensor(np.stack([encode(b) for b in boards]))
            logits, wdl, acts = net(x, return_acts=True)
            for k in range(sites):
                mm[k][i:i + len(chunk)] = acts[k].reshape(len(chunk), -1).numpy().astype(np.float16)
            for j, (it, b) in enumerate(zip(chunk, boards)):
                moves = list(b.legal_moves)
                li = torch.stack([logits[j, move_index(net, b, m)] for m in moves])
                p = torch.softmax(li, 0).numpy()
                top = int(np.argmax(p))
                it["wdl"] = [round(float(v), 5) for v in wdl[j]]
                it["q"] = round(float(wdl[j, 0] - wdl[j, 2]), 5)
                it["top_move"] = moves[top].uci()
                it["top_p"] = round(float(p[top]), 4)
                if it["source"] == "puzzle":
                    it["solved"] = it["top_move"] == it["best"]
                c = concepts(b)
                conc[i + j] = [c[n] for n in names]
            if (i // bs) % 20 == 0:
                log(f"{i + len(chunk)}/{N} ({time.time() - t0:.0f}s)")
    for m in mm:
        m.flush()
    np.savez(os.path.join(out, "concepts.npz"), X=conc, names=np.array(names))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"net": net_name, "channels": C, "sites": sites, "items": items}, f)
    pz = [it for it in items if it["source"] == "puzzle"]
    log(f"done. puzzle top-1 accuracy (no search): {np.mean([it['solved'] for it in pz]):.3f} over {len(pz)}")


if __name__ == "__main__":
    main()
