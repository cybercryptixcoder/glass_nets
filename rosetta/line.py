"""The network's own best line, three plies deep (top move, opponent's top reply, follow-up), for every position.
Usage: python3 rosetta/line.py NET   (NET = LD2 or 744706). Writes ~/kdata/rosetta/<NET>_lines.json."""
import json, os, sys, time
import chess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kepler.targets.chess.leela import LeelaNet, evaluate  # noqa: E402
K = os.path.expanduser("~/kdata")
net_name = sys.argv[1]
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
net = LeelaNet(os.path.join(K, "chess", f"{net_name}.pb.gz")).eval()
items = json.load(open(os.path.join(K, "chess", "LD2", "meta.json")))["items"]
boards = [chess.Board(it["fen"]) for it in items]
t0 = time.time(); lines = [[] for _ in boards]; cur = boards
for ply in range(3):
    ev = evaluate(net, cur, batch=256)
    nxt = []
    for i, (b, e) in enumerate(zip(cur, ev)):
        if len(e["moves"]) == 0 or (ply > 0 and lines[i] and lines[i][-1] is None):
            lines[i].append(None); nxt.append(b); continue
        m = e["moves"][int(np.argmax(e["p"]))]; lines[i].append(m.uci())
        b2 = b.copy(stack=False); b2.push(m); nxt.append(b2)
    cur = nxt
    print("ply", ply + 1, round(time.time() - t0, 1), "s", flush=True)
json.dump(lines, open(os.path.join(K, "rosetta", f"{net_name}_lines.json"), "w"))
print("agree with stored top move:", np.mean([l[0] == it["top_move"] for l, it in zip(lines, items)]))
