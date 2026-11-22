#!/bin/bash
# D1 demo: hide addition behind a secret random relabelling, train a fresh network on it, and let the decompiler
# recover the secret from the weights alone. Then compare with the secret (which the decompiler never sees).
#
#   bash scripts/decomp/demo_secret.sh SECRET_SEED [DEVICE]      e.g. bash scripts/decomp/demo_secret.sh 777 cuda
#
# Training takes about 1 minute on a GPU (8000 full-batch steps) and much longer on a CPU; decompiling takes a few
# minutes on a laptop CPU.
set -euo pipefail
S=${1:?secret seed}; DEV=${2:-$(python3 -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')")}
cd "$(dirname "$0")/../.."
M=dc_demo_secret_$S
echo "== 1. a fresh network on a+b mod 113 under secret relabelling #$S ($DEV)"
python3 - <<PY
import sys; sys.path.insert(0, "scripts/decomp"); import tasks
orig = tasks.table
tasks.table = lambda task, p=113, secret_seed=12345: orig(task, p, secret_seed=$S)
sys.argv = ["train_table.py", "--task", "secret_add", "--seed", "1", "--out", "lab/models/$M", "--device", "$DEV", "--epochs", "${EPOCHS:-8000}"]
import train_table; train_table.table = tasks.table; train_table.main()
PY
echo "== 2. decompile (reads weights.npz only)"
python3 lab/decomp/decompile.py $M
echo "== 3. compare with the secret"
python3 - <<PY
import json, numpy as np
r = json.load(open("results/decomp/$M.json")); k = json.load(open("lab/models/$M/train.json"))["answer_key"]
pi = np.array(k["secret_pi"]); true = np.argsort(pi)          # symbol t is the true number true[t]
g = {int(a): v[0] for a, v in r["element_of_symbol_a"].items()}; N = r["group"][0]
best = max((np.mean([(u * true[t] + (g[0] - u * true[0])) % N == g[t] for t in g]), u) for u in range(1, N) if np.gcd(u, N) == 1)
print(f"secret recovered for {best[0]:.1%} of the symbols (up to the group's own symmetries: x -> {best[1]}x + c)")
print("verdict:", r["verdict"])
PY
echo "== catalogue: python3 lab/server.py, then open http://localhost:8765/catalog?t=$M"
