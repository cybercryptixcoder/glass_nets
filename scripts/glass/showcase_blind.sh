#!/bin/bash
# Blind recovery demo (SHOWCASE.md): train a FRESH modular-arithmetic transformer, compute its answer key from the weights
# alone, then run the atlas pipeline without that key and score the atlas against it.
#
#   bash scripts/glass/showcase_blind.sh OP SEED            OP = add | mul;  needs a GPU for steps 1 and 4 (about 1 h on one L4)
#   EPOCHS=... STEPS=... DEVICE=... override the defaults (8000 training epochs, 30000 decomposition steps, cuda)
#
# Pass rules are the ones pre-registered for A42 and A44 (docs/glass/CHANGES.md): >= 80% of rule-labelled frequency triggers
# on the answer key (>= 20 of them), and > 50% of rule-labelled MLP-output/unembedding mechanisms writing the frequency they read.
set -euo pipefail
OP=${1:?add or mul}; SEED=${2:?seed}
EPOCHS=${EPOCHS:-8000}; STEPS=${STEPS:-30000}; DEVICE=${DEVICE:-cuda}
cd "$(dirname "$0")/../.."
T=mod${OP}_p113_blind$SEED; OUT=results/glass/atlas/spd; mkdir -p $OUT
echo "== 1. train a fresh network: $T"
python3 scripts/glass/train_modadd_gpu.py --op $OP --p 113 --frac 0.4 --epochs $EPOCHS --seed $SEED --out lab/models/$T --device $DEVICE | tail -1
(cd lab/targets && python3 -c "import sys; sys.argv=['x']; import build_registry as b; b.build_modular('$T', '$OP', 'Blind demo network (showcase_blind.sh)')")
rm -f lab/models/$T/weights.safetensors

echo "== 2. answer key, from the weights only (the atlas never sees this)"
KEY=$(python3 - <<EOF
import numpy as np
W = dict(np.load("lab/models/$T/weights.npz")); p = 113
E = W["W_E"][:p] if "$OP" == "add" else W["W_E"][[pow(3, e, p) for e in range(p - 1)]]   # discrete-log order for mul
P = (np.abs(np.fft.rfft(E, axis=0)) ** 2).sum(1)[1:]
print(",".join(str(int(k) + 1) for k in np.argsort(P)[::-1][:5]))
EOF
)
BASIS=$([ $OP = add ] && echo line || echo log)
echo "   key frequencies ($BASIS basis): $KEY"

echo "== 3. decompose (SPD, two sparsity rungs)"
for m in 3e-4 1e-4; do
  python3 scripts/glass/atlas_spd_job.py $OUT $STEPS 1024 $DEVICE --coef-min=$m --C=160 --tag=lad_$m $T | tail -1
done

echo "== 4. build the atlas (both bases offered; the atlas picks)"
L="$OUT/spd_${T}_original_${STEPS}_1024_lad_3e-4.pt,$OUT/spd_${T}_original_${STEPS}_1024_lad_1e-4.pt"
ATLAS_MODADD_VOCAB=both python3 lab/atlas.py $T x 4 --ladder $L | tail -3

echo "== 5. score against the answer key"
python3 scripts/glass/atlas_freq_known.py $T --basis=$BASIS --keys=$KEY --only-rule --min-n=20
python3 scripts/glass/atlas_writes_known.py $T both --keys=$KEY
python3 -c "import json; a=json.load(open('results/glass/atlas/${T}_d4.json')); print([s for s in a['findings']['statements'] if s.startswith('6.')][0])"
echo "== catalogue: python3 lab/server.py, then open http://localhost:8765/catalog?t=$T"
