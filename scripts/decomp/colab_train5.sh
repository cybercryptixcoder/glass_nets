#!/bin/bash
# Colab round 5 (D1): retrain what was lost when the sessions expired (held-out S4, the blind challenge set), and
# push nothing: models are downloaded by the local session. Writes lab/models/<id>/{weights.npz,train.json,meta.json}.
cd /content/helix_probe || exit 1
run() { o=lab/models/$1; shift; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --out $o "$@" 2>&1 | tail -1; }
run dc_secret_s4_s3 --task secret_s4 --seed 3 --frac 0.6 --epochs 40000
for k in 1 2 4 5 6 9 11 13 19 21; do run dc_randab${k} --task randab$k --seed 1 --frac 0.4 --epochs 10000; done
cd /content/helix_probe && tar czf /content/round5.tgz lab/models/dc_secret_s4_s3 lab/models/dc_randab* 2>/dev/null
echo ALLDONE5
