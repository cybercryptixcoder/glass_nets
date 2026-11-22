#!/bin/bash
# Colab round 6 (D1): held-out S4 networks that generalise (seed 3 did not grok at 60% data): more data.
cd /content/helix_probe || exit 1
run() { o=lab/models/$1; shift; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --out $o "$@" 2>&1 | tail -1; }
run dc_secret_s4_s4 --task secret_s4 --seed 4 --frac 0.75 --epochs 40000
run dc_secret_s4_s5 --task secret_s4 --seed 5 --frac 0.75 --epochs 40000
tar czf /content/round6.tgz lab/models/dc_secret_s4_s4 lab/models/dc_secret_s4_s5; echo ALLDONE6
