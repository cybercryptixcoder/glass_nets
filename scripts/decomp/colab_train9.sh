#!/bin/bash
# Colab round 9 (D1): held-out S4 seeds 6-9, 80k epochs (pre-registered Nov 22 00:10).
cd /content/helix_probe || exit 1
run() { o=lab/models/$1; shift; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --out $o "$@" 2>&1 | tail -1; }
for s in 6 7 8 9; do run dc_secret_s4_s$s --task secret_s4 --seed $s --frac 0.75 --epochs 80000; done
tar czf /content/round9.tgz lab/models/dc_secret_s4_s6 lab/models/dc_secret_s4_s7 lab/models/dc_secret_s4_s8 lab/models/dc_secret_s4_s9; echo ALLDONE9
