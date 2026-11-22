#!/bin/bash
# Colab round 12 (D2): prime set v2 for factor mode v2 (pre-registered Nov 22 01:56). Runs on a CPU session.
cd /content/helix_probe || exit 1
run() { o=lab/models/$1; shift; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --out $o --device cpu "$@" 2>&1 | tail -1; }
for p in 61 67 71 73 79 83 151 157; do run dc_pow_p${p}_s1 --task pow --p $p --seed 1 --frac 0.5 --epochs 20000; done
tar czf /content/round12.tgz lab/models/dc_pow_p61_s1 lab/models/dc_pow_p67_s1 lab/models/dc_pow_p71_s1 lab/models/dc_pow_p73_s1 lab/models/dc_pow_p79_s1 lab/models/dc_pow_p83_s1 lab/models/dc_pow_p151_s1 lab/models/dc_pow_p157_s1; echo ALLDONE12
