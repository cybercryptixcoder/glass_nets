#!/bin/bash
# Colab round 10 (D1): challenge set v3 (pre-registered Nov 22 02:35).
cd /content/helix_probe || exit 1
run() { o=lab/models/$1; shift; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --out $o "$@" 2>&1 | tail -1; }
for s in 47 48 50 51 52 55 56 57 60 68 86 116; do run dc_randab$s --task randab$s --seed 1 --frac 0.4 --epochs 10000; done
tar czf /content/round10.tgz lab/models/dc_randab47 lab/models/dc_randab48 lab/models/dc_randab50 lab/models/dc_randab51 lab/models/dc_randab52 lab/models/dc_randab55 lab/models/dc_randab56 lab/models/dc_randab57 lab/models/dc_randab60 lab/models/dc_randab68 lab/models/dc_randab86 lab/models/dc_randab116; echo ALLDONE10
