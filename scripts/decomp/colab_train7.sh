#!/bin/bash
# Colab round 7 (D1): challenge set v2 (fresh random abelian groups, secret labellings), pre-registered 23:19.
cd /content/helix_probe || exit 1
git pull -q origin decompiler
run() { o=lab/models/$1; shift; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --out $o "$@" 2>&1 | tail -1; }
for k in 25 38 46 35 28 30 41 33 37 27 40 26; do run dc_randab${k} --task randab$k --seed 1 --frac 0.4 --epochs 10000; done
tar czf /content/round7.tgz lab/models/dc_randab25 lab/models/dc_randab38 lab/models/dc_randab46 lab/models/dc_randab35 lab/models/dc_randab28 lab/models/dc_randab30 lab/models/dc_randab41 lab/models/dc_randab33 lab/models/dc_randab37 lab/models/dc_randab27 lab/models/dc_randab40 lab/models/dc_randab26; echo ALLDONE7
