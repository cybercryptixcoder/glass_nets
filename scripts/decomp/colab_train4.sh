#!/bin/bash
# Colab round 4 (D1): the blind challenge set, 10 random abelian groups under random labellings.
cd /content/helix_probe || exit 1
git pull -q origin decompiler
run() { o=lab/models/$1; shift; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --out $o "$@" 2>&1 | tail -1; }
for k in 1 2 4 5 6 9 11 13 19 21; do run dc_randab${k} --task randab$k --seed 1 --frac 0.4 --epochs 10000; done
echo ALLDONE4
