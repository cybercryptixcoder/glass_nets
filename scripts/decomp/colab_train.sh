#!/bin/bash
# Colab: train the decompiler discovery set (D1). Detached: setsid nohup bash scripts/decomp/colab_train.sh > /content/dc.log 2>&1 &
cd /content/helix_probe || exit 1
run() { t=$1; s=$2; shift 2; o=lab/models/dc_${t}_s$s; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --task $t --seed $s --out $o "$@" 2>&1 | tail -1; }
for s in 1 2; do
  run secret_add $s
  run xyxy $s
  run secret_z8z14 $s
  run sqsum $s --frac 0.5 --epochs 20000
  run pow $s --frac 0.5 --epochs 20000
  run secret_s4 $s --frac 0.6 --epochs 40000
done
echo ALLDONE
