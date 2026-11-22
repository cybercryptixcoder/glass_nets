#!/bin/bash
# Colab round 11 (D1): blind non-abelian challenge (pre-registered Nov 22 03:25). Three trainings at a time on one GPU.
cd /content/helix_probe || exit 1
run() { o=lab/models/$1; shift; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --out $o "$@" 2>&1 | tail -1; }
i=0
for g in A5 S3S3 A4Z3 SL23 D12 F21 F20 D10; do for s in 1 2; do
  run dc_secretg_${g}_s$s --task secretg_$g --seed $s --frac 0.75 --epochs 80000 &
  i=$((i+1)); if [ $((i % 3)) -eq 0 ]; then wait; fi
done; done; wait
tar czf /content/round11.tgz lab/models/dc_secretg_*; echo ALLDONE11
