#!/bin/bash
# Colab round 3 (D1): composite moduli and cubes.
cd /content/helix_probe || exit 1
run() { o=lab/models/$1; shift; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --out $o "$@" 2>&1 | tail -1; }
run dc_mulmod120_s1 --task mulmod120 --seed 1 --frac 0.5 --epochs 20000
run dc_mulmod105_s1 --task mulmod105 --seed 1 --frac 0.5 --epochs 20000
run dc_addmod120_s1 --task addmod120 --seed 1 --frac 0.4 --epochs 8000
run dc_cubesum_s1 --task cubesum --seed 1 --frac 0.5 --epochs 20000
run dc_mulmod120_s2 --task mulmod120 --seed 2 --frac 0.5 --epochs 20000
run dc_secret_s4_s3 --task secret_s4 --seed 3 --frac 0.6 --epochs 40000
echo ALLDONE
