#!/bin/bash
# Colab round 2 (D1): more pow networks (seeds and primes), plus the seed-2 networks lost with session dc1.
cd /content/helix_probe || exit 1
run() { o=lab/models/$1; shift; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --out $o "$@" 2>&1 | tail -1; }
run dc_pow_s2 --task pow --seed 2 --frac 0.5 --epochs 20000
run dc_pow_s3 --task pow --seed 3 --frac 0.5 --epochs 20000
run dc_pow_p109_s1 --task pow --p 109 --seed 1 --frac 0.5 --epochs 20000
run dc_pow_p101_s1 --task pow --p 101 --seed 1 --frac 0.5 --epochs 20000
run dc_pow_p97_s1 --task pow --p 97 --seed 1 --frac 0.5 --epochs 20000
run dc_sqsum_s2 --task sqsum --seed 2 --frac 0.5 --epochs 20000
run dc_secret_s4_s2 --task secret_s4 --seed 2 --frac 0.6 --epochs 40000
run dc_secret_z8z14_s2 --task secret_z8z14 --seed 2
echo ALLDONE
