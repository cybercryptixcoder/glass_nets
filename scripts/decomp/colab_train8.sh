#!/bin/bash
# Colab round 8 (D1): factor-mode challenge over primes (pre-registered Nov 22 00:05).
cd /content/helix_probe || exit 1
git fetch -q --depth 1 origin decompiler && git reset -q --hard FETCH_HEAD
run() { o=lab/models/$1; shift; [ -f $o/train.json ] && return; python3 scripts/decomp/train_table.py --out $o "$@" 2>&1 | tail -1; }
for p in 89 103 107 127 131 137 139 149; do run dc_pow_p${p}_s1 --task pow --p $p --seed 1 --frac 0.5 --epochs 20000; done
tar czf /content/round8.tgz lab/models/dc_pow_p89_s1 lab/models/dc_pow_p103_s1 lab/models/dc_pow_p107_s1 lab/models/dc_pow_p127_s1 lab/models/dc_pow_p131_s1 lab/models/dc_pow_p137_s1 lab/models/dc_pow_p139_s1 lab/models/dc_pow_p149_s1; echo ALLDONE8
