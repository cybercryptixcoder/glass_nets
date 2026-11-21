#!/bin/bash
# Colab (CPU is enough): decompile a list of networks and keep the reports. Usage: bash scripts/decomp/colab_decompile.sh id1 id2 ...
cd /content/helix_probe || exit 1
pip -q install scikit-learn sympy scipy > /dev/null 2>&1
for m in "$@"; do python3 lab/decomp/decompile.py $m 2>&1 | grep -v Warn | tail -1; done
echo ALLDONE
