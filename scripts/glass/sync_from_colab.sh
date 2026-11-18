#!/bin/bash
# Workspace side of the relay (used while Colab has no GitHub token): download Colab's latest snapshot, unpack it into
# the repo, regenerate RESULTS.md, commit and push. Run:  bash scripts/glass/sync_from_colab.sh
set -e
export OAUTHLIB_RELAX_TOKEN_SCOPE=1
cd "$(dirname "$0")/../.."
rm -f /tmp/glass_snapshot.tgz
timeout 100 colab download -s ${GLASS_SESSION:-glass2} /content/glass_snapshot.tgz /tmp/glass_snapshot.tgz >/dev/null 2>&1 || \
  timeout 100 colab download -s ${GLASS_SESSION:-glass2} /content/glass_snapshot.tgz /tmp/glass_snapshot.tgz >/dev/null 2>&1
[ -s /tmp/glass_snapshot.tgz ] || { echo "no snapshot downloaded"; exit 1; }
tar xzf /tmp/glass_snapshot.tgz
python3 scripts/glass/results.py; GLASS_ROUND=round2 python3 scripts/glass/results.py 2>/dev/null || true
git add -A results/glass lab/constructions/zoo_2nd_argmax_* lab/constructions/lc_* 2>/dev/null
if git diff --cached --quiet; then echo "nothing new"; exit 0; fi
git commit -q -m "glass: Colab results snapshot $(date -u +%H:%M)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WJHBaVqcKQoS2RVeW9xQWr"
timeout 100 git push -q 2>&1 | grep -v ^remote || true
git log --oneline -1
