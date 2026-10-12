#!/usr/bin/env bash
# Pack the codebase into a zip for sharing (e.g. uploading to a Claude chat).
#
#     scripts/pack.sh [OUT.zip]        # default: ../birdeval-YYYYMMDD.zip
#
# Contains every committed file (git archive of HEAD), plus the gitignored
# results that are small and useful for discussion: the report tables
# (results/report/*.csv) and each run's summary.csv. Leaves out the
# prediction cache, audio, .venv, the eBird taxonomy CSV and the report page.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

out=${1:-../birdeval-$(date +%Y%m%d).zip}
out=$(cd "$(dirname "$out")" && pwd)/$(basename "$out")

if [ -n "$(git status --porcelain)" ]; then
  echo "note: uncommitted changes are not included (packing HEAD $(git rev-parse --short HEAD))" >&2
fi

rm -f "$out"
git archive --format=zip -o "$out" HEAD
extras=$(ls results/report/*.csv results/*/summary.csv 2>/dev/null || true)
if [ -n "$extras" ]; then
  # shellcheck disable=SC2086
  zip -q "$out" $extras
fi

echo "wrote $out ($(du -h "$out" | cut -f1), $(unzip -l "$out" | tail -1 | awk '{print $2}') files)"
