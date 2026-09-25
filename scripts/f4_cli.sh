#!/usr/bin/env bash
# F4 CLI ladder (K43): one model over the fixed test subsets. Resume-safe; stops cleanly on quota (exit 3).
# Usage (from jev/): nohup scripts/f4_cli.sh claude claude-haiku-4-5-20251001 >> runs/logs/f4_cli.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
runner=$1; model=$2; arm=$([ "$runner" = claude ] && echo D3 || echo D4)
for d in nusax_senti nt_senti nt_emot; do
  echo; echo "=== $(date -u +%FT%TZ) $runner $model $d"
  .venv/bin/python -u -m jevexp.runners.$runner --split test --scope subset --model "$model" --dataset $d --workers 4
  rc=$?
  if [ $rc -eq 3 ]; then echo "=== PAUSED (quota/infra) at $d — rerun the same command later to resume"; exit 3; fi
  [ $rc -ne 0 ] && { echo "=== FAILED rc=$rc at $d"; exit $rc; }
  .venv/bin/python -m jevexp.report_f4 --check "test-subset-$arm-$model-$d"
done
echo "=== $(date -u +%FT%TZ) DONE $model"
