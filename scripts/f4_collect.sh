#!/usr/bin/env bash
# F4 steps 1-7 (prereg-v1 protocol): Jev arms, NLLB + A3, RQ4-min, DeepSeek D1. Resume-safe; stops on a failed check.
# Run from jev/:  nohup scripts/f4_collect.sh > runs/logs/f4_collect.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DATASETS="nusax_senti nt_senti nt_emot"
step() { echo; echo "=== $(date -u +%FT%TZ) $*"; }
check() { $PY -m jevexp.report_f4 --check "$1"; }

for d in $DATASETS; do
  step "A1 noul_en $d";   $PY -u -m jevexp.runners.jev --split test --scope full --arm A1 --formulation noul_en --dataset $d
  check "test-full-A1-noul_en-$d"
  step "A1 choice_en $d"; $PY -u -m jevexp.runners.jev --split test --scope full --arm A1 --formulation choice_en --dataset $d
  check "test-full-A1-choice_en-$d"
done
for d in nt_senti nt_emot; do
  for f in noul_en choice_en; do
    step "A4 $f $d"; $PY -u -m jevexp.runners.jev --split test --scope full --arm A4 --formulation $f --dataset $d
    check "test-full-A4-$f-$d"
  done
done
for d in $DATASETS; do
  step "NLLB $d"; $PY -u -m jevexp.runners.nllb --split test --scope full --dataset $d
  step "A3 noul_en $d"; $PY -u -m jevexp.runners.jev --split test --scope full --arm A3 --formulation noul_en --dataset $d
  check "test-full-A3-noul_en-$d"
done
for d in $DATASETS; do
  step "RQ4-min choice_id $d"; $PY -u -m jevexp.runners.jev --split test --scope subset --arm A1 --formulation choice_id --dataset $d
  check "test-subset-A1-choice_id-$d"
done
for d in $DATASETS; do
  step "D1 deepseek-flash $d"; $PY -u -m jevexp.runners.deepseek --split test --scope full --dataset $d
  check "test-full-D1-deepseek-flash-nothink-$d"
done
step "DONE"; $PY -m jevexp.report_f4
