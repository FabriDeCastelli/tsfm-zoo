#!/usr/bin/env bash
# Run the GIFT-Eval repro subset for the given family:variant list, one after another.
# Usage: scripts/repro.sh <env> <family:variant>...   (logs to outputs/repro_<variant>.log)
set -uo pipefail
cd "$(dirname "$0")/.."
export TSFM_ENV=$1; shift
source .envrc
mkdir -p outputs
for m in "$@"; do
  v=${m#*:}
  "$UV_PROJECT_ENVIRONMENT/bin/python" examples/compare.py --model "$m" --dataset gifteval:repro --device auto \
    ${BATCH_SIZE:+--batch-size $BATCH_SIZE} --out "outputs/repro_$v.jsonl" > "outputs/repro_$v.log" 2>&1
done
