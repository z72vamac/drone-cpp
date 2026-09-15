#!/usr/bin/env bash
# Launch one shard of the K-split battery.
# Usage:
#   ./run_shard.sh <shard-index> <num-shards> [extra args passed to compare_methods_k.py]
# Examples:
#   ./run_shard.sh 0 8 --ks 2,3 --nr 1,2,3 --threads 4 --time-limit 900
#   ./run_shard.sh 3 8 --ks 2 --nr 5,8,10 --E 300,2400 --threads 8 --time-limit 1800 --resume
set -euo pipefail
cd "$(dirname "$0")/.."

SHARD="${1:?shard index}"; shift
SHARDS="${1:?num shards}"; shift

if [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

python compare_methods_k.py \
  --shard-index "$SHARD" \
  --num-shards "$SHARDS" \
  "$@"
