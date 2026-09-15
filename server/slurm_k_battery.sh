#!/usr/bin/env bash
# SLURM array launcher for the K-split battery.
# Edit the SBATCH directives and the TIER block below, then:
#   sbatch server/slurm_k_battery.sh
#SBATCH --job-name=k_battery
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=compare_results_k/slurm_%A_%a.out

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- Tier 1 (core): K=2,3 on small regions, full testbed grid ----
# 270 configs x 2 K x 2 methods = 1080 runs over 8 tasks (~135 runs/task)
TIER_ARGS=(--ks 2,3 --nr 1,2,3 --threads "$SLURM_CPUS_PER_TASK" --time-limit 900 --mip-gap 0.02)

# ---- Tier 2 (large, sparse): K=2 on big regions, tight+loose endurance ----
# Uncomment to run instead (90 configs x 2 methods = 180 runs):
# TIER_ARGS=(--ks 2 --nr 5,8,10 --E 300,2400 --threads "$SLURM_CPUS_PER_TASK" --time-limit 1800 --mip-gap 0.02 --resume)

if [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

N_TASKS=8  # must match --array range size
python compare_methods_k.py \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$N_TASKS" \
  --resume \
  "${TIER_ARGS[@]}"
