#!/usr/bin/env bash
# Stage 1 multi-seed experiment: trains and evaluates every configuration below.
# Safe to stop and restart: finished runs (results file present) are skipped,
# and an interrupted run is restarted from scratch so every run has the same budget.
#
#   A  800 interior points, 3x32 interior network   (previous baseline)
#   B  2000 interior points, 3x32 interior network  (current baseline)
#   C  800 interior points, 4x48 interior network   (capacity comparison)
#   H  held particle, baseline B settings           (F_T)
#
# Usage:  bash run_all.sh            (DRY=1 bash run_all.sh  prints commands only)
set -u
SEEDS="0 1 2 3 4"
ITERS=2100
SCRIPT=pinn_stage1_jax.py
mkdir -p logs

config_args() {            # works in every bash version, including macOS bash 3.2
  case "$1" in
    A) echo "--n_int 800" ;;
    B) echo "--n_int 2000" ;;
    C) echo "--n_int 800 --int_layers 48,48,48,48" ;;
    H) echo "--n_int 2000 --branch held" ;;
  esac
}

for cfg in A B C H; do
  for seed in $SEEDS; do
    tag="${cfg}_s${seed}"
    args="$(config_args "$cfg") --seed ${seed} --tag ${tag}"
    if [ -f "results_${tag}.json" ]; then
      echo "skip ${tag} (already done)"; continue
    fi
    echo "=== ${tag}: $(date '+%H:%M:%S')"
    if [ "${DRY:-0}" = "1" ]; then
      echo "python ${SCRIPT} --mode lbfgs --iters ${ITERS} ${args}"
      echo "python ${SCRIPT} --mode eval ${args}"; continue
    fi
    rm -f "ckpt_${tag}.npz"                       # fresh start: same budget for every run
    start=$(date +%s)
    python "${SCRIPT}" --mode lbfgs --iters "${ITERS}" ${args} > "logs/${tag}_train.log" 2>&1 \
      || { echo "    training FAILED, see logs/${tag}_train.log"; continue; }
    python "${SCRIPT}" --mode eval ${args} > "logs/${tag}_eval.log" 2>&1 \
      || { echo "    evaluation FAILED, see logs/${tag}_eval.log"; continue; }
    echo "    done in $(( $(date +%s) - start )) s; $(grep -h 'rel.err' logs/${tag}_train.log | tail -1)"
    echo "${tag},$(( $(date +%s) - start ))" >> logs/timing.csv
  done
done
echo "All runs finished. Next:  python summarize_results.py"
