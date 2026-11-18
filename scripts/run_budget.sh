#!/usr/bin/env bash
# Stage 1 follow-up: is the seed-to-seed spread an optimization-budget effect?
# Baseline B settings (2000 interior points), seeds 0-4.
#   B5k   single L-BFGS call,  5000 iterations
#   B10k  single L-BFGS call, 10000 iterations
#   BR    2100 iterations as 4 x 525 with L-BFGS memory restarted between segments
# Results are directly comparable with configuration B (2100 iterations, single call).
# Usage:  bash run_budget.sh        Safe to stop and restart.
set -u
SCRIPT=pinn_stage1_jax.py
mkdir -p logs
for cfg in B5k B10k BR; do
  for seed in 0 1 2 3 4; do
    tag="${cfg}_s${seed}"; args="--n_int 2000 --seed ${seed} --tag ${tag}"
    if [ -f "results_${tag}.json" ]; then echo "skip ${tag}"; continue; fi
    echo "=== ${tag}: $(date '+%H:%M:%S')"
    rm -f "ckpt_${tag}.npz"; start=$(date +%s); ok=1
    case "$cfg" in
      B5k)  python "$SCRIPT" --mode lbfgs --iters 5000  $args > "logs/${tag}_train.log" 2>&1 || ok=0 ;;
      B10k) python "$SCRIPT" --mode lbfgs --iters 10000 $args > "logs/${tag}_train.log" 2>&1 || ok=0 ;;
      BR)   : > "logs/${tag}_train.log"
            for k in 1 2 3 4; do
              python "$SCRIPT" --mode lbfgs --iters 525 $args >> "logs/${tag}_train.log" 2>&1 || ok=0
            done ;;
    esac
    [ "$ok" = 1 ] || { echo "    training FAILED, see logs/${tag}_train.log"; continue; }
    python "$SCRIPT" --mode eval $args > "logs/${tag}_eval.log" 2>&1 \
      || { echo "    evaluation FAILED, see logs/${tag}_eval.log"; continue; }
    echo "    done in $(( $(date +%s) - start )) s; $(grep -h 'rel.err' logs/${tag}_train.log | tail -1)"
    echo "${tag},$(( $(date +%s) - start ))" >> logs/timing.csv
  done
done
echo "Finished. Next:  python summarize_results.py"
