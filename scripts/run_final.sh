#!/usr/bin/env bash
# Stage 1, final set at the 10,000-iteration budget (single L-BFGS call), seeds 0-4.
# Run in order of importance; B10k already exists.
#   H10k  held particle, 2000 interior points        -> F_T for Section 4.1
#   A10k  800 interior points, 3x32 network          -> does collocation density set the floor?
#   C10k  800 interior points, 4x48 network          -> does network capacity set the floor?
# Usage:  bash run_final.sh        Safe to stop and restart.
set -u
SCRIPT=pinn_stage1_jax.py
ITERS=10000
mkdir -p logs
config_args() {
  case "$1" in
    H10k) echo "--n_int 2000 --branch held" ;;
    A10k) echo "--n_int 800" ;;
    C10k) echo "--n_int 800 --int_layers 48,48,48,48" ;;
  esac
}
for cfg in H10k A10k C10k; do
  for seed in 0 1 2 3 4; do
    tag="${cfg}_s${seed}"; args="$(config_args "$cfg") --seed ${seed} --tag ${tag}"
    if [ -f "results_${tag}.json" ]; then echo "skip ${tag}"; continue; fi
    echo "=== ${tag}: $(date '+%H:%M:%S')"
    rm -f "ckpt_${tag}.npz"; start=$(date +%s)
    python "$SCRIPT" --mode lbfgs --iters "$ITERS" $args > "logs/${tag}_train.log" 2>&1 \
      || { echo "    training FAILED, see logs/${tag}_train.log"; continue; }
    python "$SCRIPT" --mode eval $args > "logs/${tag}_eval.log" 2>&1 \
      || { echo "    evaluation FAILED, see logs/${tag}_eval.log"; continue; }
    echo "    done in $(( $(date +%s) - start )) s; $(grep -h 'rel.err' logs/${tag}_train.log | tail -1)"
    echo "${tag},$(( $(date +%s) - start ))" >> logs/timing.csv
  done
done
echo "Finished. Next:  python summarize_results.py"
