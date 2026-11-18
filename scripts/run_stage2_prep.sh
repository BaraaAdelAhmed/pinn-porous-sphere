#!/usr/bin/env bash
# Preparation for Stage 2: where does the method stop working, and does the sine activation move that limit?
# Baseline configuration, 10,000 iterations, seeds 0-4, beta = 0.5, eta = 1, xi = 0.5, k = 1.
#   SIN16       sine activation at lam = 16   (tanh fails there: RG16)
#   RG10, RG12  tanh at lam = 10, 12          (does the baseline cover the base paper's range, lam <= 10?)
#   SIN10, SIN12  sine at lam = 10, 12
# Usage:  bash run_stage2_prep.sh        Safe to stop and restart.
set -u
SCRIPT=pinn_stage1_jax.py
ITERS=10000
mkdir -p logs
config_args() {
  case "$1" in
    SIN16) echo "--lam 16 --act sin" ;;
    RG10)  echo "--lam 10" ;;
    RG12)  echo "--lam 12" ;;
    SIN10) echo "--lam 10 --act sin" ;;
    SIN12) echo "--lam 12 --act sin" ;;
  esac
}
for cfg in SIN16 RG10 RG12 SIN10 SIN12; do
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
