#!/usr/bin/env bash
# Section 5.5: inverse demonstration. beta is unknown (learned as the slip length beta/(lam sqrt(eta)) in (0,1),
# starting mid-range); a measured U_T is the only data. lam = 2, eta = 1, xi = 0.5, k = 1, Stage 1 baseline
# configuration, 10,000 iterations, seeds 0-2.
#   IB025, IB05, IB10, IB15   true beta = 0.25, 0.5, 1.0, 1.5, exact measurement
#   IN1p, IN1m, IN5p, IN5m    true beta = 0.5, measurement in error by +1%, -1%, +5%, -5%
# Usage:  bash run_inverse.sh        Safe to stop and restart.  Then:  python summarize_inverse.py
set -u
SCRIPT=pinn_stage1_jax.py
mkdir -p logs
config_args() {
  case "$1" in
    IB025) echo "--beta 0.25" ;;
    IB05)  echo "--beta 0.5" ;;
    IB10)  echo "--beta 1.0" ;;
    IB15)  echo "--beta 1.5" ;;
    IN1p)  echo "--beta 0.5 --noise 0.01" ;;
    IN1m)  echo "--beta 0.5 --noise -0.01" ;;
    IN5p)  echo "--beta 0.5 --noise 0.05" ;;
    IN5m)  echo "--beta 0.5 --noise -0.05" ;;
  esac
}
for cfg in IB025 IB05 IB10 IB15 IN1p IN1m IN5p IN5m; do
  for seed in 0 1 2; do
    tag="${cfg}_s${seed}"; args="--inverse_beta $(config_args "$cfg") --seed ${seed} --tag ${tag}"
    if [ -f "results_${tag}.json" ]; then echo "skip ${tag}"; continue; fi
    echo "=== ${tag}: $(date '+%H:%M:%S')"
    rm -f "ckpt_${tag}.npz"; start=$(date +%s)
    python "$SCRIPT" --mode lbfgs --iters 10000 $args > "logs/${tag}_train.log" 2>&1 \
      || { echo "    training FAILED, see logs/${tag}_train.log"; continue; }
    python "$SCRIPT" --mode eval $args > "logs/${tag}_eval.log" 2>&1 \
      || { echo "    evaluation FAILED, see logs/${tag}_eval.log"; continue; }
    echo "    done in $(( $(date +%s) - start )) s; $(grep -h 'beta' logs/${tag}_train.log | tail -1)"
    echo "${tag},$(( $(date +%s) - start ))" >> logs/timing.csv
  done
done
echo "Finished. Next:  python summarize_inverse.py"
