#!/usr/bin/env bash
# Inverse demonstration at lambda = 8 (application range of Sec. 6.3): beta unknown, a measured U_T is the only data.
# eta = 1, xi = 0.5, k = 1, boundary-layer feature (Sec. 5.1), 10,000 L-BFGS iterations, seeds 0-2.
#   L8B05, L8B10, L8B20, L8B40   true beta = 0.5, 1, 2, 4, exact measurement
#   L8N05p, L8N05m, L8N1p, L8N1m  true beta = 0.5, measurement in error by +0.5%, -0.5%, +1%, -1%
#   L8B05nf                       true beta = 0.5, exact measurement, WITHOUT the boundary-layer feature (comparison)
# Usage:  bash run_inverse_lam8.sh        Safe to stop and restart.  Then:  python summarize_inverse.py
set -u
SCRIPT=pinn_stage1_jax.py
mkdir -p logs
COMMON="--inverse_beta --lam 8 --eta 1 --xi 0.5"
config_args() {
  case "$1" in
    L8B05)   echo "$COMMON --beta 0.5 --bl_feature" ;;
    L8B10)   echo "$COMMON --beta 1.0 --bl_feature" ;;
    L8B20)   echo "$COMMON --beta 2.0 --bl_feature" ;;
    L8B40)   echo "$COMMON --beta 4.0 --bl_feature" ;;
    L8N05p)  echo "$COMMON --beta 0.5 --bl_feature --noise 0.005" ;;
    L8N05m)  echo "$COMMON --beta 0.5 --bl_feature --noise -0.005" ;;
    L8N1p)   echo "$COMMON --beta 0.5 --bl_feature --noise 0.01" ;;
    L8N1m)   echo "$COMMON --beta 0.5 --bl_feature --noise -0.01" ;;
    L8B05nf) echo "$COMMON --beta 0.5" ;;
  esac
}
for cfg in L8B05 L8B10 L8B20 L8B40 L8N05p L8N05m L8N1p L8N1m L8B05nf; do
  for seed in 0 1 2; do
    tag="${cfg}_s${seed}"; args="$(config_args "$cfg") --seed ${seed} --tag ${tag}"
    if [ -f "results_${tag}.json" ]; then echo "skip ${tag}"; continue; fi
    echo "=== ${tag}: $(date '+%H:%M:%S')"
    rm -f "ckpt_${tag}.npz"; start=$(date +%s)
    python "$SCRIPT" --mode lbfgs --iters 10000 $args > "logs/${tag}_train.log" 2>&1 \
      || { echo "    training FAILED, see logs/${tag}_train.log"; continue; }
    python "$SCRIPT" --mode eval $args > "logs/${tag}_eval.log" 2>&1 \
      || { echo "    evaluation FAILED, see logs/${tag}_eval.log"; continue; }
    echo "    done in $(( $(date +%s) - start )) s; $(grep -h 'beta' logs/${tag}_train.log | tail -1)"
  done
done
echo "Finished. Next:  python summarize_inverse.py"
