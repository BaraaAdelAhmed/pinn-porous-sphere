#!/usr/bin/env bash
# Section 4.2: limiting cases. Baseline configuration (800 interior points, 3x32), 10,000 iterations, seeds 0-4.
# Run in order of importance; stop whenever time runs out and resume later.
#   DRf/DRh  viscous drop   lam = 0, beta = 0, eta = 1     exact U = 2/15,  F = 4 pi/6     (free / held)
#   BUf/BUh  bubble         eta = 0, beta = 0, lam = 2     exact U = 1/3,   F = 4 pi/3     (free / held)
#   VCf/VCh  velocity continuity  beta = 0, lam = 2        exact Eqs. (46), (48) of the base paper
#   RG8/RG16/RG32  approach to the rigid sphere, lam = 8, 16, 32 (beta = 0.5): thinning Brinkman layer
# Usage:  bash run_limits.sh        Safe to stop and restart.
set -u
SCRIPT=pinn_stage1_jax.py
ITERS=10000
mkdir -p logs
config_args() {
  case "$1" in
    DRf)  echo "--lam 0 --beta 0" ;;
    DRh)  echo "--lam 0 --beta 0 --branch held" ;;
    BUf)  echo "--eta 0 --beta 0" ;;
    BUh)  echo "--eta 0 --beta 0 --branch held" ;;
    VCf)  echo "--beta 0" ;;
    VCh)  echo "--beta 0 --branch held" ;;
    RG8)  echo "--lam 8" ;;
    RG16) echo "--lam 16" ;;
    RG32) echo "--lam 32" ;;
  esac
}
for cfg in DRf DRh BUf BUh VCf VCh RG8 RG16 RG32; do
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
