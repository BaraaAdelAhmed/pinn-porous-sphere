#!/usr/bin/env bash
# Section 5.1: remedies for the screening-layer failure. Baseline configuration otherwise,
# 10,000 iterations, seeds 0-4, beta = 0.5, eta = 1, xi = 0.5, k = 1.  Compare with RG12, RG16, RG32, A10k.
#   BL16, BL32, BL12  boundary-layer input feature exp(-lam(1-r)) for the interior network
#   BL2               same feature at lam = 2: it must not degrade the benchmark (compare A10k)
#   RS16, RS12        interior residual rescaled by (1+lam^2)^-2
#   RB16              both remedies together
# Usage:  bash run_remedies.sh        Safe to stop and restart.
set -u
SCRIPT=pinn_stage1_jax.py
ITERS=10000
mkdir -p logs
config_args() {
  case "$1" in
    BL16) echo "--lam 16 --bl_feature" ;;
    BL32) echo "--lam 32 --bl_feature" ;;
    BL12) echo "--lam 12 --bl_feature" ;;
    BL2)  echo "--bl_feature" ;;
    RS16) echo "--lam 16 --resid_scale lambda4" ;;
    RS12) echo "--lam 12 --resid_scale lambda4" ;;
    RB16) echo "--lam 16 --bl_feature --resid_scale lambda4" ;;
  esac
}
for cfg in BL16 BL32 BL12 BL2 RS16 RS12 RB16; do
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
