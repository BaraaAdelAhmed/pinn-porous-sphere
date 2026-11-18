#!/usr/bin/env bash
# Section 5.2, step 2: between the two weightings of run_param.sh.  Same sweep (lambda in [0.5, 16],
# eta = 1, xi = 0.5, beta = 0.5), 10,000 iterations, seeds 0-2.  Compare with PN and PW.
#   PH   all terms weighted by (1+lambda^2)^-1                (half the correction of PW)
#   PWu  weighted by (1+lambda^2)^-2, lambda sampled uniformly (more points at large lambda than log sampling)
# Usage:  bash run_param2.sh        Safe to stop and restart.
set -u
SCRIPT=pinn_stage2_param.py
mkdir -p logs
config_args() {
  case "$1" in
    PH)  echo "--lam 0.5 16 --param_weight lam2" ;;
    PWu) echo "--lam 0.5 16 --param_weight lam4 --lam_sampling uniform" ;;
  esac
}
for seed in 0 1 2; do
  for cfg in PH PWu; do
    tag="${cfg}_s${seed}"; args="$(config_args "$cfg") --seed ${seed} --tag ${tag}"
    if [ -f "results_${tag}.json" ]; then echo "skip ${tag}"; continue; fi
    echo "=== ${tag}: $(date '+%H:%M:%S')"
    rm -f "ckpt_${tag}.npz"; start=$(date +%s)
    python "$SCRIPT" --mode lbfgs --iters 10000 $args > "logs/${tag}_train.log" 2>&1 \
      || { echo "    training FAILED, see logs/${tag}_train.log"; continue; }
    python "$SCRIPT" --mode eval $args > "logs/${tag}_eval.log" 2>&1 \
      || { echo "    evaluation FAILED, see logs/${tag}_eval.log"; continue; }
    echo "    done in $(( $(date +%s) - start )) s"
    sed -n '/median_err/,/max_resid_interior/p' "logs/${tag}_eval.log" | sed 's/^/    /'
    echo "${tag},$(( $(date +%s) - start ))" >> logs/timing.csv
  done
done
echo "Finished. Send the results_PH_s*.json and results_PWu_s*.json files."
