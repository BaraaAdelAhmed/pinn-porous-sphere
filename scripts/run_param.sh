#!/usr/bin/env bash
# Section 5.2, step 1: one-parameter sweep, lambda in [0.5, 16] (eta = 1, xi = 0.5, beta = 0.5, k = 1),
# boundary-layer feature on, 10,000 L-BFGS iterations, evaluated against the exact U_T(lambda) curve.
#   PN  all collocation points weighted equally
#   PW  all loss terms at a point weighted by (1+lambda^2)^-2  (balances small and large lambda)
# Seeds run in the order PN_s0, PW_s0, PN_s1, PW_s1, ... so that both variants and their run time
# are known after the first two runs.  Safe to stop and restart.
# Usage:  bash run_param.sh
set -u
SCRIPT=pinn_stage2_param.py
mkdir -p logs
config_args() {
  case "$1" in
    PN) echo "--lam 0.5 16" ;;
    PW) echo "--lam 0.5 16 --param_weight lam4" ;;
  esac
}
for seed in 0 1 2; do
  for cfg in PN PW; do
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
echo "Finished. Send the logs/*_eval.log files (or the results_P*.json files)."
