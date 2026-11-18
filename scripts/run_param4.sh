#!/usr/bin/env bash
# Sections 5.2-5.4: four-parameter model, lambda in [0.5,16], eta in [0.25,4], xi in [-1,1], beta in [0,2]
# (valid domain only), weighting (1+lambda^2)^-1, boundary-layer feature, 10,000 L-BFGS iterations, seeds 0-2.
#   Q1  sweep configuration: 2000/1500/500 points, interior 3x48, exterior 2x32, U-net 2x16
#   Q2  larger: 4000/3000/1000 points, interior 4x64, exterior 3x48, U-net 2x32
# Each run is followed by
#   eval   lambda sweep at (eta, xi, beta) = (1, 0.5, 0.5)       -> results_<tag>.json
#   atlas  slices (lam,beta), (lam,eta), (lam,xi), 9x9 each     -> atlas_<tag>.json
# Order Q1_s0, Q2_s0, Q1_s1, ... so that the cost of Q2 is known early.  Safe to stop and restart.
# Usage:  bash run_param4.sh
set -u
SCRIPT=pinn_stage2_param.py
mkdir -p logs
RANGE="--lam 0.5 16 --eta 0.25 4 --xi -1 1 --beta 0 2 --param_weight lam2"
config_args() {
  case "$1" in
    Q1) echo "$RANGE" ;;
    Q2) echo "$RANGE --n_int 4000 --n_ext 3000 --n_if 1000 --int_layers 64,64,64,64 --ext_layers 48,48,48 --u_layers 32,32" ;;
  esac
}
for seed in 0 1 2; do
  for cfg in Q1 Q2; do
    tag="${cfg}_s${seed}"; args="$(config_args "$cfg") --seed ${seed} --tag ${tag}"
    if [ -f "atlas_${tag}.json" ]; then echo "skip ${tag}"; continue; fi
    echo "=== ${tag}: $(date '+%H:%M:%S')"
    start=$(date +%s)
    if [ ! -f "results_${tag}.json" ]; then
      rm -f "ckpt_${tag}.npz"
      python "$SCRIPT" --mode lbfgs --iters 10000 $args > "logs/${tag}_train.log" 2>&1 \
        || { echo "    training FAILED, see logs/${tag}_train.log"; continue; }
      echo "    trained in $(( $(date +%s) - start )) s"
      python "$SCRIPT" --mode eval $args > "logs/${tag}_eval.log" 2>&1 \
        || { echo "    evaluation FAILED, see logs/${tag}_eval.log"; continue; }
      sed -n '/median_err/,/max_resid_interior/p' "logs/${tag}_eval.log" | sed 's/^/    /'
    fi
    python "$SCRIPT" --mode atlas $args > "logs/${tag}_atlas.log" 2>&1 \
      || { echo "    atlas FAILED, see logs/${tag}_atlas.log"; continue; }
    grep "^slice" "logs/${tag}_atlas.log" | sed 's/^/    /'
    echo "    done in $(( $(date +%s) - start )) s"
    echo "${tag},$(( $(date +%s) - start ))" >> logs/timing.csv
  done
done
echo "Finished. Send the results_Q*.json and atlas_Q*.json files."
