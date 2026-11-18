#!/usr/bin/env bash
# Section 4.3: ablation of methodological choices. Baseline configuration, 10,000 iterations, seeds 0-4.
# Benchmark parameters (lam = 2) unless stated. Compare against A10k (baseline) and RG8 (baseline at lam = 8).
# Run in order of importance; safe to stop and restart.
#   NOR4   interior residual without the r^4 weight                       (claim in Sec. 3.5)
#   MA8    lam = 8, non-decaying exterior + unscaled exterior residual     (original formulation: Stokeslet mimicry)
#   MB8    lam = 8, decaying exterior + unscaled exterior residual         (decay constraint alone)
#   W1     interface weight w = 1                                          (Sec. 3.5)
#   W100   interface weight w = 100                                        (Sec. 3.5)
#   SIN    sine activation                                                 (Sec. 3.2)
#   SILU   SiLU activation                                                 (Sec. 3.2)
#   INT2   interior points concentrated towards the interface, lam = 2    (Sec. 3.6)
#   INT16  same, lam = 16: does it rescue the failure of Sec. 4.2?         (compare RG16)
# Usage:  bash run_ablation.sh
set -u
SCRIPT=pinn_stage1_jax.py
ITERS=10000
mkdir -p logs
config_args() {
  case "$1" in
    NOR4)  echo "--no_r4" ;;
    MA8)   echo "--lam 8 --exterior nondecay --resid_ext unscaled" ;;
    MB8)   echo "--lam 8 --resid_ext unscaled" ;;
    W1)    echo "--w_if 1" ;;
    W100)  echo "--w_if 100" ;;
    SIN)   echo "--act sin" ;;
    SILU)  echo "--act silu" ;;
    INT2)  echo "--int_sampling interface" ;;
    INT16) echo "--lam 16 --int_sampling interface" ;;
  esac
}
for cfg in NOR4 MA8 MB8 W1 W100 SIN SILU INT2 INT16; do
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
