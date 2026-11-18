#!/usr/bin/env bash
# Adds the residual diagnostics (Section 5.1) to existing results files. No retraining:
# each call loads the saved checkpoint ckpt_<tag>.npz and updates results_<tag>.json.
# The arguments must match those used in training, as listed below.
# Usage:  bash run_evalres.sh        (about 30-60 s per run)
set -u
SCRIPT=pinn_stage1_jax.py
config_args() {
  case "$1" in
    A10k) echo "" ;;
    BL2)  echo "--bl_feature" ;;
    BL12) echo "--lam 12 --bl_feature" ;;
    BL16) echo "--lam 16 --bl_feature" ;;
    BL32) echo "--lam 32 --bl_feature" ;;
    RS12) echo "--lam 12 --resid_scale lambda4" ;;
    RS16) echo "--lam 16 --resid_scale lambda4" ;;
    RB16) echo "--lam 16 --bl_feature --resid_scale lambda4" ;;
    RG8)  echo "--lam 8" ;;
    RG10) echo "--lam 10" ;;
    RG12) echo "--lam 12" ;;
    RG16) echo "--lam 16" ;;
    RG32) echo "--lam 32" ;;
    MA8)  echo "--lam 8 --exterior nondecay --resid_ext unscaled" ;;
    MB8)  echo "--lam 8 --resid_ext unscaled" ;;
  esac
}
for cfg in A10k BL2 BL12 BL16 BL32 RS12 RS16 RB16 RG8 RG10 RG12 RG16 RG32 MA8 MB8; do
  for seed in 0 1 2 3 4; do
    tag="${cfg}_s${seed}"
    if [ ! -f "ckpt_${tag}.npz" ]; then echo "no checkpoint for ${tag}, skipped"; continue; fi
    if grep -q resid_interior "results_${tag}.json" 2>/dev/null; then echo "skip ${tag} (already has diagnostics)"; continue; fi
    python "$SCRIPT" --mode evalres $(config_args "$cfg") --seed "$seed" --tag "$tag" 2>&1 | tail -1
  done
done
echo "Finished. Next:  python summarize_results.py"
