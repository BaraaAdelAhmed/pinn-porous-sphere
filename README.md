# PINNs for the thermocapillary migration of a Brinkman porous sphere

Code for the paper

> *Physics-Informed Neural Networks for Thermocapillary Migration of a Brinkman Porous Sphere:
> Exact Benchmarking, Failure Mapping and Inverse Identification* (authors, journal and DOI to be added).

A physics-informed neural network (PINN), written in JAX, learns the migration velocity of a porous sphere
from the governing equations alone and is verified against the exact solution of El Sayed et al.,
*Mathematics* 14 (2026) 3563. The repository contains the solvers, the scripts that ran every experiment
in the paper, and the scripts that draw every figure.

## Contents

| Path | Purpose |
|---|---|
| `pinn_stage1_jax.py` | Single-parameter PINN (Sections 3, 4, 5.1 and 5.5): free or held particle, ablation options, boundary-layer feature, inverse identification of the slip coefficient |
| `pinn_stage2_param.py` | Parametric PINN over (λ, η, ξ, β) (Sections 5.2-5.4), with a λ sweep and the failure-atlas evaluation |
| `scripts/run_*.sh` | Batch runners for every experiment (table below) |
| `summarize_results.py`, `summarize_inverse.py` | Median and range over seeds for each configuration |
| `figures/fig_*.py` | One script per figure of the paper |
| `notebooks/pinn_stage1_colab.ipynb` | Google Colab notebook used to run the experiments |
| `results/` | Results files used by the figure scripts (see `results/README.md`) |

## Requirements

Python 3.10 or later with `jax`, `numpy`, `scipy` and `matplotlib` (`pip install -r requirements.txt`).
All experiments in the paper ran on an NVIDIA T4 GPU in Google Colab; a GPU-enabled JAX installation is
strongly recommended. A single training run takes about 7 minutes (Stage 1) or 15 minutes (Stage 2) on a T4.

Both solvers check their equations against the exact solution before any training:

```
python pinn_stage1_jax.py --mode selftest
python pinn_stage2_param.py --mode selftest --lam 0.5 16 --eta 0.25 4 --xi -1 1 --beta 0 2
```

## Reproducing the paper

Run the scripts from the repository root, for example `bash scripts/run_limits.sh`. Each runner trains and
evaluates its configurations for several random seeds, writes one `results_<tag>_s<seed>.json` per run to the
working directory, skips runs that are already finished, and can be stopped and restarted at any time.

| Paper | Runner | Configuration tags |
|---|---|---|
| Sec. 4.1, Table 2, Fig. 2 | `run_final.sh` | `A10k` (free particle), `H10k` (held particle), `C10k` |
| Sec. 3.6, Table 4 (discretization, budget) | `run_all.sh`, `run_budget.sh` | `A`, `B`, `C`, `H` (2100 iterations); `B5k`, `B10k`, `BR` |
| Sec. 4.2, Table 3 | `run_limits.sh`, `run_stage2_prep.sh` | `DRf/DRh`, `BUf/BUh`, `VCf/VCh`, `RG8`-`RG32` |
| Sec. 4.3, Table 4 (ablation) | `run_ablation.sh`, `run_stage2_prep.sh` | `NOR4`, `W1`, `W100`, `SIN`, `SILU`, `INT2`, `INT16`, `MA8`, `MB8`, `SIN10`-`SIN16` |
| Sec. 5.1, Table 5 | `run_remedies.sh` | `BL2`-`BL32`, `RS12`, `RS16`, `RB16` |
| Sec. 5.1, Fig. 3 (diagnostics) | `run_evalres.sh` | adds the interior residual to the 75 runs listed in the script (no retraining) |
| Sec. 5.2, Fig. 4 | `run_param.sh`, `run_param2.sh`, `run_param4.sh` | `PN`, `PW`, `PH`, `PWu`; `Q1`, `Q2` |
| Sec. 5.3-5.4, Figs. 5-6 | `run_param4.sh` | `Q1` (atlas files `atlas_Q1_s*.json`) |
| Sec. 5.5, Fig. 7 | `run_inverse.sh` | `IB025`-`IB15`, `IN1p`, `IN1m`, `IN5p`, `IN5m` |

Then summarize with `python summarize_results.py` and `python summarize_inverse.py`.

## Figures

Move the results files into `results/` (or set `RESULTS_DIR` to their folder) and run, from the repository root,

```
python figures/fig_diagnostics.py          # or any other figures/fig_*.py
RESULTS_DIR=/path/to/results python figures/fig_atlas.py
```

Each script writes its PDF and PNG to `figures/`.

## Citation

If you use this code, please cite the paper above (full reference to be added on publication).

## License

Released under the MIT License (see `LICENSE`).
