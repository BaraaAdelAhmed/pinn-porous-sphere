# Results files

The figure scripts read the results of the experiments from this folder:

- `results_<tag>_s<seed>.json`: one file per training run, written by the solvers' evaluation mode;
- `atlas_Q1_s<seed>.json`: the failure-atlas evaluation of the four-parameter network;
- `fields_A10k_s1.npz`: the field data of the median Stage 1 run, used for Fig. 2.

Model checkpoints (`ckpt_*.npz`) are not included, because they are large and can be regenerated with the runners.
