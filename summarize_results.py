"""
Collect results_<config>_s<seed>.json files written by pinn_stage1_jax.py --mode eval
and print, for each configuration, the median and range over seeds of every metric.

Usage:  python summarize_results.py            (reads results_*.json in the current folder)
Output: a table on screen and summary.csv
"""
import glob, json, re, csv
import numpy as np

runs = {}
for f in sorted(glob.glob("results_*.json")):
    m = re.match(r"results_(.+)_s(\d+)\.json$", f)
    if not m:
        print(f"skipping {f} (name does not end in _s<seed>)")
        continue
    cfg, seed = m.group(1), int(m.group(2))
    d = json.load(open(f))
    err = d.get("U_rel_err", d.get("F_rel_err"))
    if "U_pinn" in d:
        signed = (d["U_pinn"] - d["U_exact"]) / abs(d["U_exact"])
    else:
        signed = (d["F_pinn"] - d["F_exact"]) / abs(d["F_exact"])
    fp = np.abs(list(d["force_profile"].values()))
    if d["params"]["branch"] == "held":
        fp = np.abs(fp - 1.0)                       # held: deviation from F(R)/(8 pi C) = 1
    runs.setdefault(cfg, []).append(dict(
        seed=seed, err=err, signed=signed, L2_int=d["L2_interior"], L2_ext=d["L2_exterior"],
        force=float(fp.max()), zeta=max(d["zeta_spurious_interior"], d["zeta_spurious_exterior"]),
        loss=sum(d["loss_terms"].values()), iters=d["iterations"]["lbfgs"],
        rint=d.get("resid_interior", np.nan), rext=d.get("resid_exterior", np.nan)))

cols = ["err", "L2_int", "L2_ext", "force", "zeta"]
names = {"err": "U_T or F_T error", "L2_int": "L2 interior", "L2_ext": "L2 exterior",
         "force": "force profile", "zeta": "zeta diagnostic"}
rows = []
for cfg, rs in sorted(runs.items()):
    iters = sorted({r["iters"] for r in rs})
    print(f"\n=== {cfg}: {len(rs)} seeds {sorted(r['seed'] for r in rs)}, L-BFGS iterations {iters}")
    if len(iters) > 1:
        print("    WARNING: runs in this configuration used different iteration budgets")
    for c in cols:
        v = np.array([r[c] for r in rs])
        print(f"    {names[c]:18s} median {np.median(v):.2e}   range {v.min():.2e} - {v.max():.2e}")
        rows.append([cfg, len(rs), names[c], np.median(v), v.min(), v.max()])
    for c, nm in (("rint", "interior residual"), ("rext", "exterior residual")):
        v = np.array([r[c] for r in rs], dtype=float)
        if np.isfinite(v).all():
            print(f"    {nm:18s} median {np.median(v):.2e}   range {v.min():.2e} - {v.max():.2e}")
            rows.append([cfg, len(rs), nm, np.median(v), v.min(), v.max()])
    sg = np.array([r["signed"] for r in rs])
    print(f"    signed error       median {np.median(sg):+.2e}   ({(sg < 0).sum()} of {len(sg)} runs below the exact value)")
    print(f"    (per-seed error: " + ", ".join(f"s{r['seed']} {r['err']:.1e}" for r in sorted(rs, key=lambda r: r['seed'])) + ")")
    print(f"    (per-seed loss:  " + ", ".join(f"s{r['seed']} {r['loss']:.1e}" for r in sorted(rs, key=lambda r: r['seed'])) + ")")

# does a lower final loss mean a smaller error? (rank correlation over all runs)
allr = [r for rs in runs.values() for r in rs]
if len(allr) >= 5:
    from scipy.stats import spearmanr
    rho, pval = spearmanr([r["loss"] for r in allr], [r["err"] for r in allr])
    print(f"\nSpearman rank correlation, final loss vs error, all {len(allr)} runs: rho = {rho:.2f} (p = {pval:.2g})")

with open("summary.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["configuration", "n_seeds", "metric", "median", "min", "max"])
    w.writerows(rows)
print("\nwritten: summary.csv")
