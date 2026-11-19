"""Summarize the inverse runs (Section 5.5): results_<config>_s<seed>.json files with an 'inverse' entry.
For each configuration: true beta, the beta a perfect solver would infer from the (noisy) datum, and the
PINN estimate (median and range over seeds), with relative errors against both.
Usage:  python summarize_inverse.py"""
import glob, json, re
import numpy as np
runs = {}
for f in sorted(glob.glob("results_*_s*.json")):
    d = json.load(open(f))
    if "inverse" not in d:
        continue
    cfg = re.match(r"results_(.+)_s\d+\.json$", f).group(1)
    inv = dict(d["inverse"]); inv["lam"] = d.get("params", {}).get("lam", float("nan"))
    fp = d.get("force_profile"); inv["force"] = max(abs(v) for v in fp.values()) if isinstance(fp, dict) else float("nan")
    inv["U_exact"] = d.get("U_exact", float("nan")); runs.setdefault(cfg, []).append(inv)
print(f"{'config':8s} {'lam':>4s} {'noise':>6s} {'beta_true':>9s} {'beta_ideal':>10s} {'beta_PINN median (range)':>32s} "
      f"{'err vs true':>12s} {'err vs ideal':>13s} {'seeds':>5s}")
for cfg, rs in sorted(runs.items()):
    b = np.array([r["beta_pinn"] for r in rs]); bt = rs[0]["beta_true"]; bi = rs[0]["beta_exact_inverse"]
    et = np.abs(b - bt) / bt; ei = np.abs(b - bi) / bi
    print(f"{cfg:8s} {rs[0]['lam']:4g} {rs[0]['noise']:+6.3f} {bt:9.4f} {bi:10.4f} {np.median(b):10.4f} ({b.min():.4f}-{b.max():.4f}) "
          f"{np.median(et):12.1e} {np.median(ei):13.1e} {len(rs):5d}  max force {max(r['force'] for r in rs):.1e}")
