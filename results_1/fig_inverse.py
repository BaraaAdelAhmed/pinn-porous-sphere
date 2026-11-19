"""Fig. 7: inverse identification of the slip coefficient at lambda = 2 (a, b) and lambda = 8 (c, d).
Reads results_<tag>_s<seed>.json written by scripts/run_inverse.sh and scripts/run_inverse_lam8.sh. Run from the repository root."""
import os, sys, glob, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES = os.environ.get("RESULTS_DIR", "results"); OUT = os.environ.get("FIG_DIR", "figures")
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
SRC = open("pinn_stage1_jax.py").read().split("# =============================================================================\n# 6. Collocation and loss")[0]
def exact_inverse_curve(lam, eps):
    """beta an exact solver infers from U_T(beta = 0.5) (1 + eps), eta = 1, xi = 0.5."""
    sys.argv = ["x", "--inverse_beta", "--mode", "selftest", "--lam", str(lam), "--beta", "0.5"]
    ns = {}; exec(SRC.replace('if args.mode == "selftest":', "if False:"), ns)
    return np.array([ns["exact_inverse_slip"](ns["EX"]["U"] * (1 + e)) * lam * np.sqrt(ns["ETA"]) for e in eps])
def load(cfg):
    rs = [json.load(open(f))["inverse"] for f in sorted(glob.glob(f"{RES}/results_{cfg}_s*.json"))]
    if not rs:
        raise SystemExit(f"no results_{cfg}_s*.json in {RES}/")
    b = np.array([r["beta_pinn"] for r in rs]); e = np.array([r["beta_rel_err"] for r in rs])
    return dict(bt=rs[0]["beta_true"], med=float(np.median(b)), lo=b.min(), hi=b.max(), err=float(np.median(e)), noise=rs[0]["noise"])
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
blue, orange = "#2f6f9f", "#c9732e"
def exact_panel(a, cfgs, lim, title, ann_dx):
    D = [load(c) for c in cfgs]
    bt = np.array([d["bt"] for d in D]); bm = np.array([d["med"] for d in D])
    a.plot([0, lim], [0, lim], "-", c="0.6", lw=1, label="exact")
    a.errorbar(bt, bm, yerr=[bm - [d["lo"] for d in D], [d["hi"] for d in D] - bm], fmt="o", ms=4, mfc="white",
               mec=blue, ecolor=blue, capsize=2, label="PINN (median, range)")
    for d in D:
        a.annotate(f"{100 * d['err']:.1f}%", (d["bt"], d["med"]), xytext=(ann_dx, -10), textcoords="offset points", fontsize=7)
    a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_xlabel(r"true $\beta$"); a.set_ylabel(r"inferred $\beta$")
    a.set_title(title, loc="left", fontsize=9)
def noise_panel(a, lam, cfgs, emax, title):
    D = sorted([load(c) for c in cfgs], key=lambda d: d["noise"])
    eps = np.linspace(-emax, emax, 121)
    a.plot(100 * eps, exact_inverse_curve(lam, eps), "-", c="k", lw=1.1, label="exact solver")
    a.axhline(0.5, ls=":", c="0.6", lw=0.9)
    e_ = np.array([d["noise"] for d in D]); m_ = np.array([d["med"] for d in D])
    a.errorbar(100 * e_, m_, yerr=[m_ - [d["lo"] for d in D], [d["hi"] for d in D] - m_], fmt="o", ms=4, mfc="white",
               mec=orange, ecolor=orange, capsize=2, label="PINN (median, range)")
    a.set_xlabel(r"error in measured $U_T$ (%)"); a.set_ylabel(r"inferred $\beta$"); a.legend(frameon=False, fontsize=7.5)
    a.set_title(title, loc="left", fontsize=9)
exact_panel(ax[0, 0], ["IB025", "IB05", "IB10", "IB15"], 1.7, r"(a) $\lambda=2$, exact measurement", 6)
ax[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
noise_panel(ax[0, 1], 2, ["IN5m", "IN1m", "IB05", "IN1p", "IN5p"], 0.06, r"(b) $\lambda=2$, noisy measurement, true $\beta=0.5$")
exact_panel(ax[1, 0], ["L8B05", "L8B10", "L8B20", "L8B40"], 4.5, r"(c) $\lambda=8$, exact measurement", 6)
nf = load("L8B05nf")
ax[1, 0].errorbar([nf["bt"]], [nf["med"]], yerr=[[nf["med"] - nf["lo"]], [nf["hi"] - nf["med"]]], fmt="x", ms=6, c="#b03a2e",
                  capsize=2, label="without boundary-layer feature")
ax[1, 0].set_ylim(-0.2, 4.5); ax[1, 0].set_xlim(0, 4.5)
ax[1, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
noise_panel(ax[1, 1], 8, ["L8N1m", "L8N05m", "L8B05", "L8N05p", "L8N1p"], 0.012, r"(d) $\lambda=8$, noisy measurement, true $\beta=0.5$")
fig.tight_layout(); fig.subplots_adjust(hspace=0.42, wspace=0.28)
fig.savefig(f"{OUT}/fig_inverse.pdf"); fig.savefig(f"{OUT}/fig_inverse.png", dpi=200)
