import os
RES = os.environ.get("RESULTS_DIR", "results")   # folder with the results_*.json / atlas_*.json files
OUT = "figures"                                   # figures are written here; run from the repository root
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
sys.argv = ['x', '--inverse_beta', '--mode', 'selftest']
src = open('pinn_stage1_jax.py').read().split('# =============================================================================\n# 6. Collocation and loss')[0]
exec(src.replace('if args.mode == "selftest":', 'if False:'))
LS = LAM * np.sqrt(ETA)
# results of scripts/run_inverse.sh (median, min and max over seeds)
import glob, json
def load(cfg):
    rs = [json.load(open(f))["inverse"] for f in sorted(glob.glob(f"{RES}/results_{cfg}_s*.json"))]
    if not rs:
        raise SystemExit(f"no results_{cfg}_s*.json in {RES}/")
    b = np.array([r["beta_pinn"] for r in rs]); e = np.array([r["beta_rel_err"] for r in rs])
    return rs[0]["beta_true"], float(np.median(b)), b.min(), b.max(), float(np.median(e)), rs[0]["noise"]
ex = [load(c) for c in ["IB025", "IB05", "IB10", "IB15"]]
exactdata = [x[:4] for x in ex]; med_err = [x[4] for x in ex]
noisy = sorted([(x[5],) + x[1:4] for x in [load(c) for c in ["IN5m", "IN1m", "IB05", "IN1p", "IN5p"]]])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
bt = np.array([d[0] for d in exactdata]); bm = np.array([d[1] for d in exactdata])
ax[0].plot([0, 2], [0, 2], "-", c="0.6", lw=1, label="exact")
ax[0].errorbar(bt, bm, yerr=[bm - [d[2] for d in exactdata], [d[3] for d in exactdata] - bm], fmt="o", ms=4,
               mfc="white", mec="#2f6f9f", ecolor="#2f6f9f", capsize=2, label="PINN (median, range)")
ax[0].set_xlabel(r"true $\beta$"); ax[0].set_ylabel(r"inferred $\beta$"); ax[0].set_xlim(0, 1.7); ax[0].set_ylim(0, 1.7)
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left"); ax[0].set_title("(a) Exact measurement", loc="left", fontsize=9)
for (b, m, lo, hi), e in zip(exactdata, med_err):
    ax[0].annotate(f"{100*e:.1f}%", (b, m), xytext=(6, -10), textcoords="offset points", fontsize=7)
eps = np.linspace(-0.06, 0.06, 121)
ideal = [exact_inverse_slip(EX["U"] * (1 + e)) * LS for e in eps]
ax[1].plot(100 * eps, ideal, "-", c="k", lw=1.1, label="exact solver")
ax[1].axhline(0.5, ls=":", c="0.6", lw=0.9); ax[1].text(-5.8, 0.505, r"true $\beta$", fontsize=7, color="0.4")
e_ = np.array([d[0] for d in noisy]); m_ = np.array([d[1] for d in noisy])
ax[1].errorbar(100 * e_, m_, yerr=[m_ - [d[2] for d in noisy], [d[3] for d in noisy] - m_], fmt="o", ms=4,
               mfc="white", mec="#c9732e", ecolor="#c9732e", capsize=2, label="PINN (median, range)")
ax[1].set_xlabel(r"error in measured $U_T$ (%)"); ax[1].set_ylabel(r"inferred $\beta$")
ax[1].legend(frameon=False, fontsize=7.5); ax[1].set_title(r"(b) Noisy measurement, true $\beta=0.5$", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_inverse.pdf"); fig.savefig(f"{OUT}/fig_inverse.png", dpi=200)
# sensitivity at beta = 0.5
h = 1e-4; l0 = 0.25; s_ = (np.log(U_of_slip(l0 * (1 + h))) - np.log(U_of_slip(l0 * (1 - h)))) / (2 * h)
print(f"d ln U_T / d ln beta at beta=0.5: {s_:.3f}  ->  |d ln beta / d ln U_T| = {1/abs(s_):.2f}")
