import os
RES = os.environ.get("RESULTS_DIR", "results")   # folder with the results_*.json / atlas_*.json files
OUT = "figures"                                   # figures are written here; run from the repository root
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
C_INT, C_EXT = "#c9732e", "#2f6f9f"
d = np.load(f"{RES}/fields_A10k_s1.npz")
ri, ro, zg = d["rg_in"], d["rg_out"], d["zg"]
zi = np.argmin(abs(zg))                                   # zeta = 0 (equator)
scale = max(abs(d["ex_p"]).max(), abs(d["ex_f"]).max())   # normalization for the error
ep = abs(d["Phip"] - d["ex_p"]) / scale
ef = abs(d["Phif"] - d["ex_f"]) / scale

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
# (a) amplitudes
a = ax[0]
a.plot(ri, d["ex_p"][zi], "-", c="k", lw=1.3, label="exact", zorder=1)
a.plot(ro, d["ex_f"][zi], "-", c="k", lw=1.3, zorder=1)
ki = np.unique([np.argmin(abs(ri - t)) for t in [0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.88, 0.95, 1.0]])
ko = np.unique(np.round(np.geomspace(1, len(ro) - 1, 14)).astype(int))
a.plot(ri[ki], d["Phip"][zi, ki], "o", ms=4.2, mfc="white", mec=C_INT, mew=1.2, label="PINN, interior", zorder=2)
a.plot(ro[ko], d["Phif"][zi, ko], "s", ms=4.0, mfc="white", mec=C_EXT, mew=1.2, label="PINN, exterior", zorder=2)
a.set_xscale("log"); a.axvline(1, ls=":", c="gray", lw=1)
a.set_xlabel(r"$r$"); a.set_ylabel(r"$\Phi = \psi/(1-\zeta^2)$")
a.legend(frameon=False, fontsize=8, loc="lower left")
a.set_title("(a) Stream-function amplitude", fontsize=9.5, loc="left")
# (b) error with band over all angles
b = ax[1]
for r, e, c, lab in [(ri, ep, C_INT, "interior"), (ro, ef, C_EXT, "exterior")]:
    b.plot(r, e[zi], "-", c=c, lw=1.3, label=f"{lab}, $\\zeta=0$")
    b.plot(r, e.max(0), "--", c=c, lw=1.0, label=f"{lab}, max over $\\zeta$")
b.set_xscale("log"); b.set_yscale("log"); b.axvline(1, ls=":", c="gray", lw=1)
b.set_xlabel(r"$r$"); b.set_ylabel(r"$|\Phi_{\rm PINN}-\Phi_{\rm exact}|\,/\,\max|\Phi_{\rm exact}|$")
b.set_ylim(1e-5, 1e-2)
b.legend(frameon=False, fontsize=7.5, loc="upper right")
b.set_title("(b) Error", fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_stage1_fields.pdf")
fig.savefig(f"{OUT}/fig_stage1_fields.png", dpi=220)
print("max normalized error: interior %.1e, exterior %.1e" % (ep.max(), ef.max()))
