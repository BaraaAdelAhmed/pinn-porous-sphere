"""Section 5.1: reference-free diagnostics. (a) force profiles of representative runs; (b) separation of all 75 runs."""
import os, json, glob, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
D = os.environ.get("RESULTS_DIR", "results") + "/"
OUT = "figures"
cat = {"A10k": "conv", "BL2": "conv", "BL12": "conv", "BL16": "conv", "RG8": "conv", "RG10": "conv", "RG12": "conv",
       "RG16": "repr", "RG32": "repr", "BL32": "repr", "MA8": "mimic", "MB8": "mimic", "RS12": "silent", "RS16": "silent", "RB16": "silent"}
style = {"conv": ("#3a8a3a", "o", "converged or degraded"), "mimic": ("#c9732e", "s", "Stokeslet mimicry"),
         "repr": ("#2f6f9f", "^", "representation failure"), "silent": ("#8e44ad", "D", "rescaled residual (silent failure)")}
runs = []
for f in sorted(glob.glob(D + "results_*.json")):
    c = re.match(r".*results_(.+)_s(\d)\.json$", f)
    if c is None or c.group(1) not in cat:
        continue                                   # only the 75 runs of Tables 3-5 enter this figure
    d = json.load(open(f))
    runs.append(dict(cfg=c.group(1), seed=int(c.group(2)), cat=cat[c.group(1)], err=d["U_rel_err"],
                     fp=d["force_profile"], fmax=max(abs(v) for v in d["force_profile"].values()), res=d["resid_interior"]))
def median_run(cfg):
    rr = sorted([r for r in runs if r["cfg"] == cfg], key=lambda r: r["err"]); return rr[len(rr) // 2]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
# (a) force profiles
for cfg, lab, ck in [("RG8", r"converged ($\lambda=8$)", "conv"), ("MA8", r"Stokeslet mimicry ($\lambda=8$)", "mimic"),
                     ("RG16", r"representation failure ($\lambda=16$)", "repr"), ("RS16", r"rescaled residual ($\lambda=16$)", "silent")]:
    r = median_run(cfg); Rs = [float(k.split("=")[1]) for k in r["fp"]]; v = [abs(x) for x in r["fp"].values()]
    col, mk, _ = style[ck]
    ax[0].plot(Rs, v, "-", c=col, marker=mk, ms=4, mfc="white", lw=1.2, label=lab)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel(r"$R$"); ax[0].set_ylabel(r"$|F(R)|/(6\pi|U|)$")
ax[0].set_ylim(1e-6, 1e2); ax[0].legend(frameon=False, fontsize=6.8, loc="center right")
ax[0].set_title("(a) Force transmitted across spheres", loc="left", fontsize=9)
# (b) separation of all runs
for ck in ["conv", "mimic", "repr", "silent"]:
    rr = [r for r in runs if r["cat"] == ck]; col, mk, lab = style[ck]
    ax[1].scatter([r["fmax"] for r in rr], [r["res"] for r in rr], s=18, marker=mk, facecolors="white", edgecolors=col, lw=1.0, label=f"{lab} ({len(rr)})")
ax[1].axvline(0.15, ls="--", c="gray", lw=0.8); ax[1].axhline(1e-3, ls="--", c="gray", lw=0.8)
co = min([r for r in runs if r["cat"] == "repr"], key=lambda r: r["err"])
ax[1].annotate(r"$\lambda=32$: $e_U=0.9\%$", (co["fmax"], co["res"]), xytext=(2e-3, 1.0e-6), fontsize=6.8,
               arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"))
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel(r"force profile, $\max_R|F(R)|/(6\pi|U|)$")
ax[1].set_ylabel(r"interior residual $\delta_p$"); ax[1].set_ylim(3e-7, 2e0)
ax[1].legend(frameon=True, framealpha=0.95, edgecolor="0.8", fontsize=6.3, loc="upper right", handletextpad=0.3, borderpad=0.4)
ax[1].set_title("(b) All 75 runs", loc="left", fontsize=9)
fig.tight_layout(); fig.subplots_adjust(wspace=0.32)
fig.savefig(f"{OUT}/fig_diagnostics.pdf"); fig.savefig(f"{OUT}/fig_diagnostics.png", dpi=200)
for cfg in ["RG8", "MA8", "RG16", "RS16"]:
    r = median_run(cfg); print(cfg, r["seed"], {k: f"{v:.2e}" for k, v in r["fp"].items()})
