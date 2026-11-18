import os
RES = os.environ.get("RESULTS_DIR", "results")   # folder with the results_*.json / atlas_*.json files
OUT = "figures"                                   # figures are written here; run from the repository root
import json, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
U = RES + "/"
ld = lambda f: json.load(open(U + f))
# ---------------- Figure: one-parameter sweep ----------------
V = [("PN", r"$q=0$", "#c9732e", "-"), ("PH", r"$q=1$ (adopted)", "#3a8a3a", "-"), ("PW", r"$q=2$", "#2f6f9f", "-"),
     ("PWu", r"$q=2$, uniform $\lambda$", "#999999", "--"), ("Q1", "four-parameter, $q=1$", "k", ":")]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for v, lab, c, ls in V:
    G = [ld(f"results_{v}_s{s}.json")["grid"] for s in range(3)]
    lam = np.array([r["lam"] for r in G[0]]); ex = np.array([r["exact"] for r in G[0]])
    Up = np.array([[r["pinn"] for r in g] for g in G]); E = np.array([[r["rel_err"] for r in g] for g in G])
    if v in ("PH", "Q1"):
        ax[0].plot(lam, np.median(Up, 0), "o" if v == "PH" else "s", ms=3, mfc="white", mec=c, mew=1.0, label=lab)
    ax[1].plot(lam, E.max(0), ls, c=c, lw=1.4 if ls == "-" else 1.2, label=lab)
ax[0].plot(lam, ex, "-", c="k", lw=1.0, label="exact", zorder=0)
ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\lambda$"); ax[0].set_ylabel(r"$U_T$"); ax[0].legend(frameon=False, fontsize=7.2)
ax[0].set_title("(a) Migration velocity", loc="left", fontsize=9)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_ylim(1e-4, 0.4)
ax[1].axvline(10, ls=":", c="gray", lw=0.8); ax[1].axhline(1e-2, ls=":", c="gray", lw=0.8)
ax[1].set_xlabel(r"$\lambda$"); ax[1].set_ylabel(r"relative error in $U_T$")
ax[1].legend(frameon=False, fontsize=6.8, loc="upper left", ncol=1); ax[1].set_title("(b) Error, worst of three runs", loc="left", fontsize=9)
fig.tight_layout(); fig.subplots_adjust(wspace=0.3); fig.savefig(f"{OUT}/fig_param_sweep.pdf"); fig.savefig(f"{OUT}/fig_param_sweep.png", dpi=200)
# atlas data (Q1, three seeds), used by the design-trend figure below
A = [ld(f"atlas_Q1_s{s}.json")["atlas"] for s in range(3)]
# ---------------- Figure: design trends ----------------
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
cols = ["#1b4f72", "#2f6f9f", "#5fa2c9", "#9ecae1"]
def med(sl, cond):
    out = {}
    for s in range(3):
        for r in A[s]:
            if r["slice"] == sl and cond(r):
                kk = (r["lam"], r["eta"], r["xi"], r["beta"]); out.setdefault(kk, []).append(r)
    return [(k_, np.median([x["pinn"] for x in v]), v[0]["exact"]) for k_, v in sorted(out.items())]
lam_all = sorted({r["lam"] for r in A[0] if r["slice"] == "lam-beta"})
for i, b in enumerate([0.0, 0.5, 1.0, 2.0]):
    pts = med("lam-beta", lambda r: abs(r["beta"] - b) < 1e-9)
    x = [p[0][0] for p in pts]; ax[0].plot(x, [p[2] for p in pts], "-", c=cols[i], lw=1.1)
    ax[0].plot(x, [p[1] for p in pts], "o", ms=3, mfc="white", mec=cols[i], mew=1.0, label=fr"$\beta={b:g}$")
ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\lambda$"); ax[0].set_ylabel(r"$U_T$"); ax[0].legend(frameon=False, fontsize=7)
ax[0].set_title(r"(a) slip", loc="left", fontsize=9)
for k, (sl, key, lab, logx) in enumerate([("lam-eta", "eta", r"$\eta$", True), ("lam-xi", "xi", r"$\xi$", False)], start=1):
    for i, L in enumerate([lam_all[2], lam_all[4], lam_all[6]]):
        pts = med(sl, lambda r: abs(r["lam"] - L) < 1e-9)
        idx = {"eta": 1, "xi": 2}[key]
        x = [p[0][idx] for p in pts]; ax[k].plot(x, [p[2] for p in pts], "-", c=cols[i], lw=1.1)
        ax[k].plot(x, [p[1] for p in pts], "o", ms=3, mfc="white", mec=cols[i], mew=1.0, label=fr"$\lambda={L:.2f}$")
    if logx:
        ax[k].set_xscale("log"); ax[k].set_xticks([0.25, 0.5, 1, 2, 4]); ax[k].set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        ax[k].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[k].set_xlabel(lab); ax[k].legend(frameon=False, fontsize=7)
    ax[k].set_title(["", r"(b) viscosity ratio", r"(c) stress jump"][k], loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_trends.pdf"); fig.savefig(f"{OUT}/fig_trends.png", dpi=200)
print("ok")
