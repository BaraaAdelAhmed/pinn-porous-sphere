import os
RES = os.environ.get("RESULTS_DIR", "results")   # folder with the results_*.json / atlas_*.json files
OUT = "figures"                                   # figures are written here; run from the repository root
import json, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
A = [json.load(open(f"{RES}/atlas_Q1_s{s}.json"))["atlas"] for s in range(3)]
fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.85], hspace=0.42)
gtop = gs[0].subgridspec(1, 3, wspace=0.6); tops = []
norm = LogNorm(1e-3, 3e-1)
for k, (sl, key, lab, logy) in enumerate([("lam-beta", "beta", r"$\beta$", False), ("lam-eta", "eta", r"$\eta$", True), ("lam-xi", "xi", r"$\xi$", False)]):
    a_ = fig.add_subplot(gtop[0, k]); tops.append(a_)
    R = [[r for r in a if r["slice"] == sl] for a in A]
    lam = sorted({r["lam"] for r in R[0]}); oth = sorted({r[key] for r in R[0]})
    E = np.full((3, len(oth), len(lam)), np.nan)
    for s in range(3):
        for r in R[s]:
            E[s, oth.index(r[key]), lam.index(r["lam"])] = r["rel_err"]
    with np.errstate(all="ignore"):
        M = np.nanmedian(E, 0)
    L = np.array(lam); O = np.array(oth)
    le = np.sqrt(L[:-1] * L[1:]); le = np.concatenate([[L[0]**2 / le[0]], le, [L[-1]**2 / le[-1]]])
    if logy:
        oe = np.sqrt(O[:-1] * O[1:]); oe = np.concatenate([[O[0]**2 / oe[0]], oe, [O[-1]**2 / oe[-1]]])
    else:
        d = np.diff(O)[0]; oe = np.concatenate([O - d / 2, [O[-1] + d / 2]])
    a_.set_facecolor("#dddddd")
    pc = a_.pcolormesh(le, oe, np.ma.masked_invalid(M), norm=norm, cmap="viridis", shading="flat")
    a_.set_xscale("log"); a_.set_xlabel(r"$\lambda$"); a_.set_ylabel(lab, labelpad=1)
    if logy:
        a_.set_yscale("log"); a_.set_yticks([0.25, 0.5, 1, 2, 4]); a_.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
        a_.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        lg = np.geomspace(0.4, 20, 100)
        for c_ in (4, 12):
            a_.plot(lg, (c_ / lg)**2, "w--", lw=0.9)
        a_.set_ylim(oe[0], oe[-1])
        a_.text(0.95, 0.34, r"$\lambda\sqrt{\eta}=4$", color="w", fontsize=6.5); a_.text(7.4, 0.34, r"$12$", color="w", fontsize=6.5)
    a_.set_title(f"({'abc'[k]}) " + r"$\lambda$" + "–" + lab, loc="left", fontsize=9)
cb = fig.colorbar(pc, ax=tops, pad=0.02, fraction=0.03); cb.set_label(r"relative error in $U_T$")
# (d) collapse onto a/sqrt(K)
d = fig.add_subplot(gs[1])
cols = {"lam-beta": "#c9732e", "lam-eta": "#2f6f9f", "lam-xi": "#3a8a3a"}
labs = {"lam-beta": r"$\lambda$–$\beta$ slice", "lam-eta": r"$\lambda$–$\eta$ slice", "lam-xi": r"$\lambda$–$\xi$ slice"}
allx, ally = [], []
for sl in cols:
    pts = {}
    for s in range(3):
        for r in A[s]:
            if r["slice"] == sl:
                pts.setdefault((r["lam"], r["eta"], r["xi"], r["beta"]), []).append(r["rel_err"])
                allx.append(r["lam"] * np.sqrt(r["eta"])); ally.append(r["rel_err"])
    x = [k_[0] * np.sqrt(k_[1]) for k_ in pts]; y = [np.median(v) for v in pts.values()]
    d.scatter(x, y, s=10, c=cols[sl], alpha=0.75, lw=0, label=labs[sl])
allx, ally = np.array(allx), np.array(ally)
edges = [0.25, 2, 4, 8, 12, 20, 40]
xm, ym = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (allx >= lo) & (allx < hi)
    if m.any(): xm += [lo, hi]; ym += [np.median(ally[m])] * 2
d.plot(xm, ym, "k-", lw=1.4, label="median in bins of $a/\\sqrt{K}$")
d.axhline(1e-2, ls=":", c="gray", lw=0.8)
d.set_xscale("log"); d.set_yscale("log"); d.set_xlabel(r"$a/\sqrt{K}=\lambda\sqrt{\eta}$"); d.set_ylabel(r"relative error in $U_T$")
d.legend(frameon=False, fontsize=7, loc="upper left"); d.set_title(r"(d) All slices against $a/\sqrt{K}$", loc="left", fontsize=9)
fig.savefig(f"{OUT}/fig_atlas.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_atlas.png", dpi=200, bbox_inches="tight")
print("bin medians:", [f"{v:.1e}" for v in ym[::2]])
