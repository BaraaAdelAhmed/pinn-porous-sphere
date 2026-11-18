import os
RES = os.environ.get("RESULTS_DIR", "results")   # folder with the results_*.json / atlas_*.json files
OUT = "figures"                                   # figures are written here; run from the repository root
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Arc

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5})
C_INT, C_EXT, C_IF, C_INK = "#c9732e", "#2f6f9f", "#7a2e8f", "#222222"
FILL_INT, FILL_EXT = "#f6e3d3", "#dbe9f4"

fig = plt.figure(figsize=(7.2, 7.4))

# ------------------------------------------------------------------ (a) physical problem
ax = fig.add_axes([0.0, 0.51, 0.47, 0.44]); ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-1.9, 2.75); ax.set_ylim(-2.05, 2.1)
ax.add_patch(Rectangle((-1.9, -2.0), 4.65, 4.05, fc=FILL_EXT, ec="none", zorder=0))
ax.add_patch(Circle((0, 0), 1, fc=FILL_INT, ec=C_IF, lw=2, zorder=1))
ax.plot([0, 0], [-1.95, 1.85], ls=(0, (4, 3)), c=C_INK, lw=0.9, zorder=2)
ax.text(0.07, 1.72, "axis", fontsize=7.3)
th = np.deg2rad(40)
ax.plot([0, np.sin(th)], [0, np.cos(th)], c=C_INK, lw=0.9, zorder=3)
ax.add_patch(Arc((0, 0), 0.7, 0.7, theta1=90 - 40, theta2=90, lw=0.9, zorder=3))
ax.text(0.12, 0.42, r"$\theta$", fontsize=10, zorder=3)
ax.text(np.sin(th) * 0.55 + 0.04, np.cos(th) * 0.55 - 0.16, r"$r$", fontsize=10, zorder=3)
ax.annotate("", xy=(-1.45, 1.35), xytext=(-1.45, 0.35), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_INK))
ax.text(-1.38, 0.85, r"$U$", fontsize=11)
ax.annotate("", xy=(2.35, 1.6), xytext=(2.35, 0.6), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#b03030"))
ax.text(1.95, 1.68, r"$\nabla T_\infty$", fontsize=10, color="#b03030")
ax.text(0, -0.28, "porous (Brinkman)", bbox=dict(fc=FILL_INT, ec="none", pad=1.2), ha="center", fontsize=7.5, color=C_INT, zorder=3)
ax.text(0, -0.58, r"$E^2(E^2-\lambda^2)\psi_p=0$", bbox=dict(fc=FILL_INT, ec="none", pad=1.2), ha="center", fontsize=8.5, zorder=3)
ax.text(0.45, -1.62, "fluid (Stokes)", bbox=dict(fc=FILL_EXT, ec="none", pad=1.2), ha="center", fontsize=7.5, color=C_EXT)
ax.text(0.45, -1.88, r"$E^4\psi_f=0$,  $\mathbf{u}\to 0$ as $r\to\infty$", bbox=dict(fc=FILL_EXT, ec="none", pad=1.2), ha="center", fontsize=8.5)
ax.annotate("interface $r=1$:\nBeavers–Joseph slip,\nstress jump, Marangoni",
            xy=(np.cos(np.deg2rad(-35)), np.sin(np.deg2rad(-35))), xytext=(1.12, -0.98),
            fontsize=7.2, color=C_IF, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=C_IF, lw=0.8))
fig.text(0.02, 0.985, "(a) Physical problem", fontsize=9.5, fontweight="bold", va="top")

# ------------------------------------------------------------------ (b) computational domain
bx = fig.add_axes([0.53, 0.585, 0.45, 0.33])
bx.set_xlim(0, 2); bx.set_ylim(-1, 1)
bx.add_patch(Rectangle((0, -1), 1, 2, fc=FILL_INT, ec=C_INT, lw=1.2))
bx.add_patch(Rectangle((1, -1), 1, 2, fc=FILL_EXT, ec=C_EXT, lw=1.2))
rng = np.random.default_rng(3)
ri = rng.uniform(0.02, 1, 170); zi = rng.uniform(-1, 1, 170)
bx.scatter(ri, zi, s=2.5, c=C_INT, lw=0)
se = rng.uniform(1e-3, 1, 75); ze = rng.uniform(-1, 1, 75)
bx.scatter(2 - se, ze, s=2.5, c=C_EXT, lw=0)            # plotted with s decreasing to the right
zg = np.cos(np.linspace(0, np.pi, 17))
bx.scatter(np.ones_like(zg), zg, s=11, c=C_IF, zorder=5, lw=0)
bx.axvline(1, c=C_IF, lw=2, zorder=4)
bx.set_xticks([0, 0.5, 1, 1.5, 2]); bx.set_xticklabels(["0", "", "1", "", "0"])
bx.set_yticks([-1, 0, 1]); bx.set_ylabel(r"$\zeta=\cos\theta$", labelpad=1)
bx.tick_params(labelsize=7.5, length=2.5)
bx.text(0.5, -1.17, r"$r$", ha="center", va="top", fontsize=10, color=C_INT, transform=bx.transData)
bx.text(1.5, -1.17, r"$s=1/r$", ha="center", va="top", fontsize=10, color=C_EXT, transform=bx.transData)
bx.text(0.5, 1.07, "interior: $N_p$ points", ha="center", fontsize=7.5, color=C_INT)
bx.text(1.5, 1.07, "exterior: $N_f$ points", ha="center", fontsize=7.5, color=C_EXT)
bx.text(0.03, -0.93, "centre", fontsize=7, va="bottom")
bx.text(1.97, -0.93, r"$r=\infty$", fontsize=7, va="bottom", ha="right")
bx.text(1.0, -1.37, "interface ($N_\\Gamma$ points)", ha="center", va="top", fontsize=7.3, color=C_IF)
fig.text(0.47, 0.985, "(b) Computational domains", fontsize=9.5, fontweight="bold", va="top")

# ------------------------------------------------------------------ (c) pipeline
cx = fig.add_axes([0.0, 0.0, 1.0, 0.47]); cx.axis("off"); cx.set_xlim(0, 10); cx.set_ylim(0, 6.2)
cx.text(0.15, 6.05, "(c) Representation and loss", fontsize=9.5, fontweight="bold", va="top")

def box(x, y, w, h, txt, ec, fc="white", fs=8.3, lw=1.1, ha="center", style="round,pad=0.06"):
    cx.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, ec=ec, fc=fc, lw=lw))
    tx = x + w / 2 if ha == "center" else x + 0.12
    cx.text(tx, y + h / 2, txt, ha=ha, va="center", fontsize=fs, linespacing=1.45)

def arrow(x0, y0, x1, y1, c=C_INK, ls="-", rad=0.0):
    cx.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 color=c, lw=1.0, ls=ls, connectionstyle=f"arc3,rad={rad}"))

# networks
box(0.25, 4.25, 1.55, 0.75, "network\n$M_p(r,\\zeta)$", C_INT, FILL_INT)
box(0.25, 1.75, 1.55, 0.75, "network\n$M_f(s,\\zeta)$", C_EXT, FILL_EXT)
box(0.45, 3.05, 1.15, 0.6, "$U$  or  $C$", C_IF, "#efe2f3")
cx.text(1.03, 2.92, "free / held", ha="center", va="top", fontsize=7)
# representations
box(2.4, 4.15, 3.1, 0.95, "$\\Phi_p = r^2\\left[-\\frac{1}{2}U+(1-r)\\,M_p\\right]$", C_INT, fs=8.8)
box(2.4, 1.65, 3.1, 0.95, "$\\Phi_f=-\\frac{1}{2}Us+C\\,(s^{-1}-s)+s(1-s)\\,M_f$", C_EXT, fs=8.3)
arrow(1.83, 4.62, 2.37, 4.62); arrow(1.83, 2.12, 2.37, 2.12)
arrow(1.62, 3.45, 2.37, 4.35, c=C_IF); arrow(1.62, 3.25, 2.37, 1.95, c=C_IF)
box(2.75, 3.1, 2.4, 0.55, "$\\psi_i=(1-\\zeta^2)\\,\\Phi_i$", C_INK, fs=8.8)
arrow(3.95, 4.12, 3.95, 3.70); arrow(3.95, 2.63, 3.95, 3.06)
# loss
box(6.05, 2.15, 3.75, 2.55,
    "loss (Sec. 3.5)\n"
    "$\\mathcal{R}_p=r^4E^2(E^2-\\lambda^2)\\,\\psi_p$\n"
    "$\\mathcal{R}_f=s^{-2}\\mathcal{L}_s(s^2\\mathcal{L}_s\\psi_f)$\n"
    "$\\mathcal{B}_{\\rm slip},\\ \\mathcal{B}_{\\rm stress}$  at $r=1$\n"
    "$\\mathcal{L}=\\langle\\mathcal{R}_p^2\\rangle+\\langle\\mathcal{R}_f^2\\rangle"
    "+w\\langle\\mathcal{B}_{\\rm slip}^2+\\mathcal{B}_{\\rm stress}^2\\rangle$",
    C_INK, "#f4f4f4", fs=8.2)
arrow(5.20, 3.38, 6.00, 3.38)
# optimizer feedback
cx.add_patch(FancyArrowPatch((7.9, 2.10), (1.05, 1.68), arrowstyle="-|>", mutation_scale=9, color="#666666",
                             lw=0.9, ls=(0, (4, 2)), connectionstyle="arc3,rad=-0.22"))
cx.text(4.6, 0.42, "L-BFGS updates $M_p$, $M_f$ and $U$ (or $C$)", ha="center", fontsize=7.6, color="#555555")
# built-in box
box(6.05, 4.9, 3.75, 1.1,
    "exact by construction (Secs. 3.1–3.4):\n"
    "axis $\\psi=0$,  regular centre,  normal velocity at $r=1$,\n"
    "decay as $r\\to\\infty$,  force-free ($C=0$, free particle)",
    "#4a8a4a", "#e6f2e6", fs=7.4)
fig.savefig(f"{OUT}/fig_pinn_schematic.pdf")
fig.savefig(f"{OUT}/fig_pinn_schematic.png", dpi=220)
