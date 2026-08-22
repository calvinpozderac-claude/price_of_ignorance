"""Render every figure from the cached datasets.  Run:  python experiments/plots.py

Assumes ``experiments/compute.py`` has already populated ``results/``.
Figures are written to ``figures/``.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poi import pigou_curves  # noqa: E402
from poi.metrics import efficiency, saturation_alpha  # noqa: E402
from poi.sweep import load  # noqa: E402
from style import (  # noqa: E402
    AQUA, AXIS, BLUE, DIVERGING, GRID, INK, INK_2, MUTED, ORANGE, RED, SEQ_BLUE,
    BLUE_RAMP, ORANGE_RAMP, SEQ_ORANGE, SERIES, SURFACE, VIOLET, band, direct_label, mean_sem,
    nanmean, ordinal, pc_line, use_style,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def _save(fig, name):
    path = os.path.join(FIGURES, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


# ---------------------------------------------------------------- figure 1 --
def fig_pigou():
    """Exact two-road result: the whole story in closed form."""
    a = np.linspace(0, 1, 501)
    cur = pigou_curves(a)
    C, CA, CS = cur["total_cost"], cur["cost_altruist"], cur["cost_selfish"]
    astar = cur["alpha_star"]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))

    ax = axes[0]
    ax.fill_between(a, C, 0.75, color=BLUE, alpha=0.12, lw=0)
    ax.plot(a, C, color=BLUE, zorder=3)
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.axhline(0.75, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.axvline(astar, color=AXIS, lw=0.9, ls=(0, (2, 2)))
    ax.text(0.02, 1.003, r"$C_{\rm eq}=1$", color=INK_2, fontsize=8, va="bottom")
    ax.text(0.02, 0.753, r"$C_{\rm opt}=3/4$", color=INK_2, fontsize=8, va="bottom")
    ax.text(astar + 0.02, 1.03, r"$\alpha^*=1/2$", color=INK_2, fontsize=8, va="top")
    ax.annotate(
        r"$C = 1-\alpha+\alpha^2$", xy=(0.25, 0.8125), xytext=(0.40, 0.94),
        color=BLUE, fontsize=8.5, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8, shrinkA=2, shrinkB=3),
    )
    ax.set_xlabel(r"altruistic fraction $\alpha$")
    ax.set_ylabel("average commute time $C$")
    ax.set_title("(a)  Altruism helps \u2014 then stops helping", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.70, 1.06)

    ax = axes[1]
    ax.plot(a, CA, color=ORANGE, zorder=3)
    ax.plot(a, CS, color=AQUA, zorder=3)
    ax.plot(a, C, color=BLUE, lw=1.4, ls=(0, (5, 2)), zorder=2)
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    direct_label(ax, 0.80, np.interp(0.80, a, CA), "altruists", ORANGE, dy=9)
    direct_label(ax, 0.80, np.interp(0.80, a, CS), "selfish", AQUA, dy=-9)
    direct_label(ax, 0.34, np.interp(0.34, a, C), "everyone", BLUE, dy=-10)
    ax.annotate(
        "altruists stay pinned at\nthe full-anarchy cost",
        xy=(0.36, 1.0), xytext=(0.46, 0.575), color=RED, fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=RED, lw=0.9, shrinkA=2, shrinkB=3),
    )
    ax.set_xlabel(r"altruistic fraction $\alpha$")
    ax.set_ylabel("commute time experienced")
    ax.set_title("(b)  ...and only the selfish collect", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.40, 1.12)

    ax = axes[2]
    ax.plot(a, CA - CS, color=ORANGE, zorder=3)
    ax.plot(a, cur["efficiency"], color=BLUE, zorder=3)
    ax.axvline(astar, color=AXIS, lw=0.9, ls=(0, (2, 2)))
    direct_label(ax, 0.60, np.interp(0.60, a, cur["efficiency"]), "gain captured", BLUE, dy=-10)
    direct_label(ax, 0.66, np.interp(0.66, a, CA - CS), r"gap  $C_A-C_S$", ORANGE, dy=9)
    ax.plot([astar], [0.5], "o", color=ORANGE, ms=5, zorder=4)
    ax.set_xlabel(r"altruistic fraction $\alpha$")
    ax.set_ylabel("fraction / commute time")
    ax.set_title(r"(c)  Exploitation peaks exactly at $\alpha^*$", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.12)

    handles = [
        plt.Line2D([], [], color=BLUE, lw=1.8),
        plt.Line2D([], [], color=ORANGE, lw=1.8),
        plt.Line2D([], [], color=AQUA, lw=1.8),
    ]
    fig.legend(
        handles, ["everyone (total)", "altruistic drivers", "selfish drivers"],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        "Pigou's two roads, solved exactly for a mixed population",
        x=0.008, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig1_pigou_exact.png")


# ---------------------------------------------------------------- figure 2 --
def fig_paper_poa():
    """Reproduction of the paper's Fig. 3: POA(p) peaking at the percolation threshold."""
    d = load(os.path.join(RESULTS, "paper_poa.npz"))
    p, Ls, pc = d["p"], d["Ls"], float(d["pc"])
    colors = ordinal(len(Ls))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))

    ax = axes[0]
    peaks = []
    for L, c in zip(Ls, colors):
        poa = d[f"poa_L{L}"]
        m, s = mean_sem(poa, axis=1)
        ax.plot(p, m, color=c, label=f"L = {L}")
        band(ax, p, m - s, m + s, c)
        peaks.append(p[np.nanargmax(m)])
    # The four peaks nearly coincide, so direct labels would collide; the legend
    # plus the ordered light-to-dark ramp (small to large L) carries identity.
    pc_line(ax, pc)
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel("price of anarchy, $P$")
    ax.set_title("(a)  The paper's central result, reproduced", loc="left")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(Ls, peaks, "o-", color=BLUE)
    ax.axhline(pc, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(Ls[0], pc, r" $p_c = 0.6447$", color=INK_2, fontsize=8, va="bottom")
    ax.set_xlabel("system size, $L$")
    ax.set_ylabel("location of the POA peak")
    ax.set_title("(b)  The peak converges to $p_c$", loc="left")
    ax.set_ylim(0.54, 0.68)
    ax.set_xticks(Ls)
    ax.text(
        Ls[0], 0.548, f"grid resolution {p[1] - p[0]:.3f}",
        color=MUTED, fontsize=7.5, va="bottom",
    )

    fig.suptitle(
        "Reproducing Skinner (2014) with the mixed-population solver at $\\alpha \\in \\{0, 1\\}$",
        x=0.008, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig2_paper_poa.png")


# ---------------------------------------------------------------- figure 3 --
def fig_paper_scaling():
    """Reproduction of Fig. 7: C ~ L below pc, L^m at pc, L^0 above."""
    d = load(os.path.join(RESULTS, "paper_scaling.npz"))
    L, ps, pc = d["L"], d["p"], float(d["pc"])
    m_pred = 1.097 / (1.733 + 1.097)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    labels = [r"$p = 0.30 < p_c$", r"$p = p_c$", r"$p = 0.80 > p_c$"]
    for i, (lab, c) in enumerate(zip(labels, SERIES)):
        ax.loglog(L, d["Ceq"][i], "o-", color=c, label=lab)
        ax.loglog(L, d["Copt"][i], "o--", color=c, mfc=SURFACE, lw=1.2, ms=3.5)

    ref = np.array([L[0], L[-1]], dtype=float)
    for expo, y0, txt, va in [
        (1.0, d["Ceq"][0][0], r"$C \propto L^{1}$", "bottom"),
        (m_pred, d["Ceq"][1][0], rf"$C \propto L^{{{m_pred:.3f}}}$", "bottom"),
        (0.0, d["Ceq"][2][0], r"$C \propto L^{0}$", "top"),
    ]:
        ax.loglog(ref, y0 * (ref / ref[0]) ** expo, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=0)
        ax.text(
            ref[-1] * 1.06, y0 * (ref[-1] / ref[0]) ** expo, txt,
            color=MUTED, fontsize=7.5, va=va,
        )

    fits = [np.polyfit(np.log(L), np.log(d["Ceq"][i]), 1)[0] for i in range(3)]
    handles = [plt.Line2D([], [], color=c, marker="o", ms=4) for c in SERIES]
    handles += [
        plt.Line2D([], [], color=INK_2, lw=1.6, label="_"),
        plt.Line2D([], [], color=INK_2, lw=1.2, ls="--", mfc=SURFACE, label="_"),
    ]
    ax.legend(
        handles,
        labels + ["selfish equilibrium", "social optimum"],
        loc="upper left", ncol=1,
    )
    ax.set_xlabel("system size, $L$")
    ax.set_ylabel("average commute time, $C$")
    ax.set_title(
        "Commute time scaling  "
        + rf"(fitted slopes: {fits[0]:.2f}, {fits[1]:.2f}, {fits[2]:.2f})",
        loc="left",
    )
    ax.set_xlim(4, 145)
    fig.tight_layout()
    _save(fig, "fig3_paper_scaling.png")
    return fits


# ---------------------------------------------------------------- figure 4 --
def fig_altruism_curves():
    """The headline answer, disorder-averaged, below / at / above the threshold."""
    d = load(os.path.join(RESULTS, "altruism_fine.npz"))
    al, ps, pc = d["alpha"], d["p"], float(d["pc"])
    labels = [r"$p = 0.45$  (below $p_c$)", r"$p = p_c = 0.645$", r"$p = 0.85$  (above $p_c$)"]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

    # (a) total cost, normalised so every p starts at its own POA
    ax = axes[0]
    for i, (lab, c) in enumerate(zip(labels, SERIES)):
        rel = d["C"][i] / d["C"][i][:, -1][:, None]
        m, s = mean_sem(rel, axis=0)
        ax.plot(al, m, color=c, label=lab)
        band(ax, al, m - s, m + s, c)
        lx = [0.68, 0.68, 0.40][i]
        direct_label(ax, lx, np.interp(lx, al, m), lab.split("  ")[0], c, dy=[8, -8, 8][i])
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(0.02, 1.0, "social optimum", color=INK_2, fontsize=8, va="bottom")
    ax.set_xlabel(r"altruistic fraction $\alpha$")
    ax.set_ylabel(r"$C(\alpha)\,/\,C_{\rm opt}$")
    ax.set_title("(a)  Total commute time does fall", loc="left")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right")

    # (b) who experiences what, at pc
    ax = axes[1]
    i = 1
    for key, c, lab in [("C", BLUE, "everyone"), ("CA", ORANGE, "altruists"), ("CS", AQUA, "selfish")]:
        m, s = mean_sem(d[key][i], axis=0)
        ax.plot(al, m, color=c, label=lab)
        band(ax, al, m - s, m + s, c)
    C0 = np.nanmean(d["C"][i][:, 0])
    ax.axhline(C0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    mA, _ = mean_sem(d["CA"][i], axis=0)
    ax.fill_between(al, C0, mA, where=mA >= C0, color=RED, alpha=0.13, lw=0)
    ax.text(0.30, C0 * 1.004, "worse than full anarchy", color=RED, fontsize=7.5, va="bottom")
    direct_label(ax, 0.78, np.interp(0.78, al, mA), "altruists", ORANGE)
    mS, _ = mean_sem(d["CS"][i], axis=0)
    direct_label(ax, 0.78, np.interp(0.78, al, mS), "selfish", AQUA)
    mC, _ = mean_sem(d["C"][i], axis=0)
    direct_label(ax, 0.45, np.interp(0.45, al, mC), "everyone", BLUE, dy=-9)
    ax.set_xlabel(r"altruistic fraction $\alpha$")
    ax.set_ylabel("commute time experienced")
    ax.set_title(r"(b)  ...but not for the altruists (at $p_c$)", loc="left")
    ax.set_xlim(0, 1)

    # (c) share of the achievable gain that has actually been captured
    ax = axes[2]
    for i, (lab, c) in enumerate(zip(labels, SERIES)):
        eff = np.array([efficiency(row) for row in d["C"][i]])
        m, s = mean_sem(eff, axis=0)
        ax.plot(al, m, color=c)
        band(ax, al, m - s, m + s, c)
        lx = [0.30, 0.44, 0.55][i]
        direct_label(ax, lx, np.interp(lx, al, m), lab.split("  ")[0], c, dy=[-9, 9, 8][i])
    ax.plot(al, al, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=0)
    ax.text(0.80, 0.63, "proportional\nreturn", color=MUTED, fontsize=7.5, rotation=30, ha="center")
    ax.set_xlabel(r"altruistic fraction $\alpha$")
    ax.set_ylabel(r"share of $C_{\rm eq}-C_{\rm opt}$ captured")
    ax.set_title("(c)  Early altruists do the heavy lifting", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Does adding altruistic drivers speed things up? Yes — for everyone except the altruists",
        x=0.008, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig4_altruism_curves.png")


# ---------------------------------------------------------------- figure 5 --
def fig_heatmaps():
    """The (p, alpha) plane: where altruism pays, and where it is exploited."""
    d = load(os.path.join(RESULTS, "altruism_grid.npz"))
    p, al, pc = d["p"], d["alpha"], float(d["pc"])
    ext = [al[0], al[-1], p[0], p[-1]]

    eff = np.nanmean(
        np.array([[efficiency(d["C"][i, j]) for j in range(d["C"].shape[1])] for i in range(p.size)]),
        axis=1,
    )
    gap = nanmean(d["CA"] - d["CS"], axis=1)
    regret = nanmean(d["CA"] - d["C"][:, :, [0]], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

    ax = axes[0]
    im = ax.imshow(eff, origin="lower", aspect="auto", extent=ext, cmap=SEQ_BLUE, vmin=0, vmax=1)
    cs = ax.contour(al, p, eff, levels=[0.5, 0.9], colors="white", linewidths=1.0)
    ax.clabel(cs, fmt={0.5: "50%", 0.9: "90%"}, fontsize=7, inline_spacing=2)
    fig.colorbar(im, ax=ax, label="share of achievable gain")
    ax.set_title("(a)  Gain captured", loc="left")

    ax = axes[1]
    im = ax.imshow(gap, origin="lower", aspect="auto", extent=ext, cmap=SEQ_ORANGE)
    fig.colorbar(im, ax=ax, label=r"$C_A - C_S$")
    ax.set_title("(b)  How much altruists overpay", loc="left")

    ax = axes[2]
    v = np.nanmax(np.abs(regret))
    im = ax.imshow(regret, origin="lower", aspect="auto", extent=ext, cmap=DIVERGING, vmin=-v, vmax=v)
    ax.contour(al, p, regret, levels=[0.0], colors=[INK], linewidths=1.2)
    fig.colorbar(im, ax=ax, label=r"$C_A(\alpha) - C_{\rm eq}$")
    ax.set_title("(c)  Altruists vs. full anarchy", loc="left")
    ax.text(
        0.24, 0.06, "worse off", transform=ax.transAxes, fontsize=8,
        color="white", va="bottom", ha="center", fontweight="bold",
    )
    ax.text(
        0.90, 0.06, "better off", transform=ax.transAxes, fontsize=8,
        color="white", va="bottom", ha="center", fontweight="bold",
    )
    ax.text(
        0.665, 0.955, "break-even", transform=ax.transAxes, fontsize=7.5,
        color=INK, va="top", ha="right",
    )

    for ax in axes:
        ax.axhline(pc, color="white", lw=1.0, ls=(0, (4, 3)))
        ax.text(0.012, pc, r" $p_c$", color="white", fontsize=8, va="bottom", fontweight="bold")
        ax.set_xlabel(r"altruistic fraction $\alpha$")
        ax.grid(False)
    axes[0].set_ylabel("fraction of congestible roads, $p$")

    fig.suptitle(
        f"The $(p,\\alpha)$ plane, {int(d['n_seeds'])} disorder realisations per point, $L={int(d['L'])}$",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.925,
        "The gains arrive early and saturate (a), the altruists always pay for them (b), "
        "and over most of the plane they pay more than full anarchy would have cost them (c).",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig5_heatmaps.png")


# ---------------------------------------------------------------- figure 6 --
def fig_saturation():
    """What the percolation threshold does to the *distributional* quantities.

    The price of anarchy peaks at ``pc`` (the paper's result).  The altruists'
    penalty does not: measured in absolute time it peaks far *below* ``pc``,
    simply because commutes are far longer there.  Normalised by the cost scale
    it instead plateaus and then collapses just above ``pc`` -- so the threshold
    marks the *edge* of the regime where altruism matters at all, not a peak.
    """
    d = load(os.path.join(RESULTS, "altruism_grid.npz"))
    p, al, pc = d["p"], d["alpha"], float(d["pc"])

    astar = np.array(
        [[saturation_alpha(al, d["C"][i, j]) for j in range(d["C"].shape[1])] for i in range(p.size)]
    )
    maxgap = np.nanmax(d["CA"] - d["CS"], axis=2)
    rel_gap = maxgap / d["C"][:, :, -1]
    poa = d["summary_poa"]

    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.6))

    ax = axes[0]
    m, s = mean_sem(poa, axis=1)
    ax.plot(p, m, color=BLUE)
    band(ax, p, m - s, m + s, BLUE)
    pc_line(ax, pc)
    ax.plot([p[np.nanargmax(m)]], [np.nanmax(m)], "o", color=BLUE, ms=5, zorder=4)
    ax.set_ylabel(r"price of anarchy $C_{\rm eq}/C_{\rm opt}$")
    ax.set_title("(a)  Inefficiency peaks near $p_c$", loc="left")

    ax = axes[1]
    m, s = mean_sem(maxgap, axis=1)
    ax.plot(p, m, color=ORANGE)
    band(ax, p, m - s, m + s, ORANGE)
    pc_line(ax, pc)
    ax.plot([p[np.nanargmax(m)]], [np.nanmax(m)], "o", color=ORANGE, ms=5, zorder=4)
    ax.annotate(
        "peaks well below $p_c$ —\ncommutes are simply\nlonger down here",
        xy=(p[np.nanargmax(m)], np.nanmax(m)), xytext=(0.44, 0.70),
        textcoords="axes fraction", color=INK_2, fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9, shrinkA=2, shrinkB=4),
    )
    ax.set_ylabel(r"peak gap $C_A - C_S$  (absolute)")
    ax.set_title("(b)  Absolute penalty is misleading", loc="left")

    ax = axes[2]
    m, s = mean_sem(rel_gap, axis=1)
    ax.plot(p, m, color=AQUA)
    band(ax, p, m - s, m + s, AQUA)
    pc_line(ax, pc)
    ax.set_ylabel(r"peak $(C_A - C_S)\,/\,C_{\rm opt}$")
    ax.set_title("(c)  Relative penalty dies just above $p_c$", loc="left")
    ax.set_ylim(0, None)

    ax = axes[3]
    m, s = mean_sem(astar, axis=1)
    ax.plot(p, m, color=VIOLET)
    band(ax, p, m - s, m + s, VIOLET)
    pc_line(ax, pc)
    ax.set_ylabel(r"saturating fraction $\alpha^*$")
    ax.set_title(r"(d)  ...and so does the need for altruists", loc="left")
    ax.set_ylim(0, 1)

    for ax in axes:
        ax.set_xlabel("fraction of congestible roads, $p$")
        ax.axvspan(pc, 1.0, color=GRID, alpha=0.5, lw=0, zorder=0)

    fig.suptitle(
        "The percolation threshold bounds the regime where altruism matters at all",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.935,
        "Above $p_c$ (shaded) there are many parallel congestible routes, the selfish equilibrium is already "
        "optimal, and altruism is neither useful nor costly.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    _save(fig, "fig6_saturation.png")


# ---------------------------------------------------------------- figure 7 --
def fig_backfire():
    """Where converting a selfish driver into an altruist makes the average worse."""
    d = load(os.path.join(RESULTS, "backfire.npz"))
    p, al, pc = d["p"], d["alpha"], float(d["pc"])
    C = d["C"]

    frac = np.mean(d["summary_max_backfire"] > 1e-7, axis=1)
    size = np.nanmean(d["summary_max_backfire"] / d["C"][:, :, -1], axis=1)

    # Where in alpha does the backfire happen?
    rises = np.diff(C, axis=2)
    rise_map = np.nanmean(np.maximum(rises, 0.0) / C[:, :, [-1]], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))

    ax = axes[0]
    ax.plot(p, 100 * frac, color=BLUE)
    pc_line(ax, pc)
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel("% of road networks")
    ax.set_title("(a)  Networks where altruism can backfire", loc="left")

    ax = axes[1]
    ax.plot(p, 100 * size, color=ORANGE)
    pc_line(ax, pc)
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel(r"mean worst rise, % of $C_{\rm opt}$")
    ax.set_title("(b)  ...and by how much", loc="left")

    ax = axes[2]
    mid = 0.5 * (al[1:] + al[:-1])
    im = ax.imshow(
        100 * rise_map, origin="lower", aspect="auto",
        extent=[mid[0], mid[-1], p[0], p[-1]], cmap=SEQ_ORANGE,
    )
    fig.colorbar(im, ax=ax, label=r"mean rise in $C$, % of $C_{\rm opt}$")
    ax.axhline(pc, color="white", lw=1.0, ls=(0, (4, 3)))
    ax.text(mid[0], pc, r" $p_c$", color="white", fontsize=8, va="bottom", fontweight="bold")
    ax.set_xlabel(r"altruistic fraction $\alpha$")
    ax.set_ylabel("fraction of congestible roads, $p$")
    ax.set_title("(c)  Where in the sweep it happens", loc="left")
    ax.grid(False)

    fig.suptitle(
        "Adding altruists is not always an improvement",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.925,
        f"Real but small: {100 * np.mean(d['summary_max_backfire'] > 1e-7):.0f}% of networks show some "
        f"range of $\\alpha$ over which the average commute rises, by at most "
        f"{100 * np.nanmax(d['summary_max_backfire'] / d['C'][:, :, -1]):.2f}% of $C_{{\\rm opt}}$ on any single "
        f"network. Like everything else, it vanishes above $p_c$.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig7_backfire.png")


# ---------------------------------------------------------------- figure 8 --
def fig_traffic_maps():
    """Spatial flow fields: where each class actually drives."""
    from matplotlib.collections import LineCollection

    d = load(os.path.join(RESULTS, "traffic_maps.npz"))
    pos, tail, head = d["pos"], d["tail"], d["head"]
    n = int(d["n_lattice_edges"])
    L = int(d["L"])

    segs = np.stack([pos[tail[:n]], pos[head[:n]]], axis=1)
    # Drop wrap-around segments so the periodic edge does not draw across the figure
    keep = np.abs(segs[:, 0, 1] - segs[:, 1, 1]) < 2.0
    cong = d["congestible"][:n]

    panels = [
        ("x_0.0", r"all selfish  ($\alpha=0$)", SEQ_BLUE, "solo"),
        ("x_1.0", r"all altruistic  ($\alpha=1$)", SEQ_BLUE, "solo"),
        ("fS_0.5", r"the selfish half  ($\alpha=0.5$)", SEQ_BLUE, "pair"),
        ("fA_0.5", r"the altruistic half  ($\alpha=0.5$)", SEQ_ORANGE, "pair"),
    ]
    # The two alpha = 0.5 panels share a normalisation so the classes can be
    # compared directly; the two single-class panels use their own maxima.
    pair_max = max(np.nanmax(d["fS_0.5"][:n]), np.nanmax(d["fA_0.5"][:n]))

    def concentration(f):
        """Roads carrying any traffic, and the share borne by the busiest tenth."""
        f = np.asarray(f, dtype=float)
        used = int((f > 1e-9).sum())
        top = np.sort(f)[::-1][: max(1, int(0.1 * f.size))].sum() / max(f.sum(), 1e-12)
        return used, top

    uS, tS = concentration(d["fS_0.5"][:n])
    uA, tA = concentration(d["fA_0.5"][:n])

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 5.4))
    for ax, (key, title, cmap, norm) in zip(axes.ravel(), panels):
        f = d[key][:n]
        ref = pair_max if norm == "pair" else np.nanmax(f)
        scale = np.clip(f / max(ref, 1e-12), 0, 1)
        order = np.argsort(scale[keep])
        lc = LineCollection(
            segs[keep][order],
            linewidths=0.15 + 2.4 * scale[keep][order],
            colors=cmap(0.10 + 0.90 * scale[keep][order]),
        )
        ax.add_collection(lc)
        ax.set_xlim(0, 2 * L)
        ax.set_ylim(0, L)
        ax.set_box_aspect(0.5)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(AXIS)

    for ax in axes[:, 0]:
        ax.set_ylabel("flow: left to right", color=INK_2, fontsize=8)
    fig.suptitle(
        f"Where each class actually drives, at $p \\approx p_c$  ($L={L}$)",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.935,
        "Line width and colour show road usage. Both classes travel almost entirely on congestible roads; what "
        "differs is how they spread out.\n"
        f"At $\\alpha=0.5$ the selfish squeeze onto {uS} roads with their busiest tenth carrying {100*tS:.0f}% of "
        f"their travel; the altruists use {uA} roads and only {100*tA:.0f}%.",
        ha="left", fontsize=8.5, color=INK_2, va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), h_pad=2.2)
    _save(fig, "fig8_traffic_maps.png")


ALL = {
    "pigou": fig_pigou,
    "paper_poa": fig_paper_poa,
    "paper_scaling": fig_paper_scaling,
    "altruism": fig_altruism_curves,
    "heatmaps": fig_heatmaps,
    "saturation": fig_saturation,
    "backfire": fig_backfire,
    "maps": fig_traffic_maps,
}



# ==================== user-ignorance extension (arXiv:2503.09684) ==========
# omega = ignorance level (that paper's alpha); alpha = altruistic fraction.

OMEGA_STAR = 2.0 / 3.0


def gamma_star(omega):
    """Uniform altruism that exactly offsets ignorance ``omega``."""
    return (2 - 3 * np.asarray(omega, dtype=float)) / (2 - np.asarray(omega, dtype=float))


# ---------------------------------------------------------------- figure 9 --
def fig_ignorance_surface():
    """Reproduction of the ignorance paper: the price of ignorance over (p, omega)."""
    d = load(os.path.join(RESULTS, "ignorance_surface.npz"))
    p, w, pc = d["p"], d["omega"], float(d["pc"])
    C = d["C"][:, :, :, 0]
    PI = nanmean(C / C[:, [0], :], axis=2)  # normalise each network to its own omega=0

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0))
    axes = axes.ravel()

    ax = axes[0]
    v = 0.05
    im = ax.imshow(
        PI.T, origin="lower", aspect="auto", extent=[p[0], p[-1], w[0], w[-1]],
        cmap=DIVERGING, vmin=1 - v, vmax=1 + v,
    )
    cs = ax.contour(p, w, PI.T, levels=[1.0], colors=[INK], linewidths=1.4)
    fig.colorbar(im, ax=ax, label="price of ignorance $P_I$")
    ax.axvline(pc, color="white", lw=1.0, ls=(0, (4, 3)))
    ax.axhline(OMEGA_STAR, color="white", lw=1.0, ls=(0, (2, 2)))
    ax.text(pc, 0.02, r" $p_c$", color="white", fontsize=8, fontweight="bold")
    ax.text(0.30, OMEGA_STAR, r"$\omega=2/3$", color="white", fontsize=8,
            va="bottom", fontweight="bold")
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel(r"ignorance $\omega$")
    ax.set_title("(a)  Blue = ignorance helps", loc="left")
    ax.grid(False)

    ax = axes[1]
    for i, (om, c) in enumerate(zip([0.25, OMEGA_STAR, 0.9], SERIES)):
        j = int(np.argmin(abs(w - om)))
        ax.plot(p, PI[:, j], color=c)
        direct_label(ax, 0.30, PI[int(np.argmin(abs(p - 0.30))), j],
                     rf"$\omega={w[j]:.2f}$", c, dy=[8, -9, 8][i])
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    pc_line(ax, pc)
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel("price of ignorance $P_I$")
    ax.set_title(r"(b)  Deepest benefit sits at $p_c$", loc="left")

    ax = axes[2]
    j = int(np.argmin(abs(p - pc)))
    ax.plot(w, PI[j], color=BLUE)
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.axvline(OMEGA_STAR, color=AXIS, lw=0.9, ls=(0, (2, 2)))
    ax.plot([w[np.argmin(PI[j])]], [PI[j].min()], "o", color=BLUE, ms=5, zorder=4)
    ax.annotate(
        rf"min $P_I={PI[j].min():.3f}$" "\n" rf"at $\omega={w[np.argmin(PI[j])]:.2f}$",
        xy=(w[np.argmin(PI[j])], PI[j].min()), xytext=(0.30, 0.55),
        textcoords="axes fraction", color=INK_2, fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9, shrinkA=2, shrinkB=4),
    )
    ax.set_xlabel(r"ignorance $\omega$")
    ax.set_ylabel("price of ignorance $P_I$")
    ax.set_title(r"(c)  A cut at $p=p_c$", loc="left")
    # Total ignorance is catastrophic, which would flatten the dip off-scale.
    ax.set_ylim(0.94, 1.09)
    ax.text(
        0.99, 1.085, f"$P_I\\to{PI[j][-1]:.1f}$\nat $\\omega=1$ (off scale)",
        color=RED, fontsize=7.5, ha="right", va="top",
    )

    ax = axes[3]
    dz = load(os.path.join(RESULTS, "ignorance_sizes.npz"))
    pz = dz["p"]
    for L, c in zip(dz["Ls"], ordinal(len(dz["Ls"]))):
        m, se = mean_sem(dz[f"PI_L{L}"], axis=1)
        ax.plot(pz, m, color=c, label=f"L = {L}")
        band(ax, pz, m - se, m + se, c)
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    pc_line(ax, pc)
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel("price of ignorance $P_I$")
    ax.set_title(r"(d)  At $\omega=2/3$ the dip narrows with $L$", loc="left")
    ax.legend(loc="lower left", ncol=2)

    fig.suptitle(
        "Reproducing the benefit of ignorance (arXiv:2503.09684)",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.955,
        r"Selfish drivers who cannot tell fast roads from slow ones stop piling onto the fast ones; below $\omega=2/3$ that helps at every $p$.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig9_ignorance_surface.png")


# --------------------------------------------------------------- figure 10 --
def fig_joint_surface():
    """The new result: altruism reverses sign once drivers are ignorant enough."""
    d = load(os.path.join(RESULTS, "joint_surface.npz"))
    p, w, al, pc = d["p"], d["omega"], d["alpha"], float(d["pc"])
    C = nanmean(d["C"], axis=2)  # (p, omega, alpha)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    titles = [r"(a)  $p=0.45$ (below $p_c$)", r"(b)  $p=p_c$", r"(c)  $p=0.85$ (above $p_c$)"]

    for i, (ax, title) in enumerate(zip(axes, titles)):
        # Percent above the cheapest point of this plane. Total ignorance is
        # catastrophic and would flatten everything else, so the scale is capped
        # at the 88th percentile (note the extended colour bar).
        rel = 100.0 * (C[i] / C[i].min() - 1.0)
        top = float(np.nanpercentile(rel, 88)) or 1.0
        im = ax.imshow(
            rel.T, origin="lower", aspect="auto",
            extent=[w[0], w[-1], al[0], al[-1]],
            cmap=SEQ_ORANGE, vmin=0.0, vmax=top,
        )
        fig.colorbar(im, ax=ax, label="% slower than the best mix", extend="max")
        # The empirical ridge: the best altruistic fraction at each ignorance level.
        ridge = al[np.argmin(rel, axis=1)]
        ax.plot(w, ridge, color=INK, lw=1.2, marker="o", ms=2.6,
                markerfacecolor=SURFACE, markeredgewidth=0.7, zorder=4)
        jmin = np.unravel_index(np.argmin(rel), rel.shape)
        ax.plot([w[jmin[0]]], [al[jmin[1]]], "*", color="white", ms=15,
                markeredgecolor=INK, markeredgewidth=0.9, zorder=6)
        ax.axvline(OMEGA_STAR, color="white", lw=1.1, ls=(0, (3, 2)))
        ax.set_xlabel(r"ignorance $\omega$")
        ax.set_title(title, loc="left")
        ax.set_xlim(w[0], w[-1])
        ax.set_ylim(0, 1)
        ax.grid(False)

    axes[0].set_ylabel(r"altruistic fraction $\alpha$")
    axes[0].text(
        0.03, 0.06, "line: best $\\alpha$ at each $\\omega$\n$\\star$: cheapest mix overall",
        transform=axes[0].transAxes, fontsize=7.5, color=INK, va="bottom",
    )
    axes[1].text(
        OMEGA_STAR - 0.02, 0.5, r"$\omega=2/3$", color="white", fontsize=8,
        rotation=90, ha="right", va="center", fontweight="bold",
    )

    fig.suptitle(
        "Altruism and ignorance are substitutes: using both at once overshoots",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.93,
        "Darker is slower. The best altruistic fraction falls to zero as ignorance rises: by "
        r"$\omega \approx 2/3$ the ignorant selfish are already doing the job, and altruists only add error.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig10_joint_surface.png")


# --------------------------------------------------------------- figure 11 --
def fig_altruism_reversal():
    """Where altruism flips sign, and why the two thresholds differ.

    Two different questions have two different answers, and conflating them is
    easy.  For a *uniform* population at altruism level gamma, the first
    increment stops helping exactly where gamma*(omega) = 0, i.e. omega = 2/3.
    For a *fraction* of fully-altruistic drivers there is no such guarantee, and
    the flip happens much earlier, near omega ~ 0.45.
    """
    dj = load(os.path.join(RESULTS, "joint_surface.npz"))
    ds = load(os.path.join(RESULTS, "sign_boundary.npz"))
    w, al = dj["omega"], dj["alpha"]
    C = nanmean(dj["C"], axis=2)
    ip = int(np.argmin(abs(dj["p"] - float(dj["pc"]))))

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))

    # (a) the cost curves themselves
    ax = axes[0]
    picks = [0.0, 0.35, 2 / 3, 0.85]
    cols = ordinal(len(picks), ramp=ORANGE_RAMP)
    for om, c in zip(picks, cols):
        j = int(np.argmin(abs(w - om)))
        ax.plot(al, C[ip, j], color=c, label=rf"$\omega={w[j]:.2f}$")
    ax.set_xlabel(r"altruistic fraction $\alpha$")
    ax.set_ylabel("true average commute time $C$")
    ax.set_title(r"(a)  Adding altruists, at $p=p_c$", loc="left")
    ax.legend(loc="center right", title="ignorance", fontsize=7.5, title_fontsize=8)
    jd = int(np.argmin(abs(w - 0.85)))
    ax.annotate(
        "heavy ignorance:\naltruists make it worse",
        xy=(0.62, C[ip, jd][int(0.62 * (al.size - 1))]), xytext=(0.05, 0.97),
        textcoords="axes fraction", color=RED, fontsize=7.5, va="top",
        arrowprops=dict(arrowstyle="->", color=RED, lw=0.9, shrinkA=2, shrinkB=5),
    )
    ax.annotate(
        "accurate knowledge:\nthey still help",
        xy=(0.72, C[ip, 0][int(0.72 * (al.size - 1))]), xytext=(0.05, 0.16),
        textcoords="axes fraction", color=INK_2, fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9, shrinkA=2, shrinkB=5),
    )

    # (b) the two instruments have two different thresholds
    ax = axes[1]
    try:
        di = load(os.path.join(RESULTS, "instruments.npz"))
        wi, pi_ = di["omega"], di["p"]
        k = int(np.argmin(abs(pi_ - float(di["pc"]))))
        for arr, c, lab in [
            (di["dU"][k], BLUE, r"uniform $\gamma$ (everyone mildly altruistic)"),
            (di["dF"][k], ORANGE, r"fraction $\alpha$ (a few full altruists)"),
        ]:
            y = arr / di["C0"][k]
            ax.plot(wi, 100 * y, color=c, label=lab)
            f = np.flatnonzero(y > 0)
            if f.size and f[0] > 0:
                xc = np.interp(0, [y[f[0] - 1], y[f[0]]], [wi[f[0] - 1], wi[f[0]]])
                ax.plot([xc], [0], "o", color=c, ms=6, zorder=5)
                ax.annotate(
                    rf"$\omega_c={xc:.2f}$", (xc, 0), textcoords="offset points",
                    xytext=(0, -16 if c is ORANGE else 12), color=c, fontsize=8,
                    fontweight="bold", ha="center",
                )
        ax.axvline(2 / 3, color=AXIS, lw=0.9, ls=(0, (2, 2)))
        ax.text(2 / 3, 0.03, r" $\omega=2/3$", transform=ax.get_xaxis_transform(),
                color=INK_2, fontsize=8, va="bottom")
        ax.axhline(0.0, color=INK, lw=1.0)
        ax.legend(loc="upper left")
        ax.set_ylim(-0.6, 1.4)
    except FileNotFoundError:
        ax.text(0.5, 0.5, "run compute.py instruments", ha="center", transform=ax.transAxes)
    ax.set_xlabel(r"ignorance $\omega$")
    ax.set_ylabel("% change in $C$ from a small dose")
    ax.set_title("(b)  How you deliver altruism matters", loc="left")

    # (c) sign boundary across the (p, omega) plane
    ax = axes[2]
    ps, ws = ds["p"], ds["omega"]
    Cs = nanmean(ds["C"], axis=2)
    delta = Cs[:, :, -1] - Cs[:, :, 0]  # C(alpha=1) - C(alpha=0)
    rel = 100 * delta / Cs[:, :, 0]
    v = float(np.nanpercentile(np.abs(rel), 97))
    im = ax.imshow(
        rel.T, origin="lower", aspect="auto",
        extent=[ps[0], ps[-1], ws[0], ws[-1]], cmap=DIVERGING,
        vmin=-v, vmax=v,
    )
    ax.contour(ps, ws, delta.T, levels=[0.0], colors=[INK], linewidths=1.6)
    fig.colorbar(im, ax=ax, label=r"% change in $C$, $\alpha:0\to1$", extend="both")
    ax.axvline(float(ds["pc"]), color="white", lw=1.0, ls=(0, (4, 3)))
    ax.text(0.07, 0.10, "altruism\nhelps", color=INK, fontsize=8.5, fontweight="bold")
    ax.text(0.55, 0.76, "altruism\nhurts", color="white", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel(r"ignorance $\omega$")
    ax.set_title("(c)  Where altruism backfires", loc="left")
    ax.grid(False)

    fig.suptitle(
        r"Once drivers are ignorant enough, recruiting altruists makes traffic worse",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.93,
        "Ignorance already does the job altruism was doing, so altruists over-correct. Where the flip happens "
        r"depends on how altruism is delivered: $\omega=2/3$ if spread thinly over everyone, but only "
        r"$\omega \approx 0.4$–$0.5$ if concentrated in a few drivers.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, "fig11_altruism_reversal.png")


# --------------------------------------------------------------- figure 12 --
def fig_substitution():
    """The closed-form trade-off between altruism and ignorance."""
    d = load(os.path.join(RESULTS, "substitution.npz"))
    p, w, g, C, Copt = d["p"], d["omega"], d["gamma"], d["C"], d["Copt"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    ax = axes[0]
    ww = np.linspace(0, 0.95, 200)
    ax.plot(ww, gamma_star(ww), color=INK, lw=1.6, ls=(0, (5, 2)), zorder=1,
            label=r"$\gamma^*=(2-3\omega)/(2-\omega)$")
    cols = ordinal(len(p))
    for i, c in enumerate(cols):
        meas = g[np.argmin(C[i], axis=1)]
        ax.plot(w, meas, "o", color=c, ms=4.5, label=f"$p={p[i]:.2f}$", zorder=3)
    ax.axhline(0.0, color=MUTED, lw=0.9)
    ax.axvline(OMEGA_STAR, color=AXIS, lw=0.9, ls=(0, (2, 2)))
    ax.text(OMEGA_STAR, 0.03, r" $\omega=2/3$", color=INK_2, fontsize=8,
            transform=ax.get_xaxis_transform(), va="bottom")
    ax.text(0.03, -0.62, "past $\\omega=2/3$ the optimal\n$\\gamma$ is negative: you would\nneed spite, not altruism",
            color=RED, fontsize=7.5, va="top")
    ax.set_xlabel(r"ignorance $\omega$")
    ax.set_ylabel(r"cost-minimising altruism level $\gamma$")
    ax.set_title("(a)  Measured optimum lands on the predicted curve", loc="left")
    ax.legend(loc="upper right")

    ax = axes[1]
    i = int(np.argmin(abs(p - float(d["pc"]))))
    for om, c in zip([0.0, 1 / 3, OMEGA_STAR, 0.9], ordinal(4, ramp=ORANGE_RAMP)):
        j = int(np.argmin(abs(w - om)))
        ax.plot(g, C[i, j] / Copt[i], color=c, label=rf"$\omega={w[j]:.2f}$")
        ax.plot([gamma_star(w[j])], [np.interp(gamma_star(w[j]), g, C[i, j] / Copt[i])],
                "o", color=c, ms=6, zorder=5, markeredgecolor=INK, markeredgewidth=0.7)
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(g[-1], 1.002, "true social optimum ", color=INK_2, fontsize=8,
            va="bottom", ha="right")
    ax.set_xlabel(r"uniform altruism level $\gamma$")
    ax.set_ylabel(r"$C\,/\,C_{\rm opt}$")
    ax.set_title(r"(b)  Each curve bottoms out at its $\gamma^*$ (dots)", loc="left")
    ax.legend(loc="upper center", ncol=2)
    ax.set_ylim(0.993, None)

    fig.suptitle(
        "One knob, not two: ignorance and altruism trade off along a straight-line law",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.925,
        r"Rescaling the perceived objective shows the optimum is reached whenever "
        r"$(1+\gamma)(1-\omega/2) = 2(1-\omega)$, i.e. $\gamma^*=(2-3\omega)/(2-\omega)$.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig12_substitution.png")


ALL.update({
    "ignorance": fig_ignorance_surface,
    "joint": fig_joint_surface,
    "reversal": fig_altruism_reversal,
    "substitution": fig_substitution,
})


# ============ real road networks and the equal-path-length assumption ========

def fig_real_networks():
    """Does any of this survive on an actual road network?"""
    d = load(os.path.join(RESULTS, "real_networks.npz"))
    names = [str(x) for x in d["names"]]
    w, al, C = d["omega"], d["alpha"], d["C"]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))

    ax = axes[0]
    for i, (n, c) in enumerate(zip(names, SERIES)):
        pi = C[i, :, 0] / C[i, 0, 0]
        ax.plot(w, pi, color=c, label=n.replace("-", " "))
        direct_label(ax, 0.62, np.interp(0.62, w, pi), n.split("-")[0], c, dy=[8, 8, -9][i])
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(0.02, 1.005, "ignorance helps below this line", color=INK_2, fontsize=7.5, va="bottom")
    ax.set_xlabel(r"ignorance $\omega$")
    ax.set_ylabel(r"$C(\omega)\,/\,C(0)$, selfish drivers")
    ax.set_title("(a)  On real roads, ignorance hurts", loc="left")
    ax.legend(loc="upper left")

    ax = axes[1]
    for i, (n, c) in enumerate(zip(names, SERIES)):
        eff = 100 * (C[i, :, -1] / C[i, :, 0] - 1)
        ax.plot(w, eff, color=c)
        lx = [0.55, 0.42, 0.30][i]
        direct_label(ax, lx, np.interp(lx, w, eff), n.split("-")[0], c, dy=[-10, 10, 10][i])
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.set_xlabel(r"ignorance $\omega$")
    ax.set_ylabel(r"% change in $C$, $\alpha: 0 \to 1$")
    ax.set_title("(b)  ...and altruism keeps helping", loc="left")
    ax.text(0.03, 0.06, "below zero = altruism helps", transform=ax.transAxes,
            fontsize=7.5, color=INK_2, va="bottom")

    ax = axes[2]
    i = 0
    rel = 100 * (C[i] / C[i].min() - 1)
    im = ax.imshow(rel.T, origin="lower", aspect="auto",
                   extent=[w[0], w[-1], al[0], al[-1]], cmap=SEQ_ORANGE,
                   vmin=0, vmax=float(np.nanpercentile(rel, 92)))
    fig.colorbar(im, ax=ax, label="% slower than the best mix", extend="max")
    ridge = al[np.argmin(rel, axis=1)]
    ax.plot(w, ridge, color=INK, lw=1.2, marker="o", ms=2.6,
            markerfacecolor=SURFACE, markeredgewidth=0.7)
    ax.set_xlabel(r"ignorance $\omega$")
    ax.set_ylabel(r"altruistic fraction $\alpha$")
    ax.set_title("(c)  " + names[i] + r": best $\alpha$ stays at 1", loc="left")
    ax.grid(False)

    fig.suptitle(
        "The lattice's reversal does not survive on real road networks",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.93,
        "TNTP benchmark networks with BPR costs and real OD matrices, solved by Frank-Wolfe. Ignorance is "
        "harmful here rather than helpful, so altruists repair damage instead of over-correcting.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig13_real_networks.png")


def fig_irregular():
    """The single assumption that decides the answer: equal path lengths."""
    d = load(os.path.join(RESULTS, "irregular.npz"))
    p, w, al, C = d["p"], d["omega"], d["alpha"], d["C"]
    lat = load(os.path.join(RESULTS, "ignorance_surface.npz"))

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))

    ax = axes[0]
    lp, lw_ = lat["p"], lat["omega"]
    LC = lat["C"][:, :, :, 0]
    LPI = nanmean(LC / LC[:, [0], :], axis=2)
    jl = int(np.argmin(abs(lw_ - 2 / 3)))
    ax.plot(lp, LPI[:, jl], color=BLUE, label="lattice (equal path lengths)")
    jd = int(np.argmin(abs(w - 2 / 3)))
    ax.plot(p, C[:, jd, 0] / C[:, 0, 0], color=ORANGE, label="skip-DAG (unequal)")
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel(r"$P_I$ at $\omega = 2/3$")
    ax.set_title("(a)  Universal benefit is lattice-only", loc="left")
    ax.legend(loc="upper left")

    ax = axes[1]
    rel = 100 * (C[:, :, -1] / C[:, :, 0] - 1)
    v = float(np.nanpercentile(np.abs(rel), 97))
    im = ax.imshow(rel.T, origin="lower", aspect="auto",
                   extent=[p[0], p[-1], w[0], w[-1]], cmap=DIVERGING, vmin=-v, vmax=v)
    ax.contour(p, w, rel.T, levels=[0.0], colors=[INK], linewidths=1.5)
    fig.colorbar(im, ax=ax, label=r"% change in $C$, $\alpha: 0 \to 1$", extend="both")
    ax.text(0.19, 0.10, "altruism\nhelps", color=INK, fontsize=8.5, fontweight="bold")
    ax.text(0.28, 0.72, "hurts", color="white", fontsize=8.5, fontweight="bold")
    ax.text(0.63, 0.42, "helps\nagain", color="white", fontsize=8, fontweight="bold")
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel(r"ignorance $\omega$")
    ax.set_title("(b)  A third regime appears", loc="left")
    ax.grid(False)

    ax = axes[2]
    for pp, c in zip([0.3, 0.6, 0.9], SERIES):
        k = int(np.argmin(abs(p - pp)))
        pi = C[k, :, 0] / C[k, 0, 0]
        eff = 100 * (C[k, :, -1] / C[k, :, 0] - 1)
        ax.plot(pi, eff, color=c, marker="o", ms=3)
        direct_label(ax, pi[-1], eff[-1], f"$p={p[k]:.2f}$", c, dy=-9)
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.axvline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_xlabel(r"price of ignorance $P_I$   ($>1$: ignorance hurts)")
    ax.set_ylabel(r"% change in $C$, $\alpha: 0 \to 1$")
    ax.set_title("(c)  Backfire only where ignorance helps", loc="left")

    fig.suptitle(
        "Equal path lengths is the assumption that decides the answer",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.93,
        "Same model, same cost pair, same unit demand -- only the path-length structure differs. Altruism "
        "over-corrects exactly where ignorance is already correcting, and repairs damage where it is not.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig14_irregular.png")


ALL.update({"real": fig_real_networks, "irregular": fig_irregular})


# ================= idiosyncratic uncertainty: (sigma, rho) ==================

def fig_stochastic():
    """Correlation, not magnitude, decides whether uncertainty helps."""
    d = load(os.path.join(RESULTS, "stochastic_lattice.npz"))
    sg, rh, al, C = d["sigma"], d["rho"], d["alpha"], d["C"]
    base = C[0, 0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))

    ax = axes[0]
    for j, c in zip(range(rh.size), ordinal(rh.size)):
        y = 100 * (C[:, j, 0] / base - 1)
        ax.plot(sg, y, color=c, label=rf"$\rho={rh[j]:.2f}$")
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.set_xlabel(r"error magnitude $\sigma$")
    ax.set_ylabel(r"% change in $C$ vs no error")
    ax.set_title("(a)  Same error, opposite sign", loc="left")
    ax.legend(loc="upper left", title="correlation", fontsize=7.5, title_fontsize=8)
    # The runaway damage at large sigma buries the beneficial window, so zoom it.
    ins = ax.inset_axes([0.44, 0.34, 0.53, 0.44])
    ins.set_facecolor(SURFACE)
    ins.patch.set_alpha(1.0)
    ins.set_zorder(5)
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_color(AXIS)
    keep = sg <= 0.26
    for j, c in zip(range(rh.size), ordinal(rh.size)):
        ins.plot(sg[keep], 100 * (C[keep, j, 0] / base - 1), color=c)
    ins.axhline(0.0, color=INK, lw=0.9)
    ins.set_ylim(-2.2, 3.0)
    ins.tick_params(labelsize=6.5)
    ins.set_title("zoom: small $\\sigma$", fontsize=7.5, color=INK_2, loc="left", pad=2)
    ins.grid(True, color=GRID, lw=0.5)
    ins.text(0.03, 0.06, "helps", transform=ins.transAxes, fontsize=6.5, color=INK_2)
    ins.set_xlim(0, 0.26)

    ax = axes[1]
    k = load(os.path.join(RESULTS, "stochastic_convergence.npz"))
    for key, c, lab in [("rho0", BLUE, r"$\rho = 0$"), ("rho05", ORANGE, r"$\rho = 0.5$")]:
        ax.semilogx(k["K"], 100 * (k[key] - 1), color=c, marker="o", ms=3.5, label=lab)
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.axhline(100 * (k["dial"].min() - 1), color=AQUA, lw=1.4, ls=(0, (5, 2)))
    ax.text(1.15, 100 * (k["dial"].min() - 1), "logit continuum (Dial)",
            color=AQUA, fontsize=7.5, va="bottom", ha="left")
    ax.set_xlabel("number of driver types $K$")
    ax.set_ylabel(r"% change in $C$ vs no error")
    ax.set_title(r"(b)  Too few types looks like $\rho>0$", loc="left")
    ax.legend(loc="upper right")

    ax = axes[2]
    for j, lab, c in [(0, r"$\rho=0$ (independent)", BLUE), (rh.size - 1, r"$\rho=1$ (shared)", ORANGE)]:
        eff = 100 * (C[:, j, -1] / C[:, j, 0] - 1)
        ax.plot(sg, eff, color=c, label=lab)
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.set_xlabel(r"error magnitude $\sigma$")
    ax.set_ylabel(r"% change in $C$, $\alpha: 0 \to 1$")
    ax.set_title("(c)  Does altruism still help?", loc="left")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.06, "below zero = altruism helps", transform=ax.transAxes,
            fontsize=7.5, color=INK_2, ha="right", va="bottom")

    fig.suptitle(
        "Whether ignorance helps depends on whether drivers are wrong together",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.93,
        r"$\varepsilon^k_e = \sigma(\sqrt{\rho}\,\xi_e + \sqrt{1-\rho}\,\eta^k_e)$: magnitude and correlation move "
        r"independently. Lattice, $p=p_c$, $L=" + f"{int(d['L'])}$, {int(d['n_seeds'])} networks.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig15_stochastic.png")


def fig_stochastic_transfer():
    """Does the independent-error benefit survive off the lattice?"""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    ax = axes[0]
    d = load(os.path.join(RESULTS, "stochastic_irregular.npz"))
    sg, ps, C = d["sigma"], d["p"], d["C"]
    for i, c in zip(range(ps.size), SERIES):
        ax.plot(sg, 100 * (C[i, :, 0] - 1), color=c, label=f"$p={ps[i]:.1f}$")
        ax.plot(sg, 100 * (C[i, :, 1] - 1), color=c, ls=(0, (4, 2)), lw=1.3)
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.set_xlabel(r"error magnitude $\sigma$")
    ax.set_ylabel(r"% change in $C$ vs no error")
    ax.set_title("(a)  Unequal path lengths", loc="left")
    h = [plt.Line2D([], [], color=INK_2, lw=1.6), plt.Line2D([], [], color=INK_2, lw=1.3, ls="--")]
    ax.legend(ax.get_lines()[::2][:ps.size] + h,
              [f"$p={v:.1f}$" for v in ps] + [r"$\rho=0$", r"$\rho=1$"],
              loc="upper left", ncol=2, fontsize=7.5)

    ax = axes[1]
    r = load(os.path.join(RESULTS, "stochastic_real.npz"))
    names = [str(x) for x in r["names"]]
    sg2, C2 = r["sigma"], r["C"]
    for i, (n, c) in enumerate(zip(names, SERIES)):
        b = C2[i, 0, 0, 0]
        ax.plot(sg2, 100 * (C2[i, :, 0, 0] / b - 1), color=c, label=n.split("-")[0])
        ax.plot(sg2, 100 * (C2[i, :, 1, 0] / b - 1), color=c, ls=(0, (4, 2)), lw=1.3)
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.set_xlabel(r"error magnitude $\sigma$ (coeff. of variation)")
    ax.set_ylabel(r"% change in $C$ vs no error")
    ax.set_title("(b)  Real road networks", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.text(0.97, 0.06, "solid $\\rho=0$,  dashed $\\rho=1$", transform=ax.transAxes,
            fontsize=7.5, color=INK_2, ha="right", va="bottom")

    fig.suptitle(
        "Shared error does the greater damage at realistic magnitudes",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.93,
        "Solid: independent drivers. Dashed: everyone wrong the same way, at identical magnitude. The ordering "
        r"reverses only at extreme $\sigma$, where independent drivers scatter over genuinely bad routes.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig16_stochastic_transfer.png")


ALL.update({"stochastic": fig_stochastic, "stochastic_transfer": fig_stochastic_transfer})


# --------------------------------------------------------------- figure 17 --
def fig_evolution():
    r"""Study 1: the imitation payoff landscape, lattice versus real network.

    Plots $D(\alpha) = C_A - C_S$, the journey-time penalty for being the
    altruist.  Imitation pushes $\alpha$ down wherever $D > 0$ and up wherever
    $D < 0$, so the sign of this curve is the whole dynamics.
    """
    from poi.evolution import invasion_signs, rest_points, settle

    j = load(os.path.join(RESULTS, "joint_surface.npz"))
    r = load(os.path.join(RESULTS, "real_networks.npz"))
    names = [str(x) for x in r["names"]]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))

    # (a) lattice: the penalty is positive everywhere and grows with ignorance
    ax = axes[0]
    al, w = j["alpha"], j["omega"]
    CA, CS = nanmean(j["CA"], axis=2), nanmean(j["CS"], axis=2)
    picks = [0, 6, 12, 18]
    cols = ordinal(len(picks), ORANGE_RAMP)
    for c, k in zip(cols, picks):
        D = CA[1, k] - CS[1, k]
        ax.plot(al, D, color=c, lw=1.9, label=rf"$\omega={w[k]:.2f}$")
    ax.set_yscale("log")
    leg = ax.legend(title="ignorance", loc="upper right", frameon=False,
                    fontsize=8, title_fontsize=8, handlelength=1.4, labelspacing=.3)
    leg._legend_box.align = "left"
    ax.annotate("", xy=(0.20, 0.50), xytext=(0.62, 0.50), xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
    ax.text(0.41, 0.535, "imitation drives $\\alpha$ this way", transform=ax.transAxes,
            ha="center", fontsize=8, color=INK_2)
    ax.set_xlabel(r"altruistic fraction $\alpha$")
    ax.set_ylabel(r"$C_A - C_S$   (altruist's penalty)")
    ax.set_title("a.  Lattice: altruists always pay,\n      and ignorance raises the bill", loc="left")
    ax.set_ylim(0.3, 30)

    # (b) Sioux Falls: the penalty turns into a bonus once drivers are ignorant
    ax = axes[1]
    al2, w2 = r["alpha"], r["omega"]
    picks2 = [0, 2, 4, 6, 8]
    cols2 = ordinal(len(picks2), BLUE_RAMP)
    for c, k in zip(cols2, picks2):
        D = r["CA"][0, k] - r["CS"][0, k]
        ax.plot(al2, D, color=c, lw=1.9, label=rf"$\omega={w2[k]:.1f}$")
        for q in rest_points(al2, D):
            if not q.stable:
                continue
            if q.kind == "mixed":
                ax.plot([q.alpha], [0.0], "o", color=c, ms=6.5, mec=SURFACE, mew=1.1, zorder=6)
            elif q.kind == "altruistic":
                # Altruism fixates: there is no interior rest point to mark.
                ax.plot([1.0], [0.0], ">", color=c, ms=7, mec=SURFACE, mew=1.0,
                        clip_on=False, zorder=6)
    ax.axhline(0.0, color=INK, lw=1.0)
    leg = ax.legend(title="ignorance", loc="lower right", frameon=False, ncol=2,
                    fontsize=8, title_fontsize=8, handlelength=1.4, labelspacing=.3,
                    columnspacing=1.1)
    leg._legend_box.align = "left"
    ax.set_ylim(-1.55, 1.35)
    ax.set_xlabel(r"altruistic fraction $\alpha$")
    ax.set_ylabel(r"$C_A - C_S$   (altruist's penalty)")
    ax.set_title("b.  Sioux Falls: below the line the\n      altruist is the faster driver", loc="left")

    # (c) what the settled population is worth
    ax = axes[2]
    width = 0.26
    for i, (n, c) in enumerate(zip(names, SERIES)):
        gains, alphas = [], []
        for k in range(w2.size):
            D = r["CA"][i, k] - r["CS"][i, k]
            e = settle(al2, D, r["C"][i, k], alpha0=0.5)
            gains.append(100 * e.gain)
            alphas.append(e.alpha_star)
        ax.plot(100 * w2, gains, color=c, lw=1.9, marker="o", ms=3.4)
        lab = {"SiouxFalls": "Sioux Falls",
               "Eastern-Massachusetts": "E. Massachusetts"}.get(n, n)
        kk = [2, 8, 6][i]
        direct_label(ax, 100 * w2[kk], gains[kk], lab, c, dy=[-13, -13, -13][i])
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.set_xlabel(r"ignorance $\omega$ (%)")
    ax.set_ylabel("gain captured with no enforcement (%)")
    ax.set_title("c.  Ignorance buys real efficiency\n      that nobody has to impose", loc="left")
    ax.set_ylim(-8, 108)

    fig.suptitle(
        "Altruism is evolutionarily doomed on the lattice and self-sustaining in a real city",
        x=0.008, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.925,
        "Drivers copy whichever class got home sooner. The altruistic fraction is then a state variable, not a dial: "
        r"it falls wherever $C_A > C_S$ and rises wherever $C_A < C_S$." "\n"
        "In b, a dot marks where the mixture settles; an arrow marks altruism taking over completely.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.875))
    _save(fig, "fig17_evolution.png")


# --------------------------------------------------------------- figure 18 --
def fig_evolution_stochastic():
    r"""Where random error makes altruism pay for itself.

    The same imitation dynamics as figure 17, now over the $(\sigma, \rho)$
    plane: does a lone altruist beat the crowd when every driver misjudges
    travel times independently, or only when they all misjudge them together?
    """
    from poi.evolution import invasion_signs, settle

    e = load(os.path.join(RESULTS, "evolution_stochastic.npz"))
    sg, rh, al = e["sigma"], e["rho"], e["alpha"]
    C, CA, CS = (nanmean(e[k], axis=3) for k in ("C", "CA", "CS"))

    A = np.array([[settle(al, CA[i, j] - CS[i, j], C[i, j], alpha0=0.5).alpha_star
                   for i in range(sg.size)] for j in range(rh.size)])
    G = np.array([[settle(al, CA[i, j] - CS[i, j], C[i, j], alpha0=0.5).gain
                   for i in range(sg.size)] for j in range(rh.size)])

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))

    # (a) the settled fraction over the plane
    ax = axes[0]
    im = ax.imshow(A, origin="lower", aspect="auto", cmap=SEQ_BLUE, vmin=0.0, vmax=1.0,
                   extent=(-0.5, sg.size - 0.5, -0.5, rh.size - 0.5))
    for j in range(rh.size):
        for i in range(sg.size):
            if A[j, i] > 1e-3:
                ax.text(i, j, f"{A[j, i]:.2f}", ha="center", va="center", fontsize=7.5,
                        color=SURFACE if A[j, i] > 0.55 else INK)
    ax.set_xticks(range(sg.size))
    ax.set_xticklabels([f"{v:g}" for v in sg], fontsize=8)
    ax.set_yticks(range(rh.size))
    ax.set_yticklabels([f"{v:g}" for v in rh], fontsize=8)
    ax.set_xlabel(r"error magnitude $\sigma$")
    ax.set_ylabel(r"error correlation $\rho$")
    ax.set_title("a.  Altruism survives only where\n      drivers are wrong together", loc="left")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"settled altruistic fraction $\alpha^*$", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    # (b) the payoff curves that produce it
    ax = axes[1]
    i0 = int(np.argmin(abs(sg - 0.3)))
    for j, c, lab in [(0, ORANGE, r"independent  $\rho=0$"),
                      (rh.size // 2, VIOLET, rf"$\rho={rh[rh.size // 2]:g}$"),
                      (rh.size - 1, BLUE, r"shared  $\rho=1$")]:
        D = CA[i0, j] - CS[i0, j]
        ax.plot(al, D, color=c, lw=2.0, marker="o", ms=3.4, label=lab)
        r = settle(al, D, C[i0, j], alpha0=0.5)
        if r.alpha_star > 1e-3:
            ax.plot([r.alpha_star], [0.0], "o", color=c, ms=7.5, mec=SURFACE, mew=1.2,
                    zorder=6)
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=1.5)
    ax.set_xlabel(r"altruistic fraction $\alpha$")
    ax.set_ylabel(r"$C_A - C_S$   (altruist's penalty)")
    ax.set_title(rf"b.  At $\sigma={sg[i0]:g}$: shared error turns" "\n"
                 "      the penalty into a head start", loc="left")

    # (c) what the settled population is worth
    ax = axes[2]
    dead = []
    for j, c in enumerate(ordinal(rh.size)):
        alive = G[j].max() > 1e-3
        ax.plot(sg, 100 * G[j], color=c if alive else MUTED, lw=1.9 if alive else 1.2,
                marker="o", ms=3.4 if alive else 2.6, zorder=3 if alive else 2)
        if alive:
            k = int(np.argmax(G[j] > 0.5 * G[j].max()))
            direct_label(ax, sg[k], 100 * G[j][k], rf"$\rho={rh[j]:g}$", c,
                         dx=0.015, dy=-13 if j == rh.size - 1 else 12)
        else:
            dead.append(f"{rh[j]:g}")
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.text(0.97, 0.155, rf"$\rho$ = {', '.join(dead)}: altruism never survives",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK_2)
    ax.set_ylim(-9, 112)
    ax.set_xlabel(r"error magnitude $\sigma$")
    ax.set_ylabel("gain captured with no enforcement (%)")
    ax.set_title("c.  How much efficiency arrives\n      for free", loc="left")

    fig.suptitle(
        "Shared uncertainty is what makes unselfishness pay for itself",
        x=0.008, y=0.995, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.008, 0.905,
        "When every driver misjudges the same road the same way, the altruist's detour lands on roads the "
        "crowd has mistakenly avoided, so being\nunselfish and being right coincide. Independent errors "
        "already spread the traffic, leaving the altruist nothing to gain.",
        ha="left", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.865))
    _save(fig, "fig18_evolution_stochastic.png")


ALL.update({"evolution": fig_evolution, "evolution_stochastic": fig_evolution_stochastic})


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    use_style()
    for name in sys.argv[1:] or list(ALL):
        print(f"[{name}]", flush=True)
        ALL[name]()
