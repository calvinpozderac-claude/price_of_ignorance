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
    SEQ_ORANGE, SERIES, SURFACE, VIOLET, band, direct_label, mean_sem, ordinal,
    pc_line, use_style,
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
        direct_label(ax, 0.42, np.interp(0.42, al, m), lab.split("  ")[0], c, dy=6)
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
        direct_label(ax, 0.30, np.interp(0.30, al, m), lab.split("  ")[0], c, dy=7)
    ax.plot(al, al, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=0)
    ax.text(0.62, 0.58, "proportional\nreturn", color=MUTED, fontsize=7.5, rotation=33)
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
    gap = np.nanmean(d["CA"] - d["CS"], axis=1)
    regret = np.nanmean(d["CA"] - d["C"][:, :, [0]], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

    ax = axes[0]
    im = ax.imshow(eff, origin="lower", aspect="auto", extent=ext, cmap=SEQ_BLUE, vmin=0, vmax=1)
    cs = ax.contour(al, p, eff, levels=[0.5, 0.9, 0.99], colors="white", linewidths=0.9)
    ax.clabel(cs, fmt={0.5: "50%", 0.9: "90%", 0.99: "99%"}, fontsize=7)
    fig.colorbar(im, ax=ax, label="share of achievable gain")
    ax.set_title("(a)  Gain captured", loc="left")

    ax = axes[1]
    im = ax.imshow(gap, origin="lower", aspect="auto", extent=ext, cmap=SEQ_ORANGE)
    fig.colorbar(im, ax=ax, label=r"$C_A - C_S$")
    ax.set_title("(b)  How much altruists overpay", loc="left")

    ax = axes[2]
    v = np.nanmax(np.abs(regret))
    im = ax.imshow(regret, origin="lower", aspect="auto", extent=ext, cmap=DIVERGING, vmin=-v, vmax=v)
    ax.contour(al, p, regret, levels=[0.0], colors=[INK], linewidths=1.1)
    fig.colorbar(im, ax=ax, label=r"$C_A(\alpha) - C_{\rm eq}$")
    ax.set_title("(c)  Altruists vs. full anarchy", loc="left")
    ax.text(
        0.52, 0.30, "above the black line altruists\nare worse off than if nobody\nhad cooperated at all",
        transform=ax.transAxes, fontsize=7, color=INK, va="top",
    )

    for ax in axes:
        ax.axhline(pc, color="white", lw=1.0, ls=(0, (4, 3)))
        ax.text(0.012, pc, r" $p_c$", color="white", fontsize=8, va="bottom", fontweight="bold")
        ax.set_xlabel(r"altruistic fraction $\alpha$")
        ax.grid(False)
    axes[0].set_ylabel("fraction of congestible roads, $p$")

    fig.suptitle(
        f"The $(p,\\alpha)$ plane, {int(d['n_seeds'])} disorder realisations per point, $L={int(d['L'])}$",
        x=0.008, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig5_heatmaps.png")


# ---------------------------------------------------------------- figure 6 --
def fig_saturation():
    """How the saturation point and the exploitation peak track the threshold."""
    d = load(os.path.join(RESULTS, "altruism_grid.npz"))
    p, al, pc = d["p"], d["alpha"], float(d["pc"])

    astar = np.array(
        [[saturation_alpha(al, d["C"][i, j]) for j in range(d["C"].shape[1])] for i in range(p.size)]
    )
    gap = d["CA"] - d["CS"]
    maxgap = np.nanmax(gap, axis=2)
    poa = d["summary_poa"]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))

    ax = axes[0]
    m, s = mean_sem(poa, axis=1)
    ax.plot(p, m, color=BLUE)
    band(ax, p, m - s, m + s, BLUE)
    pc_line(ax, pc)
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel(r"price of anarchy $C_{\rm eq}/C_{\rm opt}$")
    ax.set_title("(a)  Inefficiency peaks at $p_c$", loc="left")

    ax = axes[1]
    m, s = mean_sem(astar, axis=1)
    ax.plot(p, m, color=ORANGE)
    band(ax, p, m - s, m + s, ORANGE)
    pc_line(ax, pc)
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel(r"saturating fraction $\alpha^*$")
    ax.set_title(r"(b)  Altruists needed to capture the gain", loc="left")
    ax.set_ylim(0, 1)

    ax = axes[2]
    m, s = mean_sem(maxgap, axis=1)
    ax.plot(p, m, color=AQUA)
    band(ax, p, m - s, m + s, AQUA)
    pc_line(ax, pc)
    ax.set_xlabel("fraction of congestible roads, $p$")
    ax.set_ylabel(r"peak $C_A - C_S$")
    ax.set_title("(c)  Exploitation is worst at $p_c$ too", loc="left")

    fig.suptitle(
        "Everything peaks at the percolation threshold — including the cost of being nice",
        x=0.008, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
        x=0.008, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
        ("x_0.0", "all selfish  ($\\alpha=0$)", SEQ_BLUE),
        ("fS_0.5", "selfish half  ($\\alpha=0.5$)", SEQ_BLUE),
        ("fA_0.5", "altruistic half  ($\\alpha=0.5$)", SEQ_ORANGE),
        ("x_1.0", "all altruistic  ($\\alpha=1$)", SEQ_BLUE),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 5.4))
    for ax, (key, title, cmap) in zip(axes.ravel(), panels):
        f = d[key][:n]
        scale = f / max(np.nanmax(f), 1e-12)
        order = np.argsort(scale[keep])
        lc = LineCollection(
            segs[keep][order],
            linewidths=0.2 + 2.2 * scale[keep][order],
            colors=cmap(0.12 + 0.88 * scale[keep][order]),
        )
        ax.add_collection(lc)
        ax.set_xlim(0, 2 * L)
        ax.set_ylim(0, L)
        ax.set_title(title, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(GRID)
        ax.set_aspect("equal")

    for ax in axes[:, 0]:
        ax.set_ylabel("flow: left to right", color=INK_2, fontsize=8)
    fig.suptitle(
        f"Where each class drives, at $p \\approx p_c$ ($L={L}$). "
        "Altruists spread onto the slow roads the selfish abandon.",
        x=0.008, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
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

if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    use_style()
    for name in sys.argv[1:] or list(ALL):
        print(f"[{name}]", flush=True)
        ALL[name]()
