"""Plain-language summary figures.  Run:  python experiments/summary_plots.py

The figures in ``plots.py`` are technical.  These are the same results drawn for
a reader who has not read the papers: no Greek letters, no jargon on the axes.
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

from poi.sweep import load  # noqa: E402
from style import (  # noqa: E402
    AQUA, AXIS, BLUE, GRID, INK, INK_2, MUTED, ORANGE, RED, SERIES,
    band, direct_label, mean_sem, nanmean, use_style,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS, FIGURES = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")


def _save(fig, name):
    path = os.path.join(FIGURES, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def fig_lattice_summary():
    """The three lattice findings, in plain language."""
    fine = load(os.path.join(RESULTS, "altruism_fine.npz"))
    ign = load(os.path.join(RESULTS, "ignorance_surface.npz"))
    joint = load(os.path.join(RESULTS, "joint_surface.npz"))
    i = 1  # the p = p_c slice

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))

    # --- 1. altruism helps, but altruists pay --------------------------------
    ax = axes[0]
    al = 100 * fine["alpha"]
    for key, c, lab in [("CA", ORANGE, "the unselfish ones"),
                        ("CS", AQUA, "the selfish ones"),
                        ("C", BLUE, "everyone, on average")]:
        m, se = mean_sem(fine[key][i], axis=0)
        ax.plot(al, m, color=c, lw=2.0 if key == "C" else 1.8,
                ls=(0, (5, 2)) if key == "C" else "-")
        band(ax, al, m - se, m + se, c)
    anarchy = float(nanmean(fine["C"][i][:, 0]))
    ax.axhline(anarchy, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(2, anarchy * 1.003, "everyone's journey time before\nanyone became unselfish",
            color=INK_2, fontsize=7.5, va="bottom")
    mA, _ = mean_sem(fine["CA"][i], axis=0)
    mS, _ = mean_sem(fine["CS"][i], axis=0)
    mC, _ = mean_sem(fine["C"][i], axis=0)
    ax.fill_between(al, anarchy, mA, where=mA >= anarchy, color=RED, alpha=0.12, lw=0)
    direct_label(ax, 62, np.interp(62, al, mA), "the unselfish ones", ORANGE, dy=11)
    direct_label(ax, 78, np.interp(78, al, mS), "the selfish ones", AQUA, dy=-10)
    direct_label(ax, 42, np.interp(42, al, mC), "everyone", BLUE, dy=-11)
    ax.set_xlabel("% of drivers who are unselfish")
    ax.set_ylabel("journey time (model units)")
    ax.set_title("1.  Unselfish drivers speed traffic up —\n      and pay for it themselves",
                 loc="left")
    ax.set_xlim(0, 106)

    # --- 2. confusion helps selfish drivers ---------------------------------
    ax = axes[1]
    p, w = ign["p"], ign["omega"]
    C = ign["C"][:, :, :, 0]
    PI = nanmean(C / C[:, [0], :], axis=2)
    j = int(np.argmin(abs(p - 0.6447)))
    ax.plot(100 * w, 100 * (PI[j] - 1), color=BLUE)
    ax.axhline(0.0, color=INK, lw=1.0)
    lo = int(np.argmin(PI[j]))
    ax.plot([100 * w[lo]], [100 * (PI[j][lo] - 1)], "o", color=BLUE, ms=6, zorder=5)
    ax.annotate(f"best: {100*(1-PI[j][lo]):.0f}% faster\nwhen drivers are\n{100*w[lo]:.0f}% confused",
                xy=(100 * w[lo], 100 * (PI[j][lo] - 1)), xytext=(0.06, 0.42),
                textcoords="axes fraction", color=INK_2, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9, shrinkA=2, shrinkB=5))
    ax.set_ylim(-8, 12)
    ax.text(2, 11.4, "past ~85% confused, journeys\nget much worse (off scale)",
            color=RED, fontsize=7.5, ha="left", va="top")
    ax.set_xlabel("how confused drivers are about which roads are fast (%)")
    ax.set_ylabel("change in everyone's journey time (%)")
    ax.set_title("2.  A little confusion actually helps —\n      it stops the pile-up on fast roads",
                 loc="left")

    # --- 3. both at once overshoots -----------------------------------------
    ax = axes[2]
    wj, alj, Cj = joint["omega"], joint["alpha"], nanmean(joint["C"], axis=2)
    jw = int(np.argmin(abs(wj - 2 / 3)))
    best = Cj[i].min()
    vals = [Cj[i][0, 0], Cj[i][0, -1], Cj[i][jw, 0], Cj[i][jw, -1]]
    labels = ["selfish\n+ informed", "unselfish\n+ informed", "selfish\n+ confused",
              "unselfish\n+ confused"]
    cols = [MUTED, AQUA, AQUA, RED]
    bars = ax.bar(range(4), [100 * (v / best - 1) for v in vals], color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
                f"+{100*(v/best-1):.1f}%", ha="center", fontsize=8.5,
                color=INK, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("slower than the best case (%)")
    ax.set_title("3.  Either fix alone works.\n      Both together is as bad as neither.", loc="left")
    ax.set_ylim(0, max(100 * (v / best - 1) for v in vals) * 1.28)

    fig.suptitle(
        "On the idealised grid: three findings",
        x=0.008, y=0.99, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "summary_lattice.png")


def fig_real_summary():
    """The real-network findings, in plain language."""
    d = load(os.path.join(RESULTS, "real_networks.npz"))
    names = [str(x) for x in d["names"]]
    pretty = {"SiouxFalls": "Sioux Falls", "Eastern-Massachusetts": "Eastern Massachusetts",
              "Anaheim": "Anaheim"}
    w, C = d["omega"], d["C"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))

    ax = axes[0]
    for i, (n, c) in enumerate(zip(names, SERIES)):
        y = 100 * (C[i, :, 0] / C[i, 0, 0] - 1)
        ax.plot(100 * w, y, color=c)
        k = [5, 7, 7][i]
        direct_label(ax, 100 * w[k], y[k], pretty[n], c, dy=[13, -12, -12][i])
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.text(2, 0.6, "above this line, confusion makes journeys longer",
            color=INK_2, fontsize=7.5, va="bottom")
    ax.set_xlabel("how confused drivers are (%)")
    ax.set_ylabel("change in everyone's journey time (%)")
    ax.set_title("4.  In real cities, confusion only hurts", loc="left")
    ax.margins(x=0.06, y=0.10)

    ax = axes[1]
    for i, (n, c) in enumerate(zip(names, SERIES)):
        y = 100 * (C[i, :, -1] / C[i, :, 0] - 1)
        ax.plot(100 * w, y, color=c)
        k = [7, 3, 6][i]
        direct_label(ax, 100 * w[k], y[k], pretty[n], c, dy=[13, -12, 11][i])
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.text(2, -0.4, "below this line, unselfish drivers help",
            color=INK_2, fontsize=7.5, va="top")
    ax.set_xlabel("how confused drivers are (%)")
    ax.set_ylabel("change if everyone turns unselfish (%)")
    ax.set_title("5.  And unselfish drivers always help,\n      most when drivers are confused",
                 loc="left")
    ax.margins(x=0.06, y=0.10)

    fig.suptitle(
        "On three actual road networks: the grid's warning does not apply",
        x=0.008, y=0.99, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "summary_real.png")


if __name__ == "__main__":
    use_style()
    fig_lattice_summary()
    fig_real_summary()
