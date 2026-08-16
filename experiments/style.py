"""Shared plotting style: one validated palette, applied by the job each color does.

Categorical slots are taken in fixed order from a palette validated with the
data-viz colour checks (all-pairs CVD deltaE 9.2, normal-vision 24.0 on the light
surface).  Only the first three slots are used for series, which is the
all-pairs-safe cap; ordered quantities (system size, altruistic fraction) use a
single-hue ordinal ramp instead of categorical hues, and signed quantities use
the blue-red diverging pair with a neutral grey midpoint.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- categorical: identity, fixed order, never cycled -----------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
BLUE, ORANGE, AQUA = SERIES
VIOLET = "#4a3aa7"
RED = "#e34948"

# --- chrome -----------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- sequential: magnitude, one hue, light to dark --------------------------
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
# Second sequential context takes the next categorical hue as its own ramp.
ORANGE_RAMP = [
    "#fde3d5", "#fbd0ba", "#f9bd9f", "#f6a984", "#f39069", "#ef7c4e",
    "#eb6834", "#d55a26", "#c04e1e", "#a54217", "#8a3512", "#70290d", "#571f09",
]

SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP)

# --- diverging: polarity, two hues + neutral grey midpoint ------------------
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red",
    ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f0a3a2", "#d03b3b", "#8f1f1f"],
)


def ordinal(n: int, ramp=BLUE_RAMP) -> list[str]:
    """``n`` evenly spaced steps of a one-hue ramp, for an *ordered* set of series.

    Starts at step 250 so the lightest series still clears 2:1 on the light
    surface, as the ordinal rule requires.
    """
    idx = np.linspace(3, len(ramp) - 1, n).round().astype(int)
    return [ramp[i] for i in idx]


def use_style() -> None:
    """Apply the shared rcParams: recessive chrome, thin marks, no top/right spines."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
            "figure.dpi": 130,
            "savefig.dpi": 190,
            "savefig.bbox": "tight",
        }
    )


def pc_line(ax, pc: float = 0.6447, label: bool = True, y: float = 0.97) -> None:
    """Mark the directed-percolation threshold, the paper's central quantity."""
    ax.axvline(pc, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=0)
    if label:
        ax.text(
            pc, y, r" $p_c$", transform=ax.get_xaxis_transform(),
            color=MUTED, fontsize=8, ha="left", va="top",
        )


def direct_label(ax, x, y, text, color, dx=0.0, dy=0.0, **kw) -> None:
    """Selective direct label so identity is never carried by colour alone."""
    ax.annotate(
        text, (x, y), textcoords="offset points", xytext=(4 + dx, dy),
        color=color, fontsize=8, fontweight="bold", va="center", **kw,
    )


def band(ax, x, lo, hi, color, alpha=0.16):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=1)


def mean_sem(a, axis):
    """Disorder mean and standard error, ignoring any failed solves."""
    m = np.nanmean(a, axis=axis)
    n = np.sum(np.isfinite(a), axis=axis)
    s = np.nanstd(a, axis=axis) / np.sqrt(np.maximum(n, 1))
    return m, s
