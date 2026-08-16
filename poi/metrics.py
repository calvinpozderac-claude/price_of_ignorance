"""Derived quantities describing how a population responds to added altruism.

Everything here takes the cost curves produced by sweeping the altruistic
fraction ``alpha`` from 0 to 1 on one fixed network, and answers the two
questions this project is about:

1. *Does adding altruists speed things up?*  -- :func:`efficiency`,
   :func:`saturation_alpha`, :func:`is_monotone`.
2. *Or do altruists just get exploited?* -- :func:`exploitation_gap`,
   :func:`altruist_regret`, :func:`free_rider_gain`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "efficiency",
    "saturation_alpha",
    "is_monotone",
    "max_backfire",
    "exploitation_gap",
    "altruist_regret",
    "free_rider_gain",
    "price_of_ignorance",
    "summarise_curve",
]

TOL = 1e-6


def price_of_ignorance(total_cost: np.ndarray) -> np.ndarray:
    """``C(alpha) / C(1)``: cost relative to the social optimum.

    At ``alpha = 0`` this is exactly the paper's price of anarchy, so the curve
    interpolates between the POA and 1.
    """
    C = np.asarray(total_cost, dtype=float)
    return C / C[-1]


def efficiency(total_cost: np.ndarray) -> np.ndarray:
    """Share of the achievable gain ``C(0) - C(1)`` realised at each ``alpha``.

    0 at ``alpha = 0`` and 1 at ``alpha = 1`` by construction; values above 1
    mark ``alpha`` where a *mixed* population beats the pure social optimum's
    average cost, and negative values mark altruism actively backfiring.
    """
    C = np.asarray(total_cost, dtype=float)
    span = C[0] - C[-1]
    if abs(span) < TOL:
        return np.zeros_like(C)
    return (C[0] - C) / span


def saturation_alpha(alphas: np.ndarray, total_cost: np.ndarray, tol: float = 1e-6) -> float:
    """Smallest ``alpha`` whose cost already matches the fully-altruistic cost.

    This is the point beyond which recruiting more altruists buys nothing.  In
    Pigou's network it is exactly 1/2.
    """
    alphas = np.asarray(alphas, dtype=float)
    C = np.asarray(total_cost, dtype=float)
    hit = np.flatnonzero(C <= C[-1] + tol)
    return float(alphas[hit[0]]) if hit.size else float(alphas[-1])


def is_monotone(total_cost: np.ndarray, tol: float = 1e-7) -> bool:
    """Whether the total commute time falls (weakly) with every added altruist."""
    return bool(np.all(np.diff(np.asarray(total_cost, dtype=float)) <= tol))


def max_backfire(total_cost: np.ndarray) -> float:
    """Largest rise in total cost caused by *increasing* the altruistic fraction.

    Formally ``max_{i < j} C[j] - C[i]``, clipped at zero: the worst outcome
    obtainable by converting selfish drivers into altruists.  Zero when altruism
    never hurts.  A positive value is a concrete instance of added altruism
    making the average commute worse -- possible because the mixed equilibrium
    minimises a potential, not the total cost.
    """
    C = np.asarray(total_cost, dtype=float)
    if C.size < 2:
        return 0.0
    running_min = np.minimum.accumulate(C)
    return float(max(0.0, np.max(C[1:] - running_min[:-1])))


def exploitation_gap(cost_altruist: np.ndarray, cost_selfish: np.ndarray) -> np.ndarray:
    """``C_A - C_S``: how much longer an altruist's commute is than a selfish one's.

    Provably non-negative: selfish drivers sit on the minimum-latency path by
    definition, so no other class can average below them.
    """
    return np.asarray(cost_altruist, dtype=float) - np.asarray(cost_selfish, dtype=float)


def altruist_regret(cost_altruist: np.ndarray, total_cost: np.ndarray) -> np.ndarray:
    """``C_A(alpha) - C(0)``: an altruist's commute versus everyone's under pure anarchy.

    Positive means the altruists are *worse off than they would have been in a
    world with no altruists at all* -- they have not merely failed to profit
    from their own cooperation, they have paid for it.
    """
    return np.asarray(cost_altruist, dtype=float) - float(np.asarray(total_cost)[0])


def free_rider_gain(cost_selfish: np.ndarray, total_cost: np.ndarray) -> np.ndarray:
    """``C(0) - C_S(alpha)``: how much a selfish driver gains from others' altruism."""
    return float(np.asarray(total_cost)[0]) - np.asarray(cost_selfish, dtype=float)


def summarise_curve(alphas, total_cost, cost_altruist, cost_selfish) -> dict:
    """Collapse one alpha-sweep into the scalar diagnostics used by the figures."""
    alphas = np.asarray(alphas, dtype=float)
    C = np.asarray(total_cost, dtype=float)
    CA = np.asarray(cost_altruist, dtype=float)
    CS = np.asarray(cost_selfish, dtype=float)

    gap = exploitation_gap(CA, CS)
    regret = altruist_regret(CA, C)
    interior = np.isfinite(gap)
    rises = np.diff(C)

    return {
        "poa": float(C[0] / C[-1]),
        "C_eq": float(C[0]),
        "C_opt": float(C[-1]),
        "alpha_star": saturation_alpha(alphas, C),
        "monotone": is_monotone(C),
        "max_rise": float(max(0.0, rises.max())) if rises.size else 0.0,
        "max_backfire": max_backfire(C),
        "max_gap": float(np.nanmax(gap)) if interior.any() else np.nan,
        "argmax_gap": float(alphas[np.nanargmax(np.where(interior, gap, -np.inf))])
        if interior.any()
        else np.nan,
        "max_regret": float(np.nanmax(regret)) if interior.any() else np.nan,
        "regret_positive_frac": float(np.mean(regret[interior] > 0)) if interior.any() else np.nan,
        "max_free_rider_gain": float(np.nanmax(free_rider_gain(CS, C))) if interior.any() else np.nan,
    }
