"""Closed-form mixed equilibrium for Pigou's two-road network.

Road 1 is constant, ``c_1(x) = k``.  Road 2 is congestible, ``c_2(x) = a + b x``
with ``b > 0``.  The paper's Fig. 1(a) is ``k = 1, a = 0, b = 1``.  A fraction
``alpha`` of the unit demand is altruistic (responds to the marginal social cost
``a + 2 b x``) and ``1 - alpha`` is selfish (responds to ``a + b x``).

Two usage levels organise everything:

``x_eq = min(1, (k - a) / b)``
    where road 2 stops looking cheaper to a *selfish* driver;
``x_opt = min(1, (k - a) / (2 b))``
    where road 2 stops looking cheaper to an *altruistic* driver.  This is also
    the socially optimal usage of road 2, so the system optimum is reached as
    soon as the equilibrium puts ``x_opt`` on road 2.

Since ``x_opt <= x_eq``, selfish drivers always overfill road 2 relative to the
optimum, and altruists respond by retreating to the slow road.  Solving the
mixed Wardrop conditions gives three regimes, keyed off

    alpha_star = 1 - x_opt

* ``alpha < alpha_star`` (and ``1 - alpha <= x_eq``): every altruist takes the
  slow road, every selfish driver takes the fast one; ``x_2 = 1 - alpha``.
* ``alpha >= alpha_star``: altruists top road 2 up to exactly ``x_opt`` and park
  the rest on road 1; the system sits at the social optimum.
* ``1 - alpha > x_eq``: road 2 saturates on selfish drivers alone and the
  selfish split across both roads, which are then equally fast.

For the canonical case this reads ``x_2 = 1 - alpha`` and ``C = 1 - alpha +
alpha^2`` up to ``alpha = 1/2``, and ``C = 3/4`` (the optimum) beyond it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PigouSolution", "solve_pigou", "pigou_curves"]


@dataclass
class PigouSolution:
    """Exact equilibrium of the two-road network at altruistic fraction ``alpha``."""

    alpha: float
    x_slow: float  # usage of the constant road
    x_fast: float  # usage of the congestible road
    fA_slow: float
    fA_fast: float
    fS_slow: float
    fS_fast: float
    total_cost: float
    cost_altruist: float  # travel time actually experienced by an altruist
    cost_selfish: float
    regime: str

    @property
    def exploitation_gap(self) -> float:
        return self.cost_altruist - self.cost_selfish


def solve_pigou(alpha: float, k: float = 1.0, a: float = 0.0, b: float = 1.0) -> PigouSolution:
    """Exact mixed equilibrium; see the module docstring for the derivation."""
    if b <= 0:
        raise ValueError("road 2 must be congestible (b > 0)")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if a >= k:
        raise ValueError("road 2 must be faster when empty (a < k)")

    x_eq = min(1.0, (k - a) / b)
    x_opt = min(1.0, (k - a) / (2.0 * b))
    alpha_star = 1.0 - x_opt

    if 1.0 - alpha > x_eq:
        # Road 2 saturates on selfish drivers alone; the rest spill onto road 1,
        # which by construction is exactly as fast, so both classes see cost k.
        regime = "saturated"
        x_fast = x_eq
        fS_fast, fA_fast = x_eq, 0.0
        fS_slow, fA_slow = (1.0 - alpha) - x_eq, alpha
    elif alpha < alpha_star:
        # Altruists have all retreated to the slow road.
        regime = "retreat"
        x_fast = 1.0 - alpha
        fS_fast, fS_slow = 1.0 - alpha, 0.0
        fA_fast, fA_slow = 0.0, alpha
    else:
        # Enough altruists that they can top road 2 up to its optimal usage.
        regime = "optimal"
        x_fast = x_opt
        fS_fast, fS_slow = 1.0 - alpha, 0.0
        fA_fast = x_opt - (1.0 - alpha)
        fA_slow = alpha - fA_fast

    x_slow = 1.0 - x_fast
    c_slow = k
    c_fast = a + b * x_fast
    total = x_slow * c_slow + x_fast * c_fast
    cost_A = (fA_slow * c_slow + fA_fast * c_fast) / alpha if alpha > 0 else float("nan")
    cost_S = (
        (fS_slow * c_slow + fS_fast * c_fast) / (1.0 - alpha) if alpha < 1 else float("nan")
    )

    return PigouSolution(
        alpha=float(alpha),
        x_slow=x_slow,
        x_fast=x_fast,
        fA_slow=fA_slow,
        fA_fast=fA_fast,
        fS_slow=fS_slow,
        fS_fast=fS_fast,
        total_cost=total,
        cost_altruist=cost_A,
        cost_selfish=cost_S,
        regime=regime,
    )


def pigou_curves(alphas, k: float = 1.0, a: float = 0.0, b: float = 1.0) -> dict:
    """Vectorised sweep of :func:`solve_pigou` over ``alphas``.

    Also returns ``efficiency``: the share of the total achievable improvement
    ``C(0) - C(1)`` that has been realised at each ``alpha``.  For the canonical
    parameters this is ``4 alpha (1 - alpha)`` below ``alpha = 1/2`` and 1 above,
    i.e. half the drivers being altruistic already captures the entire gain.
    """
    alphas = np.asarray(alphas, dtype=float)
    sols = [solve_pigou(al, k, a, b) for al in alphas]
    C = np.array([s.total_cost for s in sols])
    C0, C1 = solve_pigou(0.0, k, a, b).total_cost, solve_pigou(1.0, k, a, b).total_cost
    return {
        "alpha": alphas,
        "total_cost": C,
        "cost_altruist": np.array([s.cost_altruist for s in sols]),
        "cost_selfish": np.array([s.cost_selfish for s in sols]),
        "gap": np.array([s.exploitation_gap for s in sols]),
        "x_fast": np.array([s.x_fast for s in sols]),
        "efficiency": (C0 - C) / (C0 - C1) if C0 > C1 else np.zeros_like(C),
        "regime": [s.regime for s in sols],
        "alpha_star": 1.0 - min(1.0, (k - a) / (2.0 * b)),
        "C_eq": C0,
        "C_opt": C1,
    }
