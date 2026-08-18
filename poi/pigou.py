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

__all__ = ["PigouSolution", "solve_pigou", "solve_pigou_ignorant", "pigou_curves"]


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

    return _two_road(alpha, k, 0.0, a, b, (k, 0.0), (a, b))


def _two_road(alpha, A1, B1, A2, B2, T1, T2):
    """Shared regime logic for two parallel roads.

    ``(A_i, B_i)`` are the *perceived* affine coefficients drivers plan against;
    ``T_i = (a_i, b_i)`` are the *true* ones that determine commute times.  Road
    1 is the slow road, road 2 the fast one.

    Selfish drivers equalise perceived cost ``A_i + B_i x_i``; altruists equalise
    perceived marginal cost ``A_i + 2 B_i x_i``.  Solving each on ``x_1 + x_2 = 1``:

        x2_S = (A1 + B1 - A2) / (B1 + B2)          selfish indifference point
        x2_A = (A1 + 2 B1 - A2) / (2 (B1 + B2))    altruist indifference point

    and ``x2_S - x2_A = (A1 - A2) / (2 (B1 + B2))``, so the selfish always want
    at least as much of road 2 as the altruists whenever ``A1 >= A2`` -- i.e.
    whenever road 1 really is the one that looks slower when empty.
    """
    denom = B1 + B2
    if denom <= 0:
        raise ValueError("at least one road must be congestible in perceived terms")
    x2_S = float(np.clip((A1 + B1 - A2) / denom, 0.0, 1.0))
    x2_A = float(np.clip((A1 + 2 * B1 - A2) / (2 * denom), 0.0, 1.0))
    alpha_star = 1.0 - x2_A

    if 1.0 - alpha > x2_S:
        # Road 2 saturates on selfish drivers alone; the rest spill onto road 1,
        # which by construction then has equal perceived cost.
        regime = "saturated"
        x_fast = x2_S
        fS_fast, fA_fast = x2_S, 0.0
        fS_slow, fA_slow = (1.0 - alpha) - x2_S, alpha
    elif alpha < alpha_star:
        regime = "retreat"
        x_fast = 1.0 - alpha
        fS_fast, fS_slow = 1.0 - alpha, 0.0
        fA_fast, fA_slow = 0.0, alpha
    else:
        regime = "optimal"
        x_fast = x2_A
        fS_fast, fS_slow = 1.0 - alpha, 0.0
        fA_fast = x2_A - (1.0 - alpha)
        fA_slow = alpha - fA_fast

    x_slow = 1.0 - x_fast
    c_slow = T1[0] + T1[1] * x_slow  # true commute times
    c_fast = T2[0] + T2[1] * x_fast
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


def solve_pigou_ignorant(alpha: float, omega: float) -> PigouSolution:
    """Pigou's two roads with both altruism ``alpha`` and ignorance ``omega``.

    True costs are ``c_1 = 1`` and ``c_2 = x``.  With ``u = omega / 2`` the
    perceived costs are ``(1 - u) + u x`` on the slow road and ``u + (1 - u) x``
    on the fast one, so ``B1 + B2 = 1`` and

        x2_S = 1 - u          (selfish target -- the paper's 1 - omega/2)
        x2_A = 1/2            (altruist target)

    The altruists' target is ``1/2`` for **every** ``omega``: in this network the
    minimiser of the *perceived* total cost coincides exactly with the minimiser
    of the true one.  Ignorance therefore never corrupts an altruist's aim here,
    which is special to Pigou -- on the lattice it does.
    """
    if not 0.0 <= omega <= 1.0:
        raise ValueError("omega must lie in [0, 1]")
    u = 0.5 * omega
    return _two_road(alpha, 1.0 - u, u, u, 1.0 - u, (1.0, 0.0), (0.0, 1.0))


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
