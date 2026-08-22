"""Does altruism survive if drivers can change their minds?

Every other sweep in this project fixes the altruistic fraction ``alpha`` by
hand.  That is a fair question to ask a planner, but not a fair question to ask
a population: the altruists are the ones paying (:mod:`poi.metrics`
``exploitation_gap``), so if drivers can watch their neighbours and copy
whoever got home sooner, ``alpha`` is not a dial -- it is a state variable with
its own dynamics.

The dynamics used here are the standard imitation / replicator equation for two
strategies.  Writing ``D(alpha) = C_A(alpha) - C_S(alpha)`` for the *experienced*
journey-time penalty of being altruistic,

    d alpha / dt  =  alpha * (1 - alpha) * (-D(alpha))  +  mu * (1/2 - alpha)

The first term is imitation: altruists grow when they are the faster class.  The
second is a small mutation rate ``mu`` -- drivers who switch for reasons outside
the model -- which makes the boundaries ``alpha = 0`` and ``alpha = 1`` leaky, so
the question "can altruism *invade*?" has an answer rather than being decided by
the initial condition.

Because ``D`` is measured on a grid of ``alpha``, everything below works from a
sampled curve ``(alphas, D)`` and interpolates.  Nothing here re-solves a
network, so the same routines apply to the lattice QP, the BPR Frank-Wolfe
networks and the stochastic solver alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "RestPoint",
    "EvolutionResult",
    "rest_points",
    "invasion_signs",
    "evolve",
    "settle",
]

TOL = 1e-12


@dataclass(frozen=True)
class RestPoint:
    """A zero of the imitation dynamics on ``[0, 1]``."""

    alpha: float
    stable: bool
    kind: str  # "selfish", "altruistic" or "mixed"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RestPoint({self.alpha:.4f}, {'stable' if self.stable else 'unstable'}, {self.kind})"


@dataclass
class EvolutionResult:
    """Outcome of running the dynamics to steady state."""

    alpha_star: float
    cost: float
    cost_anarchy: float
    cost_optimum: float
    points: list[RestPoint] = field(default_factory=list)
    converged: bool = True

    @property
    def gain(self) -> float:
        """Share of the achievable gain that the *settled* population realises.

        ``1`` means evolution lands on the social optimum, ``0`` means it lands
        back at full anarchy.  Negative values mean the settled mixture is worse
        than everybody being selfish.
        """
        span = self.cost_anarchy - self.cost_optimum
        if abs(span) < 1e-12:
            return 0.0
        return (self.cost_anarchy - self.cost) / span


def _interp(alphas: np.ndarray, y: np.ndarray):
    """Linear interpolator over ``alphas`` that tolerates NaN at the ends.

    ``C_A`` is undefined at ``alpha = 0`` and ``C_S`` at ``alpha = 1``, so ``D``
    is normally NaN at both ends of the sampled grid.  The gap is bridged by
    extrapolating from the nearest two finite samples, which is the honest
    reading of "the penalty a *lone* altruist would pay".
    """
    a = np.asarray(alphas, dtype=float)
    d = np.asarray(y, dtype=float)
    ok = np.isfinite(d)
    if ok.sum() < 2:
        raise ValueError("need at least two finite samples of D(alpha)")
    a_ok, d_ok = a[ok], d[ok]

    def f(x):
        x = np.asarray(x, dtype=float)
        out = np.interp(x, a_ok, d_ok)
        # np.interp clamps outside the sampled range; extrapolate linearly
        # instead so the boundary behaviour is driven by the data's own slope.
        lo = x < a_ok[0]
        hi = x > a_ok[-1]
        if lo.any():
            s = (d_ok[1] - d_ok[0]) / (a_ok[1] - a_ok[0])
            out = np.where(lo, d_ok[0] + s * (x - a_ok[0]), out)
        if hi.any():
            s = (d_ok[-1] - d_ok[-2]) / (a_ok[-1] - a_ok[-2])
            out = np.where(hi, d_ok[-1] + s * (x - a_ok[-1]), out)
        return out

    return f


def rest_points(alphas, D, mu: float = 0.0, n_grid: int = 2001) -> list[RestPoint]:
    """Zeros of ``alpha(1-alpha)(-D) + mu(1/2 - alpha)`` on ``[0, 1]``.

    With ``mu = 0`` the endpoints are always rest points; each is reported
    stable when the neighbouring class cannot invade it.  Interior points are
    found by sign changes of the velocity on a fine grid and refined by
    bisection; a point is stable when the velocity crosses downwards.
    """
    f = _interp(alphas, D)
    grid = np.linspace(0.0, 1.0, n_grid)

    def v(x):
        x = np.asarray(x, dtype=float)
        return x * (1.0 - x) * (-f(x)) + mu * (0.5 - x)

    pts: list[RestPoint] = []
    if mu <= TOL:
        d0, d1 = float(f(0.0)), float(f(1.0))
        pts.append(RestPoint(0.0, d0 > 0.0, "selfish"))
        pts.append(RestPoint(1.0, d1 < 0.0, "altruistic"))
        interior = grid[1:-1]
    else:
        interior = grid

    def _stable(root: float, h: float = 1e-4) -> bool:
        """Attracting iff the velocity points inward from both sides."""
        right = float(v(min(1.0, root + h)))
        left = float(v(max(0.0, root - h)))
        if root + h <= 1.0 and root - h >= 0.0:
            return right < 0.0 and left > 0.0
        return right < 0.0 if root - h < 0.0 else left > 0.0

    y = v(interior)
    sign = np.sign(y)
    # A root is either an exact zero of the sampled velocity or a strict sign
    # change between neighbours.  Both cases have to be handled: with a regular
    # grid and a piecewise-linear D, roots at round values such as 0.4 land
    # exactly on a sample and never register as a sign change.
    roots: list[float] = [float(interior[i]) for i in np.nonzero(sign == 0)[0]]
    for i in np.nonzero(sign[:-1] * sign[1:] < 0)[0]:
        lo, hi = float(interior[i]), float(interior[i + 1])
        ylo = float(y[i])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            ym = float(v(mid))
            if (ym < 0) == (ylo < 0):
                lo, ylo = mid, ym
            else:
                hi = mid
        roots.append(0.5 * (lo + hi))

    step = 1.0 / (n_grid - 1)
    for root in sorted(roots):
        if pts and any(abs(root - q.alpha) < step for q in pts):
            continue
        kind = "selfish" if root < 1e-6 else "altruistic" if root > 1 - 1e-6 else "mixed"
        pts.append(RestPoint(root, _stable(root), kind))

    pts.sort(key=lambda r: r.alpha)
    return pts


def invasion_signs(alphas, D) -> tuple[float, float]:
    """``(D(0), D(1))``: can a lone altruist invade, and can a lone defector?

    ``D(0) < 0`` means a single altruist in a selfish crowd already gets home
    sooner, so altruism invades.  ``D(1) > 0`` means a single selfish driver in
    an altruistic crowd does *better*, so altruism is invadable -- the usual
    free-rider story.
    """
    f = _interp(alphas, D)
    return float(f(0.0)), float(f(1.0))


def evolve(alphas, D, alpha0: float = 0.5, mu: float = 0.0,
           dt: float = 0.05, max_steps: int = 200_000, tol: float = 1e-10):
    """Integrate the imitation dynamics from ``alpha0`` and return the path.

    Uses a fixed-step midpoint rule, which is ample for a scalar ODE whose right
    hand side is a piecewise-linear interpolant.
    """
    f = _interp(alphas, D)

    def v(x):
        return x * (1.0 - x) * (-float(f(x))) + mu * (0.5 - x)

    path = np.empty(max_steps + 1)
    path[0] = a = float(alpha0)
    n = max_steps
    for i in range(max_steps):
        k1 = v(a)
        k2 = v(min(1.0, max(0.0, a + 0.5 * dt * k1)))
        a = min(1.0, max(0.0, a + dt * k2))
        path[i + 1] = a
        if abs(k2) < tol:
            n = i + 1
            break
    return path[: n + 1]


def settle(alphas, D, C, alpha0: float = 0.5, mu: float = 0.0, **kw) -> EvolutionResult:
    """Run the dynamics to steady state and score the mixture it lands on.

    ``C`` is the average journey time sampled on the same ``alphas`` grid, used
    only to report what the settled population costs relative to full anarchy
    and to the social optimum.
    """
    alphas = np.asarray(alphas, dtype=float)
    C = np.asarray(C, dtype=float)
    path = evolve(alphas, D, alpha0=alpha0, mu=mu, **kw)
    a_star = float(path[-1])
    return EvolutionResult(
        alpha_star=a_star,
        cost=float(np.interp(a_star, alphas, C)),
        cost_anarchy=float(C[0]),
        cost_optimum=float(np.nanmin(C)),
        points=rest_points(alphas, D, mu=mu),
        converged=path.size < 200_001,
    )
