"""Convex-QP solvers for mixed altruistic/selfish traffic equilibria.

Formulation
-----------
Every road has an affine cost ``c_e(x) = a_e + b_e x``.  A fraction ``alpha`` of
the (unit) demand is *altruistic* and a fraction ``1 - alpha`` is *selfish*.
Writing ``f^A_e`` and ``f^S_e`` for the two classes' flows and ``x_e = f^A_e +
f^S_e`` for the total:

* a selfish driver routes along a path minimising the travel time she actually
  experiences, i.e. she responds to ``c_e(x_e) = a_e + b_e x_e``;
* an altruistic driver routes so as to reduce the *total* commute time, i.e.
  she responds to the marginal social cost
  ``d/dx [x c_e(x)] = a_e + 2 b_e x_e``.

The resulting heterogeneous Wardrop equilibrium is a variational inequality
whose operator, taken at face value, has the non-symmetric Jacobian
``[[2b, 2b], [b, b]]``; its symmetric part is indefinite, so there is no
potential and no convex program.  However, an altruist only ever *compares* her
own perceived path costs, so rescaling her perceived cost by any ``lambda > 0``
leaves the equilibrium untouched.  Rescaling gives the Jacobian
``[[2*lambda*b, 2*lambda*b], [b, b]]``, whose symmetric part is PSD iff
``8*lambda >= (2*lambda + 1)**2``, i.e. iff ``(2*lambda - 1)**2 <= 0``.  So
there is exactly one rescaling, ``lambda = 1/2``, at which the Jacobian becomes
the symmetric PSD matrix ``[[b, b], [b, b]]`` and a potential exists:

    Phi(f^A, f^S) = sum_e [ (a_e/2) f^A_e + a_e f^S_e + (b_e/2) (f^A_e + f^S_e)^2 ]

Minimising ``Phi`` over the two-commodity flow polytope therefore *is* the
mixed equilibrium, and it is a convex QP.  The two endpoints come out right:

* ``alpha = 0``:  ``Phi = sum_e [a_e x_e + b_e x_e^2 / 2]`` -- Beckmann's
  potential, whose minimiser is the paper's selfish equilibrium;
* ``alpha = 1``:  ``Phi = (1/2) sum_e [a_e x_e + b_e x_e^2] = C/2`` -- so the
  minimiser is exactly the paper's social optimum.

Degeneracy
----------
``Phi`` is only positive *semi*-definite, so the minimiser need not be unique.
The gradient is constant across the solution set, which pins ``b_e x_e`` (hence
``x_e`` on every congestible road) and both classes' equilibrium path costs, but
leaves usage on the constant roads free.  A tiny ``ridge * ||z||^2`` term
selects the minimum-norm solution deterministically; :func:`cost_range` then
certifies how much the reported cost could move across the whole solution set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import clarabel
import numpy as np
import scipy.sparse as sp

from .lattice import Network

__all__ = ["Solution", "solve_mixed", "solve_single_class", "solve_uniform", "cost_range"]

# Ridge weight that breaks ties among equivalent equilibria without measurably
# moving the solution (costs here are O(1) .. O(L)).
DEFAULT_RIDGE = 1e-9


@dataclass
class Solution:
    """Result of a mixed-equilibrium solve."""

    alpha: float
    f_altruist: np.ndarray
    f_selfish: np.ndarray
    x: np.ndarray
    status: str
    total_cost: float
    cost_altruist: float
    cost_selfish: float
    mu_selfish: float
    mu_altruist: float
    residual_selfish: float
    residual_altruist: float
    solve_time: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def exploitation_gap(self) -> float:
        """How much longer an altruist's commute is than a selfish driver's."""
        return self.cost_altruist - self.cost_selfish


def _clarabel_settings(verbose: bool, tol: float):
    s = clarabel.DefaultSettings()
    s.verbose = verbose
    s.tol_gap_abs = tol
    s.tol_gap_rel = tol
    s.tol_feas = tol
    s.tol_infeas_abs = tol
    s.tol_infeas_rel = tol
    s.max_iter = 500
    s.static_regularization_enable = True
    return s


def _equality_block(net: Network, demand: np.ndarray):
    """Node-balance rows with the redundant sink row dropped."""
    keep = np.ones(net.n_nodes, dtype=bool)
    keep[net.sink] = False
    return net.incidence[keep], demand[keep]


def solve_mixed(
    net: Network,
    alpha: float,
    demand: np.ndarray | None = None,
    ridge: float = DEFAULT_RIDGE,
    tol: float = 1e-12,
    verbose: bool = False,
    true_net: Network | None = None,
) -> Solution:
    """Solve the mixed altruistic/selfish equilibrium on ``net``.

    Parameters
    ----------
    net:
        Network whose affine costs drivers *respond to*.  Under user ignorance
        these are the perceived costs from :func:`poi.ignorance.perceive`, which
        differ from the costs that actually determine commute times.
    alpha:
        Fraction of demand that is altruistic, in ``[0, 1]``.
    demand:
        Node-balance vector; defaults to a unit source-to-sink demand.
    ridge:
        Weight of the tie-breaking ``||z||^2`` term (see module docstring).
    tol:
        Clarabel convergence tolerance.
    true_net:
        Network supplying the *true* cost functions used to score the outcome.
        Defaults to ``net`` (drivers are not mistaken about anything).  Behaviour
        -- who routes where, and the equilibrium conditions -- always follows
        ``net``; only the reported commute times follow ``true_net``.
    """
    if not 0.0 - 1e-12 <= alpha <= 1.0 + 1e-12:
        raise ValueError("alpha must lie in [0, 1]")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    d = net.demand() if demand is None else np.asarray(demand, dtype=float)
    m = net.n_edges

    Aeq1, beq1 = _equality_block(net, d)
    n_eq = Aeq1.shape[0]

    B = sp.diags(net.b)
    P = sp.bmat([[B, B], [B, B]], format="csc") + ridge * sp.eye(2 * m, format="csc")
    P = sp.triu(P, format="csc")
    q = np.concatenate([0.5 * net.a, net.a])

    Aeq = sp.bmat([[Aeq1, None], [None, Aeq1]], format="csc")
    beq = np.concatenate([alpha * beq1, (1.0 - alpha) * beq1])
    A = sp.vstack([Aeq, -sp.eye(2 * m, format="csc")], format="csc")
    b = np.concatenate([beq, np.zeros(2 * m)])
    cones = [clarabel.ZeroConeT(2 * n_eq), clarabel.NonnegativeConeT(2 * m)]

    solver = clarabel.DefaultSolver(P, q, A, b, cones, _clarabel_settings(verbose, tol))
    sol = solver.solve()
    status = str(sol.status)
    z = np.asarray(sol.x)
    fA = np.maximum(z[:m], 0.0)
    fS = np.maximum(z[m:], 0.0)
    x = fA + fS

    return _summarise(net, true_net or net, alpha, fA, fS, x, status, float(sol.solve_time))


def solve_single_class(
    net: Network,
    objective: str,
    demand: np.ndarray | None = None,
    ridge: float = DEFAULT_RIDGE,
    tol: float = 1e-12,
    verbose: bool = False,
):
    """Solve the homogeneous problem directly, as a cross-check on :func:`solve_mixed`.

    ``objective="optimum"`` minimises the total commute time
    ``sum_e [a_e x_e + b_e x_e^2]``; ``objective="equilibrium"`` minimises
    Beckmann's potential ``sum_e [a_e x_e + b_e x_e^2 / 2]``.  Returns the flow
    vector ``x``, the total cost, and the solver status.
    """
    if objective not in ("optimum", "equilibrium"):
        raise ValueError("objective must be 'optimum' or 'equilibrium'")
    d = net.demand() if demand is None else np.asarray(demand, dtype=float)
    m = net.n_edges
    Aeq, beq = _equality_block(net, d)

    scale = 2.0 if objective == "optimum" else 1.0
    P = sp.triu(sp.diags(scale * net.b).tocsc() + ridge * sp.eye(m, format="csc"), format="csc")
    q = net.a.copy()
    A = sp.vstack([Aeq, -sp.eye(m, format="csc")], format="csc")
    b = np.concatenate([beq, np.zeros(m)])
    cones = [clarabel.ZeroConeT(Aeq.shape[0]), clarabel.NonnegativeConeT(m)]

    solver = clarabel.DefaultSolver(P, q, A, b, cones, _clarabel_settings(verbose, tol))
    sol = solver.solve()
    x = np.maximum(np.asarray(sol.x), 0.0)
    return x, net.total_cost(x), str(sol.status)


_TINY = 1e-300  # keeps zero-cost connector roads from being pruned as "no edge"


def solve_uniform(
    net: Network,
    gamma: float,
    demand: np.ndarray | None = None,
    ridge: float = DEFAULT_RIDGE,
    tol: float = 1e-12,
) -> np.ndarray:
    """Equilibrium of a *uniform* population with altruism level ``gamma``.

    Every driver responds to ``c_e(x) + gamma * x c_e'(x) = a_e + (1 + gamma) b_e x``,
    interpolating between purely selfish (``gamma = 0``) and marginal-cost
    routing (``gamma = 1``).  This is the homogeneous cousin of
    :func:`solve_mixed`, where instead a *fraction* of drivers is fully
    altruistic; it is a single-class potential game, so it is a plain QP:

        minimise  sum_e [ a_e x_e + (1 + gamma) b_e x_e^2 / 2 ]

    Useful because it admits the closed-form substitution curve between altruism
    and ignorance (see :mod:`poi.ignorance`).  Returns the flow vector.
    """
    if gamma < -1.0:
        raise ValueError("gamma must exceed -1 for the objective to stay convex")
    d = net.demand() if demand is None else np.asarray(demand, dtype=float)
    m = net.n_edges
    Aeq, beq = _equality_block(net, d)
    P = sp.triu(
        sp.diags((1.0 + gamma) * net.b).tocsc() + ridge * sp.eye(m, format="csc"), format="csc"
    )
    A = sp.vstack([Aeq, -sp.eye(m, format="csc")], format="csc")
    b = np.concatenate([beq, np.zeros(m)])
    cones = [clarabel.ZeroConeT(Aeq.shape[0]), clarabel.NonnegativeConeT(m)]
    solver = clarabel.DefaultSolver(P, net.a.copy(), A, b, cones, _clarabel_settings(False, tol))
    return np.maximum(np.asarray(solver.solve().x), 0.0)


def _shortest_path_cost(net: Network, weights: np.ndarray) -> float:
    """Cheapest source-to-sink path under non-negative ``weights``.

    Parallel roads (Pigou's two links share a node pair) are collapsed by taking
    the cheaper one, which is what a shortest path would use anyway.
    """
    from scipy.sparse.csgraph import dijkstra

    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    key = net.tail.astype(np.int64) * net.n_nodes + net.head
    order = np.lexsort((w, key))
    key_s, w_s = key[order], w[order]
    first = np.ones(key_s.size, dtype=bool)
    first[1:] = key_s[1:] != key_s[:-1]
    k = key_s[first]
    g = sp.csr_matrix(
        (w_s[first] + _TINY, (k // net.n_nodes, k % net.n_nodes)),
        shape=(net.n_nodes, net.n_nodes),
    )
    dist = dijkstra(g, indices=net.source, min_only=True)
    return float(dist[net.sink])


def _summarise(net, true_net, alpha, fA, fS, x, status, solve_time) -> Solution:
    # What drivers respond to (may be a mistaken view of the network)...
    c_perc = net.costs(x)
    marg_perc = 0.5 * net.a + net.b * x  # altruist's rescaled perceived marginal
    # ...versus what they actually experience.
    c_true = true_net.costs(x)

    mu_S = _shortest_path_cost(net, c_perc)
    mu_A = _shortest_path_cost(net, marg_perc)

    tot = true_net.total_cost(x)
    cost_S = float(fS @ c_true) / (1.0 - alpha) if alpha < 1.0 - 1e-12 else float("nan")
    cost_A = float(fA @ c_true) / alpha if alpha > 1e-12 else float("nan")

    # Equilibrium check, in perceived terms: every used path of a class must sit
    # at that class's minimum perceived cost, so mean perceived == min perceived.
    res_S = (
        abs(float(fS @ c_perc) / (1.0 - alpha) - mu_S) if alpha < 1.0 - 1e-12 else 0.0
    )
    res_A = (
        abs(float(fA @ marg_perc) / alpha - mu_A) if alpha > 1e-12 else 0.0
    )

    return Solution(
        alpha=alpha,
        f_altruist=fA,
        f_selfish=fS,
        x=x,
        status=status,
        total_cost=tot,
        cost_altruist=cost_A,
        cost_selfish=cost_S,
        mu_selfish=mu_S,
        mu_altruist=mu_A,
        residual_selfish=res_S,
        residual_altruist=res_A,
        solve_time=solve_time,
    )


def cost_range(
    net: Network,
    sol: Solution,
    demand: np.ndarray | None = None,
    tol: float = 1e-7,
) -> tuple[float, float]:
    """Certified ``(min, max)`` total cost over the whole set of equilibria.

    Because the potential's gradient is constant on the solution set, that set
    is exactly the optimal face of the linear program ``min grad(z*)^T z`` over
    the flow polytope; and on that face every congestible road's usage is
    already fixed, so the total cost is *linear* in the remaining freedom.  Both
    extremes are therefore LPs, solved here with HiGHS.  A narrow interval
    certifies that the reported cost does not depend on the tie-breaking ridge.
    """
    from scipy.optimize import linprog

    d = net.demand() if demand is None else np.asarray(demand, dtype=float)
    m = net.n_edges
    alpha = sol.alpha
    Aeq1, beq1 = _equality_block(net, d)

    grad = np.concatenate([0.5 * net.a + net.b * sol.x, net.a + net.b * sol.x])
    z0 = np.concatenate([sol.f_altruist, sol.f_selfish])

    A_eq = sp.bmat([[Aeq1, None], [None, Aeq1]], format="csc")
    b_eq = np.concatenate([alpha * beq1, (1.0 - alpha) * beq1])

    # The solution set is the optimal face of  min grad^T z  intersected with
    # "the quadratic part is unchanged", i.e. x_e pinned on congestible roads.
    # Both are imposed with a small slack so the LP cannot be numerically empty;
    # the resulting interval is therefore conservative (an over-estimate).
    cong = np.flatnonzero(net.b > 0)
    pin = sp.csc_matrix(
        (
            np.ones(2 * cong.size),
            (np.tile(np.arange(cong.size), 2), np.concatenate([cong, cong + m])),
        ),
        shape=(cong.size, 2 * m),
    )
    A_ub = sp.vstack([grad.reshape(1, -1), pin, -pin], format="csc")

    # On that face C = sum_cong x_e^2 (fixed) + sum_const x_e, which is linear.
    obj = np.concatenate([net.a, net.a])
    fixed = float(np.sum(net.b * sol.x**2))

    # The slack must exceed the QP's own accuracy or the LP is spuriously empty.
    # Widening only makes the certified interval more conservative, never wrong.
    scale = 1.0 + abs(float(grad @ z0))
    for widen in (1.0, 10.0, 100.0, 1000.0):
        eps = tol * widen
        b_ub = np.concatenate(
            [[grad @ z0 + eps * scale], sol.x[cong] + eps, -(sol.x[cong] - eps)]
        )
        out = []
        for sense in (1.0, -1.0):
            r = linprog(
                sense * obj,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=(0, None),
                method="highs",
            )
            if not r.success:
                out = None
                break
            out.append(sense * r.fun + fixed)
        if out is not None:
            return (min(out), max(out))
    return (float("nan"), float("nan"))
