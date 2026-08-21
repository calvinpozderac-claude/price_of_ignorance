"""Idiosyncratic driver uncertainty: stochastic user equilibrium with tunable correlation.

Both source papers model ignorance as a *systematic bias*: every driver perceives
the same wrong cost, in the same direction.  Real perception error is partly
idiosyncratic -- each driver is wrong in their own way -- and partly shared,
because drivers read the same signs and the same navigation app.  This module
spans the two with one parameter.

The error model
---------------
Driver type ``k`` perceives link ``e`` as costing ``c_e(x_e) + eps^k_e`` with

    eps^k_e = sigma * ( sqrt(rho) * xi_e  +  sqrt(1 - rho) * eta^k_e )

where ``xi_e`` is drawn once per link and shared by everyone, and ``eta^k_e`` is
drawn independently per type.  Both are standard normal, so ``Var(eps) = sigma^2``
for every ``rho``: **sigma sets the magnitude and rho sets the correlation, and
they move independently**.

* ``rho = 1`` -- everyone wrong identically.  This is the papers' regime (a
  random bias rather than their tuned one, but the same *kind* of error).
* ``rho = 0`` -- textbook stochastic user equilibrium: independent errors, no
  shared component.

Why this stays convex
---------------------
The error is **additive and flow-independent**, which is what preserves the
potential.  Writing ``f^k`` for type ``k``'s flow and ``x = sum_k f^k``, every
type's Jacobian row is ``c'(x)`` -- identical across types, hence symmetric --
so a potential exists:

    Phi({f^k}) = sum_e integral_0^{x_e} c_e  +  sum_k sum_e delta^k_e f^k_e

with ``delta^k_e = eps^k_e`` for a selfish type and, for an altruistic one,
``delta^k_e = -disc_e + eps^k_e / (beta + 1)`` (the rescaled marginal social cost
of a mis-perceived link; see :mod:`poi.bpr`).  For affine costs ``beta = 1`` and
this reduces to the familiar ``(a + eps)/2``.

The contrast matters: randomising *which type a road is*, or its capacity,
perturbs ``b_e`` and so makes the Jacobian rows differ.  That has **no**
potential and would need a variational-inequality solver.  Only flow-independent
error keeps the problem convex.

Two independent implementations
-------------------------------
* :func:`solve_stochastic` / :func:`solve_stochastic_bpr` discretise the driver
  population into ``K`` types with Gaussian link errors.  Finite ``K`` is itself a
  source of correlation -- a handful of types is a handful of coordinated groups
  -- so ``K`` must be pushed until the answer stops moving.
* :func:`dial_logit` solves the ``rho = 0`` continuum exactly for Gumbel path
  errors, by Dial's algorithm on the acyclic lattice.  No Monte-Carlo error and
  no path enumeration, which makes it a clean check on the ``K``-type results.
"""

from __future__ import annotations

from dataclasses import dataclass

import clarabel
import numpy as np
import scipy.sparse as sp

from .lattice import Network
from .qp import DEFAULT_RIDGE, _equality_block

__all__ = [
    "draw_errors",
    "StochasticSolution",
    "solve_stochastic",
    "solve_stochastic_bpr",
    "dial_logit",
]


def draw_errors(n_edges, K, sigma, rho, rng=None, mask=None) -> np.ndarray:
    """``(K, n_edges)`` perception errors with magnitude ``sigma``, correlation ``rho``.

    ``mask`` restricts the error to real roads (zero-cost connector links carry
    none).  Variance is ``sigma**2`` for every ``rho``.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = np.random.default_rng(rng)
    shared = rng.standard_normal(n_edges)
    own = rng.standard_normal((K, n_edges))
    eps = sigma * (np.sqrt(rho) * shared[None, :] + np.sqrt(1.0 - rho) * own)
    if mask is not None:
        eps = eps * np.asarray(mask, dtype=float)[None, :]
    return eps


@dataclass
class StochasticSolution:
    sigma: float
    rho: float
    K: int
    alpha: float
    x: np.ndarray
    total_cost: float
    cost_altruist: float
    cost_selfish: float
    status: str


def _class_table(net, eps, alpha, lam, disc):
    """(share, delta) for each driver class: selfish and altruistic types."""
    K = eps.shape[0]
    out = []
    for k in range(K):
        if alpha > 1e-12:
            out.append((alpha / K, -disc + lam * eps[k], True))
        if alpha < 1.0 - 1e-12:
            out.append(((1.0 - alpha) / K, eps[k], False))
    return out


def solve_stochastic(
    net: Network,
    sigma: float,
    rho: float,
    K: int,
    alpha: float = 0.0,
    rng=None,
    true_net: Network | None = None,
    tol: float = 1e-10,
    ridge: float = DEFAULT_RIDGE,
) -> StochasticSolution:
    """Stochastic mixed equilibrium on an affine-cost network, as a multi-class QP.

    ``K`` driver types each carry ``1/K`` of the demand and draw their own link
    errors.  Outcomes are scored with ``true_net`` (defaults to ``net``).
    """
    if K < 1:
        raise ValueError("K must be at least 1")
    truth = true_net or net
    m = net.n_edges
    eps = draw_errors(m, K, sigma, rho, rng, mask=net.is_real_road)
    # Affine costs: lambda = 1/2 and the altruist discount is a/2.
    classes = _class_table(net, eps, alpha, 0.5, 0.5 * net.a)
    n = len(classes)

    Aeq1, beq1 = _equality_block(net, net.demand())
    n_eq = Aeq1.shape[0]

    # Variables are [f^1 ... f^n, x] with the coupling x = sum_k f^k imposed as
    # a constraint.  Writing the quadratic on x alone keeps P diagonal with m
    # nonzeros; stacking it as an n-by-n block matrix of diag(b) instead costs
    # n^2 * m nonzeros and its factorisation runs out of memory by K ~ 100.
    nv = n * m + m
    P = sp.diags(np.concatenate([np.zeros(n * m), net.b]), format="csc")
    q = np.concatenate([d for _, d, _ in classes] + [net.a])

    couple = sp.hstack([sp.hstack([sp.eye(m, format="csc")] * n),
                        -sp.eye(m, format="csc")], format="csc")
    A_eq = sp.vstack([
        sp.hstack([sp.block_diag([Aeq1] * n, format="csc"),
                   sp.csc_matrix((n * n_eq, m))], format="csc"),
        couple,
    ], format="csc")
    b_eq = np.concatenate(
        [np.concatenate([s * beq1 for s, _, _ in classes]), np.zeros(m)]
    )
    nonneg = sp.hstack([-sp.eye(n * m, format="csc"), sp.csc_matrix((n * m, m))],
                       format="csc")
    A = sp.vstack([A_eq, nonneg], format="csc")
    b = np.concatenate([b_eq, np.zeros(n * m)])
    P = P + ridge * sp.eye(nv, format="csc")

    st = clarabel.DefaultSettings()
    st.verbose = False
    for f in ("tol_gap_abs", "tol_gap_rel", "tol_feas"):
        setattr(st, f, tol)
    sol = clarabel.DefaultSolver(
        sp.triu(P, format="csc"), q, A, b,
        [clarabel.ZeroConeT(n * n_eq + m), clarabel.NonnegativeConeT(n * m)], st,
    ).solve()

    f = np.maximum(np.asarray(sol.x)[: n * m], 0.0).reshape(n, m)
    x = f.sum(0)
    c_true = truth.costs(x)
    alt = np.array([is_a for _, _, is_a in classes])
    fA = f[alt].sum(0) if alt.any() else np.zeros(m)
    fS = f[~alt].sum(0) if (~alt).any() else np.zeros(m)
    return StochasticSolution(
        sigma=float(sigma), rho=float(rho), K=int(K), alpha=float(alpha), x=x,
        total_cost=truth.total_cost(x),
        cost_altruist=float(fA @ c_true) / alpha if alpha > 1e-12 else float("nan"),
        cost_selfish=float(fS @ c_true) / (1 - alpha) if alpha < 1 - 1e-12 else float("nan"),
        status=str(sol.status),
    )


def solve_stochastic_bpr(net, sigma: float, rho: float, K: int, alpha: float = 0.0,
                         rng=None, **kw):
    """Stochastic mixed equilibrium on a BPR network, by multi-class Frank-Wolfe.

    Errors scale with each link's free-flow time, so ``sigma`` is a coefficient of
    variation: a driver's uncertainty about a 10-minute road exceeds their
    uncertainty about a 2-minute one.
    """
    from .bpr import frank_wolfe_multiclass

    eps = draw_errors(net.n_edges, K, sigma, rho, rng) * net.t0[None, :]
    # Keep perceived free-flow time positive: a driver never believes a road
    # takes negative time, and Dijkstra could not use such a weight anyway.
    eps = np.maximum(eps, -0.9 * net.t0[None, :])
    lam = 1.0 / (net.power + 1.0)
    classes = _class_table(net, eps, alpha, lam, net.altruist_discount)
    return frank_wolfe_multiclass(net, classes, alpha=alpha, **kw)


# ------------------------------------------------------------ logit SUE ------
def _topological_order(net: Network) -> np.ndarray:
    """Kahn's algorithm; raises if the network is not acyclic."""
    n = net.n_nodes
    indeg = np.zeros(n, dtype=np.int64)
    np.add.at(indeg, net.head, 1)
    out = {i: [] for i in range(n)}
    for e, (u, v) in enumerate(zip(net.tail, net.head)):
        out[int(u)].append((e, int(v)))
    stack = [i for i in range(n) if indeg[i] == 0]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for _, v in out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(order) != n:
        raise ValueError("network contains a cycle; Dial's algorithm needs a DAG")
    return np.asarray(order, dtype=np.int64)


def dial_logit(net: Network, theta: float, max_iter: int = 500, tol: float = 1e-10,
               true_net: Network | None = None):
    """Exact logit stochastic user equilibrium on an acyclic network (all selfish).

    Path flows follow ``exp(-theta * path cost)`` -- the ``rho = 0`` continuum for
    Gumbel path errors -- computed by Dial's forward/backward weight passes in log
    space, so the lattice's ``2**(2L)`` paths never need enumerating.  The
    equilibrium fixed point ``x = load(c(x))`` is reached by the method of
    successive averages.

    ``theta -> infinity`` recovers the deterministic user equilibrium; ``theta ->
    0`` sends every path the same flow.  Very large ``theta`` is numerically
    degenerate (the path weights underflow), so take the deterministic limit from
    the QP solvers instead.  Returns ``(x, total_cost, gap)``.
    """
    if theta <= 0:
        raise ValueError("theta must be positive")
    truth = true_net or net
    order = _topological_order(net)
    rank = np.empty(net.n_nodes, dtype=np.int64)
    rank[order] = np.arange(net.n_nodes)
    e_order = np.argsort(rank[net.tail], kind="stable")

    # MSA must start from a feasible flow: an all-or-nothing loading, not zero.
    # Averaging up from x = 0 leaves the flow short of the demand for ever.
    x = None
    gap = np.inf
    for it in range(1, max_iter + 1):
        c = net.costs(x if x is not None else np.zeros(net.n_edges))
        w = -theta * c
        # forward: log sum of exp(-theta * cost) over paths source -> v
        lw = np.full(net.n_nodes, -np.inf)
        lw[net.source] = 0.0
        for e in e_order:
            u, v = net.tail[e], net.head[e]
            if lw[u] > -np.inf:
                lw[v] = np.logaddexp(lw[v], lw[u] + w[e])
        # backward: log sum over paths u -> sink
        lz = np.full(net.n_nodes, -np.inf)
        lz[net.sink] = 0.0
        for e in e_order[::-1]:
            u, v = net.tail[e], net.head[e]
            if lz[v] > -np.inf:
                lz[u] = np.logaddexp(lz[u], w[e] + lz[v])
        y = np.exp(lw[net.tail] + w + lz[net.head] - lz[net.source])
        y[~np.isfinite(y)] = 0.0

        if x is None:
            x = y
            continue
        gap = float(np.abs(y - x).max())
        if gap < tol:
            break
        x = x + (y - x) / it              # method of successive averages
    return x, truth.total_cost(x), gap
