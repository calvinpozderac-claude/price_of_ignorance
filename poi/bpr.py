"""Real road networks: BPR cost functions, TNTP data, and a Frank-Wolfe solver.

Why a new solver
----------------
The lattice study uses affine costs, so the mixed altruistic/selfish equilibrium
is a quadratic program.  Real traffic assignment uses the Bureau of Public Roads
function

    c_e(x) = t0_e * (1 + b_e * (x / cap_e) ** beta)

with ``beta = 4`` typically, which is convex but not quadratic.  The good news is
that the *structure* carries over unchanged.

Generalising the potential
--------------------------
The heterogeneous equilibrium is a variational inequality in which the selfish
respond to ``c(x)`` and altruists to the marginal social cost ``c(x) + x c'(x)``,
rescaled by a free constant ``lambda`` (an altruist only compares her own
perceived path costs, so any positive rescaling leaves her behaviour alone).  A
potential exists when the VI Jacobian is symmetric, i.e. when

    lambda * (2 c' + x c'') = c'

For any power law, ``x c'' = (beta - 1) c'``, so ``2 c' + x c'' = (beta + 1) c'``
and the condition has a *constant* solution

    lambda = 1 / (beta + 1)

which is the familiar ``lambda = 1/2`` at ``beta = 1`` (affine).  With that
choice the offset between the two classes' perceived costs is also constant:

    lambda * (c + x c') - c  =  - t0 * beta / (beta + 1)

so a potential exists, and it is Beckmann's integral plus a linear per-edge
discount that only altruists receive:

    Phi(fA, fS) = sum_e integral_0^{x_e} c_e(s) ds  -  sum_e disc_e * fA_e
    disc_e      = t0_e * beta / (beta + 1)

Sanity: at ``alpha = 0`` this is Beckmann's potential (user equilibrium) and at
``alpha = 1`` its minimiser is the system optimum.  Both are asserted in the
tests.  Altruists therefore behave exactly as if every road's *free-flow* time
were discounted by ``beta/(beta+1)`` -- they weight congestion more heavily
relative to distance, which is a pleasantly concrete reading of altruism.

The whole family requires a single ``beta`` across the network; a network with
mixed powers has no such ``lambda`` and no potential, and is rejected.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

__all__ = [
    "BPRNetwork",
    "read_tntp",
    "perceive_bpr",
    "frank_wolfe_mixed",
    "FWResult",
    "AVAILABLE",
]

AVAILABLE = (
    "SiouxFalls",
    "Eastern-Massachusetts",
    "Anaheim",
    "Berlin-Mitte-Center",
    "Chicago-Sketch",
    "Barcelona",
    "Winnipeg",
)


@dataclass
class BPRNetwork:
    """A road network with BPR link performance functions and an OD demand table."""

    tail: np.ndarray          # 0-based node indices
    head: np.ndarray
    capacity: np.ndarray
    t0: np.ndarray            # free-flow travel time
    b: np.ndarray             # BPR coefficient (usually 0.15)
    power: float              # BPR exponent, single value for the whole network
    n_nodes: int
    n_zones: int
    first_thru_node: int      # 1-based; nodes below it may not be passed through
    demand: dict              # origin (0-based) -> array over destinations
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return int(self.t0.size)

    @property
    def total_demand(self) -> float:
        return float(sum(q.sum() for q in self.demand.values()))

    def costs(self, x: np.ndarray) -> np.ndarray:
        """Realised travel time on each link."""
        return self.t0 * (1.0 + self.b * (np.maximum(x, 0.0) / self.capacity) ** self.power)

    def marginal_costs(self, x: np.ndarray) -> np.ndarray:
        """``d/dx [x c(x)] = c + x c'``, the true marginal social cost."""
        z = (np.maximum(x, 0.0) / self.capacity) ** self.power
        return self.t0 * (1.0 + self.b * (1.0 + self.power) * z)

    def beckmann(self, x: np.ndarray) -> float:
        """``sum_e integral_0^{x_e} c_e``, the user-equilibrium potential."""
        x = np.maximum(x, 0.0)
        return float(
            np.sum(
                self.t0
                * (x + self.b * x ** (self.power + 1.0)
                   / ((self.power + 1.0) * self.capacity**self.power))
            )
        )

    def total_cost(self, x: np.ndarray) -> float:
        """Total system travel time ``sum_e x_e c_e(x_e)``."""
        return float(np.sum(x * self.costs(x)))

    @property
    def altruist_discount(self) -> np.ndarray:
        """``t0 * beta / (beta + 1)``: the per-link discount altruists perceive."""
        return self.t0 * self.power / (self.power + 1.0)


# ------------------------------------------------------------------ TNTP I/O --
def _strip(line: str) -> str:
    return line.split("~")[0].strip()


def read_tntp(folder: str, name: str | None = None) -> BPRNetwork:
    """Read a network in the Transportation Networks (TNTP) benchmark format.

    ``folder`` is a directory containing ``*_net.tntp`` and ``*_trips.tntp``.
    """
    files = os.listdir(folder)
    netf = next(f for f in files if f.lower().endswith("_net.tntp"))
    tripf = next(f for f in files if f.lower().endswith("_trips.tntp"))
    name = name or os.path.basename(os.path.normpath(folder))

    meta, rows = {}, []
    with open(os.path.join(folder, netf)) as fh:
        body = False
        for line in fh:
            if not body:
                m = re.match(r"<([^>]+)>\s*(\S*)", line.strip())
                if m:
                    meta[m.group(1).upper()] = m.group(2)
                if "<END OF METADATA>" in line.upper():
                    body = True
                continue
            s = _strip(line).rstrip(";").strip()
            if not s:
                continue
            parts = s.split()
            if not parts[0].lstrip("-").replace(".", "", 1).isdigit():
                continue  # column-header comment line
            rows.append([float(v) for v in parts])

    arr = np.asarray(rows, dtype=float)
    tail = arr[:, 0].astype(np.int64) - 1
    head = arr[:, 1].astype(np.int64) - 1
    capacity = arr[:, 2]
    t0 = arr[:, 4]
    b = arr[:, 5]
    power = arr[:, 6]

    if np.ptp(power) > 1e-12:
        raise ValueError(
            f"{name}: BPR exponent varies across links ({power.min()}..{power.max()}). "
            "A single exponent is required for the mixed-class potential to exist."
        )
    if np.any(capacity <= 0):
        raise ValueError(f"{name}: {(capacity <= 0).sum()} links have non-positive capacity")

    n_nodes = int(float(meta.get("NUMBER OF NODES", head.max() + 1)))
    n_zones = int(float(meta.get("NUMBER OF ZONES", 0)))
    first_thru = int(float(meta.get("FIRST THRU NODE", 1)))

    demand: dict[int, np.ndarray] = {}
    with open(os.path.join(folder, tripf)) as fh:
        text = fh.read()
    text = text[text.upper().find("<END OF METADATA>") + len("<END OF METADATA>"):]
    for chunk in re.split(r"[Oo]rigin", text)[1:]:
        head_, _, rest = chunk.partition("\n")
        o = int(head_.strip()) - 1
        q = np.zeros(n_nodes)
        for d_s, v_s in re.findall(r"(\d+)\s*:\s*([0-9.eE+-]+)\s*;", rest):
            q[int(d_s) - 1] = float(v_s)
        q[o] = 0.0
        if q.sum() > 0:
            demand[o] = demand.get(o, np.zeros(n_nodes)) + q

    return BPRNetwork(
        tail=tail, head=head, capacity=capacity, t0=t0, b=b, power=float(power[0]),
        n_nodes=n_nodes, n_zones=n_zones, first_thru_node=first_thru,
        demand=demand, name=name,
        meta={"n_links": len(rows), **meta},
    )


# ---------------------------------------------------------------- ignorance --
def perceive_bpr(net: BPRNetwork, omega: float, shrink=("capacity",)) -> BPRNetwork:
    """Costs as *perceived* by drivers who are ignorant at level ``omega``.

    The lattice model blends each road's cost function toward the opposite road
    type.  A real network has no two types, so the natural generalisation is to
    shrink each link's parameters toward the network-wide (geometric) mean, in
    log space:

        cap_hat = exp( (1 - omega) * log cap + omega * mean(log cap) )

    ``omega = 0`` leaves the network alone; ``omega = 1`` makes every link look
    equally congestible, so drivers can no longer tell an arterial from a side
    street.  This reproduces the qualitative endpoints of the lattice model
    rather than being identical to it -- capacity is shrunk by default, on the
    view that drivers know roughly how far things are but misjudge how easily a
    road clogs.  Pass ``shrink=("capacity", "t0")`` to blur distance too.
    """
    if not 0.0 <= omega <= 1.0:
        raise ValueError("omega must lie in [0, 1]")
    bad = set(shrink) - {"capacity", "t0"}
    if bad:
        raise ValueError(f"cannot shrink unknown parameter(s): {sorted(bad)}")

    def blur(v):
        lg = np.log(v)
        return np.exp((1.0 - omega) * lg + omega * lg.mean())

    cap = blur(net.capacity) if "capacity" in shrink else net.capacity.copy()
    t0 = blur(net.t0) if "t0" in shrink else net.t0.copy()
    out = BPRNetwork(
        tail=net.tail, head=net.head, capacity=cap, t0=t0, b=net.b.copy(),
        power=net.power, n_nodes=net.n_nodes, n_zones=net.n_zones,
        first_thru_node=net.first_thru_node, demand=net.demand, name=net.name,
        meta={**net.meta, "omega": float(omega), "shrink": tuple(shrink), "perceived": True},
    )
    return out


# ------------------------------------------------------------- Frank-Wolfe --
def _dcost(net: BPRNetwork, x: np.ndarray) -> np.ndarray:
    """``c'(x)`` -- the diagonal Hessian of Beckmann's integral."""
    xs = np.maximum(x, 0.0)
    return (
        net.t0 * net.b * net.power * xs ** (net.power - 1.0) / net.capacity**net.power
    )


@dataclass
class FWResult:
    alpha: float
    f_altruist: np.ndarray
    f_selfish: np.ndarray
    x: np.ndarray
    total_cost: float
    cost_altruist: float
    cost_selfish: float
    rel_gap: float
    iterations: int
    converged: bool

    @property
    def exploitation_gap(self) -> float:
        return self.cost_altruist - self.cost_selfish


class _Loader:
    """All-or-nothing assignment, respecting centroid through-travel bans.

    Loading is vectorised by shortest-path-tree depth: every node at depth ``d``
    has its parent at depth ``d-1``, so processing depths from the deepest
    upwards respects the accumulation order while letting each level be a single
    ``np.add.at``.
    """

    def __init__(self, net: BPRNetwork):
        self.net = net
        self.n = net.n_nodes
        key = net.tail.astype(np.int64) * self.n + net.head
        self.pair_key, self.pair_of_edge = np.unique(key, return_inverse=True)
        self.pu = (self.pair_key // self.n).astype(np.int64)
        self.pv = (self.pair_key % self.n).astype(np.int64)
        # Nodes below FIRST THRU NODE are centroids: traffic may start or end
        # there but never pass through.
        self.restricted = np.arange(self.n) < (net.first_thru_node - 1)
        self.any_restricted = bool(self.restricted.any())
        self.tail_restricted = self.restricted[net.tail]

    def _graph(self, w: np.ndarray, origin: int = -1):
        ww = w
        if self.any_restricted:
            ww = np.where(self.tail_restricted & (self.net.tail != origin), np.inf, w)
        # Collapse parallel links to the cheapest one, matching what a path uses.
        order = np.lexsort((ww, self.pair_of_edge))
        pe = self.pair_of_edge[order]
        first = np.ones(order.size, dtype=bool)
        first[1:] = pe[1:] != pe[:-1]
        best_edge, pid = order[first], pe[first]
        wt = ww[best_edge]
        keep = np.isfinite(wt)
        g = sp.csr_matrix(
            (wt[keep] + 1e-12, (self.pu[pid[keep]], self.pv[pid[keep]])),
            shape=(self.n, self.n),
        )
        edge_for_pair = np.full(self.pair_key.size, -1, dtype=np.int64)
        edge_for_pair[pid[keep]] = best_edge[keep]
        return g, edge_for_pair

    def _load(self, y, dist, pred, vol, edge_for_pair):
        """Accumulate ``vol`` up the shortest-path trees onto link flows.

        ``dist``/``pred``/``vol`` are 2-D, one row per origin, so every depth
        level of every tree is handled in a single pass.
        """
        dist = np.atleast_2d(dist)
        pred = np.atleast_2d(pred).copy()
        vol = np.atleast_2d(vol).copy()
        pred[~np.isfinite(dist)] = -9999

        depth = np.zeros(pred.shape, dtype=np.int64)
        cur = pred.copy()
        for _ in range(self.n):
            live = cur >= 0
            if not live.any():
                break
            depth[live] += 1
            cur = np.where(live, np.take_along_axis(pred, np.maximum(cur, 0), 1), -1)

        rows_all = np.arange(pred.shape[0])[:, None] * np.ones(self.n, dtype=np.int64)
        cols_all = np.ones(pred.shape[0], dtype=np.int64)[:, None] * np.arange(self.n)
        for d in range(int(depth.max()), 0, -1):
            sel = (depth == d) & (vol > 0.0)
            if not sel.any():
                continue
            r, v = rows_all[sel], cols_all[sel]
            u = pred[r, v]
            e = edge_for_pair[
                np.searchsorted(self.pair_key, u.astype(np.int64) * self.n + v)
            ]
            amt = vol[r, v]
            np.add.at(y, e, amt)
            np.add.at(vol, (r, u), amt)

    def assign(self, w: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Load ``scale`` times the OD demand onto shortest paths under ``w``."""
        y = np.zeros(self.net.n_edges)
        if scale <= 0:
            return y
        origins = list(self.net.demand)
        if not self.any_restricted:
            # The graph does not depend on the origin: one build, one batched
            # Dijkstra, one batched loading pass.
            g, efp = self._graph(w)
            dist, pred = dijkstra(g, indices=origins, return_predecessors=True)
            vol = np.stack([self.net.demand[o] * scale for o in origins])
            self._load(y, dist, pred, vol, efp)
        else:
            # Centroid bans make the graph origin-specific, so these must loop.
            for o in origins:
                g, efp = self._graph(w, o)
                dist, pred = dijkstra(g, indices=o, return_predecessors=True)
                self._load(y, dist, pred, self.net.demand[o] * scale, efp)
        return y


def frank_wolfe_mixed(
    net: BPRNetwork,
    alpha: float,
    true_net: BPRNetwork | None = None,
    max_iter: int = 400,
    tol: float = 1e-5,
    verbose: bool = False,
) -> FWResult:
    """Mixed altruistic/selfish equilibrium on a BPR network, by Frank-Wolfe.

    ``net`` supplies the cost functions drivers *respond to* (pass the output of
    :func:`perceive_bpr` for ignorant drivers); ``true_net`` supplies the ones
    that determine realised travel times, defaulting to ``net``.

    Minimises ``Phi = Beckmann(x) - sum_e disc_e * fA_e`` (see the module
    docstring).  Convergence is reported as a relative duality gap, which upper
    bounds the remaining suboptimality of ``Phi``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    truth = true_net or net
    loader = _Loader(net)
    disc = net.altruist_discount

    def phi(fA, fS):
        return net.beckmann(fA + fS) - float(disc @ fA)

    # Start from a free-flow all-or-nothing loading for each class.
    c0 = net.costs(np.zeros(net.n_edges))
    fA = loader.assign(c0 - disc, alpha)
    fS = loader.assign(c0, 1.0 - alpha)

    rel_gap, it, converged = np.inf, 0, False
    ycA = ycS = None  # conjugate *target points* (feasible), not directions
    for it in range(1, max_iter + 1):
        x = fA + fS
        c = net.costs(x)
        gA, gS = c - disc, c
        yA = loader.assign(gA, alpha)
        yS = loader.assign(gS, 1.0 - alpha)

        # Duality gap uses the plain Frank-Wolfe vertex, not the conjugate one.
        gap = -(float(gA @ (yA - fA)) + float(gS @ (yS - fS)))
        denom = max(float(c @ x), 1e-12)
        rel_gap = gap / denom
        if verbose and (it % 25 == 0 or it == 1):
            print(f"  it {it:4d}  rel_gap {rel_gap:.3e}", flush=True)
        if rel_gap < tol:
            converged = True
            break

        # Conjugate Frank-Wolfe (Mitradjieva & Lindberg).  Blending the target
        # *points* keeps every iterate a convex combination of feasible points,
        # so flow conservation is preserved exactly; blending directions instead
        # can push flows negative and silently destroy conservation.
        if ycA is None:
            ycA, ycS = yA, yS
        else:
            hess = _dcost(net, x)
            pdx = (ycA - fA) + (ycS - fS)      # previous conjugate direction
            ndx = (yA - fA) + (yS - fS)        # plain FW direction
            D = float(hess @ (pdx * (ndx - pdx)))
            if abs(D) > 1e-30:
                th = float(hess @ (pdx * ndx)) / D
                th = min(max(th, 0.0), 0.9999)
            else:
                th = 0.0
            ycA = th * ycA + (1.0 - th) * yA
            ycS = th * ycS + (1.0 - th) * yS
        dA, dS = ycA - fA, ycS - fS

        # Exact line search: phi'(s) is increasing in s, so bisect on it.
        dx = dA + dS
        const = float(disc @ dA)

        def dphi(step):
            return float(net.costs(x + step * dx) @ dx) - const

        if dphi(1.0) <= 0.0:
            step = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if dphi(mid) > 0.0:
                    hi = mid
                else:
                    lo = mid
            step = 0.5 * (lo + hi)
        fA = fA + step * dA
        fS = fS + step * dS

    x = fA + fS
    c_true = truth.costs(x)
    total = truth.total_cost(x)
    cost_A = float(fA @ c_true) / (alpha * net.total_demand) if alpha > 1e-12 else float("nan")
    cost_S = (
        float(fS @ c_true) / ((1.0 - alpha) * net.total_demand)
        if alpha < 1.0 - 1e-12 else float("nan")
    )
    return FWResult(
        alpha=float(alpha), f_altruist=fA, f_selfish=fS, x=x, total_cost=total,
        cost_altruist=cost_A, cost_selfish=cost_S, rel_gap=float(rel_gap),
        iterations=it, converged=converged,
    )
